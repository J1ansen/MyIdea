"""阶段二：Top-ρ 异配候选池 + Gumbel-Softmax 可导稀疏提示邻接 EX。

整体流程（与 README §4 / idea 第二节阶段二严格对齐）：

    原图 ──► 计算每个节点局部异配度 h_local[i]
                │
                ├──► Top-ρ 取最高异配的节点构成 V_pool
                │
                ▼
    ┌───────────────────────────────────────────────┐
    │   仅 V_pool 内节点 ─► Gumbel-Softmax           │
    │     ├─ 与 P_homo (K1) 路由 → EX_homo [N, K1]   │
    │     └─ 与 P_hete (K2) 路由 → EX_hete [N, K2]   │
    └───────────────────────────────────────────────┘
                │
                ▼
    将 EX_homo / EX_hete 转换为 PyG 的 (edge_index, edge_type) 形式，
    供阶段三 PromptAwareGNNConv 在 (N+K) 节点上做异配感知消息传递。

关键设计：
    - 池外节点 EX 行恒为 0，物理含义：只对"被视为局部异配 / 需要拓扑修复"
      的节点引入提示边，避免无差别全图改图；
    - EX_homo / EX_hete 共享同一个 V_pool，保证两类提示在同一组节点上竞争，
      便于对比损失正确推理"同 prompt vs 异 prompt"；
    - hard 模式仍可反向传播（Straight-Through Gumbel），用于严格离散化拓扑分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#  低层工具：局部异配度 + Top-ρ 选择
# =============================================================================


def compute_local_heterophily(
    edge_index: torch.Tensor,
    num_nodes: int,
    y: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """每个节点的局部异配度，范围 [0, 1]，越大表示邻域越异配。

    两种估计模式：
        - **基于标签**（优先，更精准）：``h_local[i] = 1 − (邻居同类比例)``
          == 邻居异类边占比；
        - **基于特征**（兜底，标签缺失时使用）：``h_local[i] = 1 − (邻居平均余弦相似度)``。

    Args:
        edge_index: [2, E]，无向图请传入双向边。
        num_nodes: 节点总数 N。
        y: 可选标签 [N] 或 [N, 1]；若提供则走标签模式。
        x: 可选特征 [N, D]；标签缺失时用于兜底估计。

    Returns:
        scores: [N] float tensor，孤立节点为 0。
    """
    device = edge_index.device
    scores = torch.zeros(num_nodes, device=device)

    if edge_index.numel() == 0:
        return scores

    src, dst = edge_index[0], edge_index[1]

    # ---- 模式 A：基于标签 ----
    if y is not None:
        y_flat = y.view(-1)
        valid = (
            (src >= 0)
            & (dst >= 0)
            & (src < num_nodes)
            & (dst < num_nodes)
            & (y_flat[src] >= 0)
            & (y_flat[dst] >= 0)
        )
        src_v, dst_v = src[valid], dst[valid]
        if src_v.numel() == 0:
            return scores

        # 对每个目标节点 dst 统计"异类邻居 / 总邻居"
        hetero_edge = (y_flat[src_v] != y_flat[dst_v]).float()
        hetero_sum = torch.zeros(num_nodes, device=device)
        degree = torch.zeros(num_nodes, device=device)
        hetero_sum.index_add_(0, dst_v, hetero_edge)
        degree.index_add_(0, dst_v, torch.ones_like(hetero_edge))
        mask = degree > 0
        scores[mask] = hetero_sum[mask] / degree[mask]
        return scores

    # ---- 模式 B：基于特征兜底 ----
    if x is None:
        return scores

    valid = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    src_v, dst_v = src[valid], dst[valid]
    if src_v.numel() == 0:
        return scores

    x_norm = F.normalize(x, p=2, dim=-1, eps=1e-12)
    sim = (x_norm[src_v] * x_norm[dst_v]).sum(dim=-1).clamp(0.0, 1.0)
    homo_sum = torch.zeros(num_nodes, device=device)
    degree = torch.zeros(num_nodes, device=device)
    homo_sum.index_add_(0, dst_v, sim)
    degree.index_add_(0, dst_v, torch.ones_like(sim))
    mask = degree > 0
    scores[mask] = 1.0 - homo_sum[mask] / degree[mask]
    return scores


def select_top_rho_pool(
    heterophily_scores: torch.Tensor,
    rho: float,
    min_pool_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """选出异配度最高的 Top-ρ 节点作为待连接候选池 V_pool。

    Args:
        heterophily_scores: [N]，由 ``compute_local_heterophily`` 给出。
        rho: 取值 (0, 1]；≥1 表示不裁剪、全图入池。
        min_pool_size: 至少保留多少节点（防止极小图退化为空池）。

    Returns:
        pool_mask:    bool tensor [N]
        pool_indices: long tensor [|V_pool|]
    """
    num_nodes = heterophily_scores.size(0)
    if num_nodes == 0:
        empty = heterophily_scores.new_zeros(0, dtype=torch.long)
        return heterophily_scores.new_zeros(0, dtype=torch.bool), empty

    rho = float(rho)
    if rho >= 1.0:
        pool_mask = torch.ones(num_nodes, dtype=torch.bool, device=heterophily_scores.device)
        return pool_mask, torch.arange(num_nodes, device=heterophily_scores.device)

    pool_size = max(int(min_pool_size), int(round(num_nodes * rho)))
    pool_size = min(pool_size, num_nodes)
    _, pool_indices = torch.topk(heterophily_scores, k=pool_size, largest=True)
    pool_mask = torch.zeros(num_nodes, dtype=torch.bool, device=heterophily_scores.device)
    pool_mask[pool_indices] = True
    return pool_mask, pool_indices


# =============================================================================
#  低层：单方向 Gumbel 路由器
# =============================================================================


class GumbelRouter(nn.Module):
    """单方向 Gumbel-Softmax 提示路由器（用于 EX_homo 或 EX_hete 中的一路）。

    Args:
        feature_dim: 原图节点特征维度 d。
        prompt_dim:  提示原型维度 d_p。
        tau:         Gumbel 初始温度，越小越接近 one-hot。
        rho:         默认 Top-ρ 比例；前向时可通过 ``rho`` / ``pool_mask`` 覆盖。
    """

    def __init__(
        self,
        feature_dim: int,
        prompt_dim: int,
        tau: float = 1.0,
        rho: float = 1.0,
    ):
        super().__init__()
        self.tau = float(tau)
        self.rho = float(rho)
        # 把节点特征投影到提示空间，便于 logits = h_proj @ p^T 度量"匹配度"
        self.proj = nn.Linear(feature_dim, prompt_dim)
        nn.init.xavier_uniform_(self.proj.weight, gain=2.0)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def compute_pool_mask(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        y: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (pool_mask, pool_indices, local_heterophily)。"""
        scores = compute_local_heterophily(edge_index, num_nodes, y=y, x=x)
        pool_mask, pool_indices = select_top_rho_pool(
            scores, rho if rho is not None else self.rho
        )
        return pool_mask, pool_indices, scores

    def forward(
        self,
        node_features: torch.Tensor,
        prompt_features: torch.Tensor,
        hard: bool = False,
        edge_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
        pool_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        tau: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """Gumbel 路由：只有 V_pool 内节点对应行非零，池外恒为 0。

        当 ``pool_mask`` 已提供时，直接复用（多路路由器共享同一池的关键）。

        Args:
            tau: 当前退火温度；若为 None 则使用构造时设定的 ``self.tau``。
        """
        num_nodes = node_features.size(0)
        h_proj = self.proj(node_features)
        logits = torch.matmul(h_proj, prompt_features.t())  # [N, K]

        effective_rho = self.rho if rho is None else float(rho)
        effective_tau = self.tau if tau is None else float(tau)
        local_h = None

        # 准备候选池
        if pool_mask is None and effective_rho < 1.0:
            if edge_index is None:
                raise ValueError("rho<1 且未提供 pool_mask 时，必须传入 edge_index 才能计算异配度。")
            pool_mask, _, local_h = self.compute_pool_mask(
                edge_index, num_nodes, y=y, x=node_features, rho=effective_rho
            )
        elif pool_mask is None:
            pool_mask = torch.ones(num_nodes, dtype=torch.bool, device=node_features.device)
        else:
            pool_mask = pool_mask.to(node_features.device)

        # 仅对池内节点做 Gumbel-Softmax，池外保持 0
        ex = torch.zeros_like(logits)
        if pool_mask.any():
            ex[pool_mask] = F.gumbel_softmax(
                logits[pool_mask], tau=effective_tau, hard=hard, dim=-1
            )

        if not return_aux:
            return ex

        aux = {
            "pool_mask": pool_mask,
            "pool_size": int(pool_mask.sum().item()),
            "local_heterophily": local_h,
            "logits": logits,
        }
        return ex, aux


# =============================================================================
#  一站式封装：PromptRouter（推荐使用）
# =============================================================================


@dataclass
class PromptRoutingOutput:
    """``PromptRouter`` 前向的完整输出。

    Attributes:
        ex_homo:           [N, K1] 同配提示路由矩阵
        ex_hete:           [N, K2] 异配提示路由矩阵
        pool_mask:         [N] bool，本次实际使用的 V_pool
        pool_size:         |V_pool|
        local_heterophily: [N] 每个节点的局部异配度
        prompt_edge_index: [2, E_prompt] 提示边索引（已包含双向边）
        prompt_edge_class: [E_prompt] 每条提示边类别（0=homo, 1=hete），仅供分析 / 损失
        logits_homo:       [N, K1] 同配 logits（保留供分析）
        logits_hete:       [N, K2] 异配 logits
    """

    ex_homo: torch.Tensor
    ex_hete: torch.Tensor
    pool_mask: torch.Tensor
    pool_size: int
    local_heterophily: Optional[torch.Tensor]
    prompt_edge_index: torch.Tensor
    prompt_edge_class: torch.Tensor
    logits_homo: torch.Tensor
    logits_hete: torch.Tensor


class PromptRouter(nn.Module):
    """阶段二的高层封装：同时管理 EX_homo / EX_hete + Top-ρ + 边构建。

    一次 ``forward`` 即可拿到所有训练循环需要的产物，避免在 ``train.py`` 里
    手工拼装多个 ``GumbelRouter``。

    Args:
        feature_dim: 原图节点特征维度。
        prompt_dim:  提示原型维度（应与 P_homo / P_hete 对齐）。
        k_homo:      K1，同配提示数量。
        k_hete:      K2，异配提示数量。
        tau:         Gumbel 初始温度（训练开始时使用）。
        tau_end:     Gumbel 最终温度（退火目标值，默认 0.1）。
                     训练过程中由外部调用 ``anneal_tau(epoch, total_epochs)``
                     或在 ``forward`` 里传入当前 ``tau`` 覆盖。
        rho:         Top-ρ 默认比例。
    """

    def __init__(
        self,
        feature_dim: int,
        prompt_dim: int,
        k_homo: int,
        k_hete: int,
        tau: float = 1.0,
        tau_end: float = 0.1,
        rho: float = 0.3,
    ):
        super().__init__()
        self.k_homo = int(k_homo)
        self.k_hete = int(k_hete)
        self.tau = float(tau)
        self.tau_end = float(tau_end)
        self.rho = float(rho)
        self._current_tau = float(tau)  # 由外部 set_tau / anneal_tau 更新

        # 两路独立投影：保留 P_homo / P_hete 各自的几何
        self.router_homo = GumbelRouter(feature_dim, prompt_dim, tau=tau, rho=rho)
        self.router_hete = GumbelRouter(feature_dim, prompt_dim, tau=tau, rho=rho)

    # ------------------------------------------------------------------
    #  tau 退火工具
    # ------------------------------------------------------------------

    def anneal_tau(self, epoch: int, total_epochs: int) -> float:
        """指数退火：tau = tau_start * (tau_end / tau_start)^(epoch / total_epochs)。

        在训练循环每个 epoch 开始前调用，返回当前有效温度。
        """
        if total_epochs <= 1:
            self._current_tau = self.tau_end
            return self._current_tau
        progress = min(max(float(epoch - 1) / float(total_epochs - 1), 0.0), 1.0)
        if self.tau_end <= 0 or self.tau <= 0:
            self._current_tau = max(self.tau_end, 1e-6)
        else:
            log_ratio = (self.tau_end / self.tau)
            self._current_tau = float(self.tau * (log_ratio ** progress))
        return self._current_tau

    def set_tau(self, tau: float) -> None:
        """手动设置当前温度（用于外部自定义退火策略）。"""
        self._current_tau = float(tau)

    def get_current_tau(self) -> float:
        return self._current_tau

    # ------------------------------------------------------------------
    #  构造方法：与 load_data 的数据集 profile 对齐
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset_profile(
        cls,
        profile,  # load_data.DatasetProfile
        feature_dim: int,
        prompt_dim: int,
        k_homo: int,
        k_hete: int,
        tau: float = 1.0,
        tau_end: float = 0.1,
        rho: Optional[float] = None,
    ) -> "PromptRouter":
        """按 ``DatasetProfile.default_rho`` 自动构造，与 README §8 默认 ρ 对齐。"""
        effective_rho = float(rho) if rho is not None else float(profile.default_rho)
        return cls(
            feature_dim=feature_dim,
            prompt_dim=prompt_dim,
            k_homo=k_homo,
            k_hete=k_hete,
            tau=tau,
            tau_end=tau_end,
            rho=effective_rho,
        )

    # ------------------------------------------------------------------
    #  forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        p_homo: torch.Tensor,
        p_hete: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        hard: bool = False,
        topk: int = 1,
        rho: Optional[float] = None,
        pool_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> PromptRoutingOutput:
        """生成 EX_homo / EX_hete + 拼装好的提示边。

        Args:
            x:          原图节点特征 [N, d]。
            p_homo:     同配提示 [K1, d_p]。
            p_hete:     异配提示 [K2, d_p]。
            edge_index: 原图边 [2, E_orig]，用于计算异配度。
            y:          可选标签，用于标签模式异配度（推荐）。
            hard:       Gumbel 是否走 Straight-Through 离散化。
            topk:       每个池内节点最多连多少个**同类**提示（默认 1，符合一跳邻居语义）。
            rho:        覆盖默认 ρ。
            pool_mask:  覆盖默认池（高级用法）。
            tau:        当前退火温度；若为 None 则使用 ``_current_tau``。

        Returns:
            PromptRoutingOutput
        """
        num_nodes = x.size(0)
        effective_rho = self.rho if rho is None else float(rho)
        effective_tau = self._current_tau if tau is None else float(tau)

        # 1) 共享一个 V_pool（保证 homo / hete 路由在同一组节点上竞争）
        local_h = None
        if pool_mask is None:
            if effective_rho < 1.0:
                pool_mask, _, local_h = self.router_homo.compute_pool_mask(
                    edge_index, num_nodes, y=y, x=x, rho=effective_rho
                )
            else:
                pool_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
                local_h = compute_local_heterophily(edge_index, num_nodes, y=y, x=x)

        # 2) 两路 Gumbel 路由（共享池 + 当前退火温度）
        ex_homo, aux_homo = self.router_homo(
            x, p_homo, hard=hard, pool_mask=pool_mask, return_aux=True, tau=effective_tau
        )
        ex_hete, aux_hete = self.router_hete(
            x, p_hete, hard=hard, pool_mask=pool_mask, return_aux=True, tau=effective_tau
        )

        # 3) 构建提示边（PyG 形式）
        prompt_edge_index, prompt_edge_class = build_prompt_edge_index(
            ex_homo=ex_homo,
            ex_hete=ex_hete,
            num_nodes=num_nodes,
            k_homo=self.k_homo,
            topk=topk,
            pool_mask=pool_mask,
        )

        return PromptRoutingOutput(
            ex_homo=ex_homo,
            ex_hete=ex_hete,
            pool_mask=pool_mask,
            pool_size=int(pool_mask.sum().item()),
            local_heterophily=local_h,
            prompt_edge_index=prompt_edge_index,
            prompt_edge_class=prompt_edge_class,
            logits_homo=aux_homo["logits"],
            logits_hete=aux_hete["logits"],
        )


# =============================================================================
#  提示边构建（核心工具，多脚本共享）
# =============================================================================


def build_prompt_edge_index(
    ex_homo: torch.Tensor,
    ex_hete: torch.Tensor,
    num_nodes: int,
    k_homo: int,
    topk: int = 1,
    pool_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 EX_homo / EX_hete 编译成 PyG 风格的提示边。

    节点编号约定（与 ``DualBranchGNN`` / ``PromptAwareGNNConv`` 一致）：

        - 原图节点:  ``[0, num_nodes)``
        - P_homo:    ``[num_nodes, num_nodes + k_homo)``
        - P_hete:    ``[num_nodes + k_homo, num_nodes + k_homo + k_hete)``

    每个池内节点最多连 ``topk`` 个同配提示 + ``topk`` 个异配提示，
    每条边添加为**双向边**（PyG 标准）。

    Args:
        ex_homo: [N, K1]，非池节点行为 0。
        ex_hete: [N, K2]，非池节点行为 0。
        num_nodes: N。
        k_homo: K1，用于计算异配提示的偏移。
        topk: 每个节点连多少个同类提示。
        pool_mask: 可选 [N]，仅遍历池内节点；缺省时遍历全图。

    Returns:
        prompt_edge_index: [2, E_prompt]，long tensor。
        prompt_edge_class: [E_prompt]，long tensor，0=homo / 1=hete。
    """
    device = ex_homo.device
    K1 = ex_homo.size(-1)
    K2 = ex_hete.size(-1)
    topk_homo = max(1, min(topk, K1))
    topk_hete = max(1, min(topk, K2))

    # 仅遍历池内节点；其余节点 EX 全 0 不会有边
    if pool_mask is None:
        active_nodes = torch.arange(num_nodes, device=device)
    else:
        active_nodes = pool_mask.nonzero(as_tuple=False).view(-1)

    if active_nodes.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    # ---- 向量化收集 homo 边 ----
    homo_src_list: List[torch.Tensor] = []
    homo_dst_list: List[torch.Tensor] = []
    if (ex_homo[active_nodes].sum() > 0).item():
        vals_h, idx_h = torch.topk(ex_homo[active_nodes], k=topk_homo, dim=-1)  # [|pool|, topk]
        keep_h = vals_h > 0
        nodes_repeat = active_nodes.unsqueeze(-1).expand_as(idx_h)
        # 提示节点全局编号 = num_nodes + 本地 prompt 索引
        prompt_global_h = idx_h + num_nodes
        homo_src_list.append(nodes_repeat[keep_h])
        homo_dst_list.append(prompt_global_h[keep_h])

    # ---- 向量化收集 hete 边 ----
    hete_src_list: List[torch.Tensor] = []
    hete_dst_list: List[torch.Tensor] = []
    if (ex_hete[active_nodes].sum() > 0).item():
        vals_e, idx_e = torch.topk(ex_hete[active_nodes], k=topk_hete, dim=-1)
        keep_e = vals_e > 0
        nodes_repeat = active_nodes.unsqueeze(-1).expand_as(idx_e)
        # P_hete 全局编号偏移多加一个 k_homo
        prompt_global_e = idx_e + num_nodes + k_homo
        hete_src_list.append(nodes_repeat[keep_e])
        hete_dst_list.append(prompt_global_e[keep_e])

    homo_src = torch.cat(homo_src_list) if homo_src_list else torch.empty(0, dtype=torch.long, device=device)
    homo_dst = torch.cat(homo_dst_list) if homo_dst_list else torch.empty(0, dtype=torch.long, device=device)
    hete_src = torch.cat(hete_src_list) if hete_src_list else torch.empty(0, dtype=torch.long, device=device)
    hete_dst = torch.cat(hete_dst_list) if hete_dst_list else torch.empty(0, dtype=torch.long, device=device)

    if homo_src.numel() == 0 and hete_src.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    # 构造双向边：每条原始边变成 (u→p) + (p→u)
    homo_edges = torch.stack(
        [
            torch.cat([homo_src, homo_dst]),
            torch.cat([homo_dst, homo_src]),
        ],
        dim=0,
    )  # [2, 2*|homo|]
    hete_edges = torch.stack(
        [
            torch.cat([hete_src, hete_dst]),
            torch.cat([hete_dst, hete_src]),
        ],
        dim=0,
    )

    prompt_edge_index = torch.cat([homo_edges, hete_edges], dim=1).long()
    prompt_edge_class = torch.cat(
        [
            torch.zeros(homo_edges.size(1), dtype=torch.long, device=device),
            torch.ones(hete_edges.size(1), dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return prompt_edge_index, prompt_edge_class


# =============================================================================
#  向后兼容：函数式 API（旧测试脚本仍在用）
# =============================================================================


def route_homo_hete_prompts(
    router_homo: GumbelRouter,
    router_hete: GumbelRouter,
    node_features: torch.Tensor,
    p_homo: torch.Tensor,
    p_hete: torch.Tensor,
    hard: bool = False,
    edge_index: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    rho: Optional[float] = None,
    pool_mask: Optional[torch.Tensor] = None,
    return_aux: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, dict]]:
    """[兼容旧脚本] 用两个独立 ``GumbelRouter`` 拼出 EX_homo / EX_hete + 共享池。

    新代码请优先使用 :class:`PromptRouter`，它一站式返回完整产物。
    """
    if pool_mask is None and edge_index is not None and (rho is None or float(rho) < 1.0):
        effective_rho = router_homo.rho if rho is None else float(rho)
        pool_mask, _, local_h = router_homo.compute_pool_mask(
            edge_index, node_features.size(0), y=y, x=node_features, rho=effective_rho
        )
    else:
        local_h = None

    ex_homo = router_homo(
        node_features,
        p_homo,
        hard=hard,
        pool_mask=pool_mask,
        rho=1.0 if pool_mask is not None else rho,
    )
    ex_hete = router_hete(
        node_features,
        p_hete,
        hard=hard,
        pool_mask=pool_mask,
        rho=1.0 if pool_mask is not None else rho,
    )

    if not return_aux:
        return ex_homo, ex_hete

    aux = {
        "pool_mask": pool_mask,
        "pool_size": int(pool_mask.sum().item()) if pool_mask is not None else node_features.size(0),
        "local_heterophily": local_h,
    }
    return ex_homo, ex_hete, aux


# =============================================================================
#  自测脚本
# =============================================================================


if __name__ == "__main__":
    N, d, K1, K2 = 2708, 1433, 5, 5
    torch.manual_seed(0)

    node_x = torch.randn(N, d)
    edge_index = torch.randint(0, N, (2, N * 4))
    y = torch.randint(0, 7, (N,))

    p_homo = torch.randn(K1, d)
    p_hete = torch.randn(K2, d)

    # 推荐用法：PromptRouter
    router = PromptRouter(feature_dim=d, prompt_dim=d, k_homo=K1, k_hete=K2, tau=0.5, rho=0.3)
    out = router(node_x, p_homo, p_hete, edge_index=edge_index, y=y, topk=1)
    print(f"V_pool size: {out.pool_size} / {N}")
    print(f"EX_homo non-zero rows: {(out.ex_homo.sum(-1) > 0).sum().item()}")
    print(f"EX_hete non-zero rows: {(out.ex_hete.sum(-1) > 0).sum().item()}")
    print(f"prompt_edge_index shape: {out.prompt_edge_index.shape}")
    print(f"homo edges: {(out.prompt_edge_class == 0).sum().item()}")
    print(f"hete edges: {(out.prompt_edge_class == 1).sum().item()}")
