"""阶段三：异配度感知消息传递 PromptAwareGNNConv。

整体公式（与 README §5 / idea 第二节阶段三对齐）：

    对每条边 (j → i)，令：
        h_i = local_heterophily[i]  ∈ [0, 1]   # 目的节点的局部异配度
        t   = edge_type             ∈ {0, 1}   # 0=原图边, 1=提示边

    边权重：
        w_ji =  α · (1 − γ · h_i)   if t = 0   # 异配度越高，原图邻居越被弱化
              β · (1 + γ · h_i)     if t = 1   # 异配度越高，提示邻居越被强化

    输出：
        H_i' = W_root · H_i + Σ_{j∈N(i)} w_ji · (W · H_j) + b

其中 α / β 是基础权重，γ 控制"按异配度差异化"的幅度。
γ = 0 时退化为静态加权 GNN；γ = 1 时差异化最强。

边类型约定（与 ``gumbel_route.build_prompt_edge_index`` 保持一致）：
    edge_type == 0  ↔  原图边（含 self-loop）
    edge_type == 1  ↔  提示边（无论 P_homo 还是 P_hete；二者再细分由对比损失负责）

物理意义：
    - 对一个被异配邻居淹没的节点 i（h_i 接近 1），原图聚合反而拉低其表征质量，
      因此 α(1 − γ·h_i) 缩小；
    - 同一时刻，它通过 E_X 连接到的提示节点更可能提供"同类同盟"或"异类捷径"
      信号，因此 β(1 + γ·h_i) 放大。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


# 边类型常量（导出供训练循环使用，避免魔法数字）
ORIG_EDGE: int = 0
PROMPT_EDGE: int = 1


class PromptAwareGNNConv(MessagePassing):
    """异配度感知消息传递层。

    Args:
        in_channels:  输入特征维度。
        out_channels: 输出特征维度。
        alpha:    原图边的基础权重（默认 1.0；与 ``beta`` 一起决定基线强度）。
        beta:     提示边的基础权重（默认 1.0）。
        strength: 异配度差异化幅度 γ ∈ [0, 1]（默认 1.0）。
        bias:     是否添加偏置。
        aggr:     PyG 聚合方式，默认 ``"add"``。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        strength: float = 1.0,
        bias: bool = True,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.strength = float(strength)

        # 邻居线性变换（W·H_j）与根节点变换（W_root·H_i）解耦
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.root_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.root_lin.weight)

    # ------------------------------------------------------------------
    #  局部异配度估计（fallback 路径；推荐外部传入）
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_node_heterophily_from_features(
        x_raw: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """基于原始特征余弦相似度估计每节点异配度（fallback 路径）。

        ``h_i = 1 − mean_{j∈N(i)} cos(x_i, x_j)``。

        Note:
            推荐由 ``prompts.gumbel_route.compute_local_heterophily`` 在数据预处理
            阶段计算（可选基于标签，更精准），并通过 ``forward(node_heterophily=...)``
            注入；该函数仅作没有外部信号时的兜底。
        """
        device = edge_index.device
        h = torch.zeros(num_nodes, device=device)
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index[0], edge_index[1]
        x_norm = F.normalize(x_raw, p=2, dim=-1, eps=1e-12)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=-1).clamp(0.0, 1.0)

        # 按 dst 聚合，得到每节点的邻居平均相似度
        sim_sum = torch.zeros(num_nodes, device=device)
        degree = torch.zeros(num_nodes, device=device)
        sim_sum.index_add_(0, dst, sim)
        degree.index_add_(0, dst, torch.ones_like(sim))
        mask = degree > 0
        h[mask] = 1.0 - sim_sum[mask] / degree[mask]
        return h

    # ------------------------------------------------------------------
    #  forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        node_heterophily: Optional[torch.Tensor] = None,
        return_edge_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prompt-Biased MP 前向。

        Args:
            x:                节点特征 [N, in_channels]。
            edge_index:       PyG 边索引 [2, E]；通常 = 原图边 ∪ 提示边。
            edge_type:        每条边的类型 [E]，0=原图 / 1=提示；缺省视为全为原图。
            node_heterophily: 每节点局部异配度 [N]（推荐外部传入；缺省时用特征估计）。
            return_edge_weights: 是否返回每条边最终生效的权重，便于调试 / 可视化。

        Returns:
            out: 卷积后的节点表征 [N, out_channels]；若 ``return_edge_weights`` 为
                 True，则同时返回 [E']（含 self-loop）的边权重。
        """
        num_nodes = x.size(0)
        x_raw = x  # 保留原始特征用于 fallback 异配度估计

        # 1) 邻居线性变换：x ← W · x
        x = self.lin(x)

        # 2) 准备 edge_type；缺省视为全部原图边
        if edge_type is None:
            edge_type = edge_index.new_zeros(edge_index.size(1), dtype=torch.long)
        else:
            edge_type = edge_type.to(edge_index.device).long()

        # 3) 加 self-loop（边类型记为原图）；保证孤立节点也有自身信号
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        self_loop_type = edge_index.new_zeros(num_nodes, dtype=torch.long)
        edge_type = torch.cat([edge_type, self_loop_type], dim=0)

        # 4) 局部异配度：优先外部传入；否则按原始特征估计
        if node_heterophily is None:
            local_h = self._estimate_node_heterophily_from_features(
                x_raw, edge_index, num_nodes=num_nodes
            )
        else:
            local_h = node_heterophily.to(edge_index.device).clamp(0.0, 1.0)

        # 5) 把节点级异配度广播到边上（按目的节点 dst）
        dst = edge_index[1]
        h_dst = local_h[dst]  # shape: [E']

        # 6) 边权重（核心公式）
        #    原图边: w = α · (1 − γ·h_i)   → 异配越强、原图聚合越弱
        #    提示边: w = β · (1 + γ·h_i)   → 异配越强、提示聚合越强
        orig_mask = (edge_type == ORIG_EDGE).float()
        prompt_mask = (edge_type == PROMPT_EDGE).float()
        w_orig = self.alpha * (1.0 - self.strength * h_dst)
        w_prompt = self.beta * (1.0 + self.strength * h_dst)
        # 保证非负
        w_orig = w_orig.clamp(min=0.0)
        edge_weight = w_orig * orig_mask + w_prompt * prompt_mask  # shape: [E']

        # 7) 消息传播 + 根节点变换 + 偏置
        #    root_lin 作用于"原始 x"（in_channels → out_channels），与邻居信号解耦
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + self.root_lin(x_raw)
        if self.bias is not None:
            out = out + self.bias

        if return_edge_weights:
            return out, edge_weight
        return out

    # ------------------------------------------------------------------
    #  PyG message / aggregate hooks
    # ------------------------------------------------------------------

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """对每条边 (j → i)：传递 w_ji · x_j。"""
        return x_j * edge_weight.unsqueeze(-1)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
