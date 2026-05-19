"""阶段五：动态多任务损失 + 分析指标。

综合损失（与 README §7 / idea 第五节对齐）：

    L = L_CE
      + λ1 · L_sparse
      + λ2(is_homophilic) · L_consist
      + λ3_homo · L_contrastive_homo
      + λ3_hete · L_contrastive_hete

关键设计：
    1. **homo / hete 对比损失分路**：两类提示在物理语义上不同（同盟枢纽 vs
       跨界桥梁），损失独立计算，便于在不同数据集族上施加不同权重；
    2. **动态 λ2**：异配图上一致性约束显著增大（默认 0.25 → 2.0），用源域先验
       压制异配噪声；同配图上则放小以避免冻结分支束缚适应分支；
    3. **prompt_edge_homophily 指标**：按提示节点的多数伪标签衡量"提示边
       是否真的把同类节点聚到一起"，分 P_homo / P_hete 汇报，是论文的核心
       分析指标之一；
    4. **大图内存友好**：对比损失支持随机采样 ``contrastive_sample_size`` 个
       池内节点而非全 N 对，避免 Flickr / Amazon-ratings 上 O(N^2) 爆显存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# =============================================================================
#  配置
# =============================================================================


@dataclass
class LossConfig:
    """损失函数的所有超参数。

    若 `is_homophilic = True`，consistency 实际权重 = ``lambda2 * lambda2_homo_mult``；
    否则 = ``lambda2 * lambda2_hete_mult``。

    训练稳定性（默认开启）：
        - ``normalize_contrastive``：对比前 L2 归一化嵌入，避免 dist 爆炸；
        - ``consist_max``：对称 KL 上限，防止 Gate 升高时 Consist 失控；
        - ``loss_warmup_epochs``：前若干 epoch 线性增大 λ1/λ2/λ3，先学好 CE。
    """

    # 总体权重
    lambda1: float = 0.01         # L_sparse
    lambda2: float = 0.05         # L_consist 基准权重（同配图上再 ×0.25）
    lambda3_homo: float = 0.1     # L_contrastive_homo
    lambda3_hete: float = 0.1     # L_contrastive_hete

    # 动态 λ2 的两档乘数（同配 / 异配）
    lambda2_homo_mult: float = 0.25
    lambda2_hete_mult: float = 2.0

    # 对比损失
    contrastive_margin: float = 1.0
    contrastive_sample_size: int = 512  # 0 表示不采样、用全图（小图）
    contrastive_use_hard: bool = False  # False: 软 EX 相似度（更稳）

    # 稳定性
    normalize_contrastive: bool = True
    consist_max: float = 1.0
    loss_warmup_epochs: int = 50
    current_epoch: int = 1


# =============================================================================
#  损失项（函数式）
# =============================================================================


def sparsity_loss(ex: torch.Tensor) -> torch.Tensor:
    """提示边稀疏正则：鼓励 EX 行更接近 one-hot（即每个节点尽量只用少量提示）。

    形式：``L_sparse = mean(EX)``；由于每行 softmax 求和为 1（池内），
    该项不会强迫所有边为 0；其作用是配合 Gumbel-Softmax 温度趋近 hard 模式。
    """
    if ex is None or ex.numel() == 0:
        return torch.tensor(0.0)
    return ex.mean()


def loss_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """前 ``warmup_epochs`` 个 epoch 将辅助损失权重从 0 线性升到 1。"""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(float(epoch), 1.0) / float(warmup_epochs))


def consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """对称 KL：拉近两支分支的输出分布（一致性约束）。

    要求两 logits 同 shape。``train.py`` 调用前需保证已对齐
    （冻结分支 N 维 + 适应分支截断后 N 维）。
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f"consistency_loss expects same-shaped tensors, "
            f"got {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}"
        )
    p_a_log = F.log_softmax(logits_a, dim=-1)
    p_b_log = F.log_softmax(logits_b, dim=-1)
    p_a = p_a_log.exp()
    p_b = p_b_log.exp()
    kl_ab = F.kl_div(p_a_log, p_b, reduction="batchmean")
    kl_ba = F.kl_div(p_b_log, p_a, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def _same_prompt_matrix(
    ex: torch.Tensor,
    use_hard: bool,
) -> torch.Tensor:
    """根据 EX 计算 [|S|, |S|] 的"是否连向同一 prompt"指示矩阵。

    - ``use_hard=True``：取每行 argmax，判别两个节点是否落到同一 prompt（推荐，
      与 Gumbel hard 模式语义一致）；
    - ``use_hard=False``：用 ``EX @ EX^T`` 作软相似度，等价于
      "共享同一 prompt 的概率"（适合 soft Gumbel）。
    """
    if use_hard:
        labels = ex.argmax(dim=-1)                            # [|S|]
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        return same
    # 软相似度：sum_k EX[i,k] * EX[j,k]
    return ex @ ex.t()


def prompt_edge_contrastive_loss(
    h: torch.Tensor,
    ex: torch.Tensor,
    margin: float = 1.0,
    sample_size: int = 512,
    use_hard: bool = True,
    pool_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """提示边对比损失：同 prompt → 拉近；异 prompt → 推远。

    Args:
        h:        节点嵌入 [N, D]（建议取适应分支输出的前 N 行）。
        ex:       提示路由矩阵 [N, K]（EX_homo 或 EX_hete）。
        margin:   负样本的分离 margin。
        sample_size: 池内随机采样多少节点参与计算；``0`` 表示全用（仅适合小图）。
        use_hard: 见 ``_same_prompt_matrix``。
        pool_mask: 可选 [N] bool，仅在池内节点上计算（推荐传入，节省显存）。

    Note:
        Flickr / Amazon-ratings 上 N≈10⁴–10⁵；不采样会 O(N²) 爆显存。
        推荐 ``sample_size=512``（与 SimCLR 经验值一致）。
    """
    if ex is None or ex.numel() == 0:
        return h.new_tensor(0.0)

    # 1) 先按 pool_mask 限定到池内节点（其它行 EX 全 0，参与计算只会拖累）
    if pool_mask is not None:
        idx_all = pool_mask.nonzero(as_tuple=False).view(-1)
    else:
        # 兜底：仅取 EX 有非零的行
        idx_all = (ex.abs().sum(dim=-1) > 0).nonzero(as_tuple=False).view(-1)

    if idx_all.numel() < 2:
        return h.new_tensor(0.0)

    # 2) 随机采样以控制 O(S^2) 复杂度
    if sample_size > 0 and idx_all.numel() > sample_size:
        perm = torch.randperm(idx_all.numel(), device=h.device)[:sample_size]
        idx = idx_all[perm]
    else:
        idx = idx_all

    h_s = h[idx]  # [S, D]
    ex_s = ex[idx]  # [S, K]

    same = _same_prompt_matrix(ex_s, use_hard=use_hard)       # [S, S]
    eye = torch.eye(same.size(0), device=same.device)
    same = same * (1.0 - eye)                                 # 去除自对

    pos_count = same.sum()
    if pos_count.item() == 0:
        return h.new_tensor(0.0)

    # 3) 安全成对距离：避免 cdist/sqrt 在 dist=0 时 backward 出现 NaN
    diff = h_s.unsqueeze(1) - h_s.unsqueeze(0)                # [S, S, D]
    dist_sq = (diff ** 2).sum(dim=-1)                         # [S, S]
    eps = h_s.new_tensor(1e-8)
    dist = torch.sqrt(dist_sq + eps)

    # 同 prompt → 拉近（直接用 dist_sq，不经过 sqrt）
    pos_loss = (same * dist_sq).sum() / (pos_count + 1e-8)
    # 不同 prompt → 推远（hinge: max(0, margin - dist)²）
    neg_mask = (1.0 - same) * (1.0 - eye)
    neg_count = neg_mask.sum()
    if neg_count.item() == 0:
        neg_loss = h.new_tensor(0.0)
    else:
        neg_loss = (neg_mask * F.relu(margin - dist).pow(2)).sum() / (neg_count + 1e-8)

    return pos_loss + neg_loss


def dynamic_lambda2(
    is_homophilic: bool,
    base_lambda2: float = 1.0,
    homo_mult: float = 0.25,
    hete_mult: float = 2.0,
) -> float:
    """根据图族返回实际 λ2：同配图小、异配图大。"""
    return float(base_lambda2 * (homo_mult if is_homophilic else hete_mult))


# =============================================================================
#  分析指标
# =============================================================================


def prompt_majority_labels(
    ex: torch.Tensor,
    y: torch.Tensor,
    pool_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算每个 prompt 的"伪标签"（其连接的原图节点的多数类）。

    Args:
        ex: [N, K]，提示路由矩阵（hard 一行 one-hot；soft 取 argmax）。
        y:  [N] 真实标签。
        pool_mask: 仅在池内节点上统计。

    Returns:
        labels: [K] long tensor，第 k 项是 prompt k 的多数伪标签；
                若 prompt k 没有任何节点连过来，记为 -1。
    """
    K = ex.size(-1)
    device = ex.device
    pseudo = torch.full((K,), -1, dtype=torch.long, device=device)
    if ex.numel() == 0 or y is None:
        return pseudo

    y_flat = y.view(-1)
    assigned = ex.argmax(dim=-1)                                # [N]
    valid = (y_flat >= 0)
    if pool_mask is not None:
        valid = valid & pool_mask.to(device)

    # 仅在池内、且 EX 有非零的位置算
    has_edge = ex.abs().sum(dim=-1) > 0
    valid = valid & has_edge

    if valid.sum().item() == 0:
        return pseudo

    num_classes = int(y_flat[valid].max().item()) + 1
    # one-hot 累加：counts[k, c] = #{ nodes assigned to k with label c }
    counts = torch.zeros(K, num_classes, device=device)
    idx_k = assigned[valid]
    idx_c = y_flat[valid]
    ones = torch.ones_like(idx_k, dtype=torch.float)
    counts.index_put_((idx_k, idx_c), ones, accumulate=True)

    has_any = counts.sum(dim=-1) > 0
    pseudo[has_any] = counts[has_any].argmax(dim=-1)
    return pseudo


def prompt_edge_homophily(
    ex: torch.Tensor,
    y: torch.Tensor,
    pool_mask: Optional[torch.Tensor] = None,
) -> float:
    """提示边同配率：连边节点真实标签与 prompt 多数伪标签一致的比例。

    与 README §8.4 的同名指标一致；越高说明提示边把同类节点拉到了同一 prompt。

    Returns:
        scalar ∈ [0, 1]；若没有有效连边返回 0。
    """
    if ex is None or ex.numel() == 0 or y is None:
        return 0.0

    pseudo = prompt_majority_labels(ex, y, pool_mask=pool_mask)  # [K]
    y_flat = y.view(-1)

    assigned = ex.argmax(dim=-1)                                # [N]
    has_edge = ex.abs().sum(dim=-1) > 0
    valid = has_edge & (y_flat >= 0)
    if pool_mask is not None:
        valid = valid & pool_mask.to(ex.device)
    if valid.sum().item() == 0:
        return 0.0

    pseudo_per_node = pseudo[assigned[valid]]
    match = (pseudo_per_node == y_flat[valid]) & (pseudo_per_node >= 0)
    return float(match.float().mean().item())


# =============================================================================
#  统一入口：compute_total_loss
# =============================================================================


@dataclass
class LossOutputs:
    """单次前向的所有损失项 + 用于日志的标量。

    Attributes:
        total: 反向传播用的总损失。
        ce, sparse, consist: 各项标量（已 detach 用于日志）。
        contrastive_homo, contrastive_hete: 分路对比损失。
        lambda2_effective: 实际生效的一致性损失权重。
        prompt_homo_homophily, prompt_hete_homophily: 提示边同配率指标。
    """

    total: torch.Tensor
    ce: torch.Tensor
    sparse: torch.Tensor
    consist: torch.Tensor
    contrastive_homo: torch.Tensor
    contrastive_hete: torch.Tensor
    lambda2_effective: float
    prompt_homo_homophily: float = 0.0
    prompt_hete_homophily: float = 0.0

    def to_log_dict(self) -> Dict[str, float]:
        """便于训练循环打印或写 TensorBoard。"""
        return {
            "total": float(self.total.detach().item()),
            "ce": float(self.ce.detach().item()),
            "sparse": float(self.sparse.detach().item()),
            "consist": float(self.consist.detach().item()),
            "contrast_homo": float(self.contrastive_homo.detach().item()),
            "contrast_hete": float(self.contrastive_hete.detach().item()),
            "lambda2": float(self.lambda2_effective),
            "prompt_homo_homophily": float(self.prompt_homo_homophily),
            "prompt_hete_homophily": float(self.prompt_hete_homophily),
        }


def compute_total_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    h_for_contrast: torch.Tensor,
    ex_homo: torch.Tensor,
    ex_hete: torch.Tensor,
    logits_frozen: torch.Tensor,
    logits_adapted: torch.Tensor,
    is_homophilic: bool,
    pool_mask: Optional[torch.Tensor] = None,
    full_y_for_metrics: Optional[torch.Tensor] = None,
    cfg: Optional[LossConfig] = None,
) -> LossOutputs:
    """组合所有损失项，**仅在 train_mask 上**计算交叉熵（5-shot 协议）。

    Args:
        logits:          融合后 logits [N, C]，用于评估。
        labels:          完整标签 [N]，但仅在 ``train_mask`` 上参与 CE。
        train_mask:      [N] bool，5-shot 训练 mask。
        h_for_contrast:  节点嵌入 [N, D]（一般取适应分支输出前 N 行）。
        ex_homo / ex_hete: 提示路由矩阵 [N, K1] / [N, K2]。
        logits_frozen / logits_adapted: 双分支各自的 logits [N, C]，用于一致性损失。
        is_homophilic:   图族；影响 λ2 自适应缩放。
        pool_mask:       V_pool，用于对比损失采样与指标。
        full_y_for_metrics: 全图标签，用于 prompt_edge_homophily 指标（不参与 loss）。
        cfg:             ``LossConfig``；缺省取默认值。

    Returns:
        LossOutputs
    """
    cfg = cfg or LossConfig()

    # 1) 5-shot 交叉熵
    if train_mask.sum().item() == 0:
        ce = logits.new_tensor(0.0, requires_grad=True)
    else:
        ce = F.cross_entropy(logits[train_mask], labels[train_mask])

    # 2) 稀疏正则（对 EX_homo + EX_hete 同时施加；均匀加权）
    sparse_h = sparsity_loss(ex_homo)
    sparse_e = sparsity_loss(ex_hete)
    sparse = 0.5 * (sparse_h + sparse_e)

    # 3) 双分支一致性（动态 λ2，带上限避免 KL 爆炸）
    consist = consistency_loss(logits_frozen, logits_adapted)
    if cfg.consist_max > 0:
        consist = consist.clamp(max=float(cfg.consist_max))
    lambda2_eff = dynamic_lambda2(
        is_homophilic,
        base_lambda2=cfg.lambda2,
        homo_mult=cfg.lambda2_homo_mult,
        hete_mult=cfg.lambda2_hete_mult,
    )

    aux_scale = loss_warmup_factor(cfg.current_epoch, cfg.loss_warmup_epochs)

    # 4) 分路对比损失（嵌入在函数内已 L2 归一化）
    h_contrast = h_for_contrast
    if cfg.normalize_contrastive:
        h_contrast = F.normalize(h_for_contrast, p=2, dim=-1, eps=1e-8)

    contrast_homo = prompt_edge_contrastive_loss(
        h_contrast,
        ex_homo,
        margin=cfg.contrastive_margin,
        sample_size=cfg.contrastive_sample_size,
        use_hard=cfg.contrastive_use_hard,
        pool_mask=pool_mask,
    )
    contrast_hete = prompt_edge_contrastive_loss(
        h_contrast,
        ex_hete,
        margin=cfg.contrastive_margin,
        sample_size=cfg.contrastive_sample_size,
        use_hard=cfg.contrastive_use_hard,
        pool_mask=pool_mask,
    )

    # 5) 组装总损失（辅助项 × warmup）
    total = (
        ce
        + aux_scale * cfg.lambda1 * sparse
        + aux_scale * lambda2_eff * consist
        + aux_scale * cfg.lambda3_homo * contrast_homo
        + aux_scale * cfg.lambda3_hete * contrast_hete
    )

    # 6) 可选：提示边同配率指标（不进损失）
    homo_hp = 0.0
    hete_hp = 0.0
    if full_y_for_metrics is not None:
        homo_hp = prompt_edge_homophily(ex_homo, full_y_for_metrics, pool_mask=pool_mask)
        hete_hp = prompt_edge_homophily(ex_hete, full_y_for_metrics, pool_mask=pool_mask)

    return LossOutputs(
        total=total,
        ce=ce,
        sparse=sparse,
        consist=consist,
        contrastive_homo=contrast_homo,
        contrastive_hete=contrast_hete,
        lambda2_effective=lambda2_eff,
        prompt_homo_homophily=homo_hp,
        prompt_hete_homophily=hete_hp,
    )
