from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def prompt_edge_contrastive_loss(
    x: torch.Tensor,
    prompt_assignments: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Prompt-edge contrastive objective.

    If two nodes connect to the same prompt, pull them together.
    Otherwise push them apart.

    Args:
        x: Node embeddings, shape [N, D]
        prompt_assignments: Binary/soft assignment matrix, shape [N, K]
        margin: Separation margin for negative pairs.
    """
    if prompt_assignments is None or prompt_assignments.numel() == 0:
        return x.new_tensor(0.0)

    same_prompt = prompt_assignments @ prompt_assignments.t()  # shape: [N, N]
    same_prompt = (same_prompt > 0).float()
    eye = torch.eye(same_prompt.size(0), device=same_prompt.device)
    same_prompt = same_prompt * (1.0 - eye)

    if same_prompt.sum() == 0:
        return x.new_tensor(0.0)

    dist = torch.cdist(x, x, p=2)  # shape: [N, N]
    pos_loss = (same_prompt * dist.pow(2)).sum() / (same_prompt.sum() + 1e-8)
    neg_mask = (1.0 - same_prompt) * (1.0 - eye)
    neg_loss = (neg_mask * F.relu(margin - dist).pow(2)).sum() / (neg_mask.sum() + 1e-8)
    return pos_loss + neg_loss


def dynamic_lambda2(is_homophilic: bool, base_lambda2: float = 1.0) -> float:
    """Larger consistency weight for heterophilic graphs."""
    return float(base_lambda2 * (0.25 if is_homophilic else 2.0))


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL consistency for tensors with the same shape.

    If shapes mismatch, this function should not be called directly.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f"consistency_loss expects same-shaped tensors, got {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}"
        )
    p_a = F.log_softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    p_b_log = F.log_softmax(logits_b, dim=-1)
    p_a_prob = F.softmax(logits_a, dim=-1)
    kl_ab = F.kl_div(p_a, p_b, reduction="batchmean")
    kl_ba = F.kl_div(p_b_log, p_a_prob, reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def compute_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sparse_loss: torch.Tensor,
    consist_loss: torch.Tensor,
    contrastive_loss: torch.Tensor,
    is_homophilic: bool,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    ce_loss = F.cross_entropy(logits, labels)
    dyn_lambda2 = dynamic_lambda2(is_homophilic, lambda2)
    total = ce_loss + lambda1 * sparse_loss + dyn_lambda2 * consist_loss + lambda3 * contrastive_loss
    stats = {
        "ce_loss": ce_loss.detach(),
        "sparse_loss": sparse_loss.detach(),
        "consist_loss": consist_loss.detach(),
        "contrastive_loss": contrastive_loss.detach(),
        "lambda2": torch.tensor(dyn_lambda2, device=logits.device),
    }
    return total, stats
