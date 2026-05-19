"""训练前 Prompt 路由 / EX / 提示边诊断（只读统计，不参与反传）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from loss import prompt_edge_homophily, prompt_majority_labels
from prompts.gumbel_route import PromptRoutingOutput


def _tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float()
    if t.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()) if t.numel() > 1 else 0.0,
    }


def _row_entropy(ex_rows: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """每行 Shannon 熵（自然对数）；one-hot 行 → 0，均匀行 → log(K)。"""
    p = ex_rows.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def _summarize_ex_block(
    ex: torch.Tensor,
    name: str,
    pool_mask: torch.Tensor,
    y: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """单路 EX 的分布与退化检测。"""
    device = ex.device
    k = ex.size(-1)
    pool = pool_mask.to(device)
    pool_rows = ex[pool] if pool.any() else ex.new_zeros((0, k))

    active_in_pool = (pool_rows.abs().sum(dim=-1) > 1e-8).sum().item() if pool_rows.numel() else 0
    row_sums = pool_rows.sum(dim=-1) if pool_rows.numel() else ex.new_zeros(0)

    out: Dict[str, Any] = {
        "shape": tuple(ex.shape),
        "pool_active_rows": int(active_in_pool),
        "row_sum": _tensor_stats(row_sums) if row_sums.numel() else _tensor_stats(row_sums),
    }

    if pool_rows.numel() > 0:
        ent = _row_entropy(pool_rows)
        argmax = pool_rows.argmax(dim=-1)
        counts = torch.bincount(argmax, minlength=k).float()
        max_share = float((counts.max() / counts.sum()).item()) if counts.sum() > 0 else 0.0
        empty_prompts = int((counts == 0).sum().item())

        out["entropy"] = _tensor_stats(ent)
        out["max_weight"] = _tensor_stats(pool_rows.max(dim=-1).values)
        out["argmax_dominant_share"] = max_share
        out["empty_prompt_slots"] = empty_prompts
        out["near_uniform_rows"] = int((ent > 0.9 * torch.log(torch.tensor(float(k)))).sum().item())
        out["near_onehot_rows"] = int((ent < 0.05).sum().item())

        if y is not None:
            out["prompt_edge_homophily"] = prompt_edge_homophily(ex, y, pool_mask=pool_mask)
            pseudo = prompt_majority_labels(ex, y, pool_mask=pool_mask)
            out["prompt_pseudo_labels"] = pseudo.detach().cpu().tolist()

    return out


def _summarize_prompt_edges(
    routing: PromptRoutingOutput,
    num_nodes: int,
) -> Dict[str, Any]:
    ei = routing.prompt_edge_index
    if ei.numel() == 0:
        return {
            "num_directed_edges": 0,
            "num_undirected_pairs": 0,
            "homo_edges": 0,
            "hete_edges": 0,
            "nodes_with_prompt_edge": 0,
            "mean_prompt_degree": 0.0,
            "max_prompt_degree": 0,
        }

    src, dst = ei[0], ei[1]
    real_src = src[src < num_nodes]
    degree = torch.bincount(real_src, minlength=num_nodes).float()
    used = (degree > 0).sum().item()

    cls = routing.prompt_edge_class
    homo_e = int((cls == 0).sum().item()) if cls.numel() else 0
    hete_e = int((cls == 1).sum().item()) if cls.numel() else 0

    return {
        "num_directed_edges": int(ei.size(1)),
        "num_undirected_pairs": int(ei.size(1) // 2),
        "homo_edges": homo_e,
        "hete_edges": hete_e,
        "nodes_with_prompt_edge": int(used),
        "mean_prompt_degree": float(degree[degree > 0].mean().item()) if used > 0 else 0.0,
        "max_prompt_degree": int(degree.max().item()),
    }


def collect_routing_debug(
    routing: PromptRoutingOutput,
    *,
    num_nodes: int,
    y: Optional[torch.Tensor],
    rho: float,
    topk: int,
    k_homo: int,
    k_hete: int,
    mode: str = "soft",
) -> Dict[str, Any]:
    """汇总一次 ``PromptRouter`` 前向的诊断字典。"""
    pool_mask = routing.pool_mask
    local_h = routing.local_heterophily
    pool_h = local_h[pool_mask] if local_h is not None and pool_mask.any() else None

    return {
        "mode": mode,
        "num_nodes": num_nodes,
        "rho": float(rho),
        "topk": int(topk),
        "k_homo": int(k_homo),
        "k_hete": int(k_hete),
        "pool_size": routing.pool_size,
        "pool_fraction": float(routing.pool_size) / max(num_nodes, 1),
        "pool_heterophily": _tensor_stats(pool_h) if pool_h is not None and pool_h.numel() else None,
        "ex_homo": _summarize_ex_block(routing.ex_homo, "homo", pool_mask, y),
        "ex_hete": _summarize_ex_block(routing.ex_hete, "hete", pool_mask, y),
        "logits_homo": _tensor_stats(routing.logits_homo[pool_mask]) if pool_mask.any() else _tensor_stats(routing.logits_homo[:0]),
        "logits_hete": _tensor_stats(routing.logits_hete[pool_mask]) if pool_mask.any() else _tensor_stats(routing.logits_hete[:0]),
        "prompt_edges": _summarize_prompt_edges(routing, num_nodes),
    }


def _fmt_stats(d: Optional[Dict[str, float]]) -> str:
    if not d:
        return "n/a"
    return f"mean={d['mean']:.4f} std={d['std']:.4f} min={d['min']:.4f} max={d['max']:.4f}"


def _print_ex_block(title: str, block: Dict[str, Any]) -> None:
    print(f"  [{title}] shape={block['shape']} | pool 内有效行={block['pool_active_rows']}")
    print(f"    row_sum (池内): {_fmt_stats(block.get('row_sum'))}")
    if "entropy" in block:
        print(f"    entropy:        {_fmt_stats(block['entropy'])} | near_uniform={block['near_uniform_rows']} near_onehot={block['near_onehot_rows']}")
        print(f"    max_weight:     {_fmt_stats(block['max_weight'])}")
        print(
            f"    argmax 最大桶占比={block['argmax_dominant_share']:.3f} | "
            f"空 prompt 槽={block['empty_prompt_slots']}/{block['shape'][1]}"
        )
        if "prompt_edge_homophily" in block:
            print(f"    prompt_edge_homophily={block['prompt_edge_homophily']:.4f} | pseudo_labels={block.get('prompt_pseudo_labels')}")


def print_routing_debug(summary: Dict[str, Any]) -> None:
    """格式化打印 ``collect_routing_debug`` 的结果。"""
    print("\n" + "=" * 72)
    print(f"[Routing Debug] mode={summary['mode']} | N={summary['num_nodes']} | "
          f"rho={summary['rho']:.3f} topk={summary['topk']} | "
          f"K_homo={summary['k_homo']} K_hete={summary['k_hete']}")
    print(
        f"  V_pool: {summary['pool_size']} nodes ({summary['pool_fraction']:.1%}) | "
        f"pool heterophily: {_fmt_stats(summary.get('pool_heterophily'))}"
    )
    _print_ex_block("EX_homo", summary["ex_homo"])
    _print_ex_block("EX_hete", summary["ex_hete"])
    print(f"  [logits_homo 池内] {_fmt_stats(summary['logits_homo'])}")
    print(f"  [logits_hete 池内] {_fmt_stats(summary['logits_hete'])}")
    pe = summary["prompt_edges"]
    print(
        f"  [prompt edges] directed={pe['num_directed_edges']} "
        f"(undirected≈{pe['num_undirected_pairs']}) | homo={pe['homo_edges']} hete={pe['hete_edges']}"
    )
    print(
        f"    有 prompt 边的原图节点={pe['nodes_with_prompt_edge']}/{summary['num_nodes']} | "
        f"prompt度 mean={pe['mean_prompt_degree']:.2f} max={pe['max_prompt_degree']}"
    )

    # 退化提示
    homo = summary["ex_homo"]
    hete = summary["ex_hete"]
    warnings = []
    if homo.get("pool_active_rows", 0) == 0 and hete.get("pool_active_rows", 0) == 0:
        warnings.append("池内 EX 全零：无提示边可建")
    for tag, blk in [("homo", homo), ("hete", hete)]:
        if blk.get("argmax_dominant_share", 0) > 0.8:
            warnings.append(f"EX_{tag} 可能退化：>80% 节点连向同一 prompt")
        if blk.get("near_uniform_rows", 0) > 0.5 * max(blk.get("pool_active_rows", 1), 1):
            warnings.append(f"EX_{tag} 大量行近似均匀（路由过软）")
    if pe["num_directed_edges"] == 0:
        warnings.append("prompt_edge_index 为空")
    if warnings:
        print("  [WARN] " + " | ".join(warnings))
    print("=" * 72 + "\n")
