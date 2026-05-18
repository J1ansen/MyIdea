"""Prompt topology ablation: homophily before/after prompt edges (Stage 1-2).

Validates Top-rho candidate pooling + Gumbel routing on 8 benchmark datasets.
Uses unified ``load_data`` profiles and ``PromptRouter`` (same as ``train.py``).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_data import (
    ALL_DATASETS,
    estimate_edge_homophily,
    get_dataset_profile,
    load_dataset,
    resolve_dataset_name,
)
from loss import prompt_edge_homophily
from prompts.cluster_generator import PromptGenerator
from prompts.gumbel_route import PromptRouter


@dataclass
class HomophilyStats:
    edge_homophily: float
    feature_homophily: float
    prompt_homo_homophily: float = 0.0
    prompt_hete_homophily: float = 0.0


@dataclass
class RunResult:
    dataset: str
    family: str
    rho: float
    pool_size: int
    num_nodes: int
    before: HomophilyStats
    after: HomophilyStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_homophily(edge_index: torch.Tensor, x: torch.Tensor) -> float:
    src, dst = edge_index
    valid = (src >= 0) & (dst >= 0) & (src < x.size(0)) & (dst < x.size(0))
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    x_norm = F.normalize(x, p=2, dim=-1)
    return float((x_norm[src] * x_norm[dst]).sum(dim=-1).mean().item())


class FrozenBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return torch.relu(self.lin2(torch.relu(self.lin1(x))))


class IdentityAligner(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _align_prompt_dim(prompt: torch.Tensor, target_dim: int) -> torch.Tensor:
    if prompt.size(-1) == target_dim:
        return prompt
    if prompt.size(-1) > target_dim:
        return prompt[:, :target_dim]
    pad = torch.zeros(
        prompt.size(0),
        target_dim - prompt.size(-1),
        device=prompt.device,
        dtype=prompt.dtype,
    )
    return torch.cat([prompt, pad], dim=-1)


def run_once(args, dataset_name: str, seed_offset: int = 0) -> RunResult:
    set_seed(args.seed + seed_offset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = resolve_dataset_name(dataset_name)
    profile = get_dataset_profile(dataset_name)
    rho = args.rho if args.rho is not None else profile.default_rho

    dataset, _ = load_dataset(dataset_name, args.data_root)
    data = dataset[0].to(device)

    backbone = FrozenBackbone(data.num_features, args.hidden_dim).to(device).eval()
    aligner = IdentityAligner().to(device).eval()
    generator = PromptGenerator(
        k_homo=args.k_homo,
        k_hete=args.k_hete,
        device=str(device),
        homophily_threshold=args.homophily_threshold,
        hetero_clusterer=args.hetero_clusterer,
        random_state=args.seed + seed_offset,
        graph_family=profile.family,
    )

    before = HomophilyStats(
        edge_homophily=estimate_edge_homophily(data.edge_index, data.y),
        feature_homophily=feature_homophily(data.edge_index, data.x),
    )

    p_homo, p_hete = generator.generate(data, backbone, aligner)
    p_homo = _align_prompt_dim(p_homo.detach(), data.x.size(-1))
    p_hete = _align_prompt_dim(p_hete.detach(), data.x.size(-1))
    prompt_nodes = torch.cat([p_homo, p_hete], dim=0)

    feat_dim = data.x.size(-1)
    router = PromptRouter.from_dataset_profile(
        profile,
        feature_dim=feat_dim,
        prompt_dim=feat_dim,
        k_homo=args.k_homo,
        k_hete=args.k_hete,
        tau=args.tau,
        rho=rho,
    ).to(device)

    routing = router(
        data.x,
        p_homo,
        p_hete,
        edge_index=data.edge_index,
        y=data.y,
        hard=args.hard_route,
        topk=args.topk_prompts,
        rho=rho,
    )

    aug_edge_index = data.edge_index
    if routing.prompt_edge_index.numel() > 0:
        aug_edge_index = torch.cat([data.edge_index, routing.prompt_edge_index], dim=1)

    aug_x = torch.cat([data.x, prompt_nodes], dim=0)
    aug_y = torch.cat(
        [data.y, torch.full((prompt_nodes.size(0),), -1, device=device, dtype=data.y.dtype)],
        dim=0,
    )

    after = HomophilyStats(
        edge_homophily=estimate_edge_homophily(aug_edge_index, aug_y),
        feature_homophily=feature_homophily(aug_edge_index, aug_x),
        prompt_homo_homophily=prompt_edge_homophily(
            routing.ex_homo, data.y, pool_mask=routing.pool_mask
        ),
        prompt_hete_homophily=prompt_edge_homophily(
            routing.ex_hete, data.y, pool_mask=routing.pool_mask
        ),
    )

    return RunResult(
        dataset=dataset_name,
        family=profile.family,
        rho=rho,
        pool_size=routing.pool_size,
        num_nodes=int(data.num_nodes),
        before=before,
        after=after,
    )


def _delta(before: float, after: float) -> float:
    return after - before


def print_run_summary(results: Sequence[RunResult]) -> None:
    if not results:
        return

    ds = results[0].dataset
    profile = get_dataset_profile(ds)
    edge_before = sum(r.before.edge_homophily for r in results) / len(results)
    edge_after = sum(r.after.edge_homophily for r in results) / len(results)
    feat_before = sum(r.before.feature_homophily for r in results) / len(results)
    feat_after = sum(r.after.feature_homophily for r in results) / len(results)
    pool_size = int(sum(r.pool_size for r in results) / len(results))
    homo_hp = sum(r.after.prompt_homo_homophily for r in results) / len(results)
    hete_hp = sum(r.after.prompt_hete_homophily for r in results) / len(results)

    print(
        f"[{ds}] family={results[0].family} | rho={results[0].rho:.2f} | "
        f"pool={pool_size}/{results[0].num_nodes} | "
        f"Edge Homophily {edge_before:.4f} -> {edge_after:.4f} ({_delta(edge_before, edge_after):+.4f}) | "
        f"Feature Homophily {feat_before:.4f} -> {feat_after:.4f} ({_delta(feat_before, feat_after):+.4f}) | "
        f"Prompt-Edge H(E) {homo_hp:.4f}/{hete_hp:.4f}"
    )
    print(f"  P_homo: {profile.p_homo_role}")
    print(f"  P_hete: {profile.p_hete_role}")


def evaluate_dataset(args, dataset_name: str) -> Dict[str, float]:
    results = [run_once(args, dataset_name, seed_offset=i) for i in range(args.runs)]
    print_run_summary(results)

    edge_gain = _delta(
        sum(r.before.edge_homophily for r in results) / len(results),
        sum(r.after.edge_homophily for r in results) / len(results),
    )
    feat_gain = _delta(
        sum(r.before.feature_homophily for r in results) / len(results),
        sum(r.after.feature_homophily for r in results) / len(results),
    )
    return {
        "edge_gain": edge_gain,
        "feat_gain": feat_gain,
        "edge_after": sum(r.after.edge_homophily for r in results) / len(results),
        "feat_after": sum(r.after.feature_homophily for r in results) / len(results),
        "prompt_homo_homophily": sum(r.after.prompt_homo_homophily for r in results) / len(results),
        "prompt_hete_homophily": sum(r.after.prompt_hete_homophily for r in results) / len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt homophily ablation (Top-rho + multi-dataset)")
    parser.add_argument("--dataset", type=str, default="Cora", help="Single dataset name")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Explicit dataset list, e.g. --datasets Cora Minesweeper Actor",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all 8 benchmark datasets")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--k_homo", type=int, default=10)
    parser.add_argument("--k_hete", type=int, default=10)
    parser.add_argument("--homophily_threshold", type=float, default=0.5)
    parser.add_argument("--hetero_clusterer", type=str, default="gmm", choices=["gmm", "spectral"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=None, help="Top-rho pool ratio; default from dataset profile")
    parser.add_argument("--hard_route", action="store_true")
    parser.add_argument("--topk_prompts", type=int, default=1)
    args = parser.parse_args()

    if args.all:
        dataset_list = list(ALL_DATASETS)
    elif args.datasets:
        dataset_list = [resolve_dataset_name(d) for d in args.datasets]
    else:
        dataset_list = [resolve_dataset_name(args.dataset)]

    print("=" * 72)
    print("Prompt Homophily Ablation | load_data + PromptRouter")
    print(f"Datasets: {', '.join(dataset_list)} | runs={args.runs}")
    print("=" * 72)

    summary_rows: List[Dict[str, object]] = []
    for ds in dataset_list:
        print(f"\n--- {ds} ---")
        metrics = evaluate_dataset(args, ds)
        summary_rows.append({"dataset": ds, **metrics})

    if len(summary_rows) > 1:
        print("\n" + "=" * 72)
        print("Cross-dataset summary (mean gain over runs)")
        print(f"{'Dataset':<16} {'Family':<14} {'Edge Δ':>10} {'Feature Δ':>12} {'Pass?':>8}")
        print("-" * 72)
        pass_count = 0
        for row in summary_rows:
            ds = str(row["dataset"])
            family = get_dataset_profile(ds).family
            edge_gain = float(row["edge_gain"])
            feat_gain = float(row["feat_gain"])
            ok = edge_gain > 0 and feat_gain > 0
            pass_count += int(ok)
            print(
                f"{ds:<16} {family:<14} {edge_gain:>+10.4f} {feat_gain:>+12.4f} "
                f"{'YES' if ok else 'NO':>8}"
            )
        print("-" * 72)
        print(f"Passed {pass_count}/{len(summary_rows)} datasets (both homophily metrics improved)")


if __name__ == "__main__":
    main()
