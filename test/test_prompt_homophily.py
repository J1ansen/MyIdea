from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork
import torch_geometric.transforms as T

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts.cluster_generator import PromptGenerator
from prompts.gumbel_route import GumbelRouter


@dataclass
class HomophilyStats:
    edge_homophily: float
    feature_homophily: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str, root: str):
    transform = T.NormalizeFeatures()
    if name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(root=root, name=name, transform=transform)
    return WikipediaNetwork(root=root, name=name, transform=transform)


def edge_homophily_ratio(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    src, dst = edge_index
    valid = (src >= 0) & (dst >= 0) & (src < y.numel()) & (dst < y.numel()) & (y[src] >= 0) & (y[dst] >= 0)
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    return float((y[src] == y[dst]).float().mean().item())


def feature_homophily(edge_index: torch.Tensor, x: torch.Tensor) -> float:
    src, dst = edge_index
    valid = (src >= 0) & (dst >= 0) & (src < x.size(0)) & (dst < x.size(0))
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    x_norm = F.normalize(x, p=2, dim=-1)
    return float((x_norm[src] * x_norm[dst]).sum(dim=-1).mean().item())


class FrozenBackbone(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin2(torch.relu(self.lin1(x))))


class IdentityAligner(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _align_prompt_dim(prompt: torch.Tensor, target_dim: int) -> torch.Tensor:
    if prompt.size(-1) == target_dim:
        return prompt
    if prompt.size(-1) > target_dim:
        return prompt[:, :target_dim]
    pad = torch.zeros(prompt.size(0), target_dim - prompt.size(-1), device=prompt.device, dtype=prompt.dtype)
    return torch.cat([prompt, pad], dim=-1)


def build_prompt_edges(ex_weights: torch.Tensor, num_nodes: int, prompt_offset: int, topk: int = 1) -> torch.Tensor:
    # ex_weights shape: [N, K]
    topk = max(1, min(topk, ex_weights.size(-1)))
    vals, idx = torch.topk(ex_weights, k=topk, dim=-1)
    edges = []
    for i in range(num_nodes):
        for j in idx[i].tolist():
            edges.append([i, prompt_offset + j])
            edges.append([prompt_offset + j, i])
    return torch.tensor(edges, dtype=torch.long, device=ex_weights.device).t().contiguous()


def run_once(args, seed_offset: int = 0) -> Dict[str, HomophilyStats]:
    set_seed(args.seed + seed_offset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset, args.data_root)
    data = dataset[0].to(device)

    backbone = FrozenBackbone(data.num_features, args.hidden_dim).to(device).eval()
    aligner = IdentityAligner().to(device).eval()
    generator = PromptGenerator(
        k_homo=args.k_homo,
        k_hete=args.k_hete,
        device=str(device),
        homophily_threshold=args.homophily_threshold,
        hetero_clusterer=args.hetero_clusterer,
    )

    before = HomophilyStats(edge_homophily_ratio(data.edge_index, data.y), feature_homophily(data.edge_index, data.x))
    p_homo, p_hete = generator.generate(data, backbone, aligner)
    p_homo = _align_prompt_dim(p_homo, data.x.size(-1))  # shape: [K1, F]
    p_hete = _align_prompt_dim(p_hete, data.x.size(-1))  # shape: [K2, F]
    prompt_nodes = torch.cat([p_homo, p_hete], dim=0)  # shape: [K1+K2, F]

    router = GumbelRouter(feature_dim=data.x.size(-1), prompt_dim=prompt_nodes.size(-1), tau=args.tau).to(device)
    ex_weights = router(data.x, prompt_nodes, hard=args.hard_route)  # shape: [N, K]
    prompt_edges = build_prompt_edges(ex_weights, data.num_nodes, data.num_nodes, topk=args.topk_prompts)
    aug_edge_index = torch.cat([data.edge_index, prompt_edges], dim=1)
    aug_x = torch.cat(
        [data.x, prompt_nodes],
        dim=0,
    )  # shape: [N+K, F]
    aug_y = torch.cat([data.y, torch.full((prompt_nodes.size(0),), -1, device=device, dtype=data.y.dtype)], dim=0)

    after = HomophilyStats(edge_homophily_ratio(aug_edge_index, aug_y), feature_homophily(aug_edge_index, aug_x))
    return {"before": before, "after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt homophily ablation test")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--k_homo", type=int, default=10)
    parser.add_argument("--k_hete", type=int, default=10)
    parser.add_argument("--homophily_threshold", type=float, default=0.5)
    parser.add_argument("--hetero_clusterer", type=str, default="gmm", choices=["gmm", "spectral"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard_route", action="store_true")
    parser.add_argument("--topk_prompts", type=int, default=1)
    args = parser.parse_args()

    before_edge, after_edge = [], []
    before_feat, after_feat = [], []
    for i in range(args.runs):
        stats = run_once(args, seed_offset=i)
        before_edge.append(stats["before"].edge_homophily)
        after_edge.append(stats["after"].edge_homophily)
        before_feat.append(stats["before"].feature_homophily)
        after_feat.append(stats["after"].feature_homophily)
        print(
            f"Run {i+1}/{args.runs} | Edge Homophily: {before_edge[-1]:.4f} -> {after_edge[-1]:.4f} | "
            f"Feature Homophily: {before_feat[-1]:.4f} -> {after_feat[-1]:.4f}"
        )

    print("\nAveraged over runs")
    print(f"Edge Homophily  before/after: {sum(before_edge)/len(before_edge):.4f} -> {sum(after_edge)/len(after_edge):.4f}")
    print(f"Feature Homophily before/after: {sum(before_feat)/len(before_feat):.4f} -> {sum(after_feat)/len(after_feat):.4f}")


if __name__ == "__main__":
    main()
