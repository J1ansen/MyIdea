from __future__ import annotations

import argparse
import random
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

from models.dual_branch import DualBranchGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str, root: str):
    transform = T.NormalizeFeatures()
    if name in {"Cora", "CiteSeer", "PubMed"}:
        dataset = Planetoid(root=root, name=name, transform=transform)
    else:
        dataset = WikipediaNetwork(root=root, name=name, transform=transform)
    return dataset


class InputAligner(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SimpleBackbone(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin2(torch.relu(self.lin1(x))))


def build_masks(y: torch.Tensor, shots: int = 5):
    num_classes = int(y.max().item() + 1)
    train_mask = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    for c in range(num_classes):
        idx = torch.where(y == c)[0]
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), device=y.device)]
        train = perm[:shots]
        val = perm[shots:shots + 20]
        test = perm[shots + 20:]
        train_mask[train] = True
        val_mask[val] = True
        test_mask[test] = True
    return train_mask, val_mask, test_mask


def orthogonal_alignment_check(model: DualBranchGNN, x: torch.Tensor, edge_index: torch.Tensor) -> float:
    with torch.no_grad():
        _, h_frozen, h_adapted = model(x, edge_index)
    cos = F.cosine_similarity(h_frozen, h_adapted, dim=-1)
    return float(cos.mean().item())


def run_once(args, seed_offset: int = 0) -> Dict[str, float]:
    seed = args.seed + seed_offset
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_ds = load_dataset(args.source_dataset, args.data_root)
    tgt_ds = load_dataset(args.target_dataset, args.data_root)
    tgt = tgt_ds[0].to(device)
    tgt.train_mask, tgt.val_mask, tgt.test_mask = build_masks(tgt.y, shots=args.shots)

    input_aligner = InputAligner(tgt.num_features, src_ds.num_features).to(device)
    backbone = SimpleBackbone(src_ds.num_features, args.hidden_dim).to(device)
    model = DualBranchGNN(
        pretrained_backbone=backbone,
        hidden_dim=args.hidden_dim,
        out_dim=tgt_ds.num_classes,
        prompt_dim=args.prompt_dim,
        bottleneck_dim=args.bottleneck_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(input_aligner.parameters()),
        lr=args.lr,
    )
    best_val, best_test = 0.0, 0.0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        x_align = input_aligner(tgt.x)
        logits, _, _ = model(x_align, tgt.edge_index)
        loss = F.cross_entropy(logits[tgt.train_mask], tgt.y[tgt.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x_align = input_aligner(tgt.x)
            logits, _, _ = model(x_align, tgt.edge_index)
            pred = logits.argmax(dim=-1)
            val_acc = float((pred[tgt.val_mask] == tgt.y[tgt.val_mask]).float().mean().item())
            test_acc = float((pred[tgt.test_mask] == tgt.y[tgt.test_mask]).float().mean().item())
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

    align_score = orthogonal_alignment_check(model, input_aligner(tgt.x), tgt.edge_index)
    return {
        "best_val": best_val,
        "best_test": best_test,
        "alignment_score": align_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure dual-branch baseline test")
    parser.add_argument("--source_dataset", type=str, default="Cora")
    parser.add_argument("--target_dataset", type=str, default="CiteSeer")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--prompt_dim", type=int, default=128)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    metrics = []
    for i in range(args.runs):
        out = run_once(args, seed_offset=i)
        metrics.append(out)
        print(f"Run {i+1}/{args.runs} | Best Val: {out['best_val']:.4f} | Final Test: {out['best_test']:.4f} | Align: {out['alignment_score']:.4f}")

    mean_test = sum(m["best_test"] for m in metrics) / len(metrics)
    mean_align = sum(m["alignment_score"] for m in metrics) / len(metrics)
    print(f"Mean Final Test Acc over {args.runs} runs: {mean_test:.4f}")
    print(f"Mean Alignment Score over {args.runs} runs: {mean_align:.4f}")


if __name__ == "__main__":
    main()
