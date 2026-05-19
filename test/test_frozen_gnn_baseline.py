"""Frozen GNN + linear probe baseline (prompts ❌, dual-branch ❌).

Cross-domain 5-shot node classification:
    1. Align target features to source dimension (orthogonal projection).
    2. Extract node embeddings with a **frozen** pretrained GCN.
    3. Train only a linear classifier on 5-shot labels.

Uses unified ``load_cross_domain`` (same protocol as ``train.py``).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_data import load_cross_domain, resolve_dataset_name
from models.base_gnn import load_pretrained_backbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class InputAligner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FrozenGNNProbe(nn.Module):
    """Frozen backbone + trainable linear head."""

    def __init__(self, backbone: nn.Module, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.classifier(h)


def infer_checkpoint_dims(weight_path: str, device: torch.device) -> Tuple[int, int]:
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

    for key, tensor in state_dict.items():
        if key.endswith("conv1.lin.weight") and tensor.ndim == 2:
            return int(tensor.shape[1]), int(tensor.shape[0])
        if key.endswith("conv1.weight") and tensor.ndim == 2:
            return int(tensor.shape[1]), int(tensor.shape[0])
        if key.endswith("lin1.weight") and tensor.ndim == 2:
            return int(tensor.shape[1]), int(tensor.shape[0])
    raise ValueError(f"Cannot infer dimensions from checkpoint: {weight_path}")


def run_once(args, seed_offset: int = 0) -> Dict[str, float]:
    seed = args.seed + seed_offset
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_cross_domain(
        source_name=resolve_dataset_name(args.source_dataset),
        target_name=resolve_dataset_name(args.target_dataset),
        root=args.data_root,
        shots=args.shots,
        val_per_class=args.val_per_class,
        seed=seed,
        device=device,
    )
    data = bundle.target_data
    split = bundle.target_split

    pretrained_path = os.path.join(args.pretrained_dir, args.pretrained_name)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    source_dim, hidden_dim = infer_checkpoint_dims(pretrained_path, device)
    backbone = load_pretrained_backbone(
        pretrained_path,
        in_channels=source_dim,
        hidden_channels=hidden_dim,
        device=device,
    )

    input_aligner = InputAligner(bundle.target_in_dim, source_dim).to(device)
    model = FrozenGNNProbe(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_classes=bundle.target_num_classes,
    ).to(device)

    trainable_params = list(input_aligner.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        input_aligner.train()
        optimizer.zero_grad()

        x_align = input_aligner(data.x)
        logits = model(x_align, data.edge_index)
        loss = F.cross_entropy(logits[split.train_mask], data.y[split.train_mask])
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        model.eval()
        input_aligner.eval()
        with torch.no_grad():
            x_align = input_aligner(data.x)
            logits = model(x_align, data.edge_index)
            pred = logits.argmax(dim=-1)
            val_acc = (
                float((pred[split.val_mask] == data.y[split.val_mask]).float().mean().item())
                if split.val_mask.any()
                else 0.0
            )
            test_acc = (
                float((pred[split.test_mask] == data.y[split.test_mask]).float().mean().item())
                if split.test_mask.any()
                else 0.0
            )
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

    return {
        "best_val": best_val,
        "best_test": best_test,
        "final_loss": float(loss.detach().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen GNN + linear probe baseline (no prompts, no dual-branch)"
    )
    parser.add_argument("--source_dataset", type=str, default="PubMed")
    parser.add_argument("--target_dataset", type=str, default="Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_gnns")
    parser.add_argument("--pretrained_name", type=str, default="PubMed_SimGRACE_GCN_1.pth")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--val_per_class", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 72)
    print(
        f"Frozen GNN + Linear Probe | {resolve_dataset_name(args.source_dataset)} -> "
        f"{resolve_dataset_name(args.target_dataset)} | "
        f"pretrained={args.pretrained_name}"
    )
    print("Trainable: InputAligner + linear head | Frozen: GCN backbone")
    print("=" * 72)

    metrics = []
    for i in range(args.runs):
        out = run_once(args, seed_offset=i)
        metrics.append(out)
        print(
            f"Run {i + 1}/{args.runs} | Best Val: {out['best_val']:.4f} | "
            f"Best Test: {out['best_test']:.4f}"
        )

    mean_test = sum(m["best_test"] for m in metrics) / len(metrics)
    mean_val = sum(m["best_val"] for m in metrics) / len(metrics)
    print(f"\nMean Best Val Acc over {args.runs} runs: {mean_val:.4f}")
    print(f"Mean Best Test Acc over {args.runs} runs: {mean_test:.4f}")


if __name__ == "__main__":
    main()
