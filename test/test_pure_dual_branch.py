"""Pure dual-branch ablation (Stage 3-4): no prompt module.

Cross-domain 5-shot classification with frozen + adapted branches only.
Uses unified ``load_cross_domain`` and optional pretrained GCN weights.
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
from models.dual_branch import DualBranchGNN, GateSchedule


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


class SimpleBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return torch.relu(self.lin2(torch.relu(self.lin1(x))))


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


def orthogonal_alignment_check(model: DualBranchGNN, x: torch.Tensor, edge_index: torch.Tensor) -> float:
    with torch.no_grad():
        _, h_frozen, h_adapted = model(x, edge_index)
    cos = F.cosine_similarity(h_frozen, h_adapted, dim=-1)
    return float(cos.mean().item())


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

    source_dim = bundle.source_in_dim
    hidden_dim = args.hidden_dim
    if args.pretrained_name:
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
    else:
        backbone = SimpleBackbone(bundle.source_in_dim, args.hidden_dim).to(device)
        hidden_dim = args.hidden_dim

    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    input_aligner = InputAligner(bundle.target_in_dim, source_dim).to(device)
    model = DualBranchGNN(
        pretrained_backbone=backbone,
        hidden_dim=hidden_dim,
        out_dim=bundle.target_num_classes,
        prompt_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        input_dim=source_dim,
        gate_schedule=GateSchedule(
            start=args.gate_start,
            end=args.gate_end,
            warmup_epochs=args.gate_warmup_epochs,
            mode=args.gate_schedule,
            learnable_scale=args.gate_learnable_scale,
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(input_aligner.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        input_aligner.train()
        optimizer.zero_grad()

        x_align = input_aligner(data.x)
        logits, _, _ = model(x_align, data.edge_index, epoch=epoch)
        loss = F.cross_entropy(logits[split.train_mask], data.y[split.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        input_aligner.eval()
        with torch.no_grad():
            x_align = input_aligner(data.x)
            logits, _, _ = model(x_align, data.edge_index, epoch=epoch)
            pred = logits.argmax(dim=-1)
            val_acc = float((pred[split.val_mask] == data.y[split.val_mask]).float().mean().item()) if split.val_mask.any() else 0.0
            test_acc = float((pred[split.test_mask] == data.y[split.test_mask]).float().mean().item()) if split.test_mask.any() else 0.0
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

    align_score = orthogonal_alignment_check(model, input_aligner(data.x), data.edge_index)
    return {
        "best_val": best_val,
        "best_test": best_test,
        "alignment_score": align_score,
        "final_gate": float(model.get_gate(args.epochs).detach().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure dual-branch baseline (no prompts)")
    parser.add_argument("--source_dataset", type=str, default="PubMed")
    parser.add_argument("--target_dataset", type=str, default="Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_gnns")
    parser.add_argument("--pretrained_name", type=str, default="PubMed_SimGRACE_GCN_1.pth")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--val_per_class", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gate_start", type=float, default=0.2)
    parser.add_argument("--gate_end", type=float, default=0.8)
    parser.add_argument("--gate_warmup_epochs", type=int, default=100)
    parser.add_argument("--gate_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--gate_learnable_scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 72)
    print(
        f"Pure Dual-Branch | {resolve_dataset_name(args.source_dataset)} -> "
        f"{resolve_dataset_name(args.target_dataset)} | "
        f"pretrained={args.pretrained_name or 'random backbone'}"
    )
    print("=" * 72)

    metrics = []
    for i in range(args.runs):
        out = run_once(args, seed_offset=i)
        metrics.append(out)
        print(
            f"Run {i + 1}/{args.runs} | Best Val: {out['best_val']:.4f} | "
            f"Best Test: {out['best_test']:.4f} | Align: {out['alignment_score']:.4f} | "
            f"Gate: {out['final_gate']:.3f}"
        )

    mean_test = sum(m["best_test"] for m in metrics) / len(metrics)
    mean_align = sum(m["alignment_score"] for m in metrics) / len(metrics)
    print(f"\nMean Best Test Acc over {args.runs} runs: {mean_test:.4f}")
    print(f"Mean Alignment Score over {args.runs} runs: {mean_align:.4f}")


if __name__ == "__main__":
    main()
