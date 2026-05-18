from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss import consistency_loss, prompt_edge_contrastive_loss
from models.base_gnn import load_pretrained_backbone
from models.dual_branch import DualBranchGNN
from prompts.cluster_generator import PromptGenerator
from prompts.gumbel_route import GumbelRouter


@dataclass
class Split:
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


class InputAligner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptFeatureAligner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptLossWeights:
    def __init__(self, lambda1: float, lambda2: float, lambda3: float) -> None:
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str, root: str):
    transform = T.NormalizeFeatures()
    if name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(root=root, name=name, transform=transform)
    if name in {"Chameleon", "Squirrel"}:
        return WikipediaNetwork(root=root, name=name, transform=transform)
    if name == "Cornell":
        return WebKB(root=root, name=name, transform=transform)
    if name == "Actor":
        return Actor(root=root, transform=transform)
    return WikipediaNetwork(root=root, name=name, transform=transform)


def build_few_shot_masks(y: torch.Tensor, shots: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    num_classes = int(y.max().item() + 1)
    train_mask = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    for c in range(num_classes):
        idx = torch.where(y == c)[0].cpu().tolist()
        rng.shuffle(idx)
        train_idx = idx[:shots]
        val_idx = idx[shots : shots + 30]
        test_idx = idx[shots + 30 :]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return Split(train_mask, val_mask, test_mask)


def infer_edge_homophily(data) -> float:
    if data.y is None or data.edge_index is None:
        return 0.0
    src, dst = data.edge_index
    valid = (src >= 0) & (dst >= 0) & (src < data.y.numel()) & (dst < data.y.numel())
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    return float((data.y[src] == data.y[dst]).float().mean().item())


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


def normalize_prompt_dim(prompt: torch.Tensor, target_dim: int) -> torch.Tensor:
    if prompt.size(-1) == target_dim:
        return prompt
    if prompt.size(-1) > target_dim:
        return prompt[:, :target_dim]
    pad = torch.zeros(prompt.size(0), target_dim - prompt.size(-1), device=prompt.device, dtype=prompt.dtype)
    return torch.cat([prompt, pad], dim=-1)


def build_prompt_edges(
    ex_weights: torch.Tensor,
    num_nodes: int,
    topk_prompts: int,
    device: torch.device,
) -> torch.Tensor:
    topk_prompts = max(1, min(topk_prompts, ex_weights.size(-1)))
    topk_idx = torch.topk(ex_weights, k=topk_prompts, dim=-1).indices
    src, dst = [], []
    for i in range(num_nodes):
        for j in topk_idx[i].tolist():
            src.extend([i, num_nodes + j])
            dst.extend([num_nodes + j, i])
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def evaluate(model: DualBranchGNN, x: torch.Tensor, edge_index: torch.Tensor, split: Split, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x, edge_index)
        pred = logits.argmax(dim=-1)
        return {
            "train": float((pred[split.train_mask] == y[split.train_mask]).float().mean().item()) if split.train_mask.any() else 0.0,
            "val": float((pred[split.val_mask] == y[split.val_mask]).float().mean().item()) if split.val_mask.any() else 0.0,
            "test": float((pred[split.test_mask] == y[split.test_mask]).float().mean().item()) if split.test_mask.any() else 0.0,
        }


def run_one(args, seed: int) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(args.dataset, args.data_root)
    data = dataset[0].to(device)
    split = build_few_shot_masks(data.y, shots=args.shots, seed=seed)

    pretrained_path = os.path.join(args.pretrained_dir, args.pretrained_name)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    inferred_source_dim, inferred_hidden_dim = infer_checkpoint_dims(pretrained_path, device)
    source_dim = args.source_dim if args.source_dim is not None else inferred_source_dim
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else inferred_hidden_dim

    backbone = load_pretrained_backbone(pretrained_path, in_channels=source_dim, hidden_channels=hidden_dim, device=device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    input_aligner = InputAligner(data.num_features, source_dim).to(device)

    # Prompt generation uses source-aligned inputs and hidden-sized prompt prototypes.
    generator = PromptGenerator(
        k_homo=args.k_homo,
        k_hete=args.k_hete,
        device=str(device),
        homophily_threshold=args.homophily_threshold,
        hetero_clusterer=args.hetero_clusterer,
    )
    p_homo, p_hete = generator.generate(data, backbone, input_aligner)
    p_homo = normalize_prompt_dim(p_homo, hidden_dim)  # shape: [K1, hidden_dim]
    p_hete = normalize_prompt_dim(p_hete, hidden_dim)  # shape: [K2, hidden_dim]
    prompt_nodes_init = torch.cat([p_homo, p_hete], dim=0).detach()  # shape: [K1+K2, hidden_dim]

    # Stable prompt feature alignment into hidden space. Recomputed every epoch.
    prompt_feature_aligner = PromptFeatureAligner(prompt_nodes_init.size(-1), hidden_dim).to(device)

    # Differentiable routing from source-aligned node features to prompt nodes.
    router = GumbelRouter(feature_dim=source_dim, prompt_dim=hidden_dim, tau=args.tau).to(device)

    model = DualBranchGNN(
        pretrained_backbone=backbone,
        hidden_dim=hidden_dim,
        out_dim=dataset.num_classes,
        prompt_dim=hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        input_dim=source_dim,
    ).to(device)

    trainable_params = list(model.parameters()) + list(input_aligner.parameters()) + list(prompt_feature_aligner.parameters()) + list(router.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    best_test = 0.0
    final_test = 0.0
    is_homophilic = infer_edge_homophily(data) >= args.homophily_threshold

    for epoch in range(1, args.epochs + 1):
        model.train()
        input_aligner.train()
        prompt_feature_aligner.train()
        router.train()
        optimizer.zero_grad()

        x_aligned = input_aligner(data.x)  # shape: [N, source_dim]
        prompt_nodes = prompt_feature_aligner(prompt_nodes_init)  # shape: [K, hidden_dim]
        ex_weights = router(x_aligned, prompt_nodes, hard=args.hard_route)  # shape: [N, K]
        prompt_edge_index = build_prompt_edges(ex_weights, data.num_nodes, args.topk_prompts, device=device)
        full_edge_index = torch.cat([data.edge_index, prompt_edge_index], dim=1) if prompt_edge_index.numel() > 0 else data.edge_index
        edge_type = torch.cat(
            [
                torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device),
                torch.ones(prompt_edge_index.size(1), dtype=torch.long, device=device),
            ],
            dim=0,
        ) if prompt_edge_index.numel() > 0 else torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device)

        logits, _, h_adapted = model(
            x_aligned,
            data.edge_index,
            prompt_nodes=prompt_nodes,
            prompt_edge_index=full_edge_index,
            edge_type=edge_type,
        )

        train_logits = logits[split.train_mask]
        train_y = data.y[split.train_mask]

        ce_loss = F.cross_entropy(train_logits, train_y)
        sparse_loss = ex_weights.mean()
        consist_loss = consistency_loss(logits, logits.detach()) if logits.size(0) > 1 else torch.tensor(0.0, device=device)
        contrastive_loss = prompt_edge_contrastive_loss(h_adapted, ex_weights)

        total_loss = ce_loss + args.lambda1 * sparse_loss + (args.lambda2 * (0.25 if is_homophilic else 2.0)) * consist_loss + args.lambda3 * contrastive_loss
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        acc = evaluate(model, x_aligned.detach(), data.edge_index, split, data.y)
        final_test = acc["test"]
        if acc["val"] > best_val:
            best_val = acc["val"]
            best_test = acc["test"]

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {float(total_loss.item()):.4f} | CE {float(ce_loss.item()):.4f} | "
                f"Sparse {float(sparse_loss.item()):.4f} | Consist {float(consist_loss.item()):.4f} | "
                f"Contrast {float(contrastive_loss.item()):.4f} | Val {acc['val']:.4f} | Test {acc['test']:.4f}"
            )

    return {
        "best_val": best_val,
        "best_test": best_test,
        "final_test": final_test,
        "homophily": infer_edge_homophily(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable end-to-end 5-shot training pipeline")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_gnns")
    parser.add_argument("--pretrained_name", type=str, default="PubMed_SimGRACE_GCN_1.pth")
    parser.add_argument("--source_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    parser.add_argument("--k_homo", type=int, default=10)
    parser.add_argument("--k_hete", type=int, default=10)
    parser.add_argument("--homophily_threshold", type=float, default=0.5)
    parser.add_argument("--hetero_clusterer", type=str, default="gmm", choices=["gmm", "spectral"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard_route", action="store_true")
    parser.add_argument("--topk_prompts", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    results = []
    for r in range(args.runs):
        out = run_one(args, seed=args.seed + r)
        results.append(out)
        print(
            f"Run {r+1}/{args.runs} | Best Val {out['best_val']:.4f} | "
            f"Best Test {out['best_test']:.4f} | Final Test {out['final_test']:.4f} | Homophily {out['homophily']:.4f}"
        )

    mean_best = sum(x["best_test"] for x in results) / len(results)
    mean_final = sum(x["final_test"] for x in results) / len(results)
    print("\n===== Summary =====")
    print(f"Dataset: {args.dataset}")
    print(f"5-shot runs: {args.runs}")
    print(f"Mean Best Test Acc: {mean_best:.4f}")
    print(f"Mean Final Test Acc: {mean_final:.4f}")


if __name__ == "__main__":
    main()
