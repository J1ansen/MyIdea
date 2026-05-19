from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_data import load_cross_domain
from loss import LossConfig, compute_total_loss
from models.base_gnn import load_pretrained_backbone
from models.dual_branch import DualBranchGNN, GateSchedule
from models.prompt_conv import ORIG_EDGE, PROMPT_EDGE
from prompts.cluster_generator import PromptGenerator, TargetSSLPromptGenerator
from prompts.gumbel_route import PromptRouter
from prompts.routing_debug import collect_routing_debug, print_routing_debug


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


def parse_routing_debug_epochs(spec: str) -> Tuple[set, bool]:
    """解析 ``--routing_debug_epochs``：整数 epoch + 可选 ``end``/``final``。"""
    epoch_ids: set = set()
    dump_end = False
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("end", "final", "post"):
            dump_end = True
        else:
            epoch_ids.add(int(part))
    return epoch_ids, dump_end


def dump_routing_debug(
    label: str,
    *,
    router: PromptRouter,
    input_aligner: InputAligner,
    data,
    p_homo: nn.Parameter,
    p_hete: nn.Parameter,
    bundle,
    args,
) -> None:
    """打印一次路由诊断（logits / EX / 提示边）；``label`` 如 ``init`` / ``epoch-050``。"""
    device = data.x.device
    effective_rho = float(args.rho) if args.rho is not None else float(bundle.target_profile.default_rho)
    train_mode = args.hard_route
    print(f"\n>>> Routing debug [{label}] (train hard_route={train_mode}) <<<")
    with torch.no_grad():
        input_aligner.eval()
        router.eval()
        x_dbg = input_aligner(data.x)
        pool_y = data.y if args.router_use_labels else None
        dbg_routing = router(
            x_dbg,
            p_homo,
            p_hete,
            edge_index=data.edge_index,
            y=pool_y,
            hard=train_mode,
            topk=args.topk_prompts,
            rho=args.rho,
        )
        print_routing_debug(
            collect_routing_debug(
                dbg_routing,
                num_nodes=data.num_nodes,
                y=data.y,
                rho=effective_rho,
                topk=args.topk_prompts,
                k_homo=args.k_homo,
                k_hete=args.k_hete,
                mode="hard" if train_mode else "soft",
            )
        )
        pool = dbg_routing.pool_mask
        for name, logits in [("logits_homo", dbg_routing.logits_homo), ("logits_hete", dbg_routing.logits_hete)]:
            if pool is not None and pool.any():
                t = logits[pool].detach().float()
            else:
                t = logits.detach().float()
            print(
                f"  [{label}] {name} pool std={t.std(unbiased=False).item():.6f} "
                f"|range|={float((t.max() - t.min()).item()):.6f}"
            )
        if args.debug_routing_hard and not train_mode:
            dbg_hard = router(
                x_dbg,
                p_homo,
                p_hete,
                edge_index=data.edge_index,
                y=pool_y,
                hard=True,
                topk=args.topk_prompts,
                rho=args.rho,
            )
            print(f"\n>>> Routing debug [{label}] (extra hard) <<<")
            print_routing_debug(
                collect_routing_debug(
                    dbg_hard,
                    num_nodes=data.num_nodes,
                    y=data.y,
                    rho=effective_rho,
                    topk=args.topk_prompts,
                    k_homo=args.k_homo,
                    k_hete=args.k_hete,
                    mode="hard",
                )
            )


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


def resolve_router_tau(router: Optional[PromptRouter], args, epoch: int) -> float:
    if router is None:
        return 1.0
    if args.anneal_tau:
        return float(router.anneal_tau(epoch, args.epochs))
    return float(args.tau_start)


def save_trainable_checkpoint(
    path: str,
    *,
    model: DualBranchGNN,
    input_aligner: InputAligner,
    router: Optional[PromptRouter],
    p_homo: Optional[torch.Tensor],
    p_hete: Optional[torch.Tensor],
    best_epoch: int,
    best_val: float,
    best_test: float,
) -> None:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "input_aligner": input_aligner.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
        "best_test": float(best_test),
    }
    if router is not None:
        payload["router"] = router.state_dict()
    if isinstance(p_homo, nn.Parameter):
        payload["p_homo"] = p_homo.detach().cpu()
    if isinstance(p_hete, nn.Parameter):
        payload["p_hete"] = p_hete.detach().cpu()
    torch.save(payload, path)


def load_trainable_checkpoint(
    path: str,
    *,
    model: DualBranchGNN,
    input_aligner: InputAligner,
    router: Optional[PromptRouter],
    p_homo: Optional[torch.Tensor],
    p_hete: Optional[torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    input_aligner.load_state_dict(payload["input_aligner"])
    if router is not None and "router" in payload:
        router.load_state_dict(payload["router"])
    if isinstance(p_homo, nn.Parameter) and "p_homo" in payload:
        p_homo.data.copy_(payload["p_homo"].to(device))
    if isinstance(p_hete, nn.Parameter) and "p_hete" in payload:
        p_hete.data.copy_(payload["p_hete"].to(device))
    return {
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_val": float(payload.get("best_val", 0.0)),
        "best_test": float(payload.get("best_test", 0.0)),
    }


@torch.no_grad()
def evaluate_accuracies(
    *,
    model: DualBranchGNN,
    input_aligner: InputAligner,
    data,
    split: Split,
    device: torch.device,
    epoch: int,
    use_prompt_edges: bool,
    router: Optional[PromptRouter] = None,
    p_homo: Optional[torch.Tensor] = None,
    p_hete: Optional[torch.Tensor] = None,
    router_y: Optional[torch.Tensor] = None,
    args=None,
) -> Dict[str, float]:
    model.eval()
    input_aligner.eval()
    x_eval = input_aligner(data.x)
    if not use_prompt_edges:
        logits, _, _ = model(x_eval, data.edge_index, epoch=epoch)
    else:
        assert router is not None and p_homo is not None and p_hete is not None and args is not None
        router.eval()
        current_tau = resolve_router_tau(router, args, epoch)
        eval_routing = router(
            x_eval,
            p_homo,
            p_hete,
            edge_index=data.edge_index,
            y=router_y,
            hard=args.hard_route,
            topk=args.topk_prompts,
            rho=args.rho,
            tau=current_tau,
        )
        eval_prompt_nodes = torch.cat([p_homo, p_hete], dim=0)
        eval_full_edge_index = (
            torch.cat([data.edge_index, eval_routing.prompt_edge_index], dim=1)
            if eval_routing.prompt_edge_index.numel() > 0
            else data.edge_index
        )
        eval_edge_type = torch.cat(
            [
                torch.full((data.edge_index.size(1),), ORIG_EDGE, dtype=torch.long, device=device),
                torch.full(
                    (eval_routing.prompt_edge_index.size(1),),
                    PROMPT_EDGE,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=0,
        ) if eval_routing.prompt_edge_index.numel() > 0 else torch.full(
            (data.edge_index.size(1),), ORIG_EDGE, dtype=torch.long, device=device
        )
        logits, _, _ = model(
            x_eval,
            data.edge_index,
            prompt_nodes=eval_prompt_nodes,
            prompt_edge_index=eval_full_edge_index,
            edge_type=eval_edge_type,
            node_heterophily=eval_routing.local_heterophily,
            epoch=epoch,
        )
    pred = logits.argmax(dim=-1)
    return {
        "train": float((pred[split.train_mask] == data.y[split.train_mask]).float().mean().item()) if split.train_mask.any() else 0.0,
        "val": float((pred[split.val_mask] == data.y[split.val_mask]).float().mean().item()) if split.val_mask.any() else 0.0,
        "test": float((pred[split.test_mask] == data.y[split.test_mask]).float().mean().item()) if split.test_mask.any() else 0.0,
    }


def run_one(args, seed: int) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_dataset = args.target_dataset or args.dataset
    bundle = load_cross_domain(
        source_name=args.source_dataset,
        target_name=target_dataset,
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

    inferred_source_dim, inferred_hidden_dim = infer_checkpoint_dims(pretrained_path, device)
    source_dim = args.source_dim if args.source_dim is not None else inferred_source_dim
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else inferred_hidden_dim

    backbone = load_pretrained_backbone(pretrained_path, in_channels=source_dim, hidden_channels=hidden_dim, device=device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    input_aligner = InputAligner(bundle.target_in_dim, source_dim).to(device)

    use_prompt_edges = not args.no_prompt_edges
    if args.no_prompt_edges:
        print("[Ablation] --no_prompt_edges: 原图-only 双分支（对齐 test_pure_dual_branch）")

    if use_prompt_edges:
        # 若指定 --use_profile_k，优先读数据集 profile 中的 K 设定（按类别数设计）
        k_homo = args.k_homo
        k_hete = args.k_hete
        if args.use_profile_k:
            k_homo = bundle.target_profile.default_k_homo
            k_hete = bundle.target_profile.default_k_hete
            if k_hete < args.min_k_hete:
                print(
                    f"[Profile K] k_hete={k_hete} < min_k_hete={args.min_k_hete}，"
                    f"提升至 {args.min_k_hete}（避免单 prompt 退化）"
                )
                k_hete = args.min_k_hete
            print(
                f"[Profile K] {bundle.target_profile.name}: "
                f"k_homo={k_homo}, k_hete={k_hete} (from dataset profile)"
            )
        elif k_hete < args.min_k_hete:
            print(f"[K] k_hete={k_hete} 提升至 min_k_hete={args.min_k_hete}")
            k_hete = args.min_k_hete

        effective_rho = float(args.rho) if args.rho is not None else float(bundle.target_profile.default_rho)
        if args.prompt_init == "target_ssl":
            ssl_hidden = args.ssl_hidden_dim if args.ssl_hidden_dim is not None else hidden_dim
            ssl_out = args.ssl_out_dim if args.ssl_out_dim is not None else ssl_hidden
            generator = TargetSSLPromptGenerator(
                k_homo=k_homo,
                k_hete=k_hete,
                device=str(device),
                ssl_method=args.ssl_method,
                ssl_epochs=args.ssl_epochs,
                ssl_lr=args.ssl_lr,
                ssl_hidden_dim=ssl_hidden,
                ssl_out_dim=ssl_out,
                rho=effective_rho,
                feat_drop=args.ssl_feat_drop,
                edge_drop=args.ssl_edge_drop,
                random_state=seed,
                graph_family=bundle.target_profile.family,
                homophily_threshold=args.homophily_threshold,
            )
        else:
            generator = PromptGenerator(
                k_homo=k_homo,
                k_hete=k_hete,
                device=str(device),
                homophily_threshold=args.homophily_threshold,
                hetero_clusterer=args.hetero_clusterer,
                random_state=seed,
                graph_family=bundle.target_profile.family,
            )
        p_homo_init, p_hete_init, prompt_signals = generator.generate(
            data, backbone, input_aligner, return_signals=True
        )

        # 按 --freeze_prompts 决定 prompt 节点向量是否参与反传
        _p_homo_tensor = normalize_prompt_dim(p_homo_init, hidden_dim)
        _p_hete_tensor = normalize_prompt_dim(p_hete_init, hidden_dim)
        if args.freeze_prompts:
            # 冻结：只学 router.proj（路由权重 / EX），不更新 prompt 原型
            p_homo = _p_homo_tensor.detach().to(device)
            p_hete = _p_hete_tensor.detach().to(device)
        else:
            p_homo = nn.Parameter(_p_homo_tensor.to(device))
            p_hete = nn.Parameter(_p_hete_tensor.to(device))

        router = PromptRouter.from_dataset_profile(
            bundle.target_profile,
            feature_dim=source_dim,
            prompt_dim=hidden_dim,
            k_homo=k_homo,
            k_hete=k_hete,
            tau=args.tau_start,
            tau_end=args.tau_end,
            rho=args.rho,
        ).to(device)
    else:
        k_homo = k_hete = 0
        prompt_signals = type("PromptSignalsStub", (), {"edge_homophily": infer_edge_homophily(data)})()
        p_homo = p_hete = None
        router = None

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

    loss_cfg = LossConfig(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3_homo=args.lambda3,
        lambda3_hete=args.lambda3,
        contrastive_margin=args.contrastive_margin,
        contrastive_sample_size=args.contrastive_sample_size,
        contrastive_use_hard=args.hard_contrastive,
        normalize_contrastive=not args.no_normalize_contrastive,
        consist_max=args.consist_max,
        loss_warmup_epochs=args.loss_warmup_epochs,
    )

    trainable_params = list(model.parameters()) + list(input_aligner.parameters())
    if use_prompt_edges:
        # freeze_prompts=True：只训 router.proj（logit 头），不更新 prompt 原型向量
        if args.freeze_prompts:
            trainable_params = trainable_params + list(router.parameters())
            print(
                f"[freeze_prompts] p_homo/p_hete 已冻结，"
                f"只优化 router.proj (logits / EX) + dual_branch"
            )
        else:
            trainable_params = trainable_params + [p_homo, p_hete] + list(router.parameters())
    if use_prompt_edges and args.prompt_lr_scale != 1.0:
        prompt_param_ids = set()
        if not args.freeze_prompts:
            prompt_param_ids.update(id(p) for p in (p_homo, p_hete))
        prompt_param_ids.update(id(p) for p in router.parameters())
        prompt_params = [p for p in trainable_params if id(p) in prompt_param_ids]
        base_params = [p for p in trainable_params if id(p) not in prompt_param_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": args.lr},
                {"params": prompt_params, "lr": args.lr * args.prompt_lr_scale},
            ],
            weight_decay=args.weight_decay,
        )
        print(
            f"[Optimizer] base_lr={args.lr}, prompt/router_lr={args.lr * args.prompt_lr_scale:.6f}"
        )
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    routing_debug_epoch_set: set = set()
    routing_debug_at_end = False
    if use_prompt_edges and args.routing_debug_epochs:
        routing_debug_epoch_set, routing_debug_at_end = parse_routing_debug_epochs(args.routing_debug_epochs)

    router_y = data.y if (use_prompt_edges and args.router_use_labels) else None

    if use_prompt_edges and not args.no_routing_debug:
        dump_routing_debug(
            "init (before train)",
            router=router,
            input_aligner=input_aligner,
            data=data,
            p_homo=p_homo,
            p_hete=p_hete,
            bundle=bundle,
            args=args,
        )

    best_val = 0.0
    best_test = 0.0
    final_test = 0.0
    best_epoch = 0
    stopped_epoch = 0
    epochs_no_improve = 0
    early_stopped = False
    is_homophilic = bundle.target_profile.family == "homophilic"
    final_prompt_homo_homophily = 0.0
    final_prompt_hete_homophily = 0.0
    nan_streak = 0

    ckpt_path: Optional[str] = None
    if args.early_stop_patience > 0:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"seed_{seed}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        input_aligner.train()
        if router is not None:
            router.train()
        optimizer.zero_grad()

        x_aligned = input_aligner(data.x)  # shape: [N, source_dim]

        if not use_prompt_edges:
            logits, _, h_adapted, logits_frozen, logits_adapted = model(
                x_aligned,
                data.edge_index,
                epoch=epoch,
                return_branch_logits=True,
            )
            del h_adapted, logits_frozen, logits_adapted
            loss = F.cross_entropy(logits[split.train_mask], data.y[split.train_mask])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            acc = evaluate_accuracies(
                model=model,
                input_aligner=input_aligner,
                data=data,
                split=split,
                device=device,
                epoch=epoch,
                use_prompt_edges=False,
            )
            final_test = acc["test"]
            stopped_epoch = epoch
            if acc["val"] > best_val:
                best_val = acc["val"]
                best_test = acc["test"]
                best_epoch = epoch
                epochs_no_improve = 0
                if ckpt_path is not None:
                    save_trainable_checkpoint(
                        ckpt_path,
                        model=model,
                        input_aligner=input_aligner,
                        router=None,
                        p_homo=None,
                        p_hete=None,
                        best_epoch=best_epoch,
                        best_val=best_val,
                        best_test=best_test,
                    )
            else:
                epochs_no_improve += 1
            if epoch % args.log_every == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | Loss {float(loss.detach().item()):.4f} | CE {float(loss.detach().item()):.4f} | "
                    f"Gate {float(model.get_gate(epoch).detach().item()):.3f} | "
                    f"Val {acc['val']:.4f} | Test {acc['test']:.4f} | "
                    f"Best@{best_epoch} | [no_prompt_edges]"
                )
            if (
                ckpt_path is not None
                and epoch >= args.early_stop_min_epochs
                and epochs_no_improve >= args.early_stop_patience
            ):
                early_stopped = True
                print(
                    f"[EarlyStop] epoch {epoch}: val 连续 {epochs_no_improve} epoch 未提升 "
                    f"(best val={best_val:.4f} @ epoch {best_epoch})"
                )
                break
            continue

        loss_cfg.current_epoch = epoch
        current_tau = resolve_router_tau(router, args, epoch)

        routing = router(
            x_aligned,
            p_homo,
            p_hete,
            edge_index=data.edge_index,
            y=router_y,
            hard=args.hard_route,
            topk=args.topk_prompts,
            rho=args.rho,
            tau=current_tau,
        )
        prompt_nodes = torch.cat([p_homo, p_hete], dim=0)  # shape: [K1+K2, hidden_dim]
        full_edge_index = (
            torch.cat([data.edge_index, routing.prompt_edge_index], dim=1)
            if routing.prompt_edge_index.numel() > 0
            else data.edge_index
        )
        edge_type = torch.cat(
            [
                torch.full((data.edge_index.size(1),), ORIG_EDGE, dtype=torch.long, device=device),
                torch.full(
                    (routing.prompt_edge_index.size(1),),
                    PROMPT_EDGE,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=0,
        ) if routing.prompt_edge_index.numel() > 0 else torch.full(
            (data.edge_index.size(1),), ORIG_EDGE, dtype=torch.long, device=device
        )

        logits, _, h_adapted, logits_frozen, logits_adapted = model(
            x_aligned,
            data.edge_index,
            prompt_nodes=prompt_nodes,
            prompt_edge_index=full_edge_index,
            edge_type=edge_type,
            node_heterophily=routing.local_heterophily,
            epoch=epoch,
            return_branch_logits=True,
        )
        if not (
            torch.isfinite(logits).all()
            and torch.isfinite(logits_frozen).all()
            and torch.isfinite(logits_adapted).all()
        ):
            nan_streak += 1
            print(
                f"[WARN] Epoch {epoch:03d}: non-finite logits "
                f"(streak={nan_streak}), skip optimizer step"
            )
            if nan_streak >= args.nan_patience:
                print(f"[STOP] logits NaN/Inf 连续 {nan_streak} 次，提前结束本 run")
                break
            continue

        loss_out = compute_total_loss(
            logits=logits,
            labels=data.y,
            train_mask=split.train_mask,
            h_for_contrast=h_adapted,
            ex_homo=routing.ex_homo,
            ex_hete=routing.ex_hete,
            logits_frozen=logits_frozen,
            logits_adapted=logits_adapted,
            is_homophilic=is_homophilic,
            pool_mask=routing.pool_mask,
            full_y_for_metrics=data.y,
            cfg=loss_cfg,
        )
        if not torch.isfinite(loss_out.total):
            nan_streak += 1
            print(
                f"[WARN] Epoch {epoch:03d}: non-finite loss "
                f"(streak={nan_streak}), skip optimizer step"
            )
            if nan_streak >= args.nan_patience:
                print(f"[STOP] NaN/Inf 连续 {nan_streak} 次，提前结束本 run")
                break
            continue
        nan_streak = 0
        loss_out.total.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        acc = evaluate_accuracies(
            model=model,
            input_aligner=input_aligner,
            data=data,
            split=split,
            device=device,
            epoch=epoch,
            use_prompt_edges=True,
            router=router,
            p_homo=p_homo,
            p_hete=p_hete,
            router_y=router_y,
            args=args,
        )
        final_test = acc["test"]
        stopped_epoch = epoch
        log_items = loss_out.to_log_dict()
        final_prompt_homo_homophily = log_items["prompt_homo_homophily"]
        final_prompt_hete_homophily = log_items["prompt_hete_homophily"]
        if acc["val"] > best_val:
            best_val = acc["val"]
            best_test = acc["test"]
            best_epoch = epoch
            epochs_no_improve = 0
            if ckpt_path is not None:
                save_trainable_checkpoint(
                    ckpt_path,
                    model=model,
                    input_aligner=input_aligner,
                    router=router,
                    p_homo=p_homo,
                    p_hete=p_hete,
                    best_epoch=best_epoch,
                    best_val=best_val,
                    best_test=best_test,
                )
        else:
            epochs_no_improve += 1

        if epoch % args.log_every == 0 or epoch == 1:
            aux_scale = min(
                1.0,
                float(epoch) / float(max(loss_cfg.loss_warmup_epochs, 1)),
            )
            print(
                f"Epoch {epoch:03d} | Loss {log_items['total']:.4f} | CE {log_items['ce']:.4f} | "
                f"Sparse {log_items['sparse']:.4f} | Consist {log_items['consist']:.4f} | "
                f"Contrast(H/E) {log_items['contrast_homo']:.4f}/{log_items['contrast_hete']:.4f} | "
                f"Aux×{aux_scale:.2f} | P-H(E) {log_items['prompt_homo_homophily']:.3f}/"
                f"{log_items['prompt_hete_homophily']:.3f} | "
                f"Gate {float(model.get_gate(epoch).detach().item()):.3f} | "
                f"Tau {current_tau:.4f} | Pool {routing.pool_size} | "
                f"Val {acc['val']:.4f} | Test {acc['test']:.4f} | Best@{best_epoch}"
            )

        if (
            ckpt_path is not None
            and epoch >= args.early_stop_min_epochs
            and epochs_no_improve >= args.early_stop_patience
        ):
            early_stopped = True
            print(
                f"[EarlyStop] epoch {epoch}: val 连续 {epochs_no_improve} epoch 未提升 "
                f"(best val={best_val:.4f} @ epoch {best_epoch})"
            )
            break

        if use_prompt_edges and epoch in routing_debug_epoch_set:
            dump_routing_debug(
                f"epoch-{epoch:03d}",
                router=router,
                input_aligner=input_aligner,
                data=data,
                p_homo=p_homo,
                p_hete=p_hete,
                bundle=bundle,
                args=args,
            )

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        meta = load_trainable_checkpoint(
            ckpt_path,
            model=model,
            input_aligner=input_aligner,
            router=router if use_prompt_edges else None,
            p_homo=p_homo if use_prompt_edges else None,
            p_hete=p_hete if use_prompt_edges else None,
            device=device,
        )
        best_epoch = meta["best_epoch"]
        best_val = meta["best_val"]
        best_test = meta["best_test"]
        acc_best = evaluate_accuracies(
            model=model,
            input_aligner=input_aligner,
            data=data,
            split=split,
            device=device,
            epoch=best_epoch,
            use_prompt_edges=use_prompt_edges,
            router=router if use_prompt_edges else None,
            p_homo=p_homo if use_prompt_edges else None,
            p_hete=p_hete if use_prompt_edges else None,
            router_y=router_y if use_prompt_edges else None,
            args=args if use_prompt_edges else None,
        )
        final_test = acc_best["test"]
        if not args.keep_checkpoint:
            os.remove(ckpt_path)
        print(
            f"[Checkpoint] 已恢复 best @ epoch {best_epoch} | "
            f"Val {acc_best['val']:.4f} | Test {acc_best['test']:.4f}"
            + (" | early stop" if early_stopped else "")
        )

    if use_prompt_edges and routing_debug_at_end:
        dump_routing_debug(
            "post-train",
            router=router,
            input_aligner=input_aligner,
            data=data,
            p_homo=p_homo,
            p_hete=p_hete,
            bundle=bundle,
            args=args,
        )

    return {
        "best_val": best_val,
        "best_test": best_test,
        "final_test": final_test,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "early_stopped": early_stopped,
        "homophily": prompt_signals.edge_homophily,
        "prompt_homo_homophily": final_prompt_homo_homophily,
        "prompt_hete_homophily": final_prompt_hete_homophily,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable end-to-end 5-shot training pipeline")
    parser.add_argument("--source_dataset", type=str, default="PubMed")
    parser.add_argument("--target_dataset", type=str, default=None)
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
    parser.add_argument("--val_per_class", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=40,
        help="验证集连续 N epoch 无提升则早停；0 表示关闭（训满 epochs，不存 checkpoint）",
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        type=int,
        default=20,
        help="至少训练 N epoch 后才允许早停（避免 warmup 前误停）",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="早停时 best 权重保存目录（按 seed 命名）",
    )
    parser.add_argument(
        "--keep_checkpoint",
        action="store_true",
        help="run 结束后保留 checkpoint 文件（默认恢复后删除）",
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--prompt_lr_scale",
        type=float,
        default=0.25,
        help="router + prompt 原型相对 backbone/adapter 的学习率倍率",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda1", type=float, default=0.01, help="L_sparse 权重")
    parser.add_argument("--lambda2", type=float, default=0.05, help="L_consist 基准权重")
    parser.add_argument("--lambda3", type=float, default=0.1, help="L_contrastive 分路权重")
    parser.add_argument("--contrastive_margin", type=float, default=1.0)
    parser.add_argument("--contrastive_sample_size", type=int, default=512)
    parser.add_argument(
        "--hard_contrastive",
        action="store_true",
        help="对比损失用 argmax 判同 prompt（更抖，默认用软 EX 相似度）",
    )
    parser.add_argument(
        "--soft_contrastive",
        action="store_true",
        help="[已废弃] 请改用默认软对比；仅保留兼容",
    )
    parser.add_argument(
        "--no_normalize_contrastive",
        action="store_true",
        help="关闭对比损失前的嵌入 L2 归一化",
    )
    parser.add_argument(
        "--consist_max",
        type=float,
        default=1.0,
        help="对称 KL 一致性损失上限（≤0 表示不截断）",
    )
    parser.add_argument(
        "--loss_warmup_epochs",
        type=int,
        default=50,
        help="前 N epoch 线性增大 λ1/λ2/λ3，先稳定 CE",
    )
    parser.add_argument(
        "--nan_patience",
        type=int,
        default=3,
        help="连续多少次 non-finite loss 后提前结束本 run",
    )
    parser.add_argument(
        "--tau_start", type=float, default=1.0,
        help="Gumbel 温度（默认固定，不退火）",
    )
    parser.add_argument(
        "--tau_end", type=float, default=1.0,
        help="Gumbel 终态温度；仅 --anneal_tau 时从 tau_start 退火到此值",
    )
    parser.add_argument(
        "--anneal_tau",
        action="store_true",
        help="开启 Gumbel 温度指数退火（默认关闭，避免对比梯度爆炸）",
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="[兼容旧用法] 固定温度；若设置则覆盖 tau_start/tau_end（等同 tau_start=tau_end=该值）",
    )
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--hard_route", action="store_true")
    parser.add_argument("--topk_prompts", type=int, default=1)
    parser.add_argument("--gate_start", type=float, default=0.2)
    parser.add_argument("--gate_end", type=float, default=0.65)
    parser.add_argument("--gate_warmup_epochs", type=int, default=150)
    parser.add_argument("--gate_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--gate_learnable_scale", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--freeze_prompts",
        action="store_true",
        help="冻结 p_homo/p_hete，只优化 router.proj + 双分支",
    )
    parser.add_argument(
        "--no_freeze_prompts",
        dest="freeze_prompts",
        action="store_false",
        help="允许 p_homo/p_hete 参与反传（默认）",
    )
    parser.set_defaults(freeze_prompts=False)
    parser.add_argument(
        "--router_use_labels",
        action="store_true",
        help="Top-ρ 池选使用标签异配度（默认仅用特征，避免 5-shot 泄漏）",
    )
    parser.add_argument(
        "--min_k_hete",
        type=int,
        default=2,
        help="异配 prompt 最少数量（profile 给出 1 时自动抬高）",
    )
    parser.add_argument(
        "--use_profile_k",
        action="store_true",
        help="忽略 --k_homo/--k_hete，改用各数据集 profile 中的 default_k_homo/k_hete",
    )
    parser.add_argument(
        "--prompt_init",
        type=str,
        default="dual_freq",
        choices=["dual_freq", "target_ssl"],
        help="提示初始化：dual_freq=源域冻结GNN双频聚类；target_ssl=目标域GRACE/GAE/BGRL+均衡KMeans",
    )
    parser.add_argument(
        "--ssl_method",
        type=str,
        default="grace",
        choices=["grace", "gae", "bgrl"],
        help="--prompt_init target_ssl 时使用的无监督方法",
    )
    parser.add_argument("--ssl_epochs", type=int, default=200, help="目标域 SSL 预训练 epoch 数")
    parser.add_argument("--ssl_lr", type=float, default=0.01, help="目标域 SSL 学习率")
    parser.add_argument("--ssl_hidden_dim", type=int, default=None, help="SSL 编码器隐层维（默认=checkpoint hidden）")
    parser.add_argument("--ssl_out_dim", type=int, default=None, help="Z_target 维（默认=ssl_hidden_dim）")
    parser.add_argument("--ssl_feat_drop", type=float, default=0.2, help="GRACE/BGRL 特征 dropout")
    parser.add_argument("--ssl_edge_drop", type=float, default=0.2, help="GRACE/BGRL 边 dropout")
    parser.add_argument(
        "--no_prompt_edges",
        action="store_true",
        help="Ablation: 关闭 prompt 边与 router，仅用原图双分支 + CE（对齐 test_pure_dual_branch）",
    )
    parser.add_argument(
        "--no_routing_debug",
        action="store_true",
        help="关闭训练前 EX / 提示边统计打印",
    )
    parser.add_argument(
        "--debug_routing_hard",
        action="store_true",
        help="额外打印 hard Gumbel 路由下的 EX 统计（默认同训练 hard_route 一致）",
    )
    parser.add_argument(
        "--routing_debug_epochs",
        type=str,
        default="",
        help="训练过程中额外 dump 路由的 epoch，逗号分隔，如 50,end（end=训练结束后）",
    )
    args = parser.parse_args()

    # --tau 兼容旧用法：若显式设置则覆盖 tau_start/tau_end（固定温度，不退火）
    if args.tau is not None:
        args.tau_start = args.tau
        args.tau_end = args.tau

    if args.soft_contrastive and not args.hard_contrastive:
        print("[NOTE] --soft_contrastive 已为默认行为，无需再指定")

    if args.no_prompt_edges:
        print("=" * 72)
        print("Ablation mode: NO prompt edges (train.py ≈ test_pure_dual_branch)")
        print("=" * 72)

    results = []
    for r in range(args.runs):
        out = run_one(args, seed=args.seed + r)
        results.append(out)
        es_flag = "Y" if out["early_stopped"] else "N"
        print(
            f"Run {r+1}/{args.runs} | Best Val {out['best_val']:.4f} | "
            f"Best Test {out['best_test']:.4f} | ES Test {out['final_test']:.4f} | "
            f"Best@Ep {out['best_epoch']} | Stop@Ep {out['stopped_epoch']} | EarlyStop {es_flag} | "
            f"Homophily {out['homophily']:.4f}"
        )

    mean_best = sum(x["best_test"] for x in results) / len(results)
    mean_final = sum(x["final_test"] for x in results) / len(results)
    n_es = sum(1 for x in results if x["early_stopped"])
    print("\n===== Summary =====")
    print(f"Source -> Target: {args.source_dataset} -> {args.target_dataset or args.dataset}")
    print(f"5-shot runs: {args.runs}")
    if args.early_stop_patience > 0:
        print(
            f"Early stop: patience={args.early_stop_patience}, "
            f"min_epochs={args.early_stop_min_epochs} ({n_es}/{len(results)} runs stopped early)"
        )
    print(f"Mean Best Test Acc: {mean_best:.4f}")
    print(f"Mean ES Test Acc (best checkpoint): {mean_final:.4f}")


if __name__ == "__main__":
    main()
