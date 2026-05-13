# train.py (项目根目录)
import os
import copy
import random
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入图同配性计算工具
from torch_geometric.utils import homophily

# ==========================================
# 导入自定义模块
# ==========================================
from models.dual_branch import DualBranchCrossDomainGNN
from prompts.cluster_generator import PromptGenerator
from models.base_gnn import load_pretrained_backbone
from load_data import load_node_data


def set_seed(seed):
    """固定随机性，保证多次实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_masks(data, shots, val_node_num, test_node_num, seed, device):
    """
    构建 train/val/test 掩码：
    - train: 每类采样 shots 个节点（few-shot）
    - val/test: 从剩余节点中随机划分
    """
    rng = random.Random(seed)
    y = data.y.squeeze()
    num_classes = int(y.max().item() + 1)
    train_nodes = []

    for cls in range(num_classes):
        cls_indices = torch.where(y == cls)[0].tolist()
        if len(cls_indices) <= shots:
            train_nodes.extend(cls_indices)
        else:
            train_nodes.extend(rng.sample(cls_indices, k=shots))

    remain_nodes = [idx for idx in range(data.num_nodes) if idx not in set(train_nodes)]
    rng.shuffle(remain_nodes)

    val_nodes = remain_nodes[:val_node_num]
    test_nodes = remain_nodes[val_node_num:val_node_num + test_node_num]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    return train_mask, val_mask, test_mask


def compute_acc(logits, labels, mask):
    """在指定 mask 上计算准确率。"""
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def get_consistency_weight(epoch, args):
    """一致性损失权重调度"""
    base = args.lambda_consist
    warmup = args.consist_warmup_epochs
    min_ratio = args.consist_min_ratio
    if warmup <= 0 or epoch <= warmup:
        return base
    tail_epochs = max(1, args.epochs - warmup)
    decay_progress = min(1.0, (epoch - warmup) / tail_epochs)
    current_ratio = 1.0 - decay_progress * (1.0 - min_ratio)
    return base * current_ratio


def analyze_prompt_homophily(model, aligner, data, p_homo, p_hete, args):
    """训练后分析提示模块质量"""
    model.eval()
    aligner.eval()
    with torch.no_grad():
        x_eval = aligner(data.x)
        _, _, _, ex_total = model(
            x_eval, data.edge_index, p_homo, p_hete, hard_route=args.eval_hard_route
        )
        prompt_assignments = ex_total.argmax(dim=1)
        true_labels = data.y.squeeze()

        total_prompt_nodes = args.k_homo + args.k_hete
        purity_scores = []
        prompt_majority_labels = torch.full(
            (total_prompt_nodes,), -1, dtype=true_labels.dtype, device=true_labels.device
        )
        for k in range(total_prompt_nodes):
            connected_nodes = (prompt_assignments == k).nonzero(as_tuple=True)[0]
            if len(connected_nodes) > 1:
                labels_in_k = true_labels[connected_nodes]
                bincount = torch.bincount(labels_in_k)
                majority_class = bincount.argmax()
                prompt_majority_labels[k] = majority_class
                purity_scores.append(bincount[majority_class].item() / len(connected_nodes))

        original_edge_homophily = homophily(data.edge_index, true_labels)
        if len(purity_scores) == 0:
            return original_edge_homophily, 0.0, 0.0

        avg_prompt_purity = float(sum(purity_scores) / len(purity_scores))
        node_indices = torch.arange(data.num_nodes, device=true_labels.device)
        valid_node_mask = prompt_majority_labels[prompt_assignments] >= 0
        if valid_node_mask.sum().item() > 0:
            matched = (
                true_labels[node_indices[valid_node_mask]]
                == prompt_majority_labels[prompt_assignments[valid_node_mask]]
            )
            prompt_edge_homophily = matched.float().mean().item()
        else:
            prompt_edge_homophily = 0.0
        return original_edge_homophily, prompt_edge_homophily, avg_prompt_purity


def train_once(args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== Seed {seed} =====")
    print(f"🚀 实验环境: {args.source_dataset} -> {args.target_dataset}")

    # 1) 加载数据
    data, target_in_dim, target_out_dim = load_node_data(args.target_dataset, "./data")
    data = data.to(device)
    train_mask, val_mask, test_mask = build_masks(
        data, args.shots, args.val_node_num, args.test_node_num, seed, device
    )

    # 2) 初始化维度对齐层 (Aligner)
    # 使用 Sequential 结构，稍微增加复杂度以更好对齐源域特征空间
    aligner = nn.Sequential(
        nn.Linear(target_in_dim, args.source_dim),
        nn.ReLU(),
        nn.Linear(args.source_dim, args.source_dim)
    ).to(device)

    # 3) 加载源域预训练模型
    weight_file = f"./pretrained_gnns/{args.source_dataset}_SimGRACE_GCN_1.pth"
    pretrained_backbone = load_pretrained_backbone(
        weight_path=weight_file,
        in_channels=args.source_dim,
        hidden_channels=args.hidden_dim,
        device=device,
    )

    # 4) 【核心修改】：使用真实的 GNN 提取表征并生成静态提示
    print(f"\n🔮 正在基于预训练 GNN 提取 {args.target_dataset} 的同配/异配神经锚点...")
    generator = PromptGenerator(k_homo=args.k_homo, k_hete=args.k_hete, device=device)
    # 按照你的 idea：传入模型和对齐层来计算残差并聚类
    p_homo, p_hete = generator.generate(data, pretrained_backbone, aligner)

    # 5) 构建双分支模型
    model = DualBranchCrossDomainGNN(
        pretrained_gnn=pretrained_backbone,
        hidden_channels=args.hidden_dim,
        out_channels=target_out_dim,
        # 维度由 generator 决定：homo 是 hidden_dim, hete 是 2 * hidden_dim
        prompt_homo_dim=p_homo.size(-1),
        prompt_hete_dim=p_hete.size(-1),
        tau=args.tau,
        adapter_r=args.adapter_r
    ).to(device)

    # 6) 优化器
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(aligner.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 7) 训练循环
    best_val_acc = -1.0
    best_test_acc = 0.0
    best_state = None
    wait = 0
    labels = data.y.squeeze()

    for epoch in range(1, args.epochs + 1):
        model.train()
        aligner.train()
        optimizer.zero_grad()

        x_aligned = aligner(data.x)
        # 前向传播得到：预测值，冻结表征，纯净的适应表征(m_real)，路由矩阵
        logits, h_frozen, m_real, ex_total = model(
            x_aligned, data.edge_index, p_homo, p_hete, hard_route=args.train_hard_route
        )

        loss_cls = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss_sparse = -torch.mean(torch.sum(ex_total * torch.log(ex_total + 1e-8), dim=-1))
        
        # 【修复】：一致性损失只约束 Adapter 出来的 m_real，不干涉 Prompt 的纠偏功能
        if args.consist_on_train_only:
            loss_consist = F.mse_loss(m_real[train_mask], h_frozen[train_mask])
        else:
            loss_consist = F.mse_loss(m_real, h_frozen)

        # 路由分布一致性
        route_consist_terms = []
        train_labels = labels[train_mask]
        for cls in train_labels.unique():
            cls_mask = train_mask & (labels == cls)
            if cls_mask.sum().item() > 1:
                ex_cls = ex_total[cls_mask]
                cls_center = ex_cls.mean(dim=0, keepdim=True)
                route_consist_terms.append(F.mse_loss(ex_cls, cls_center.expand_as(ex_cls)))
        loss_route_consist = torch.stack(route_consist_terms).mean() if route_consist_terms else torch.tensor(0.0, device=device)

        consist_weight = get_consistency_weight(epoch, args)
        loss = (
            loss_cls
            + args.lambda_sparse * loss_sparse
            + consist_weight * loss_consist
            + args.lambda_route_consist * loss_route_consist
        )
        loss.backward()
        optimizer.step()
        
        # 路由器温度退火
        model.global_router.tau = max(args.min_tau, model.global_router.tau * args.tau_decay)

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            aligner.eval()
            with torch.no_grad():
                x_eval = aligner(data.x)
                logits_eval, _, _, _ = model(
                    x_eval, data.edge_index, p_homo, p_hete, hard_route=args.eval_hard_route
                )
                train_acc = compute_acc(logits_eval, labels, train_mask)
                val_acc = compute_acc(logits_eval, labels, val_mask)
                test_acc = compute_acc(logits_eval, labels, test_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                wait = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "aligner": copy.deepcopy(aligner.state_dict()),
                }
            else:
                wait += 1

            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} "
                f"(Cls: {loss_cls.item():.4f}, Sparse: {loss_sparse.item():.4f}, "
                f"Consist: {loss_consist.item():.4f}) "
                f"| Train/Val/Test: {train_acc*100:.2f}/{val_acc*100:.2f}/{test_acc*100:.2f} "
                f"| BestVal: {best_val_acc*100:.2f} | BestTest@BestVal: {best_test_acc*100:.2f}"
            )

            if wait >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        aligner.load_state_dict(best_state["aligner"])

    print(f"\n🎉 Seed {seed} 完成！Best Test Acc: {best_test_acc*100:.2f}%")
    edge_h, prompt_edge_h, prompt_purity = analyze_prompt_homophily(model, aligner, data, p_homo, p_hete, args)
    print(f"📉 原图边级同配率: {edge_h:.4f} | 📈 提示边级同配率: {prompt_edge_h:.4f} | 🧪 提示簇纯度 (Prompt Purity): {prompt_purity:.4f}")
    return best_test_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Amazon-ratings')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--adapter_r', type=int, default=8)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42')

    parser.add_argument('--k_homo', type=int, default=10)
    parser.add_argument('--k_hete', type=int, default=10)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau_decay', type=float, default=0.98)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--lambda_sparse', type=float, default=0.05)
    parser.add_argument('--lambda_consist', type=float, default=0.1)
    parser.add_argument('--consist_warmup_epochs', type=int, default=30)
    parser.add_argument('--consist_min_ratio', type=float, default=0.1)
    parser.add_argument('--consist_on_train_only', action='store_true')
    parser.add_argument('--lambda_route_consist', type=float, default=0.1)
    parser.add_argument('--train_hard_route', action='store_true')
    parser.add_argument('--eval_hard_route', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    active_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    all_test_accs = []
    for seed in active_seeds:
        acc = train_once(args, seed)
        all_test_accs.append(acc)
    
    print("\n" + "=" * 44)
    print("📦 最终复现实验汇总")
    print("=" * 44)
    print(f"Test Acc: {np.mean(all_test_accs)*100:.2f}% ± {np.std(all_test_accs)*100:.2f}%")