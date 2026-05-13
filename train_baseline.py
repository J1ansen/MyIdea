#测试baseline模块：无提示无Adapter
# train_baseline.py (项目根目录)
import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 导入你的自定义数据加载模块
# ==========================================
from models.base_gnn import load_pretrained_backbone
from load_data import load_node_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_masks(data, shots, val_node_num, test_node_num, seed, device):
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
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()

# ==========================================
# 极简 Baseline 模型定义
# ==========================================
class PureBaselineGNN(nn.Module):
    """
    纯天然 Baseline: 没有任何 Adapter 和 Prompt
    仅仅使用冻结的老专家提取特征，加一个线性分类器
    """
    def __init__(self, pretrained_gnn, hidden_channels, out_channels):
        super(PureBaselineGNN, self).__init__()
        
        # 1. 冻结的老专家
        self.frozen_branch = copy.deepcopy(pretrained_gnn)
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
            
        # 2. 简单的线性分类器
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def train(self, mode=True):
        """物理锁死老专家，防范 Dropout 随机性"""
        super().train(mode)
        self.frozen_branch.eval()
        return self

    def forward(self, x, edge_index):
        # 纯净前向传播
        with torch.no_grad():
            h_frozen = self.frozen_branch(x, edge_index)
            
        logits = self.classifier(h_frozen)
        return logits

# ==========================================
# 训练主干
# ==========================================
def train_once(args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== [Baseline] Seed {seed} =====")
    print(f"🚀 实验环境: {args.source_dataset} -> {args.target_dataset} (无提示, 无Adapter)")

    # 1. 加载数据
    data, target_in_dim, target_out_dim = load_node_data(args.target_dataset, "./data")
    data = data.to(device)
    train_mask, val_mask, test_mask = build_masks(
        data, args.shots, args.val_node_num, args.test_node_num, seed, device
    )

    # 2. 维度对齐层 (Aligner)
    aligner = nn.Linear(target_in_dim, args.source_dim).to(device)

    # 3. 加载预训练模型
    weight_file = f"./pretrained_gnns/{args.source_dataset}_SimGRACE_GCN_1.pth"
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"找不到权重文件 {weight_file}")
    pretrained_backbone = load_pretrained_backbone(
        weight_path=weight_file,
        in_channels=args.source_dim,
        hidden_channels=args.hidden_dim,
        device=device,
    )

    # 4. 初始化 Baseline 模型
    model = PureBaselineGNN(
        pretrained_gnn=pretrained_backbone,
        hidden_channels=args.hidden_dim,
        out_channels=target_out_dim
    ).to(device)

    # 5. 优化器 (极少参数：仅仅训练 aligner 和 classifier)
    trainable_params = list(model.classifier.parameters()) + list(aligner.parameters())
    print(f"📉 [Baseline] 可训练参数极小化，仅包含维度对齐和分类器头。")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 6. 训练循环
    best_val_acc = -1.0
    best_test_acc = 0.0
    wait = 0
    labels = data.y.squeeze()

    for epoch in range(1, args.epochs + 1):
        model.train()
        aligner.train()
        optimizer.zero_grad()

        x_aligned = aligner(data.x)
        logits = model(x_aligned, data.edge_index)
        
        # 唯一的监督信号
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            aligner.eval()
            with torch.no_grad():
                x_eval = aligner(data.x)
                logits_eval = model(x_eval, data.edge_index)
                train_acc = compute_acc(logits_eval, labels, train_mask)
                val_acc = compute_acc(logits_eval, labels, val_mask)
                test_acc = compute_acc(logits_eval, labels, test_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                wait = 0
            else:
                wait += 1

            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train/Val/Test: {train_acc*100:.2f}/{val_acc*100:.2f}/{test_acc*100:.2f} | BestVal: {best_val_acc*100:.2f} | BestTest: {best_test_acc*100:.2f}")

            if wait >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    print(f"\n🎉 Baseline Seed {seed} 完成！最终 Test Acc: {best_test_acc*100:.2f}%")
    return best_test_acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Amazon-ratings')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    active_seeds = [int(s.strip()) for s in args.seeds.split(',')]
    all_test_accs = []
    for seed in active_seeds:
        acc = train_once(args, seed)
        all_test_accs.append(acc)
        
    print("\n" + "=" * 44)
    print("📦 纯天然 Baseline (无Prompt/Adapter) 实验汇总")
    print("=" * 44)
    print(f"Test Acc: {np.mean(all_test_accs)*100:.2f}% ± {np.std(all_test_accs)*100:.2f}%")