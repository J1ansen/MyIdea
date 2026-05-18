# test/test_dual_branch.py
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_data import load_node_data
from models.base_gnn import load_pretrained_backbone
from prompts.prototype_generator import PrototypePromptGenerator
from models.dual_branch import DualBranchGNN

# ==========================================
# 残差门控稀疏路由器 (实现 差异二: TOP x% 待连接池筛选)
# ==========================================
class SparseDiagnosisRouter(nn.Module):
    def __init__(self, residual_dim, tau=1.0, top_rate=0.3):
        super().__init__()
        self.tau = tau
        self.top_rate = top_rate
        
        # 极轻量级残差评估器 (病灶诊断)
        self.diagnosis_gate = nn.Sequential(
            nn.Linear(residual_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, h_query, p_anchor, residual, hard=False):
        # 1. 计算原图节点和提示节点之间的连接意愿分数 Logits 并做 GumbelSoftmax
        logits = torch.matmul(h_query, p_anchor.t())
        base_ex_weights = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)

        # 2. 计算每个节点的病灶严重程度 (求助倾向)
        g = self.diagnosis_gate(residual)

        # 3. 动态构建 TOP X% 待连接池 (完全贴合你的算法步骤 3)
        if self.top_rate < 1.0:
            k = max(1, int(g.size(0) * self.top_rate))
            # 挑出高频残差激活性最强的前 X% 个困难节点进入待连接池
            threshold = torch.topk(g.squeeze(), k).values[-1]
            
            # 其余好节点掩码设为 0，实行稀疏拓扑截断
            mask = (g >= threshold).float()
            g_sparse = mask * g
        else:
            g_sparse = g

        # 4. 生成最终稀疏化的提示邻接矩阵 EX
        sparse_ex_weights = base_ex_weights * g_sparse
        return sparse_ex_weights, base_ex_weights

# ==========================================
# 通用设置与工具
# ==========================================
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
        if len(cls_indices) <= shots: train_nodes.extend(cls_indices)
        else: train_nodes.extend(rng.sample(cls_indices, k=shots))
    remain = [i for i in range(data.num_nodes) if i not in set(train_nodes)]
    rng.shuffle(remain)
    masks = []
    for nodes in [train_nodes, remain[:val_node_num], remain[val_node_num:val_node_num+test_node_num]]:
        m = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        m[nodes] = True
        masks.append(m)
    return masks

def compute_acc(logits, labels, mask):
    if mask.sum().item() == 0: return 0.0
    return (logits.argmax(dim=1)[mask] == labels[mask]).float().mean().item()

# ==========================================
# 训练主循环
# ==========================================
def run_ablation_test(args, seed=42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 GP2F 双分支 + Top-{int(args.top_rate*100)}% 稀疏路由 有效性测试 | Target: {args.target_dataset}")
    print(f"{'='*60}")

    data_dir = os.path.join(project_root, "data")
    data, in_dim, out_dim = load_node_data(args.target_dataset, data_dir)
    data = data.to(device)
    train_m, val_m, test_m = build_masks(data, args.shots, args.val_node_num, args.test_node_num, seed, device)
    labels = data.y.squeeze()

    aligner = nn.Linear(in_dim, args.source_dim).to(device)
    weight_path = os.path.join(project_root, "pretrained_gnns", f"{args.source_dataset}_SimGRACE_GCN_1.pth")
    backbone = load_pretrained_backbone(weight_path, args.source_dim, args.hidden_dim, device).to(device)
    backbone.eval()

    # ==========================================
    # 🔄 阶段 0: 统一双频预热 & 生成原型
    # ==========================================
    print("\n🔄 准备阶段: 正在执行特征预热与原型初始化...")
    temp_clf_wu = nn.Linear(args.hidden_dim * 2, out_dim).to(device)
    opt_wu = torch.optim.Adam(list(aligner.parameters()) + list(temp_clf_wu.parameters()), lr=0.01)
    for _ in range(100):
        aligner.train()
        opt_wu.zero_grad()
        x_align = aligner(data.x)
        h_temp = backbone(x_align, data.edge_index)
        res = x_align[:, :h_temp.shape[-1]] - h_temp if x_align.shape[-1] != h_temp.shape[-1] else x_align - h_temp
        F.cross_entropy(temp_clf_wu(torch.cat([h_temp, res], dim=-1))[train_m], labels[train_m]).backward()
        opt_wu.step()
        
    aligner.eval()
    for param in aligner.parameters(): param.requires_grad = False

    generator = PrototypePromptGenerator(num_classes=out_dim, device=device)
    p_homo, p_hete = generator.generate(data, backbone, aligner, train_m)
    
    is_heterophilic = (args.target_dataset in ['Amazon-ratings', 'Minesweeper', 'Roman-empire', 'Squirrel', 'Cornell', 'Chameleon', 'Actor'])
    init_prompt = p_hete if is_heterophilic else p_homo
    prompt_dim = init_prompt.size(-1)

    # ==========================================
    # [基线] 仅冻结分支 + 线性分类器
    # ==========================================
    print("\n[Baseline] 评估仅冻结分支 (使用预热后的特征，无双分支)...")
    clf_b1 = nn.Linear(args.hidden_dim, out_dim).to(device)
    opt_b1 = torch.optim.Adam(clf_b1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_b1 = 0
    with torch.no_grad(): x_aligned_static = aligner(data.x)
    for _ in range(150):
        clf_b1.train()
        opt_b1.zero_grad()
        with torch.no_grad():
            h_froz = backbone(x_aligned_static, data.edge_index)
        loss = F.cross_entropy(clf_b1(h_froz)[train_m], labels[train_m])
        loss.backward()
        opt_b1.step()
        best_b1 = max(best_b1, compute_acc(clf_b1(h_froz), labels, test_m))
    print(f"📉 Baseline 最终准确率: {best_b1*100:.2f}%")

    # ==========================================
    # [主角] 完整 Dual-Branch (引入 Top-X% 稀疏路由)
    # ==========================================
    print("\n[Ours] 启动完整 Dual-Branch (显式重构 + 稀疏门控 + Adapter 微调)...")
    
    # 🔥 彻底消除 Bug 的核心：显式指定参数名初始化，绝不踩错位坑！
    dual_model = DualBranchGNN(
        pretrained_backbone=backbone, 
        hidden_dim=args.hidden_dim, 
        out_dim=out_dim, 
        prompt_dim=prompt_dim, 
        bottleneck_dim=32
    ).to(device)
    
    router = SparseDiagnosisRouter(residual_dim=args.hidden_dim, tau=args.tau, top_rate=args.top_rate).to(device)
    learnable_prompt = nn.Parameter(init_prompt.clone())
    
    # 仅微调轻量级组件及门控网络
    optimizer = torch.optim.Adam([
        {'params': dual_model.W_adapter.parameters()},
        {'params': dual_model.adapter.parameters()},
        {'params': dual_model.prompt_gate.parameters()},   # 内部提示门控
        {'params': dual_model.fusion_gate.parameters()},   # 外部融合门控
        {'params': dual_model.classifier.parameters()},
        {'params': router.parameters()},
        {'params': [learnable_prompt], 'lr': args.lr * 0.5} 
    ], lr=args.lr, weight_decay=args.weight_decay)

    best_ours = 0.0

    for epoch in range(1, 151): 
        dual_model.train()
        optimizer.zero_grad()
        
        with torch.no_grad():
            h_real = backbone(x_aligned_static, data.edge_index)
            
        res = x_aligned_static[:, :h_real.shape[-1]] - h_real if x_aligned_static.shape[-1] != h_real.shape[-1] else x_aligned_static - h_real
        
        if is_heterophilic:
            h_query = torch.cat([h_real, res], dim=-1)
        else:
            h_query = h_real
            
        # 激活稀疏连接，将残差特征残差矩阵送入路由器进行病灶挑选
        ex_weights_sparse, base_ex_weights = router(h_query, learnable_prompt, residual=res, hard=False)
        
        # 适应分支只处理被选中的前 X% 个极度受异配干扰的困难节点
        logits, _, _ = dual_model(x_aligned_static, data.edge_index, learnable_prompt, ex_weights_sparse)
        
        loss_cls = F.cross_entropy(logits[train_m], labels[train_m])
        loss_router = F.cross_entropy(base_ex_weights[train_m], labels[train_m])
        
        loss = loss_cls + 0.5 * loss_router
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            dual_model.eval()
            with torch.no_grad():
                ex_w_sparse_eval, _ = router(h_query, learnable_prompt, residual=res, hard=True)
                pred_logits, _, _ = dual_model(x_aligned_static, data.edge_index, learnable_prompt, ex_w_sparse_eval)
                t_acc = compute_acc(pred_logits, labels, test_m)
                best_ours = max(best_ours, t_acc)
                print(f"  Epoch {epoch:03d} | Ours Test Acc: {t_acc*100:.1f}%")

    print(f"\n🌟 Ours (Sparse Router + Adapter) 最终准确率: {best_ours*100:.2f}%")
    print(f"🚀 对比 Baseline 提升: +{(best_ours - best_b1)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Amazon-ratings')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--top_rate', type=float, default=0.3)
    args = parser.parse_args()
    run_ablation_test(args, 42)