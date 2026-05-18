# test/test_prompt_module.py
import os
import sys

# ==========================================
# 🔥 核心修复1：把项目的根目录加入 Python 搜索路径
# 获取当前脚本的绝对路径，向上推两层得到项目根目录 (MyIdea/)
# 这使得 Python 无论在哪个目录下运行该脚本，都能正确 import 模型和数据
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入基础加载模块 (现在 Python 能在 MyIdea/ 根目录下找到它们了)
from models.base_gnn import load_pretrained_backbone
from load_data import load_node_data
from prompts.prototype_generator import PrototypePromptGenerator

# ==========================================
# 简化的零参数 Gumbel 路由器
# ==========================================
class SimpleGumbelRouter(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, h_query, p_anchor, hard=False):
        # h_query 和 p_anchor 必须在同一个特征维度空间
        logits = torch.matmul(h_query, p_anchor.t())
        ex_weights = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        return ex_weights

# ==========================================
# 实验模型：检验提示特征融合的极简骨架
# ==========================================
class PromptValidationGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_dim, tau=1.0):
        super().__init__()
        self.backbone = copy.deepcopy(pretrained_gnn)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.router = SimpleGumbelRouter(tau=tau)
        
        # 将提示节点特征降维，以便与 h_real (低频特征) 相加融合
        self.w_adapter = nn.Linear(prompt_dim, hidden_channels)
        self.prompt_scale = nn.Parameter(torch.tensor(0.01))
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_aligned, edge_index, prompt_nodes, is_hete=False, hard_route=False):
        with torch.no_grad():
            h_real = self.backbone(x_aligned, edge_index)
            
        # 🔥 核心：确保路由查询特征与提示节点的维度“门当户对”
        if is_hete:
            # 异配寻址：使用双频 [H, R]
            if x_aligned.shape[-1] != h_real.shape[-1]:
                residual = x_aligned[:, :h_real.shape[-1]] - h_real
            else:
                residual = x_aligned - h_real
            h_query = torch.cat([h_real, residual], dim=-1)
        else:
            # 同配寻址：使用低频 [H]
            h_query = h_real
            
        # 1. 在高维空间精准路由
        ex_weights = self.router(h_query, prompt_nodes, hard=hard_route)
        
        # 2. 消息降维与融合
        p_aligned = self.w_adapter(prompt_nodes)
        m_prompt = torch.matmul(ex_weights, p_aligned)
        
        h_fused = h_real + self.prompt_scale * m_prompt
        logits = self.classifier(h_fused)
        
        return logits, ex_weights

# ==========================================
# 辅助与探针函数
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

def probe_prototype_quality(h_features, p_nodes, labels, mask, name="Prompt"):
    sim_matrix = torch.matmul(h_features[mask], p_nodes.t())
    pseudo_labels = sim_matrix.argmax(dim=-1)
    acc = (pseudo_labels == labels[mask]).float().mean().item()
    print(f"  -> [{name}] 在测试集上的零样本最近邻匹配准确率: {acc*100:.2f}%")
    return acc

# ==========================================
# 训练主循环
# ==========================================
def train_once(args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 提示模块有效性沙盒验证 | Target: {args.target_dataset}")

    # 🔥 核心修复2：使用基于根目录的绝对路径
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
    # 🔥 阶段 0: 双频跨域语义对齐预热 (Dual-Freq Warm-up)
    # ==========================================
    print("\n🔥 阶段 0: 正在执行 [双频] 跨域语义对齐预热...")
    temp_clf = nn.Linear(args.hidden_dim * 2, out_dim).to(device)
    opt_warmup = torch.optim.Adam(list(aligner.parameters()) + list(temp_clf.parameters()), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(1, 101):
        aligner.train()
        opt_warmup.zero_grad()
        x_align = aligner(data.x)
        h_temp = backbone(x_align, data.edge_index)
        
        # 提取高频残差
        if x_align.shape[-1] != h_temp.shape[-1]:
            res = x_align[:, :h_temp.shape[-1]] - h_temp
        else:
            res = x_align - h_temp
            
        # 拼接双频特征
        h_aug = torch.cat([h_temp, res], dim=-1)
        out_temp = temp_clf(h_aug)
        
        loss = F.cross_entropy(out_temp[train_m], labels[train_m])
        loss.backward()
        opt_warmup.step()
    print("✅ 预热完成！翻译器已学会保留高低频融合特征。")

    # ==========================================
    # 阶段 1: 提取原型提示 
    # ==========================================
    generator = PrototypePromptGenerator(num_classes=out_dim, device=device)
    p_homo, p_hete = generator.generate(data, backbone, aligner, train_m)

    # ==========================================
    # 阶段 2: 探针测试
    # ==========================================
    print("\n🔬 正在执行探针测试 (Prototype Quality Probing)...")
    with torch.no_grad():
        aligner.eval()
        x_aligned = aligner(data.x)
        h_real = backbone(x_aligned, data.edge_index)
        if x_aligned.shape[-1] != h_real.shape[-1]:
            residual = x_aligned[:, :h_real.shape[-1]] - h_real
        else:
            residual = x_aligned - h_real
        h_aug = torch.cat([h_real, residual], dim=-1)

        probe_prototype_quality(h_real, p_homo, labels, test_m, "同配提示 P_homo")
        probe_prototype_quality(h_aug, p_hete, labels, test_m, "异配提示 P_hete")
    print("-" * 50)

    # ==========================================
    # 阶段 3: 主模型路由验证训练
    # ==========================================
    is_heterophilic = (args.target_dataset in ['Amazon-ratings', 'Minesweeper', 'Roman-empire', 'Squirrel', 'Cornell', 'Chameleon', 'Actor'])
    test_p = p_hete if is_heterophilic else p_homo
    target_prompt_name = "P_hete" if is_heterophilic else "P_homo"
    
    print(f"🚀 开始训练验证路由器的连接能力 (当前测试目标: {target_prompt_name})...")
    
    model = PromptValidationGNN(backbone, args.hidden_dim, out_dim, test_p.size(-1), args.tau).to(device)

    # 🔥 核心修复 1: 将静态的提示节点转化为可学习的参数！
    learnable_prompt = nn.Parameter(test_p.clone())
    
    # 🔥 核心修复 2: 冻结 Aligner！预热完就不要再动它了，防止特征空间崩溃
    aligner.eval()
    for param in aligner.parameters():
        param.requires_grad = False

    # 优化器现在只负责更新模型(如 classifier 等) 和 我们刚刚激活的可学习提示节点
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [learnable_prompt], 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    best_test = 0.0

    for epoch in range(1, 101): 
        model.train()
        # 注意: aligner 不再调用 .train()
        optimizer.zero_grad()
        
        # 将 learnable_prompt 传给模型
        logits, ex_weights = model(aligner(data.x), data.edge_index, learnable_prompt, is_hete=is_heterophilic)
        
        loss_cls = F.cross_entropy(logits[train_m], labels[train_m])
        loss_router = F.cross_entropy(ex_weights[train_m], labels[train_m])
        loss = loss_cls + 0.5 * loss_router
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                # 测试时也使用更新后的 learnable_prompt
                pred, ex_w = model(aligner(data.x), data.edge_index, learnable_prompt, is_hete=is_heterophilic, hard_route=True)
                t_acc = compute_acc(pred, labels, test_m)
                
                route_preds = ex_w.argmax(dim=-1)[test_m]
                route_purity = (route_preds == labels[test_m]).float().mean().item()
                
                best_test = max(best_test, t_acc)
                print(f"Epoch {epoch:03d} | Test Acc: {t_acc*100:.1f}% | 🔗 Router分配纯度: {route_purity*100:.1f}%")

    print(f"\n🎉 验证结束！极简验证 GNN 最终最高准确率: {best_test*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Cora')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--tau', type=float, default=1.0)
    args = parser.parse_args()
    train_once(args, 42)