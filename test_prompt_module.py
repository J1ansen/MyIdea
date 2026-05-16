# test_prompt_module.py
import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans 

# 导入图同配性计算工具
from torch_geometric.utils import homophily

# ==========================================
# 导入基础加载模块 (请确保这些文件在同级目录)
# ==========================================
from models.base_gnn import load_pretrained_backbone
from load_data import load_node_data

# ==========================================
# 简化版只支持异配的 Gumbel 路由器
# ==========================================
class SimpleHeteroGumbelRouter(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, h_node, p_hete, hard=False):
        # 纯点积计算相似度 [N, K_hete]
        logits = torch.matmul(h_node, p_hete.t())
        
        # Gumbel Softmax 路由分配
        ex_hete = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        
        return ex_hete

# ==========================================
# 1. 提示生成器：仅生成异配神经表征
# ==========================================
class PromptGenerator:
    def __init__(self, k_hete=3, device='cpu'):
        self.k_hete = k_hete
        self.device = device

    @torch.no_grad()
    def generate(self, data, model, aligner):
        model.eval()
        aligner.eval()
        
        x_aligned = aligner(data.x)
        h = model(x_aligned, data.edge_index)
        
        # 计算残差并拼接 [H, R]
        feat_dim = h.shape[-1]
        residual = x_aligned[:, :feat_dim] - h
        h_aug = torch.cat([h, residual], dim=-1)

        print(f"--- 正在执行聚类初始化 (Seed 42) ---")
        print(f"同配提示节点已关闭 (K1=0)")
        print(f"正在生成异配提示节点 (K2={self.k_hete})...")
        kt = KMeans(n_clusters=self.k_hete, random_state=42, n_init='auto')
        kt.fit(h_aug.cpu().numpy())
        p_hete = torch.tensor(kt.cluster_centers_, dtype=torch.float32).to(self.device)

        return p_hete

# ==========================================
# 2. 实验模型：HeteroPromptOnlyGNN 
# ==========================================
class HeteroPromptOnlyGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_hete_dim, tau=1.0):
        super(HeteroPromptOnlyGNN, self).__init__()
        # 冻结源域骨架
        self.backbone = copy.deepcopy(pretrained_gnn)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 统一映射到 hidden_channels
        self.w_adapter_hete = nn.Linear(prompt_hete_dim, hidden_channels)
        
        self.global_router = SimpleHeteroGumbelRouter(tau=tau)
        
        # 残差纠偏缩放因子
        self.prompt_scale = nn.Parameter(torch.tensor(0.01))
        
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x_aligned, edge_index, p_hete, hard_route=False):
        with torch.no_grad():
            h_real = self.backbone(x_aligned, edge_index)
            
        # 对齐提示节点
        p_hete_aligned = self.w_adapter_hete(p_hete)
        
        # 计算连接概率
        ex_hete = self.global_router(h_real, p_hete_aligned, hard=hard_route)
        
        # 聚合提示特征
        m_prompt = torch.matmul(ex_hete, p_hete_aligned)
        
        # 融合原图特征与异配提示信息
        h_fused = h_real + self.prompt_scale * m_prompt
        
        logits = self.classifier(h_fused)
        return logits, ex_hete

# ==========================================
# 3. 辅助函数
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

def analyze_purity(model, aligner, data, p_hete, args):
    model.eval()
    aligner.eval()
    with torch.no_grad():
        out, ex_hete = model(aligner(data.x), data.edge_index, p_hete, hard_route=True)
        assignments = ex_hete.argmax(dim=1)
        y = data.y.squeeze()
        purities = []
        for k in range(args.k_hete):
            nodes = (assignments == k).nonzero(as_tuple=True)[0]
            if len(nodes) > 1:
                purities.append(torch.bincount(y[nodes]).max().item() / len(nodes))
        return homophily(data.edge_index, y), np.mean(purities) if purities else 0.0

# ==========================================
# 4. 训练主循环
# ==========================================
def train_once(args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 异配提示剥离验证启动 | Target: {args.target_dataset} | K_hete={args.k_hete}")

    data, in_dim, out_dim = load_node_data(args.target_dataset, "./data")
    data = data.to(device)
    train_m, val_m, test_m = build_masks(data, args.shots, args.val_node_num, args.test_node_num, seed, device)

    # 补零对齐 (防止引入随机噪声污染特征)
    class PaddingAligner(nn.Module):
        def __init__(self, target_dim, source_dim):
            super().__init__()
            self.target_dim = target_dim
            self.source_dim = source_dim
        def forward(self, x):
            if self.target_dim < self.source_dim:
                return F.pad(x, (0, self.source_dim - self.target_dim))
            elif self.target_dim > self.source_dim:
                return x[:, :self.source_dim]
            return x

    aligner = PaddingAligner(in_dim, args.source_dim).to(device)

    # 预训练骨架
    weight_path = f"./pretrained_gnns/{args.source_dataset}_SimGRACE_GCN_1.pth"
    backbone = load_pretrained_backbone(weight_path, args.source_dim, args.hidden_dim, device)

    # 仅提取异配提示
    generator = PromptGenerator(k_hete=args.k_hete, device=device)
    p_hete = generator.generate(data, backbone, aligner)

    # 初始化模型
    model = HeteroPromptOnlyGNN(backbone, args.hidden_dim, out_dim, p_hete.size(-1), args.tau).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_test = -1.0, 0.0
    wait, patience = 0, 30
    labels = data.y.squeeze()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, ex_hete = model(aligner(data.x), data.edge_index, p_hete)
        loss_cls = F.cross_entropy(logits[train_m], labels[train_m])
        
        # 异配稀疏损失
        loss_sparse = -torch.mean(torch.sum(ex_hete * torch.log(ex_hete + 1e-8), dim=-1))
        
        loss = loss_cls + args.lambda_sparse * loss_sparse
        loss.backward()
        optimizer.step()
        
        model.global_router.tau = max(args.min_tau, model.global_router.tau * args.tau_decay)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred, _ = model(aligner(data.x), data.edge_index, p_hete, hard_route=True)
                v_acc = compute_acc(pred, labels, val_m)
                t_acc = compute_acc(pred, labels, test_m)
                if v_acc > best_val:
                    best_val, best_test, wait = v_acc, t_acc, 0
                else: wait += 1
                
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val: {v_acc*100:.1f} | Test: {t_acc*100:.1f} | Tau: {model.global_router.tau:.2f} | Scale: {model.prompt_scale.item():.4f}")

        if wait >= patience: break

    orig_h, purity = analyze_purity(model, aligner, data, p_hete, args)
    print(f"\n🎉 实验完成！Best Test Acc (在最高 Val 时): {best_test*100:.2f}%")
    print(f"📊 最终异配缩放因子 (Prompt Scale): {model.prompt_scale.item():.4f}")
    print(f"📉 原图同配率 {orig_h:.4f} | 📈 异配提示纯度 {purity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Cora')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--k_hete', type=int, default=3)  # 只设极少量的异配提示点
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--lambda_sparse', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau_decay', type=float, default=0.95)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--seeds', type=str, default='42')
    train_once(parser.parse_args(), 42)