#测试提示模块
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
from prompts.global_router import GlobalGumbelRouter

# ==========================================
# 1. 提示生成器：按照你的 Idea 提取神经表征
# ==========================================
class PromptGenerator:
    def __init__(self, k_homo=10, k_hete=10, device='cpu'):
        self.k_homo = k_homo
        self.k_hete = k_hete
        self.device = device

    @torch.no_grad()
    def generate(self, data, model, aligner):
        """
        Idea实现：提取低频表征 H，计算残差 R，拼接 [H, R] 后聚类
        """
        model.eval()
        aligner.eval()
        
        # 步骤 1：特征空间对齐
        x_aligned = aligner(data.x)
        
        # 步骤 2：提取 GNN 输出的平滑表征 H (低频)
        h = model(x_aligned, data.edge_index)
        
        # 步骤 3：计算残差 R (捕捉被 GNN 抹掉的高频/个性化信号)
        # 鲁棒性处理：确保维度一致
        feat_dim = h.shape[-1]
        residual = x_aligned[:, :feat_dim] - h
            
        # 步骤 4：拼接 [H, R] 用于异配提示节点聚类
        h_aug = torch.cat([h, residual], dim=-1)

        print(f"--- 正在执行聚类初始化 (Seed 42) ---")
        # 同配聚类 (基于 H)
        print(f"正在生成同配提示节点 (K1={self.k_homo})...")
        kh = KMeans(n_clusters=self.k_homo, random_state=42, n_init='auto')
        kh.fit(h.cpu().numpy())
        p_homo = torch.tensor(kh.cluster_centers_, dtype=torch.float32).to(self.device)

        # 异配聚类 (基于 [H, R])
        print(f"正在生成异配提示节点 (K2={self.k_hete})...")
        kt = KMeans(n_clusters=self.k_hete, random_state=42, n_init='auto')
        kt.fit(h_aug.cpu().numpy())
        p_hete = torch.tensor(kt.cluster_centers_, dtype=torch.float32).to(self.device)

        return p_homo, p_hete

# ==========================================
# 2. 实验模型：PromptOnlyGNN (剥离双分支)
# ==========================================
class PromptOnlyGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_homo_dim, prompt_hete_dim, tau=1.0):
        super(PromptOnlyGNN, self).__init__()
        # 冻结源域骨架
        self.backbone = copy.deepcopy(pretrained_gnn)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 提示维度对齐层
        self.w_adapter_homo = nn.Linear(prompt_homo_dim, hidden_channels)
        self.w_adapter_hete = nn.Linear(prompt_hete_dim, hidden_channels)
        
        # 零参数路由器
        self.global_router = GlobalGumbelRouter(tau=tau)
        
        # 残差纠偏缩放因子 (可学习)
        self.prompt_scale = nn.Parameter(torch.tensor(0.1))
        
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x, edge_index, p_homo, p_hete, hard_route=False):
        with torch.no_grad():
            h_real = self.backbone(x, edge_index)
            
        p_homo_aligned = self.w_adapter_homo(p_homo)
        p_hete_aligned = self.w_adapter_hete(p_hete)
        
        # 计算连接概率
        ex_homo, ex_hete, ex_total = self.global_router(h_real, p_homo_aligned, p_hete_aligned, hard=hard_route)
        
        # 聚合提示特征
        m_prompt = torch.matmul(ex_homo, p_homo_aligned) + torch.matmul(ex_hete, p_hete_aligned)
        
        # 融合原图特征与提示信息 (纠正异配)
        h_fused = h_real + self.prompt_scale * m_prompt
        
        logits = self.classifier(h_fused)
        return logits, ex_total

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

def analyze_purity(model, aligner, data, p_homo, p_hete, args):
    model.eval()
    aligner.eval()
    with torch.no_grad():
        out, ex_total = model(aligner(data.x), data.edge_index, p_homo, p_hete, hard_route=True)
        assignments = ex_total.argmax(dim=1)
        y = data.y.squeeze()
        purities = []
        for k in range(args.k_homo + args.k_hete):
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
    print(f"\n🚀 实验启动: Seed {seed} | Target: {args.target_dataset} | K1={args.k_homo}, K2={args.k_hete}")

    data, in_dim, out_dim = load_node_data(args.target_dataset, "./data")
    data = data.to(device)
    train_m, val_m, test_m = build_masks(data, args.shots, args.val_node_num, args.test_node_num, seed, device)

    # 对齐层：单层线性，显式初始化防止噪声
    aligner = nn.Linear(in_dim, args.source_dim).to(device)
    nn.init.xavier_uniform_(aligner.weight)

    # 预训练骨架
    weight_path = f"./pretrained_gnns/{args.source_dataset}_SimGRACE_GCN_1.pth"
    backbone = load_pretrained_backbone(weight_path, args.source_dim, args.hidden_dim, device)

    # 【你的Idea核心】：执行表征提取与聚类
    print(f"🔮 正在通过预训练模型提取神经提示锚点...")
    generator = PromptGenerator(k_homo=args.k_homo, k_hete=args.k_hete, device=device)
    p_homo, p_hete = generator.generate(data, backbone, aligner)

    # 初始化实验模型
    model = PromptOnlyGNN(backbone, args.hidden_dim, out_dim, p_homo.size(-1), p_hete.size(-1), args.tau).to(device)

    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aligner.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_test = -1.0, 0.0
    wait, patience = 0, 30
    labels = data.y.squeeze()

    for epoch in range(1, args.epochs + 1):
        model.train(); aligner.train(); optimizer.zero_grad()
        
        logits, ex = model(aligner(data.x), data.edge_index, p_homo, p_hete)
        loss_cls = F.cross_entropy(logits[train_m], labels[train_m])
        loss_sparse = -torch.mean(torch.sum(ex * torch.log(ex + 1e-8), dim=-1))
        
        (loss_cls + args.lambda_sparse * loss_sparse).backward()
        optimizer.step()
        
        model.global_router.tau = max(args.min_tau, model.global_router.tau * args.tau_decay)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred, _ = model(aligner(data.x), data.edge_index, p_homo, p_hete)
                v_acc = compute_acc(pred, labels, val_m)
                t_acc = compute_acc(pred, labels, test_m)
                if v_acc > best_val:
                    best_val, best_test, wait = v_acc, t_acc, 0
                else: wait += 1
                print(f"Epoch {epoch:03d} | Loss: {loss_cls.item():.4f} | Val: {v_acc*100:.1f} | Test: {t_acc*100:.1f} | Tau: {model.global_router.tau:.2f}")

        if wait >= patience: break

    orig_h, purity = analyze_purity(model, aligner, data, p_homo, p_hete, args)
    print(f"\n🎉 实验完成！Best Test: {best_test*100:.2f}% | Scale: {model.prompt_scale.item():.4f}")
    print(f"📊 指标：原图同配率 {orig_h:.4f} | 提示纯度 {purity:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Cora')
    parser.add_argument('--source_dataset', type=str, default='PubMed')
    parser.add_argument('--source_dim', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--k_homo', type=int, default=7)  # Cora 默认 7 类
    parser.add_argument('--k_hete', type=int, default=2)  # Cora 异配需求低，设小一点
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--val_node_num', type=int, default=1000)
    parser.add_argument('--test_node_num', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--lambda_sparse', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--tau_decay', type=float, default=0.98)
    parser.add_argument('--min_tau', type=float, default=0.1)
    parser.add_argument('--seeds', type=str, default='42')
    train_once(parser.parse_args(), 42)