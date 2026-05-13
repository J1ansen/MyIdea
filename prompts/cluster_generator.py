# prompts/cluster_generator.py
import torch
import torch.nn as nn
from sklearn.cluster import KMeans 

class PromptGenerator:
    def __init__(self, k_homo=10, k_hete=10, device='cpu'):
        self.k_homo = k_homo
        self.k_hete = k_hete
        self.device = device

    @torch.no_grad()
    def generate(self, data, model, aligner):
        """
        使用真实的 GNN 模型生成表征并聚类
        参数:
            data: 目标域数据 (data.x, data.edge_index)
            model: 预训练的 GNN 骨架 (frozen_branch)
            aligner: 维度对齐层 (把 target_in_dim 映射到 source_dim)
        """
        model.eval()
        aligner.eval()
        
        # 1. 特征对齐：将原始特征映射到源域空间
        x_aligned = aligner(data.x)  # [N, source_dim]
        
        # 2. 提取 GNN 表征 H (低频信号)
        # 注意：这里得到的是经过 GNN 卷积层平滑后的隐层表征
        h = model(x_aligned, data.edge_index) # [N, hidden_dim]
        
        # 3. 计算残差 R (高频信号)
        # R = 原始输入(对齐后) - GNN输出
        # 注意：这要求 x_aligned 和 h 的维度一致 (通常 SimGRACE 预训练模型 hidden=source_dim=128)
        if x_aligned.shape[-1] != h.shape[-1]:
            # 如果维度不一致（比如 500 vs 128），需要对 x_aligned 做截断或线性投影，
            # 但最标准的情况是两者维度相同。这里做一个鲁棒性处理：
            residual = x_aligned[:, :h.shape[-1]] - h
        else:
            residual = x_aligned - h
            
        # 4. 准备异配聚类特征：拼接 [H, R]
        h_aug = torch.cat([h, residual], dim=-1) # 维度变为 2 * hidden_dim

        # --- 执行聚类 ---
        print(f"正在基于 GNN 表征进行同配聚类 (K1={self.k_homo})...")
        kmeans_homo = KMeans(n_clusters=self.k_homo, random_state=42, n_init='auto')
        kmeans_homo.fit(h.cpu().numpy())
        p_homo = torch.tensor(kmeans_homo.cluster_centers_, dtype=torch.float32).to(self.device)

        print(f"正在基于拼接残差进行异配聚类 (K2={self.k_hete})...")
        kmeans_hete = KMeans(n_clusters=self.k_hete, random_state=42, n_init='auto')
        kmeans_hete.fit(h_aug.cpu().numpy())
        p_hete = torch.tensor(kmeans_hete.cluster_centers_, dtype=torch.float32).to(self.device)

        return p_homo, p_hete