import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalGumbelRouter(nn.Module):
    def __init__(self, feature_dim, tau=1.0):
        super().__init__()
        self.tau = tau
        
        # 残差门控网络
        self.residual_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, x_aligned, h_frozen, p_homo, p_hete, hard=False):
        # 1. 拼接所有提示节点 [K, 128]
        p_total = torch.cat([p_homo, p_hete], dim=0) 
        
        # 2. 计算点积相似度 Logits
        # 【极其关键的修复】：使用 h_frozen (128维) 和 p_total (128维) 进行点积
        # 因为提示节点本身就是在 h_frozen 的低频空间聚类出来的，在这个空间算距离最纯粹
        logits_total = torch.matmul(h_frozen, p_total.t()) 

        # 3. Gumbel-Softmax 基础路由
        ex_base = F.gumbel_softmax(logits_total, tau=self.tau, hard=hard, dim=-1)

        # 4. 计算高频残差 (异配病灶)
        # 截断对齐 x_aligned 以便与 h_frozen 做减法
        if x_aligned.shape[-1] != h_frozen.shape[-1]:
            residual = x_aligned[:, :h_frozen.shape[-1]] - h_frozen
        else:
            residual = x_aligned - h_frozen
        
        # 5. 生成稀疏门控掩码 g
        g = self.residual_gate(residual)

        # 6. 自适应稀疏截断
        ex_optimized = ex_base * g 

        return ex_optimized, g, ex_base, p_total