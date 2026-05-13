# models/global_router.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalGumbelRouter(nn.Module):
    """
    全局互斥的 Gumbel-Softmax 路由器
    确保每个原图节点在 (同配 + 异配) 的所有候选池中，最终只选择唯一的一个提示节点。
    """
    def __init__(self, feature_dim, prompt_homo_dim, prompt_hete_dim, tau=1.0):
        super(GlobalGumbelRouter, self).__init__()
        self.tau = tau
        
        # 分别定义映射矩阵，用于计算特征相似度
        self.proj_homo = nn.Linear(feature_dim, prompt_homo_dim)
        self.proj_hete = nn.Linear(feature_dim, prompt_hete_dim)

    def forward(self, node_features, p_homo, p_hete, hard=False):
        """
        输出:
            ex_homo: [N, K1] (大部分全为0)
            ex_hete: [N, K2] (大部分全为0)
            确保对于节点 i, sum(ex_homo[i]) + sum(ex_hete[i]) == 1
        """
        K1 = p_homo.size(0)
        K2 = p_hete.size(0)
        
        # 1. 分别计算与两类提示节点的连接分数 (Logits)
        h_proj_homo = self.proj_homo(node_features)  # [N, d_homo]
        h_proj_hete = self.proj_hete(node_features)  # [N, d_hete]
        
        logits_homo = torch.matmul(h_proj_homo, p_homo.t())  # [N, K1]
        logits_hete = torch.matmul(h_proj_hete, p_hete.t())  # [N, K2]
        
        # 2. 【核心修改】将 Logits 拼接成全局大池子
        # 维度变成: [N, K1 + K2]
        logits_total = torch.cat([logits_homo, logits_hete], dim=-1)
        
        # 3. 在全局池子里进行唯一一次 Gumbel-Softmax
        # 这保证了在 hard=True 时，K1+K2 个候选中，只有唯一一个是 1，其余全为 0
        ex_total = F.gumbel_softmax(logits_total, tau=self.tau, hard=hard, dim=-1)
        
        # 4. 将输出的概率矩阵拆分回去，方便后续与不同维度的提示特征相乘
        ex_homo = ex_total[:, :K1]
        ex_hete = ex_total[:, K1:]
        
        # 同时返回拆分后的矩阵和合并的矩阵(合并的矩阵用于算稀疏损失)
        return ex_homo, ex_hete, ex_total