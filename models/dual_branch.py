import torch
import torch.nn as nn
import copy
from prompts.global_router import GlobalGumbelRouter

class DualBranchCrossDomainGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_homo_dim, prompt_hete_dim, tau=1.0, adapter_r=8):
        super().__init__()
        self.hidden_dim = hidden_channels
        
        # 1. 冻结分支 (老专家，保留源域知识)
        self.frozen_branch = pretrained_gnn
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
            
        # 2. 适应分支 (学习目标域知识)
        # 这里使用深拷贝预训练模型作为适应分支基础。如果你有专用的 GP2FAdapter，也可以在这替换。
        self.adapted_branch = copy.deepcopy(pretrained_gnn)
        for param in self.adapted_branch.parameters():
            param.requires_grad = True
            
        # 3. 初始化残差稀疏路由器
        # 注意：由于提示节点是在隐层空间 (128维) 聚类出来的，这里的特征维度使用隐层维度
        self.global_router = GlobalGumbelRouter(feature_dim=prompt_homo_dim, tau=tau)
        
        # 4. 特征自适应门控融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 5. 最终分类器
        self.classifier = nn.Linear(self.hidden_dim, out_channels)
        
        # 动态投影层 (用于将 128 维的 prompt 投回 500 维的原图特征空间)
        self.prompt_up_proj = None

    def forward(self, x_aligned, edge_index, p_homo, p_hete, hard_route=False):
        # 1. 冻结分支前向传播，得到先验表征 [N, 128]
        with torch.no_grad():
            self.frozen_branch.eval()
            h_frozen = self.frozen_branch(x_aligned, edge_index)
        
        # 统一提示特征维度 (因为 p_hete 可能是拼接了残差的 256 维，需对齐回 128 维)
        if p_hete.shape[-1] != p_homo.shape[-1]:
            p_hete_aligned = p_hete[:, :p_homo.shape[-1]]
        else:
            p_hete_aligned = p_hete
        
        # 2. 获取全局路由与门控权重
        ex_optimized, gate_weight, ex_base, p_total = self.global_router(
            x_aligned, h_frozen, p_homo, p_hete_aligned, hard=hard_route
        )
        
        # 3. 【核心拓扑增强】通过路由矩阵拉取提示空间的同类特征 [N, 128]
        m_prompt = torch.matmul(ex_optimized, p_total) 
        
        # 4. 【早期特征融合】在图卷积之前增强节点输入特征
        # 因为 x_aligned 是 500 维，m_prompt 是 128 维，自动初始化投影层对齐
        if self.prompt_up_proj is None or self.prompt_up_proj.in_features != m_prompt.shape[-1]:
            self.prompt_up_proj = nn.Linear(m_prompt.shape[-1], x_aligned.shape[-1], bias=False).to(x_aligned.device)
        
        m_prompt_projected = self.prompt_up_proj(m_prompt)
        x_enhanced = x_aligned + m_prompt_projected 
        
        # 5. 适应分支前向传播 (在增强后的同配特征上进行 Message Passing)
        h_adapted = self.adapted_branch(x_enhanced, edge_index)
        
        # 6. 后期双分支门控融合
        combined_features = torch.cat([h_frozen, h_adapted], dim=-1)
        beta = self.fusion_gate(combined_features)
        h_final = beta * h_adapted + (1 - beta) * h_frozen
        
        # 7. 分类器输出
        logits = self.classifier(h_final)
        
        return logits, h_final, h_frozen, h_adapted, gate_weight, ex_optimized