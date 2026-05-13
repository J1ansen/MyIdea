# models/dual_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from prompts.global_router import GlobalGumbelRouter
from models.gp2f_adapter import GP2FAdapter  # 引入刚才写的 Adapter

class DualBranchCrossDomainGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_homo_dim, prompt_hete_dim, tau=1.0, adapter_r=32):
        super(DualBranchCrossDomainGNN, self).__init__()
        
        # ==========================================
        # 1. 冻结分支 (Frozen Branch) - 彻底锁死
        # ==========================================
        self.frozen_branch = copy.deepcopy(pretrained_gnn)
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
            
        # ==========================================
        # 2. 适应分支 (Adapted Branch) - 替换为轻量级逐层 Adapter
        # ==========================================
        # 针对 2 层 GCN，我们在每一层后接入一个 Adapter (维度与隐藏层对齐)
        self.adapter_layer1 = GP2FAdapter(feature_dim=hidden_channels, r=adapter_r)
        self.adapter_layer2 = GP2FAdapter(feature_dim=hidden_channels, r=adapter_r)
        
        # ==========================================
        # 3. 零参数全局路由器 (Zero-param Global Router)
        # ==========================================
        self.global_router = GlobalGumbelRouter(tau=tau)
        
        # ==========================================
        # 4. 提示节点维度适应器 (Aligners)
        # ==========================================
        self.w_adapter_homo = nn.Linear(prompt_homo_dim, hidden_channels)
        self.w_adapter_hete = nn.Linear(prompt_hete_dim, hidden_channels)
        
        # ==========================================
        # 5. 门控融合与分类器 (Gating & Classifier)
        # ==========================================
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(hidden_channels * 2, out_channels)

    def train(self, mode=True):
        """物理防范 Dropout 陷阱，确保老专家彻底被锁死在评估模式"""
        super().train(mode)
        self.frozen_branch.eval()
        return self

    def forward(self, x, edge_index, p_homo, p_hete, hard_route=False):
        # -------------------------------------------
        # 步骤 A：双分支特征提取 (GP2F 逐层推演逻辑)
        # -------------------------------------------
        # 【分支 1：冻结分支】 直接利用老专家提取纯净的源域先验
        with torch.no_grad():
            h_frozen = self.frozen_branch(x, edge_index)
            
        # 【分支 2：适应分支】 借用老专家权重，逐层注入目标域偏差
        # 第一层：卷积 -> Adapter微调 -> 激活 -> Dropout
        h1 = self.frozen_branch.conv1(x, edge_index)
        h1_adapted = self.adapter_layer1(h1)
        h1_adapted = F.relu(h1_adapted)
        h1_adapted = F.dropout(h1_adapted, p=0.5, training=self.training)
        
        # 第二层：卷积 -> Adapter微调 (得到目标域专属表征 m_real)
        h2 = self.frozen_branch.conv2(h1_adapted, edge_index)
        m_real = self.adapter_layer2(h2)
        
        # -------------------------------------------
        # 步骤 B：提示对齐与零参数全局路由 (解决结构异配)
        # -------------------------------------------
        # 1. 先对齐提示节点到统一维度
        p_homo_aligned = self.w_adapter_homo(p_homo)
        p_hete_aligned = self.w_adapter_hete(p_hete)
        
        # 2. 算纯净点积并进行联合全局路由
        ex_homo, ex_hete, ex_total = self.global_router(m_real, p_homo_aligned, p_hete_aligned, hard=hard_route)
        
        # 3. 吸收提示特征
        m_prompt = torch.matmul(ex_homo, p_homo_aligned) + torch.matmul(ex_hete, p_hete_aligned)
        
        # -------------------------------------------
        # 步骤 C & D：门控融合与双分支对齐输出
        # -------------------------------------------
        gate_input = torch.cat([m_real, m_prompt], dim=-1)
        g = self.gate(gate_input)
        h_adapted_final = g * m_real + (1 - g) * m_prompt
        
        final_rep = torch.cat([h_frozen, h_adapted_final], dim=-1)
        logits = self.classifier(final_rep)
        
        return logits, h_frozen, h_adapted_final, ex_total