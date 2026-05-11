# 双分支跨域 GNN 框架
# 包含：冻结分支（源域知识）、适应分支（提示聚合）、门控融合

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv  # 这里以 GCN 为例，你可以随时替换为 GraphSAGE 或 GAT

# 假设这与你的项目结构一致
from prompts.gumbel_route import GumbelRouter

class DualBranchCrossDomainGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, prompt_homo_dim, prompt_hete_dim, tau=0.5):
        super(DualBranchCrossDomainGNN, self).__init__()
        
        # ==========================================
        # 1. 冻结分支 (Frozen Branch) - 保留源域先验
        # ==========================================
        # 这里的参数在预训练后应当被冻结 (requires_grad=False)
        self.frozen_conv1 = GCNConv(in_channels, hidden_channels)
        self.frozen_conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # ==========================================
        # 2. 适应分支 (Adaptive Branch) - 学习目标域知识
        # ==========================================
        self.adapted_conv1 = GCNConv(in_channels, hidden_channels)
        self.adapted_conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # ==========================================
        # 3. 提示拓扑路由模块
        # ==========================================
        # 为同配和异配提示分别实例化路由器
        self.router_homo = GumbelRouter(feature_dim=hidden_channels, prompt_dim=prompt_homo_dim, tau=tau)
        self.router_hete = GumbelRouter(feature_dim=hidden_channels, prompt_dim=prompt_hete_dim, tau=tau)
        
        # ==========================================
        # 4. 适应分支的低参数权重矩阵 (W_adapter)
        # 用于将不同维度的提示特征映射到统一维度
        # ==========================================
        self.w_adapter_homo = nn.Linear(prompt_homo_dim, hidden_channels)
        self.w_adapter_hete = nn.Linear(prompt_hete_dim, hidden_channels)
        
        # ==========================================
        # 5. 门控融合机制 (Gating Mechanism)
        # ==========================================
        # 输入原图信息 (M_real) 和提示信息 (M_prompt) 的拼接，输出 0~1 的缩放标量
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 最终分类器（融合双分支特征）
        self.classifier = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index, p_homo, p_hete, hard_route=False):
        """
        x: 原图节点初始特征 [N, in_channels]
        edge_index: 原图拓扑结构
        p_homo: 静态同配提示节点特征 [K1, prompt_homo_dim]
        p_hete: 静态异配提示节点特征 [K2, prompt_hete_dim]
        """
        
        # -------------------------------------------
        # 步骤 A：冻结分支前向传播 (M_frozen)
        # -------------------------------------------
        # 在实际训练脚本中，需确保使用 torch.no_grad() 或将此分支参数设为不可导
        h_frozen = self.frozen_conv1(x, edge_index).relu()
        h_frozen = self.frozen_conv2(h_frozen, edge_index).relu()  # [N, hidden_channels]
        
        # -------------------------------------------
        # 步骤 B：适应分支前向传播
        # -------------------------------------------
        # 1. 提取原图当前的目标域表征 (M_real)
        h_adapted = self.adapted_conv1(x, edge_index).relu()
        m_real = self.adapted_conv2(h_adapted, edge_index).relu()  # [N, hidden_channels]
        
        # 2. 生成连接拓扑 (EX 矩阵)
        ex_homo = self.router_homo(m_real, p_homo, hard=hard_route)  # [N, K1]
        ex_hete = self.router_hete(m_real, p_hete, hard=hard_route)  # [N, K2]
        
        # 3. 沿着 EX 边聚合提示特征 (矩阵乘法实现信息流传递)
        # (N, K1) @ (K1, d_homo) -> [N, d_homo]
        m_prompt_homo_raw = torch.matmul(ex_homo, p_homo)
        # (N, K2) @ (K2, d_hete) -> [N, d_hete]
        m_prompt_hete_raw = torch.matmul(ex_hete, p_hete)
        
        # 4. 使用 W_adapter 统一维度并相加得到最终提示信息流 (M_prompt)
        m_prompt_homo = self.w_adapter_homo(m_prompt_homo_raw)     # [N, hidden_channels]
        m_prompt_hete = self.w_adapter_hete(m_prompt_hete_raw)     # [N, hidden_channels]
        m_prompt = m_prompt_homo + m_prompt_hete                   # [N, hidden_channels]
        
        # -------------------------------------------
        # 步骤 C：门控机制融合 (Gating)
        # -------------------------------------------
        # 计算门控系数 g (基于节点自身特征和聚合来的提示特征)
        gate_input = torch.cat([m_real, m_prompt], dim=-1)         # [N, hidden_channels * 2]
        g = self.gate(gate_input)                                  # [N, 1]
        
        # 自适应融合：g 决定保留多少原图信息，(1-g) 决定吸收多少提示信息
        h_adapted_final = g * m_real + (1 - g) * m_prompt          # [N, hidden_channels]
        
        # -------------------------------------------
        # 步骤 D：双分支对齐与最终预测
        # -------------------------------------------
        # 拼接冻结分支（源域知识）和适应分支（目标域知识）
        final_rep = torch.cat([h_frozen, h_adapted_final], dim=-1) # [N, hidden_channels * 2]
        logits = self.classifier(final_rep)                        # [N, out_channels]
        
        return logits, h_frozen, h_adapted_final