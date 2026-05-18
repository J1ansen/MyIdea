import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GCNConv

# ==========================================
# 0. 先进的跨域维度对齐模块
# ==========================================
class RobustFeatureAligner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 使用不带偏置的线性层
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # 🔥 核心：正交初始化。在降维/升维时最大程度保留特征空间的几何流形，这比随机初始化好得多
        nn.init.orthogonal_(self.proj.weight)
        # 使用 LayerNorm 对齐特征分布（缓解跨域 Domain Shift）
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(self.norm(self.proj(x)))

# ==========================================
# 1. 低参数目标域特征提取器 (Tiny GNN)
# ==========================================
class TinyGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 极轻量级，仅用于提炼目标图独有的局部结构信息
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# ==========================================
# 2. 纯净版双分支架构 (Pure Dual-Branch GNN)
# ==========================================
class PureDualBranchGNN(nn.Module):
    def __init__(self, pretrained_backbone, source_dim, target_dim, hidden_dim, out_dim, bottleneck_dim=32):
        super().__init__()
        
        # 0. 维度对齐器
        if source_dim != target_dim:
            self.aligner = RobustFeatureAligner(target_dim, source_dim)
        else:
            self.aligner = nn.Identity()
            
        # ==========================================
        # 1. 冻结分支 (源域先验, 绝对不更新参数)
        # ==========================================
        self.frozen_branch = copy.deepcopy(pretrained_backbone)
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
        self.frozen_branch.eval() 
        
        # ==========================================
        # 2. 适应分支模块 (目标域自适应)
        # ==========================================
        # 2a. GP2F 风格的 Bottleneck Adapter (极低参数)
        # 专门用于将源域的冻结表征适配到目标域分布
        self.adapter_down = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.adapter_up = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        self.adapter_act = nn.ReLU()
        
        # 2b. Tiny GNN (提取目标域特有拓扑信息)
        self.tiny_gnn = TinyGNN(source_dim, hidden_dim)
        
        # 2c. 适应分支内部融合门控 (融合 Adapter 特征和 TinyGNN 特征)
        self.adapt_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # ==========================================
        # 3. 最终全局融合模块
        # ==========================================
        # 外部自适应门控: 用于加权融合 H_frozen 和 H_adapted
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # ------------------------------------------
        # 步骤 0: 维度自适应对齐 (Target Dim -> Source Dim)
        # ------------------------------------------
        x_aligned = self.aligner(x)
        
        # ------------------------------------------
        # 步骤 1: 冻结分支获取通用先验 H_frozen
        # ------------------------------------------
        with torch.no_grad():
            self.frozen_branch.eval() # 强制 eval，防止 BatchNorm/Dropout 影响
            h_frozen = self.frozen_branch(x_aligned, edge_index) 
            
        # ------------------------------------------
        # 步骤 2: 适应分支生成目标域专属表征 H_adapted
        # ------------------------------------------
        # 动作A: 通过 GP2F Adapter 提炼冻结表征
        h_adapter = self.adapter_down(h_frozen)
        h_adapter = self.adapter_act(h_adapter)
        h_adapter = self.adapter_up(h_adapter)
        h_adapter = h_frozen + h_adapter # 残差连接，保障基础性能
        
        # 动作B: 通过 Tiny GNN 提炼目标图的真实拓扑表征
        h_tiny = self.tiny_gnn(x_aligned, edge_index)
        
        # 动作C: 内部融合生成最终的适应分支表征
        g_adapt = torch.sigmoid(self.adapt_gate(torch.cat([h_adapter, h_tiny], dim=-1)))
        h_adapted = g_adapt * h_adapter + (1 - g_adapt) * h_tiny
        
        # ------------------------------------------
        # 步骤 3: 全局双分支自适应融合
        # ------------------------------------------
        # 拼接 h_frozen 和 h_adapted，由网络自行决定信赖谁
        g_fusion = torch.sigmoid(self.fusion_gate(torch.cat([h_frozen, h_adapted], dim=-1)))
        
        # 软加权融合
        h_final = g_fusion * h_frozen + (1 - g_fusion) * h_adapted
        
        # 最终分类
        logits = self.classifier(h_final)
        
        return logits, h_frozen, h_adapted