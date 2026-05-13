# models/dual_branch.py
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv 

# 导入我们新设计的全局互斥路由器
from prompts.global_router import GlobalGumbelRouter

class DualBranchCrossDomainGNN(nn.Module):
    def __init__(self, pretrained_gnn, hidden_channels, out_channels, prompt_homo_dim, prompt_hete_dim, tau=1.0):
        super(DualBranchCrossDomainGNN, self).__init__()
        
        # ==========================================
        # 1. 冻结分支 (Frozen Branch) - 保留源域先验知识
        # ==========================================
        self.frozen_branch = copy.deepcopy(pretrained_gnn)
        # 彻底物理锁死参数梯度
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
            
        # ==========================================
        # 2. 适应分支 (Adaptive Branch) - 学习目标域新知识
        # ==========================================
        # 热启动：以预训练权重为起点进行微调
        self.adaptive_branch = copy.deepcopy(pretrained_gnn)
        
        # ==========================================
        # 3. 全局联合路由器 (Global Mutually Exclusive Router)
        # ==========================================
        # 核心逻辑：将同配和异配提示合并，进行唯一一次 Gumbel-Softmax
        self.global_router = GlobalGumbelRouter(
            feature_dim=hidden_channels, 
            prompt_homo_dim=prompt_homo_dim, 
            prompt_hete_dim=prompt_hete_dim, 
            tau=tau
        )
        
        # ==========================================
        # 4. 适应器投影矩阵 (W_adapter)
        # ==========================================
        self.w_adapter_homo = nn.Linear(prompt_homo_dim, hidden_channels)
        self.w_adapter_hete = nn.Linear(prompt_hete_dim, hidden_channels)
        
        # ==========================================
        # 5. 自适应门控融合 (Gating Mechanism)
        # ==========================================
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.GELU(), # 使用平滑的 GELU
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # 最终预测层：融合双分支信息 [源域表征 + 目标域适配表征]
        self.classifier = nn.Linear(hidden_channels * 2, out_channels)

    def train(self, mode=True):
        """
        强制修复：确保冻结分支永远处于 eval 模式。
        防止预训练模型中的 Dropout 在训练时产生随机表征偏移。
        """
        super(DualBranchCrossDomainGNN, self).train(mode)
        self.frozen_branch.eval()
        return self

    def forward(self, x, edge_index, p_homo, p_hete, hard_route=False):
        """
        参数:
            x: 原图节点特征 (已对齐维度)
            p_homo: 静态同配提示节点 [K1, d]
            p_hete: 静态异配提示节点 [K2, 2d]
            hard_route: True 则生成 0/1 离散连接，False 则生成软概率
        """
        # 维度安全检查
        assert p_homo.size(-1) == self.w_adapter_homo.in_features, "同配提示维度不匹配"
        assert p_hete.size(-1) == self.w_adapter_hete.in_features, "异配提示维度不匹配"
        
        # -------------------------------------------
        # 步骤 A：双分支特征并行提取
        # -------------------------------------------
        # 分支 1: 源域老专家 (Frozen)
        with torch.no_grad():
            h_frozen = self.frozen_branch(x, edge_index)
        
        # 分支 2: 目标域探索者 (Adaptive)
        m_real = self.adaptive_branch(x, edge_index)
        
        # -------------------------------------------
        # 步骤 B：互斥路由与提示信息聚合
        # -------------------------------------------
        # 1. 联合路由：在所有提示节点中选出唯一一个最优连接
        # ex_total 的维度是 [N, K1 + K2], 用于后续计算稀疏损失
        ex_homo, ex_hete, ex_total = self.global_router(m_real, p_homo, p_hete, hard=hard_route)
        
        # 2. 沿着 EX 边吸收提示特征
        m_prompt_homo_raw = torch.matmul(ex_homo, p_homo)  # [N, d]
        m_prompt_hete_raw = torch.matmul(ex_hete, p_hete)  # [N, 2d]
        
        # 3. 映射到隐藏层统一维度并求和
        m_prompt_homo = self.w_adapter_homo(m_prompt_homo_raw)
        m_prompt_hete = self.w_adapter_hete(m_prompt_hete_raw)
        m_prompt = m_prompt_homo + m_prompt_hete
        
        # -------------------------------------------
        # 步骤 C：自适应门控融合 (核心决策)
        # -------------------------------------------
        gate_input = torch.cat([m_real, m_prompt], dim=-1)
        g = self.gate(gate_input)
        
        # 若 g 趋近 0，说明原图异配严重，模型更依赖提示节点
        h_adapted_final = g * m_real + (1 - g) * m_prompt
        
        # -------------------------------------------
        # 步骤 D：双分支对齐输出
        # -------------------------------------------
        final_rep = torch.cat([h_frozen, h_adapted_final], dim=-1)
        logits = self.classifier(final_rep)
        
        # 返回 ex_total 以便在 train.py 中计算极化稀疏损失
        return logits, h_frozen, h_adapted_final, ex_total

# ==========================================
# 独立测试模块 (运行此文件可验证逻辑)
# ==========================================
if __name__ == "__main__":
    # 模拟环境
    N, in_dim, hid_dim, num_classes = 100, 1433, 128, 7
    K1, K2 = 5, 5
    
    # 模拟预训练 GNN
    class SimpleGNN(nn.Module):
        def __init__(self, in_d, hid_d):
            super().__init__()
            self.c = GCNConv(in_d, hid_d)
        def forward(self, x, ei):
            return self.c(x, ei).relu()

    dummy_gnn = SimpleGNN(in_dim, hid_dim)
    
    # 实例化我们的框架
    model = DualBranchCrossDomainGNN(
        pretrained_gnn=dummy_gnn,
        hidden_channels=hid_dim,
        out_channels=num_classes,
        prompt_homo_dim=in_dim,
        prompt_hete_dim=in_dim * 2
    )

    # 模拟输入
    tx = torch.randn(N, in_dim)
    tei = torch.randint(0, N, (2, 200))
    ph = torch.randn(K1, in_dim)
    pe = torch.randn(K2, in_dim * 2)

    # 前向传播测试
    model.train()
    logits, h_f, h_a, ex_t = model(tx, tei, ph, pe)

    print(f"测试通过！")
    print(f"路由矩阵维度: {ex_t.shape}")
    print(f"路由矩阵每一行的和（应为1）: {ex_t.sum(dim=-1)[0].item():.2f}")
    print(f"最终预测 Logits 维度: {logits.shape}")