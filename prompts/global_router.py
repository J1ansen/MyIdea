# prompts/global_router.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalGumbelRouter(nn.Module):
    """
    零参数的全局互斥 Gumbel-Softmax 路由器
    """
    def __init__(self, tau=1.0):
        # 删除了所有 proj 线性层，彻底斩断过拟合！
        super(GlobalGumbelRouter, self).__init__()
        self.tau = tau

    def forward(self, m_real, p_homo_aligned, p_hete_aligned, hard=False):
        """
        注意：传入的提示节点特征必须是已经经过 w_adapter 对齐维度的！
        """
        K1 = p_homo_aligned.size(0)
        
        # 直接计算点积相似度 (两个空间维度已经一致，无需学习参数)
        logits_homo = torch.matmul(m_real, p_homo_aligned.t())  # [N, K1]
        logits_hete = torch.matmul(m_real, p_hete_aligned.t())  # [N, K2]
        
        # 拼接并计算全局唯一的 Softmax
        logits_total = torch.cat([logits_homo, logits_hete], dim=-1)
        ex_total = F.gumbel_softmax(logits_total, tau=self.tau, hard=hard, dim=-1)
        
        # 拆分返回，以便分别乘以同配和异配的提示特征
        ex_homo = ex_total[:, :K1]
        ex_hete = ex_total[:, K1:]
        
        return ex_homo, ex_hete, ex_total