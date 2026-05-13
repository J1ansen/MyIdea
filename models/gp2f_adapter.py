# models/gp2f_adapter.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GP2FAdapter(nn.Module):
    """
    GP2F 论文标准的 Layer-wise 轻量级适应器
    公式: H_adp = H + beta * UP(ReLU(DOWN(H)))
    """
    def __init__(self, feature_dim, r=32):
        super(GP2FAdapter, self).__init__()
        # 降维与升维矩阵 (r 是瓶颈维度，远小于 feature_dim)
        self.down_proj = nn.Linear(feature_dim, r, bias=False)
        self.up_proj = nn.Linear(r, feature_dim, bias=False)
        
        # 核心设计：可学习的缩放因子 beta，初始化为一个极小值 (0.01)
        self.beta = nn.Parameter(torch.tensor(0.01))
        
        # 【致胜关键：零初始化】
        # DOWN 层随机初始化，UP 层全 0 初始化。
        # 确保在训练的第 0 轮，Adapter 输出绝对为 0，模型完美等同于预训练老专家。
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, h):
        # 提取目标域特定的偏移量
        adapter_out = self.up_proj(F.relu(self.down_proj(h)))
        # 残差相加
        return h + self.beta * adapter_out