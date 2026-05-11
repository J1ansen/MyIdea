# 基于 Gumbel-Softmax 的提示节点路由连边模块
# 输入: 原图节点特征, 提示节点特征
# 输出: 节点到提示节点的权重/连接矩阵

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelRouter(nn.Module):
    """
    基于 Gumbel-Softmax 的提示节点路由连边模块
    """
    def __init__(self, feature_dim, prompt_dim, tau=1.0):
        super(GumbelRouter, self).__init__()
        self.tau = tau  # Gumbel 温度参数，越小分布越尖锐（越接近 one-hot 离散选择）
        
        # 定义一个特征投影矩阵，用于计算原图节点和提示节点的匹配度
        self.proj = nn.Linear(feature_dim, prompt_dim)

    def forward(self, node_features, prompt_features, hard=False):
        """
        前向传播
        输入:
            node_features: 原图节点特征 [N, d]
            prompt_features: 提示节点特征 [K, d_p]
            hard: 是否输出绝对的 0/1 离散边（Gumbel的特有魔法，即使输出0和1，依然能反向传播梯度！）
        输出:
            EX: 节点到提示节点的权重/连接矩阵 [N, K]
        """
        # 1. 投影原图节点特征以对齐提示节点维度
        h_proj = self.proj(node_features) # [N, prompt_dim]
        
        # 2. 计算内积作为连接意愿分数 (Logits): (N, prompt_dim) @ (prompt_dim, K) -> [N, K]
        logits = torch.matmul(h_proj, prompt_features.t())

        # 3. Gumbel-Softmax 魔法
        # dim=-1 表示在每一行（也就是针对每一个节点）进行 Softmax，保证权重和为 1
        EX = F.gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)

        return EX

# ==========================================
# 独立测试模块
# ==========================================
if __name__ == "__main__":
    # 模拟我们刚刚在 Cora 上得到的数据维度
    N = 2708      # 原图节点数
    d = 1433      # 特征维度
    K = 5         # 假设我们现在要给那 5 个同配提示节点连边

    # 伪造一些原图的特征数据用来测试
    test_node_features = torch.randn((N, d))
    # 伪造一些提示节点的特征数据用来测试 (同配提示节点的维度为 d)
    test_prompt_features = torch.randn((K, d))

    print(f"正在初始化 Gumbel Router ...")
    print(f"特征维度: {d}, 候选提示节点数: {K}")
    # 实例化时传入原图特征维度和提示节点维度
    router = GumbelRouter(feature_dim=d, prompt_dim=d, tau=0.5)

    print("\n" + "="*40)
    # 测试一：Soft 模式（连续权重，适合融入类似注意力机制的信息聚合）
    EX_soft = router(test_node_features, test_prompt_features, hard=False)
    print(f"[Soft 模式] 生成的 EX 矩阵维度: {EX_soft.shape} (期望是 [2708, 5])")
    print(f"节点 0 到 5 个提示节点的连续连接权重 (和为1): \n{EX_soft[0].detach().numpy()}")

    print("\n" + "="*40)
    # 测试二：Hard 模式（绝对离散连边，适合修改严格的拓扑结构）
    EX_hard = router(test_node_features, test_prompt_features, hard=True)
    print(f"[Hard 模式] 生成的 EX 矩阵维度: {EX_hard.shape} (期望是 [2708, 5])")
    print(f"节点 0 到 5 个提示节点的绝对连接状态 (只有一个是1，其余是0): \n{EX_hard[0].detach().numpy()}")