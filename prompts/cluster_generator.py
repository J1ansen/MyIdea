# 特征提示节点生成器
# 基于高低频信号分离与 K-means 聚类
# 输入: PyG 的 Data 对象，包含 data.x 和 data.edge_index
# 输出: p_homo (同配提示特征), p_hete (异配提示特征)

import torch
import numpy as np
# 如果 sklearn 太慢，可以换成 faiss-gpu
from sklearn.cluster import KMeans 
from torch_geometric.utils import add_self_loops, scatter

class PromptGenerator:
    """
    图提示节点生成器：基于高低频信号分离与 K-means 聚类
    """
    def __init__(self, k_homo=10, k_hete=10, device='cpu'):
        self.k_homo = k_homo  # 同配提示节点数量 K1
        self.k_hete = k_hete  # 异配提示节点数量 K2
        self.device = device

    def _get_low_freq_signals(self, features, edge_index):
        """
        内部方法：计算低频信号（同配特征）
        （这里需要实现特征的平滑操作，比如简单的邻居特征聚合）
        补全代码：
        """
        # 给每个节点加自环，确保“节点自身 + 一阶邻居”共同参与均值聚合
        edge_index_with_loops, _ = add_self_loops(
            edge_index,
            num_nodes=features.size(0),
        )

        src, dst = edge_index_with_loops  # src -> dst
        # 对每个 dst 节点，聚合其 src（邻居+自身）特征并取均值
        low_freq = scatter(
            features[src],
            dst,
            dim=0,
            dim_size=features.size(0),
            reduce='mean',
        )
        return low_freq

    def _get_high_freq_signals(self, features, edge_index):
        """
        内部方法：计算高频信号（异配特征）
        （原特征减去低频特征，或者使用拉普拉斯矩阵进行高频滤波）
        补全代码：
        """
        low_freq = self._get_low_freq_signals(features, edge_index)
        high_freq = features - low_freq
        return high_freq

    def generate(self, data):
        """
        主控方法：生成提示节点
        输入: data (PyG 的 Data 对象，包含 data.x 和 data.edge_index)
        输出: p_homo (同配提示特征), p_hete (异配提示特征)
        """
        features = data.x
        edge_index = data.edge_index

        # 1. 提取信号
        low_freq_feats = self._get_low_freq_signals(features, edge_index)
        high_freq_feats = self._get_high_freq_signals(features, edge_index)

        # 2. 生成同配提示节点 (对低频特征聚类)
        print(f"正在进行同配聚类，生成 {self.k_homo} 个节点...")
        kmeans_homo = KMeans(n_clusters=self.k_homo, random_state=42)
        kmeans_homo.fit(low_freq_feats.cpu().numpy())
        p_homo = torch.tensor(kmeans_homo.cluster_centers_, dtype=torch.float32).to(self.device)

        # 3. 生成异配提示节点 (对高低频拼接特征聚类)
        print(f"正在进行异配聚类，生成 {self.k_hete} 个节点...")
        # 拼接低频和高频信号得到 H_aug
        h_aug = torch.cat([low_freq_feats, high_freq_feats], dim=-1)
        kmeans_hete = KMeans(n_clusters=self.k_hete, random_state=42)
        kmeans_hete.fit(h_aug.cpu().numpy())
        p_hete = torch.tensor(kmeans_hete.cluster_centers_, dtype=torch.float32).to(self.device)

        return p_homo, p_hete

# ==========================================
# 独立测试模块 
# 只有直接运行这个文件时，下面的代码才会执行
# ==========================================
if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    
    # 1. 随便加载一个小型数据集（比如 Cora）用来做测试
    dataset = Planetoid(root='./data/Cora', name='Cora')
    test_data = dataset[0]
    print(f"原始图节点数量: {test_data.num_nodes}, 特征维度: {test_data.num_features}")

    # 2. 实例化你写的生成器
    generator = PromptGenerator(k_homo=5, k_hete=3)

    # 3. 试运行！
    try:
        p_homo, p_hete = generator.generate(test_data)
        print(f"成功！同配提示节点维度: {p_homo.shape}") # 期望输出: [5, 特征维度]
        # 注意：由于拼接了高低频，异配节点的维度是原特征维度的 2 倍
        print(f"成功！异配提示节点维度: {p_hete.shape}") # 期望输出: [3, 2 * 特征维度] 
    except Exception as e:
        print(f"报错啦，赶紧让 Cursor 帮你看看: {e}")