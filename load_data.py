# load_data.py
# 负责跨域图数据集的自动下载、解析以及 Few-shot 样本划分

import random
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr, HeterophilousGraphDataset

def load_node_data(dataset_name, data_folder):
    """
    按名称加载节点分类数据集，并返回图数据对象与特征/分类维度信息。
    支持同配图 (Planetoid) 和 异配图 (HeterophilousGraphDataset)。
    """
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=f'{data_folder}/Planetoid', name=dataset_name)
    elif dataset_name in ['Amazon-ratings', 'Minesweeper', 'Roman-empire', 'Questions']:
        # 天然支持异配图数据集，完美契合你的 Idea！
        dataset = HeterophilousGraphDataset(root=f'{data_folder}/HeterophilousGraphDataset', name=dataset_name)
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=f'{data_folder}/Flickr')
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    data = dataset[0]
    input_dim = dataset.num_features
    output_dim = dataset.num_classes

    return data, input_dim, output_dim


def NodeDownstream(data, shots=5, test_node_num=1000):
    """
    构造 Few-shot (少样本) 的训练节点与测试节点划分。
    
    参数:
        data: PyG 图数据对象
        shots: 每个类别在目标域中能“看到”的标签数量 (跨域微调通常标签极少)
        test_node_num: 用于测试模型性能的节点数量
    返回:
        train_node_list: 用于微调的节点索引列表
        test_node_list: 用于测试的节点索引列表
    """
    num_classes = data.y.max().item() + 1
    node_list = []
    
    # 1. 抽取每个类别的 shots 个节点作为训练集
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            node_list.extend(indices)
        else:
            node_list.extend(random.sample(indices, k=shots))
            
    # 2. 剩余节点打乱用于测试
    random_node_list = random.sample(range(data.num_nodes), k=data.num_nodes)
    for node in node_list:
        random_node_list.remove(node)
        
    train_node_list = node_list
    
    # 3. 截取指定数量的测试节点
    if test_node_num > 1:
        test_node_list = random_node_list[:test_node_num]
    else:
        test_node_list = random_node_list[:int(test_node_num * data.num_nodes)]

    return train_node_list, test_node_list


# ==========================================
# 🌟 未来功能扩展区 (预留给我们的新 Idea)
# ==========================================
def normalize_high_freq_adj(edge_index, num_nodes):
    """
    【预留位置】：如果你发现在 `cluster_generator.py` 中直接用 `features - low_freq` 
    得到的高频特征不够好，后续我们可以在这里实现一个严格的“高通图滤波器”(High-pass Filter)。
    例如：计算拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}，然后用它来乘节点特征。
    
    如果有需要，我们随时可以在这里补充数学计算代码。
    """
    pass