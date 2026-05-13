# models/base_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class BaseGCN(nn.Module):
    """
    复刻 GraphTOP 预训练模型的 2 层 GCN 骨架。
    必须保证变量名（conv1, conv2）与 .pth 文件中的键值完全对应！
    """
    def __init__(self, in_channels, hidden_channels):
        super(BaseGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_pretrained_backbone(weight_path, in_channels, hidden_channels=128, device='cpu'):
    """
    万能智能加载器：读取 .pth 文件并将权重安全地注入到 BaseGCN 中
    """
    # 1. 实例化空骨架
    model = BaseGCN(in_channels=in_channels, hidden_channels=hidden_channels)
    
    # 2. 读取磁盘上的权重文件 (安全映射到指定设备)
    print(f"正在加载预训练权重: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    
    # 3. 智能解包：很多论文保存的 .pth 是一个大字典，里面不仅有权重，还有 epoch、optimizer 等
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # 假设整个字典就是权重
    else:
        state_dict = checkpoint
        
    # 4. 剔除可能的命名空间前缀 (有些代码保存时会带上 'gnn.' 或 'encoder.' 的前缀)
    clean_state_dict = {}
    for k, v in state_dict.items():
        # 如果带有前缀，截取掉 (例如把 'gnn.conv1.weight' 变成 'conv1.weight')
        clean_key = k.replace('gnn.', '').replace('encoder.', '').replace('backbone.', '')
        clean_state_dict[clean_key] = v

    # 5. 将清洗好的权重注入到模型中 (strict=False 允许少许无关参数的不匹配)
    model.load_state_dict(clean_state_dict, strict=False)
    print("预训练权重注入成功！\n")
    
    return model