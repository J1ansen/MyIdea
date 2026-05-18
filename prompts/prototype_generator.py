# prompts/prototype_generator.py
import torch
import torch.nn as nn

class PrototypePromptGenerator:
    def __init__(self, num_classes, device='cpu'):
        """
        使用目标域的 Few-shot 样本进行有监督的原型初始化
        """
        self.num_classes = num_classes
        self.device = device

    @torch.no_grad()
    def generate(self, data, model, aligner, train_mask):
        """
        生成高纯度的神经原型作为提示节点
        """
        model.eval()
        aligner.eval()
        
        # 1. 特征对齐
        x_aligned = aligner(data.x)  
        
        # 2. 提取低频信号 H
        h = model(x_aligned, data.edge_index) 
        
        # 3. 提取高频残差 R
        if x_aligned.shape[-1] != h.shape[-1]:
            residual = x_aligned[:, :h.shape[-1]] - h
        else:
            residual = x_aligned - h
            
        # 4. 拼接异配双频特征 [H, R]
        h_aug = torch.cat([h, residual], dim=-1) 

        print(f"🌟 正在使用 {train_mask.sum().item()} 个 Few-shot 样本进行原型初始化...")
        
        p_homo_list = []
        p_hete_list = []
        
        h_train = h[train_mask]
        h_aug_train = h_aug[train_mask]
        y_train = data.y[train_mask].squeeze()

        # 5. 基于类别的原型聚合
        for c in range(self.num_classes):
            class_mask = (y_train == c)
            if class_mask.sum() > 0:
                p_homo_list.append(h_train[class_mask].mean(dim=0))
                p_hete_list.append(h_aug_train[class_mask].mean(dim=0))
            else:
                p_homo_list.append(h_train.mean(dim=0))
                p_hete_list.append(h_aug_train.mean(dim=0))

        p_homo = torch.stack(p_homo_list).to(self.device)
        p_hete = torch.stack(p_hete_list).to(self.device)

        return p_homo, p_hete