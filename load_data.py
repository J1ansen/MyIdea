"""统一数据加载层：8 数据集 + 拓扑语义 profile + 5-shot 划分 + 跨域加载。

本模块是所有训练 / 测试脚本共享的数据底座，确保：
    1. 8 个 benchmark 使用一致的加载与归一化协议；
    2. 每个数据集自带 `DatasetProfile`（family / 默认 Top-ρ / 拓扑语义）；
    3. Few-shot（默认 shots=5）划分按类均衡 + 可复现；
    4. 跨域实验 (source → target) 有统一入口。

数据集别名见 `DATASET_ALIASES`；规范名以 PyG 类参数为准（如 ``Amazon-ratings``, ``squirrel``）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import (
    Actor,
    Flickr,
    HeterophilousGraphDataset,
    Planetoid,
    WikipediaNetwork,
)


# =============================================================================
#  数据集拓扑语义 profile（与 README 第八节 / idea 第四节严格对齐）
# =============================================================================


@dataclass(frozen=True)
class DatasetProfile:
    """单个数据集的拓扑语义与默认超参。

    Attributes:
        name: 规范名称（PyG 类参数）。
        family: ``"homophilic"`` 或 ``"heterophilic"``，决定 λ2、聚类策略等行为。
        default_rho: 阶段二 Top-ρ 候选池比例的默认值（异配族取较大值）。
        default_k_homo: 同配提示节点数量，默认与目标域类别数对齐（同配图），
                        或取小值（1）表示"仅需一个跨类桥梁"。
        default_k_hete: 异配提示节点数量；异配丰富的数据集取较大值（=类别数），
                        同配图仅需少量（如 1）隔离综述/跨界节点。
        p_homo_role: 同配提示节点在该数据集上的拓扑语义。
        p_hete_role: 异配提示节点在该数据集上的拓扑语义。
    """

    name: str
    family: str
    default_rho: float
    default_k_homo: int
    default_k_hete: int
    p_homo_role: str
    p_hete_role: str

    @property
    def is_heterophilic(self) -> bool:
        return self.family == "heterophilic"


# 8 个 benchmark 的 profile（顺序仅作展示，无功能影响）
# default_k_homo / default_k_hete 设计原则：
#   同配图：k_homo = 类别数（每类一个语义枢纽），k_hete = 1（少量跨界桥梁足矣）
#   异配图：k_homo = k_hete = 类别数（homo/hete 语义模式同样丰富）
DATASET_PROFILES: Dict[str, DatasetProfile] = {
    # ---- 强同配 / 隐含异配族（idea 第四节 8.2）----
    "Cora": DatasetProfile(
        name="Cora",
        family="homophilic",
        default_rho=0.15,
        default_k_homo=7,   # 7 个论文领域，每类一个同配枢纽
        default_k_hete=1,   # 仅需 1 个"综述隔离"节点
        p_homo_role="语义填补器：在同领域但未互引的论文之间补连边（虚拟主题会议）",
        p_hete_role="跨学科综述隔离：聚合跨领域综述论文，避免它们在 MP 中导致过平滑",
    ),
    "CiteSeer": DatasetProfile(
        name="CiteSeer",
        family="homophilic",
        default_rho=0.15,
        default_k_homo=6,   # 6 类
        default_k_hete=1,
        p_homo_role="语义填补器：补全同主题论文的缺失引用",
        p_hete_role="跨主题桥梁：隔离跨学科节点，缓解噪声扩散",
    ),
    "PubMed": DatasetProfile(
        name="PubMed",
        family="homophilic",
        default_rho=0.15,
        default_k_homo=3,   # 3 类（Diabetes I/II/Experimental）
        default_k_hete=1,
        p_homo_role="语义填补器：聚合同领域生医论文",
        p_hete_role="跨领域综述隔离：避免综述类论文导致的特征平滑",
    ),
    "Amazon-ratings": DatasetProfile(
        name="Amazon-ratings",
        family="homophilic",
        default_rho=0.20,
        default_k_homo=5,   # 5 个评分等级
        default_k_hete=2,   # 跨品类桥梁略多（百搭配件模式多样）
        p_homo_role="共购同配填补：相似商品在评分图上的同盟",
        p_hete_role="功能跨界桥梁：将跨品类「百搭爆款配件」隔离聚合，阻断跨主品类污染",
    ),
    "Flickr": DatasetProfile(
        name="Flickr",
        family="homophilic",
        default_rho=0.20,
        default_k_homo=7,   # Flickr 标签体系丰富，取 7 作默认
        default_k_hete=2,
        p_homo_role="视觉相似图像聚合：补连无元数据关联的视觉相似图像",
        p_hete_role="跨标签功能桥梁：隔离跨主题高度互联的图像",
    ),
    # ---- 极端异配 / 多部图族（idea 第四节 8.1）----
    "Minesweeper": DatasetProfile(
        name="Minesweeper",
        family="heterophilic",
        default_rho=0.35,
        default_k_homo=2,   # 二分类（mine/safe），homo/hete 各 2
        default_k_hete=2,
        p_homo_role="长尾同盟：将边缘安全节点在输入层聚拢",
        p_hete_role="孤岛直连通道：打破网格隔离，让地雷节点共享『危险模式矩阵』",
    ),
    "Actor": DatasetProfile(
        name="Actor",
        family="heterophilic",
        default_rho=0.35,
        default_k_homo=5,   # 5 类演员角色，homo 每类一个
        default_k_hete=5,   # 跨界模式丰富，hete 同样取 5
        p_homo_role="小众题材同盟：聚拢特定流派的边缘演员",
        p_hete_role="跨流派配角通道：捕获『万金油』配角，隔离其共演噪声",
    ),
    "squirrel": DatasetProfile(
        name="squirrel",
        family="heterophilic",
        default_rho=0.40,
        default_k_homo=5,   # 5 类维基页面
        default_k_hete=5,
        p_homo_role="长尾页面同盟：聚拢冷门维基词条",
        p_hete_role="超级枢纽：直连跨领域高流量页面，防止被长尾稀释",
    ),
}


# 支持论文 / 代码中常见的拼写变体
DATASET_ALIASES: Dict[str, str] = {
    "amazon": "Amazon-ratings",
    "Amazon": "Amazon-ratings",
    "Amazon-Ratings": "Amazon-ratings",
    "AmazonRatings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "MINESWEEPER": "Minesweeper",
    "actor": "Actor",
    "ACTOR": "Actor",
    "Squirrel": "squirrel",
    "SQUIRREL": "squirrel",
    "SQUIRREL-F": "squirrel",
    "squirrel-f": "squirrel",
}

ALL_DATASETS: List[str] = list(DATASET_PROFILES.keys())


def resolve_dataset_name(name: str) -> str:
    """把别名 / 大小写变体规范化为 `DATASET_PROFILES` 的 key。"""
    key = name.strip()
    if key in DATASET_PROFILES:
        return key
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    # 大小写不敏感兜底
    lowered = {k.lower(): k for k in DATASET_PROFILES}
    if key.lower() in lowered:
        return lowered[key.lower()]
    raise ValueError(
        f"未知数据集 '{name}'。支持: {ALL_DATASETS}（亦支持别名: {sorted(DATASET_ALIASES)}）"
    )


def get_dataset_profile(name: str) -> DatasetProfile:
    """获取规范化后的数据集 profile（含 family / default_rho / 拓扑语义）。"""
    return DATASET_PROFILES[resolve_dataset_name(name)]


# =============================================================================
#  数据集加载（PyG 原生 API + NormalizeFeatures）
# =============================================================================


def load_dataset(
    name: str,
    root: str = "./data",
    normalize: bool = True,
) -> Tuple[Dataset, str]:
    """加载 8 个 benchmark 中的一个。

    Args:
        name: 数据集名称（支持别名）。
        root: 数据下载 / 缓存根目录。所有子集会按数据集类型分目录存放。
        normalize: 是否对节点特征做行归一化（默认 True，符合主流图分类协议）。

    Returns:
        (dataset, canonical_name)：PyG Dataset 对象 + 规范化后的数据集名称。
    """
    name = resolve_dataset_name(name)
    transform = T.NormalizeFeatures() if normalize else None
    root_path = str(Path(root))

    if name in {"Cora", "CiteSeer", "PubMed"}:
        dataset = Planetoid(root=f"{root_path}/Planetoid", name=name, transform=transform)
    elif name in {"Amazon-ratings", "Minesweeper"}:
        dataset = HeterophilousGraphDataset(
            root=f"{root_path}/HeterophilousGraphDataset",
            name=name,
            transform=transform,
        )
    elif name == "Flickr":
        dataset = Flickr(root=f"{root_path}/Flickr", transform=transform)
    elif name == "Actor":
        dataset = Actor(root=f"{root_path}/Actor", transform=transform)
    elif name == "squirrel":
        # PyG WikipediaNetwork 的 squirrel 子集对应论文中 SQUIRREL-F
        dataset = WikipediaNetwork(
            root=f"{root_path}/WikipediaNetwork",
            name="squirrel",
            transform=transform,
        )
    else:
        # 理论上不会走到这里，resolve_dataset_name 已经过滤
        raise ValueError(f"未实现的数据集加载逻辑: {name}")

    return dataset, name


# =============================================================================
#  Few-shot 划分（默认 shots=5，与论文 5-shot 协议对齐）
# =============================================================================


@dataclass
class FewShotSplit:
    """每类均衡的 5-shot 划分结果。

    Attributes:
        train_mask: bool tensor [N]，每类各 ``shots`` 个标注样本（用于训练）。
        val_mask:   bool tensor [N]，每类 ``val_per_class`` 个验证样本。
        test_mask:  bool tensor [N]，其余样本，用于测试。
        shots:      实际使用的 shots 数。
    """

    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    shots: int


def build_few_shot_masks(
    y: torch.Tensor,
    shots: int = 5,
    val_per_class: int = 30,
    seed: int = 42,
) -> FewShotSplit:
    """构造 5-shot 训练 + 类均衡验证 + 剩余测试的 mask。

    协议说明：
        - 每类先随机抽 ``shots`` 个作为训练；
        - 再抽 ``val_per_class`` 个作为验证（若某类样本不足，按可用量裁剪）；
        - 剩余全部进入测试集；
        - 仅使用类别 ``>= 0`` 的标签（-1 等占位标签自动忽略）。

    Args:
        y: 标签 tensor，形状 [N] 或 [N, 1]。
        shots: 每类训练样本数（默认 5，符合论文设定）。
        val_per_class: 每类验证样本数（默认 30）。
        seed: 随机种子，保证不同 run 之间可复现。

    Returns:
        FewShotSplit
    """
    y = y.view(-1)
    device = y.device
    num_nodes = int(y.numel())
    rng = np.random.default_rng(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    if num_nodes == 0:
        return FewShotSplit(train_mask, val_mask, test_mask, shots)

    classes = sorted(int(c) for c in torch.unique(y).tolist() if int(c) >= 0)

    for c in classes:
        idx = torch.where(y == c)[0].cpu().numpy()
        if idx.size == 0:
            continue
        rng.shuffle(idx)

        # 按 shots / val_per_class 切片；若样本不足，按实际数量裁剪
        train_take = min(shots, idx.size)
        val_take = min(val_per_class, max(0, idx.size - train_take))

        train_idx = idx[:train_take]
        val_idx = idx[train_take : train_take + val_take]
        test_idx = idx[train_take + val_take :]

        if train_idx.size > 0:
            train_mask[torch.from_numpy(train_idx).to(device)] = True
        if val_idx.size > 0:
            val_mask[torch.from_numpy(val_idx).to(device)] = True
        if test_idx.size > 0:
            test_mask[torch.from_numpy(test_idx).to(device)] = True

    return FewShotSplit(train_mask, val_mask, test_mask, shots)


# =============================================================================
#  跨域加载入口（source 仅提供维度信息；目标域用于训练 / 评估）
# =============================================================================


@dataclass
class CrossDomainBundle:
    """跨域实验所需的一站式数据包。

    Attributes:
        source_data: 源域 PyG `Data`（仅用于推断维度；通常不参与训练）。
        target_data: 目标域 PyG `Data`。
        target_split: 目标域的 5-shot 划分。
        source_profile: 源域 profile。
        target_profile: 目标域 profile。
        source_in_dim / target_in_dim: 节点特征维度。
        source_num_classes / target_num_classes: 类别数。
    """

    source_data: Data
    target_data: Data
    target_split: FewShotSplit
    source_profile: DatasetProfile
    target_profile: DatasetProfile
    source_in_dim: int
    target_in_dim: int
    source_num_classes: int
    target_num_classes: int


def load_cross_domain(
    source_name: str,
    target_name: str,
    root: str = "./data",
    shots: int = 5,
    val_per_class: int = 30,
    seed: int = 42,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> CrossDomainBundle:
    """同时加载源域 + 目标域 + 目标 5-shot 划分。

    源域信息主要用于：
        - 维度对齐（OrthogonalProjection 的输入 / 输出维）；
        - 加载预训练 backbone 时确定 in_channels；
    目标域承担实际训练与评估。

    Note:
        预训练权重的加载交给 `models.base_gnn.load_pretrained_backbone`，
        本函数不负责权重 IO。
    """
    src_ds, src_name = load_dataset(source_name, root=root, normalize=normalize)
    tgt_ds, tgt_name = load_dataset(target_name, root=root, normalize=normalize)

    src_data = src_ds[0]
    tgt_data = tgt_ds[0]
    if device is not None:
        src_data = src_data.to(device)
        tgt_data = tgt_data.to(device)

    target_split = build_few_shot_masks(
        tgt_data.y,
        shots=shots,
        val_per_class=val_per_class,
        seed=seed,
    )

    return CrossDomainBundle(
        source_data=src_data,
        target_data=tgt_data,
        target_split=target_split,
        source_profile=DATASET_PROFILES[src_name],
        target_profile=DATASET_PROFILES[tgt_name],
        source_in_dim=int(src_ds.num_features),
        target_in_dim=int(tgt_ds.num_features),
        source_num_classes=int(src_ds.num_classes),
        target_num_classes=int(tgt_ds.num_classes),
    )


# =============================================================================
#  辅助工具
# =============================================================================


def estimate_edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """估计整图边同配率（相连节点同类的边比例）。"""
    if edge_index is None or edge_index.numel() == 0 or y is None or y.numel() == 0:
        return 0.0
    y_flat = y.view(-1)
    src, dst = edge_index[0], edge_index[1]
    valid = (
        (src >= 0)
        & (dst >= 0)
        & (src < y_flat.numel())
        & (dst < y_flat.numel())
        & (y_flat[src] >= 0)
        & (y_flat[dst] >= 0)
    )
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    return float((y_flat[src] == y_flat[dst]).float().mean().item())


# =============================================================================
#  向后兼容（旧代码 / 用户脚本仍在用的 API；新代码请使用上面的接口）
# =============================================================================


def load_node_data(dataset_name: str, data_folder: str):
    """[Deprecated] 旧接口；返回 (data, in_dim, out_dim)。新代码请用 `load_dataset`。"""
    dataset, _ = load_dataset(dataset_name, root=data_folder)
    data = dataset[0]
    return data, int(dataset.num_features), int(dataset.num_classes)


def NodeDownstream(data, shots: int = 5, test_node_num: int = 1000):
    """[Deprecated] 旧接口；返回 (train_node_list, test_node_list)。

    新代码请用 `build_few_shot_masks`，可直接得到 train/val/test 三套 mask。
    """
    num_classes = int(data.y.max().item()) + 1
    node_list: List[int] = []
    for c in range(num_classes):
        indices = torch.where(data.y.squeeze() == c)[0].tolist()
        if len(indices) < shots:
            node_list.extend(indices)
        else:
            node_list.extend(random.sample(indices, k=shots))

    random_node_list = random.sample(range(data.num_nodes), k=data.num_nodes)
    for node in node_list:
        random_node_list.remove(node)

    train_node_list = node_list
    if test_node_num > 1:
        test_node_list = random_node_list[:test_node_num]
    else:
        test_node_list = random_node_list[: int(test_node_num * data.num_nodes)]

    return train_node_list, test_node_list


def normalize_high_freq_adj(edge_index, num_nodes):
    """预留：严格高通图滤波器 L = I - D^{-1/2} A D^{-1/2} 的实现位。

    当前 `prompts.cluster_generator.PromptGenerator` 使用 ``X - H_low`` 作为高频残差，
    若需要更严格的频谱视角，可在此实现 L · X 并替换 `_build_residual` 的输入。
    """
    raise NotImplementedError("如需严格高通滤波，请在此函数中实现并接入 cluster_generator。")
