"""目标域无监督图表征（GRACE / GAE / BGRL），用于 prompt 初始化前的 Z_target 学习。

仅使用目标图的 ``x`` 与 ``edge_index``，不使用节点标签，避免跨域聚类泄漏。
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge, negative_sampling

SSLMethod = Literal["grace", "gae", "bgrl"]


def _drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0.0:
        return x
    mask = torch.rand(x.size(1), device=x.device) > drop_prob
    return x * mask.float()


class GCNEncoder(nn.Module):
    """两层 GCN 编码器（与 BaseGCN 结构相近，参数独立训练）。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class GRACEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.projector = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)


class GAEModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode_logits(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


class BGRLModel(nn.Module):
    """简化 BGRL：online 编码 + predictor，target 为 EMA 副本。"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.online_encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.target_encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
        self.predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self._copy_target_from_online()

    def _copy_target_from_online(self) -> None:
        for t, o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            t.data.copy_(o.data)

    @torch.no_grad()
    def update_target(self, momentum: float = 0.99) -> None:
        for t, o in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            t.data.mul_(momentum).add_(o.data, alpha=1.0 - momentum)

    def forward_online(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x, edge_index)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = F.normalize(z1, p=2, dim=-1, eps=1e-12)
    z2 = F.normalize(z2, p=2, dim=-1, eps=1e-12)
    n = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.t() / temperature
    mask = torch.eye(2 * n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    pos = torch.cat([torch.arange(n, 2 * n, device=sim.device), torch.arange(0, n, device=sim.device)])
    targets = torch.arange(2 * n, device=sim.device)
    logits = sim
    return F.cross_entropy(logits, targets)


def _train_grace(
    model: GRACEModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    feat_drop: float,
    edge_drop: float,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        x1 = _drop_feature(x, feat_drop)
        x2 = _drop_feature(x, feat_drop)
        ei1, _ = dropout_edge(edge_index, p=edge_drop, force_undirected=True)
        ei2, _ = dropout_edge(edge_index, p=edge_drop, force_undirected=True)
        h1 = model.project(model(x1, ei1))
        h2 = model.project(model(x2, ei2))
        loss = _nt_xent(h1, h2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model.encoder(x, edge_index)


def _train_gae(
    model: GAEModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    num_neg: int | None = None,
) -> torch.Tensor:
    num_nodes = x.size(0)
    num_neg = num_neg or edge_index.size(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        pos_logits = model.decode_logits(z, edge_index)
        neg_edge = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
            force_undirected=True,
            method="sparse",
        )
        neg_logits = model.decode_logits(z, neg_edge)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model.encode(x, edge_index)


def _train_bgrl(
    model: BGRLModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    feat_drop: float,
    edge_drop: float,
    momentum: float = 0.99,
) -> torch.Tensor:
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr,
        weight_decay=0.0,
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        x1 = _drop_feature(x, feat_drop)
        x2 = _drop_feature(x, feat_drop)
        ei1, _ = dropout_edge(edge_index, p=edge_drop, force_undirected=True)
        ei2, _ = dropout_edge(edge_index, p=edge_drop, force_undirected=True)

        online1 = model.predictor(model.forward_online(x1, ei1))
        online2 = model.predictor(model.forward_online(x2, ei2))
        with torch.no_grad():
            target1 = model.target_encoder(x1, ei1)
            target2 = model.target_encoder(x2, ei2)

        loss = (
            2.0
            - F.cosine_similarity(online1, target2.detach(), dim=-1).mean()
            - F.cosine_similarity(online2, target1.detach(), dim=-1).mean()
        )
        loss.backward()
        optimizer.step()
        model.update_target(momentum=momentum)

    model.eval()
    with torch.no_grad():
        return model.target_encoder(x, edge_index)


def fit_target_ssl_embeddings(
    data,
    *,
    method: SSLMethod = "grace",
    hidden_dim: int = 128,
    out_dim: int = 128,
    epochs: int = 200,
    lr: float = 0.01,
    feat_drop: float = 0.2,
    edge_drop: float = 0.2,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.Tensor, str]:
    """在目标图上训练无监督编码器并返回 Z_target [N, out_dim]。"""
    device = torch.device(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    in_dim = x.size(-1)

    if verbose:
        print(
            f"[TargetSSL] method={method} | in_dim={in_dim} hidden={hidden_dim} "
            f"out={out_dim} | epochs={epochs} lr={lr}"
        )

    if method == "grace":
        model = GRACEModel(in_dim, hidden_dim, out_dim).to(device)
        z = _train_grace(
            model, x, edge_index, epochs=epochs, lr=lr, feat_drop=feat_drop, edge_drop=edge_drop
        )
    elif method == "gae":
        model = GAEModel(in_dim, hidden_dim, out_dim).to(device)
        z = _train_gae(model, x, edge_index, epochs=epochs, lr=lr)
    elif method == "bgrl":
        model = BGRLModel(in_dim, hidden_dim, out_dim).to(device)
        z = _train_bgrl(
            model, x, edge_index, epochs=epochs, lr=lr, feat_drop=feat_drop, edge_drop=edge_drop
        )
    else:
        raise ValueError(f"Unknown SSL method: {method}")

    z = z.detach()
    if verbose:
        print(f"[TargetSSL] Z_target shape={tuple(z.shape)}")
    return z, method
