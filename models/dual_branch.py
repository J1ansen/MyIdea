# models/dual_branch.py
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.gp2f_adapter import GP2FAdapter
from models.prompt_conv import PromptAwareGNNConv


class OrthogonalProjection(nn.Module):
    """Feature-preserving orthogonal projection for dimension alignment."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptBranchEncoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(PromptAwareGNNConv(hidden_dim, hidden_dim, alpha=0.5, beta=1.5))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, PromptAwareGNNConv):
                x = layer(x, edge_index, edge_type=edge_type)  # shape: [N+K, D]
            else:
                x = layer(x)
        return self.out_proj(x)


class DualBranchGNN(nn.Module):
    def __init__(
        self,
        pretrained_backbone,
        hidden_dim: int,
        out_dim: int,
        prompt_dim: int,
        bottleneck_dim: int = 32,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # 1. frozen branch
        self.frozen_branch = copy.deepcopy(pretrained_backbone)
        for param in self.frozen_branch.parameters():
            param.requires_grad = False
        self.frozen_branch.eval()

        # 2. adaptive branch
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim if input_dim is not None else hidden_dim
        self.x_proj = nn.Linear(self.input_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.x_proj.weight)
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.prompt_proj.weight)

        self.adapter = GP2FAdapter(hidden_dim, r=bottleneck_dim)
        self.prompt_branch = PromptBranchEncoder(hidden_dim, hidden_dim)

        # gate initialized toward frozen branch: h_final = (1-gate)*frozen + gate*adapted
        self.gate = nn.Parameter(torch.tensor(0.2))
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.domain_align = OrthogonalProjection(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prompt_nodes: Optional[torch.Tensor] = None,
        prompt_edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [N, F]
        with torch.no_grad():
            h_frozen = self.frozen_branch(x, edge_index)  # shape: [N, D]

        h_frozen = self.domain_align(h_frozen)  # shape: [N, D]

        if prompt_nodes is None:
            prompt_nodes = x.new_zeros((0, self.hidden_dim))  # shape: [0, D]

        x_base = self.x_proj(x) if x.size(-1) != self.hidden_dim else x  # shape: [N, D]

        if prompt_nodes.numel() > 0:
            if prompt_nodes.size(-1) != self.hidden_dim:
                prompt_nodes = self.prompt_proj(prompt_nodes)  # shape: [K, D]
            x_adapt = torch.cat([x_base, prompt_nodes], dim=0)  # shape: [N+K, D]
        else:
            x_adapt = x_base  # shape: [N, D]

        if prompt_edge_index is None:
            prompt_edge_index = edge_index
        if edge_type is None:
            edge_type = edge_index.new_zeros(edge_index.size(1), dtype=torch.long)

        h_adapted_full = self.prompt_branch(x_adapt, prompt_edge_index, edge_type=edge_type)  # shape: [N+K, D]
        h_adapted = h_adapted_full[: x.size(0), :]  # shape: [N, D]

        h_adapted = self.adapter(h_adapted)  # shape: [N, D]

        gate = torch.clamp(self.gate, 0.0, 1.0)
        h_final = (1.0 - gate) * h_frozen + gate * h_adapted  # shape: [N, D]
        logits = self.classifier(h_final)  # shape: [N, C]

        return logits, h_frozen, h_adapted
