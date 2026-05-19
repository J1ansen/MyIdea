# models/dual_branch.py
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from models.gp2f_adapter import GP2FAdapter
from models.prompt_conv import PromptAwareGNNConv


@dataclass
class GateSchedule:
    """Epoch-aware fusion gate schedule.

    The scheduled value is the base gate in:
        H_final = (1 - g) * H_frozen + g * H_adapted

    It starts close to the frozen branch and gradually trusts the adapted branch.
    """

    start: float = 0.2
    end: float = 0.8
    warmup_epochs: int = 100
    mode: str = "cosine"  # "linear" or "cosine"
    learnable_scale: float = 0.05

    def value(self, epoch: Optional[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if epoch is None:
            progress = 0.0
        elif self.warmup_epochs <= 0:
            progress = 1.0
        else:
            progress = min(max(float(epoch) / float(self.warmup_epochs), 0.0), 1.0)

        if self.mode == "linear":
            factor = progress
        elif self.mode == "cosine":
            factor = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            raise ValueError(f"Unsupported gate schedule mode: {self.mode}")

        gate = self.start + (self.end - self.start) * factor
        return torch.tensor(gate, device=device, dtype=dtype).clamp(0.0, 1.0)


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
            # beta 过大易在 (N+K) 提示图上放大激活导致 NaN；略降并加 LayerNorm
            layers.append(PromptAwareGNNConv(hidden_dim, hidden_dim, alpha=0.8, beta=1.0))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        node_heterophily: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, PromptAwareGNNConv):
                x = layer(
                    x,
                    edge_index,
                    edge_type=edge_type,
                    node_heterophily=node_heterophily,
                )  # shape: [N+K, D]
            else:
                x = layer(x)
            if not torch.isfinite(x).all():
                x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
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
        gate_schedule: Optional[Union[GateSchedule, dict]] = None,
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

        if gate_schedule is None:
            gate_schedule = GateSchedule()
        elif isinstance(gate_schedule, dict):
            gate_schedule = GateSchedule(**gate_schedule)
        self.gate_schedule = gate_schedule
        self.gate_offset = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.domain_align = OrthogonalProjection(hidden_dim, hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_branch.eval()
        return self

    def get_gate(self, epoch: Optional[int] = None) -> torch.Tensor:
        base_gate = self.gate_schedule.value(
            epoch,
            device=self.gate_offset.device,
            dtype=self.gate_offset.dtype,
        )
        gate = base_gate + self.gate_schedule.learnable_scale * torch.tanh(self.gate_offset)
        return gate.clamp(0.0, 1.0)

    @staticmethod
    def _expand_node_heterophily(
        node_heterophily: Optional[torch.Tensor],
        num_real_nodes: int,
        num_total_nodes: int,
    ) -> Optional[torch.Tensor]:
        if node_heterophily is None:
            return None
        if node_heterophily.size(0) == num_total_nodes:
            return node_heterophily
        if node_heterophily.size(0) != num_real_nodes:
            raise ValueError(
                "node_heterophily must have length N or N+K, "
                f"got {node_heterophily.size(0)} for N={num_real_nodes}, N+K={num_total_nodes}"
            )
        if num_total_nodes == num_real_nodes:
            return node_heterophily
        pad = node_heterophily.new_zeros(num_total_nodes - num_real_nodes)
        return torch.cat([node_heterophily, pad], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prompt_nodes: Optional[torch.Tensor] = None,
        prompt_edge_index: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None,
        node_heterophily: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        return_branch_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
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

        node_heterophily_full = self._expand_node_heterophily(
            node_heterophily,
            num_real_nodes=x.size(0),
            num_total_nodes=x_adapt.size(0),
        )
        h_adapted_full = self.prompt_branch(
            x_adapt,
            prompt_edge_index,
            edge_type=edge_type,
            node_heterophily=node_heterophily_full,
        )  # shape: [N+K, D]
        h_adapted = h_adapted_full[: x.size(0), :]  # shape: [N, D]

        h_adapted = self.adapter(h_adapted)  # shape: [N, D]

        gate = self.get_gate(epoch)
        h_final = (1.0 - gate) * h_frozen + gate * h_adapted  # shape: [N, D]
        logits = self.classifier(h_final)  # shape: [N, C]

        if return_branch_logits:
            logits_frozen = self.classifier(h_frozen)
            logits_adapted = self.classifier(h_adapted)
            return logits, h_frozen, h_adapted, logits_frozen, logits_adapted
        return logits, h_frozen, h_adapted
