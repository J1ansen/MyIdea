from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class PromptAwareGNNConv(MessagePassing):
    """Prompt-aware message passing layer.

    Edge weights are dynamically reweighted by:
    1) the local homophily score of the destination/source node;
    2) the edge type indicator (original edge vs prompt edge).

    Edge types:
        0 -> original graph edge
        1 -> prompt edge
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float = 0.5,
        beta: float = 1.5,
        bias: bool = True,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.root_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.local_gate = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid(),
        )

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.root_lin.weight)

    @staticmethod
    def _compute_local_homophily(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-12)
        sim = (x_norm[src] * x_norm[dst]).sum(dim=-1)
        return sim.clamp(min=0.0, max=1.0)  # shape: [E]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        prompt_bias: Optional[torch.Tensor] = None,
        return_edge_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # shape: [N, F]
        x = self.lin(x)

        if edge_type is None:
            edge_type = x.new_zeros(edge_index.size(1), dtype=torch.long)
        else:
            edge_type = edge_type.to(x.device)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loops = edge_index.new_zeros((1, x.size(0))).view(-1)
        edge_type = torch.cat([edge_type, self_loops], dim=0)

        local_h = self._compute_local_homophily(x, edge_index)  # shape: [E]
        prompt_mask = (edge_type == 1).float()
        orig_mask = (edge_type == 0).float()

        # Stronger prompt influence for heterophilic neighborhoods.
        # shape: [E]
        hetero_strength = 1.0 - local_h
        dyn_prompt = self.beta * (1.0 + hetero_strength)
        dyn_orig = self.alpha * (1.0 - 0.5 * hetero_strength)
        edge_weight = dyn_orig * orig_mask + dyn_prompt * prompt_mask

        # Node-local gating further suppresses noisy original aggregation.
        node_gate = self.local_gate(local_h.unsqueeze(-1)).squeeze(-1)  # shape: [E]
        edge_weight = edge_weight * node_gate

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out + self.root_lin(x)
        if self.bias is not None:
            out = out + self.bias

        if return_edge_weights:
            return out, edge_weight
        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(-1)

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr=None, dim_size=None) -> torch.Tensor:
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
