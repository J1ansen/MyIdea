# prompts/cluster_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


@dataclass
class HomophilyConfig:
    """Configuration for graph homophily estimation."""

    threshold: float = 0.5
    min_edges: int = 1


class PromptGenerator:
    """Generate prompt nodes for homophilic / heterophilic subspaces.

    The generator follows the project constraints:
    1) always estimate graph homophily;
    2) use spherical K-Means for the low-frequency homophilic signal;
    3) use GMM or Spectral Clustering for the heterophilic high-frequency augmented signal;
    4) return prompt prototypes wrapped as ``nn.Parameter``.
    """

    def __init__(
        self,
        k_homo: int = 10,
        k_hete: int = 10,
        device: str = "cpu",
        homophily_threshold: float = 0.5,
        hetero_clusterer: str = "gmm",
        random_state: int = 42,
    ) -> None:
        self.k_homo = int(k_homo)
        self.k_hete = int(k_hete)
        self.device = torch.device(device)
        self.homophily_cfg = HomophilyConfig(threshold=homophily_threshold)
        self.hetero_clusterer = hetero_clusterer.lower()
        self.random_state = int(random_state)

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @staticmethod
    def _safe_l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=eps)

    def _estimate_edge_homophily(self, data) -> float:
        if not hasattr(data, "edge_index") or not hasattr(data, "y"):
            return 0.0
        if data.edge_index is None or data.y is None:
            return 0.0

        edge_index = data.edge_index
        y = data.y.view(-1)
        if edge_index.numel() == 0 or y.numel() == 0:
            return 0.0

        src, dst = edge_index[0], edge_index[1]
        valid = (src >= 0) & (dst >= 0) & (src < y.size(0)) & (dst < y.size(0))
        src, dst = src[valid], dst[valid]
        if src.numel() < self.homophily_cfg.min_edges:
            return 0.0

        same = (y[src] == y[dst]).float().mean().item()
        return float(same)

    def _build_residual(self, x_aligned: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if x_aligned.size(-1) == h.size(-1):
            return x_aligned - h
        min_dim = min(x_aligned.size(-1), h.size(-1))
        return x_aligned[:, :min_dim] - h[:, :min_dim]

    def _spherical_kmeans(
        self,
        x: torch.Tensor,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """Cosine-distance K-Means via L2-normalized iterations."""
        if x.size(0) < n_clusters:
            raise ValueError(f"Cannot cluster {x.size(0)} samples into {n_clusters} clusters.")

        x_norm = self._safe_l2_normalize(x)
        indices = torch.randperm(x_norm.size(0), device=x_norm.device)[:n_clusters]
        centroids = x_norm[indices].clone()

        for _ in range(max_iter):
            similarity = x_norm @ centroids.t()  # shape: [N, K]
            labels = similarity.argmax(dim=1)

            new_centroids = []
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    cluster_vec = x_norm[mask].mean(dim=0)
                    new_centroids.append(self._safe_l2_normalize(cluster_vec.unsqueeze(0)).squeeze(0))
                else:
                    fallback_idx = torch.randint(0, x_norm.size(0), (1,), device=x_norm.device).item()
                    new_centroids.append(x_norm[fallback_idx])
            new_centroids = torch.stack(new_centroids, dim=0)

            delta = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()
            centroids = new_centroids
            if delta < tol:
                break

        return centroids

    def _hetero_clustering(self, h_aug: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(h_aug)
        n_samples = x_np.shape[0]
        if n_samples < self.k_hete:
            raise ValueError(f"Cannot cluster {n_samples} samples into {self.k_hete} heterophilic prompts.")

        if self.hetero_clusterer == "spectral":
            clustering = SpectralClustering(
                n_clusters=self.k_hete,
                affinity="nearest_neighbors",
                random_state=self.random_state,
                assign_labels="kmeans",
            )
            labels = clustering.fit_predict(x_np)
            centers = []
            for k in range(self.k_hete):
                mask = labels == k
                if mask.any():
                    centers.append(torch.from_numpy(x_np[mask].mean(axis=0)))
                else:
                    centers.append(torch.from_numpy(x_np[np.random.randint(0, n_samples)]))
            return torch.stack([c.float() for c in centers], dim=0)

        if self.hetero_clusterer == "gmm":
            gmm = GaussianMixture(
                n_components=self.k_hete,
                covariance_type="full",
                random_state=self.random_state,
                reg_covar=1e-6,
            )
            gmm.fit(x_np)
            return torch.from_numpy(gmm.means_).float()

        # Fallback: standard KMeans is intentionally avoided for heterophilic prompts.
        gmm = GaussianMixture(
            n_components=self.k_hete,
            covariance_type="full",
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        gmm.fit(x_np)
        return torch.from_numpy(gmm.means_).float()

    @torch.no_grad()
    def generate(self, data, model, aligner) -> Tuple[nn.Parameter, nn.Parameter]:
        """Generate homophilic and heterophilic prompt prototypes.

        Returns:
            p_homo: nn.Parameter with shape [K1, D_h]
            p_hete: nn.Parameter with shape [K2, D_hete]
        """
        model.eval()
        aligner.eval()

        homophily = self._estimate_edge_homophily(data)
        is_homophilic_graph = homophily >= self.homophily_cfg.threshold

        x_aligned = aligner(data.x)  # shape: [N, source_dim]
        h = model(x_aligned, data.edge_index)  # shape: [N, hidden_dim]
        residual = self._build_residual(x_aligned, h)  # shape: [N, hidden_dim]
        h_aug = torch.cat([h, residual], dim=-1)  # shape: [N, 2 * hidden_dim]

        print(f"[PromptGenerator] estimated edge homophily = {homophily:.4f}")
        print(f"[PromptGenerator] graph type = {'homophilic' if is_homophilic_graph else 'heterophilic'}")

        # Stage-1 constraint: low-frequency prompts always use spherical K-Means.
        print(f"[PromptGenerator] spherical K-Means for homophilic prompts (K1={self.k_homo})...")
        p_homo = self._spherical_kmeans(h, self.k_homo)  # shape: [K1, hidden_dim]

        # Stage-1 constraint: heterophilic graphs must not use standard K-Means.
        print(f"[PromptGenerator] heterophilic clustering for augmented prompts (K2={self.k_hete})...")
        if is_homophilic_graph:
            # For homophilic graphs, heterophilic noise is mild; GMM still captures the augmented manifold.
            p_hete = self._hetero_clustering(h_aug)  # shape: [K2, 2 * hidden_dim]
        else:
            # For heterophilic graphs, prefer non-convex clustering explicitly.
            original_choice = self.hetero_clusterer
            self.hetero_clusterer = "spectral" if original_choice not in {"gmm", "spectral"} else original_choice
            p_hete = self._hetero_clustering(h_aug)  # shape: [K2, 2 * hidden_dim]
            self.hetero_clusterer = original_choice

        # Unify prompt prototype dimension so homo/hete prompts can be concatenated safely.
        if p_hete.size(-1) != p_homo.size(-1):
            if p_hete.size(-1) > p_homo.size(-1):
                p_hete = p_hete[:, : p_homo.size(-1)]
            else:
                pad = torch.zeros(p_hete.size(0), p_homo.size(-1) - p_hete.size(-1), device=p_hete.device, dtype=p_hete.dtype)
                p_hete = torch.cat([p_hete, pad], dim=-1)

        p_homo = nn.Parameter(p_homo.to(self.device), requires_grad=True)
        p_hete = nn.Parameter(p_hete.to(self.device), requires_grad=True)

        return p_homo, p_hete
