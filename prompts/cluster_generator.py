"""阶段一：双频提示初始化与自适应语义簇生成。

整体流程（与 README §3 / idea 第二节阶段一严格对齐）：

    目标图 ──► 冻结 GNN ──► 低频信号 H_low ──► 球形 K-Means ──► P_homo (K1 个)
                              │
                              └──► 残差 H_high = X - H_low
                                       │
                                       └──► 拼接 H_aug = [H_low ; H_high]
                                                 │
                                                 ├── 同配图 → 基础 K-Means
                                                 └── 异配图 → GMM / 谱聚类（禁止 K-Means）
                                                            ──► P_hete (K2 个)

物理意义：
    - 低频信号 H_low 对应"同质性平滑"，聚类得到的 P_homo 充当**语义簇枢纽**；
    - 高频残差 H_high = X - H_low 捕获节点偏离邻域均值的部分，
      在异配图上即"异常 / 高频特征"，拼接后的 H_aug 让 P_hete 同时
      看到节点的"原始身份"与"邻域差异"，便于发现非凸的异配流形。

返回：
    p_homo, p_hete 均包裹为 ``nn.Parameter``（可被反向传播更新）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from load_data import estimate_edge_homophily
from prompts.gumbel_route import compute_local_heterophily, select_top_rho_pool
from prompts.target_ssl import SSLMethod, fit_target_ssl_embeddings


# =============================================================================
#  数据结构
# =============================================================================


@dataclass
class HomophilyConfig:
    """图同配率估计配置。"""

    threshold: float = 0.5  # 高于此阈值视为同配图
    min_edges: int = 1


@dataclass
class PromptSignals:
    """提示生成阶段的中间产物，可被下游模块（路由 / 损失）复用。

    Attributes:
        h_low:           [N, D_low]  低频平滑信号
        h_high:          [N, D_high] 高频残差（X 与 H_low 维度对齐后相减）
        h_aug:           [N, D_low + D_high] 双频拼接
        z_target:        [N, D] 目标域无监督嵌入（仅 target_ssl 模式）
        edge_homophily:  float, 整图边同配率
        is_homophilic:   bool, 是否被视为同配图（依据 ``homophily_threshold`` 或外部强制）
        hetero_clusterer_used: 实际使用的异配聚类器名称（kmeans / gmm / spectral / target_ssl）
        pool_size:       Top-ρ 候选池大小（target_ssl 模式）
    """

    h_low: torch.Tensor
    h_high: torch.Tensor
    h_aug: torch.Tensor
    edge_homophily: float
    is_homophilic: bool
    hetero_clusterer_used: str
    z_target: Optional[torch.Tensor] = None
    pool_size: int = 0


# =============================================================================
#  PromptGenerator
# =============================================================================


class PromptGenerator:
    """生成 P_homo / P_hete 两类提示原型（无监督，仅依赖目标图与冻结 GNN）。

    与 idea 严格一致的约束：
        1. 始终估计图同配率，必要时由 ``graph_family`` 外部覆盖；
        2. 同配提示一律使用 **球形 K-Means**（cosine 距离更稳健于尺度）；
        3. 异配提示的聚类策略取决于图族：
             - 同配图：基础 K-Means（噪声轻，凸结构足够）
             - 异配图：**禁止 K-Means**，使用 GMM 或谱聚类捕捉非凸流形；
        4. 返回的两个原型均为 ``nn.Parameter``，可在阶段二/三中参与梯度更新。
    """

    HETERO_CLUSTERERS = {"gmm", "spectral"}

    def __init__(
        self,
        k_homo: int = 10,
        k_hete: int = 10,
        device: str = "cpu",
        homophily_threshold: float = 0.5,
        hetero_clusterer: str = "gmm",
        random_state: int = 42,
        graph_family: Optional[str] = None,
    ) -> None:
        """
        Args:
            k_homo: P_homo 的个数 K1。
            k_hete: P_hete 的个数 K2。
            device: 提示原型最终所在设备（``"cpu"`` / ``"cuda"``）。
            homophily_threshold: 边同配率高于该值视为同配图。
            hetero_clusterer: **仅作用于异配图族**的聚类器，``"gmm"`` 或 ``"spectral"``；
                对同配图族，异配簇固定走基础 K-Means（idea 阶段一约束）。
            random_state: GMM / Spectral / K-Means 的随机种子。
            graph_family: 可选，外部强制指定图族（``"homophilic"`` / ``"heterophilic"``），
                用于由 ``load_data.DATASET_PROFILES`` 驱动聚类策略，
                避免极端 few-shot 下同配率估计噪声过大。
        """
        self.k_homo = int(k_homo)
        self.k_hete = int(k_hete)
        self.device = torch.device(device)
        self.homophily_cfg = HomophilyConfig(threshold=float(homophily_threshold))

        hetero_clusterer = hetero_clusterer.lower()
        if hetero_clusterer not in self.HETERO_CLUSTERERS:
            raise ValueError(
                f"hetero_clusterer 仅支持 {sorted(self.HETERO_CLUSTERERS)}，得到: {hetero_clusterer}"
            )
        self.hetero_clusterer = hetero_clusterer
        self.random_state = int(random_state)

        if graph_family is not None and graph_family not in {"homophilic", "heterophilic"}:
            raise ValueError("graph_family 必须是 'homophilic' / 'heterophilic' / None")
        self.graph_family = graph_family

    # ------------------------------------------------------------------
    #  工具
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @staticmethod
    def _safe_l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=eps)

    def _build_residual(self, x_aligned: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """计算高频残差 H_high = X - H_low（维度不匹配时按 min_dim 对齐）。

        Note:
            严格的高通图滤波 L · X 已预留于 ``load_data.normalize_high_freq_adj``，
            此处采用工程友好的"输入 − 低频"近似，足以让 H_aug 同时包含原始身份与邻域差。
        """
        if x_aligned.size(-1) == h.size(-1):
            return x_aligned - h
        min_dim = min(x_aligned.size(-1), h.size(-1))
        return x_aligned[:, :min_dim] - h[:, :min_dim]

    # ------------------------------------------------------------------
    #  球形 K-Means（同配提示固定使用）
    # ------------------------------------------------------------------

    def _spherical_kmeans(
        self,
        x: torch.Tensor,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """对 L2 归一化向量做 cosine-K-Means，返回 [K, D] 球面质心。

        与普通 K-Means 相比，球形 K-Means 在 cosine 几何下衡量相似度，
        对低频特征的整体尺度更鲁棒（GCN 输出常有量级漂移）。
        """
        if x.size(0) < n_clusters:
            raise ValueError(
                f"球形 K-Means 失败：样本数 {x.size(0)} < 聚类数 {n_clusters}"
            )

        x_norm = self._safe_l2_normalize(x)
        generator = torch.Generator(device=x_norm.device).manual_seed(self.random_state)
        indices = torch.randperm(x_norm.size(0), generator=generator, device=x_norm.device)[
            :n_clusters
        ]
        centroids = x_norm[indices].clone()

        for _ in range(max_iter):
            similarity = x_norm @ centroids.t()  # [N, K]
            labels = similarity.argmax(dim=1)

            new_centroids = []
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    cluster_vec = x_norm[mask].mean(dim=0)
                    new_centroids.append(
                        self._safe_l2_normalize(cluster_vec.unsqueeze(0)).squeeze(0)
                    )
                else:
                    # 空簇时随机重置一个球面点，避免 collapse
                    fallback_idx = torch.randint(
                        0, x_norm.size(0), (1,), generator=generator, device=x_norm.device
                    ).item()
                    new_centroids.append(x_norm[fallback_idx])
            new_centroids = torch.stack(new_centroids, dim=0)

            delta = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()
            centroids = new_centroids
            if delta < tol:
                break

        return centroids

    # ------------------------------------------------------------------
    #  均衡球形 K-Means（目标域 SSL 嵌入 → P_homo）
    # ------------------------------------------------------------------

    def _balanced_spherical_kmeans(
        self,
        x: torch.Tensor,
        n_clusters: int,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> torch.Tensor:
        """球形 K-Means + 每簇近似均衡分配（greedy capacity）。"""
        if x.size(0) < n_clusters:
            raise ValueError(
                f"均衡球形 K-Means 失败：样本数 {x.size(0)} < 聚类数 {n_clusters}"
            )

        n_samples = x.size(0)
        base = n_samples // n_clusters
        rem = n_samples % n_clusters
        capacities = [base + (1 if i < rem else 0) for i in range(n_clusters)]

        x_norm = self._safe_l2_normalize(x)
        generator = torch.Generator(device=x_norm.device).manual_seed(self.random_state)
        indices = torch.randperm(x_norm.size(0), generator=generator, device=x_norm.device)[
            :n_clusters
        ]
        centroids = x_norm[indices].clone()

        for _ in range(max_iter):
            similarity = x_norm @ centroids.t()
            flat_scores = similarity.reshape(-1)
            order = torch.argsort(flat_scores, descending=True)
            assigned = torch.full((n_samples,), -1, dtype=torch.long, device=x.device)
            remaining = capacities.copy()

            for flat_idx in order.tolist():
                n = flat_idx // n_clusters
                c = flat_idx % n_clusters
                if assigned[n] >= 0:
                    continue
                if remaining[c] <= 0:
                    continue
                assigned[n] = c
                remaining[c] -= 1

            unassigned = (assigned < 0).nonzero(as_tuple=False).view(-1)
            for n in unassigned.tolist():
                for c in range(n_clusters):
                    if remaining[c] > 0:
                        assigned[n] = c
                        remaining[c] -= 1
                        break

            new_centroids = []
            for k in range(n_clusters):
                mask = assigned == k
                if mask.any():
                    cluster_vec = x_norm[mask].mean(dim=0)
                    new_centroids.append(
                        self._safe_l2_normalize(cluster_vec.unsqueeze(0)).squeeze(0)
                    )
                else:
                    fallback_idx = torch.randint(
                        0, x_norm.size(0), (1,), generator=generator, device=x_norm.device
                    ).item()
                    new_centroids.append(x_norm[fallback_idx])
            new_centroids = torch.stack(new_centroids, dim=0)

            delta = torch.norm(new_centroids - centroids, p=2, dim=1).max().item()
            centroids = new_centroids
            if delta < tol:
                break

        return centroids

    # ------------------------------------------------------------------
    #  基础 K-Means（仅用于同配图族的异配簇）
    # ------------------------------------------------------------------

    def _basic_kmeans(self, h_aug: torch.Tensor) -> torch.Tensor:
        """同配图族异配簇专用：基础欧氏 K-Means。

        同配图上的"异配残差"分布相对凸、噪声有限，普通 K-Means 已足够提取
        H_aug 上的少量异常模式，无需 GMM 的高计算成本。
        """
        x_np = self._to_numpy(h_aug)
        n_samples = x_np.shape[0]
        if n_samples < self.k_hete:
            raise ValueError(
                f"基础 K-Means 失败：样本数 {n_samples} < k_hete={self.k_hete}"
            )

        kmeans = KMeans(
            n_clusters=self.k_hete,
            n_init=10,
            random_state=self.random_state,
        )
        kmeans.fit(x_np)
        return torch.from_numpy(kmeans.cluster_centers_).float()

    # ------------------------------------------------------------------
    #  GMM / 谱聚类（仅用于异配图族的异配簇，idea 阶段一硬约束）
    # ------------------------------------------------------------------

    def _nonconvex_clustering(self, h_aug: torch.Tensor) -> torch.Tensor:
        """异配图族异配簇专用：GMM 或谱聚类，**禁止使用普通 K-Means**。

        - GMM：以高斯混合显式建模 H_aug 的多峰分布，质心为各分量均值；
        - 谱聚类：在 kNN 图上做拉普拉斯嵌入再 K-Means，捕获非凸流形结构，
                  随后用每簇成员均值作为"聚类原型"（避免直接落到嵌入空间）。
        """
        x_np = self._to_numpy(h_aug)
        n_samples = x_np.shape[0]
        if n_samples < self.k_hete:
            raise ValueError(
                f"非凸聚类失败：样本数 {n_samples} < k_hete={self.k_hete}"
            )

        if self.hetero_clusterer == "gmm":
            gmm = GaussianMixture(
                n_components=self.k_hete,
                covariance_type="full",
                random_state=self.random_state,
                reg_covar=1e-6,
            )
            gmm.fit(x_np)
            return torch.from_numpy(gmm.means_).float()

        # spectral
        # nearest_neighbors affinity 在大图上比 RBF 更稀疏 / 稳健
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
                # 空簇时随机选一个样本兜底
                fallback = np.random.default_rng(self.random_state).integers(0, n_samples)
                centers.append(torch.from_numpy(x_np[fallback]))
        return torch.stack([c.float() for c in centers], dim=0)

    # ------------------------------------------------------------------
    #  维度统一（P_homo 与 P_hete 拼接前对齐）
    # ------------------------------------------------------------------

    @staticmethod
    def _align_prompt_dim(prompt: torch.Tensor, target_dim: int) -> torch.Tensor:
        if prompt.size(-1) == target_dim:
            return prompt
        if prompt.size(-1) > target_dim:
            return prompt[:, :target_dim]
        pad = torch.zeros(
            prompt.size(0),
            target_dim - prompt.size(-1),
            device=prompt.device,
            dtype=prompt.dtype,
        )
        return torch.cat([prompt, pad], dim=-1)

    # ------------------------------------------------------------------
    #  主入口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        data,
        model: nn.Module,
        aligner: nn.Module,
        return_signals: bool = False,
    ) -> Union[
        Tuple[nn.Parameter, nn.Parameter],
        Tuple[nn.Parameter, nn.Parameter, PromptSignals],
    ]:
        """生成 P_homo / P_hete 提示原型。

        Args:
            data:    PyG `Data`，需含 ``x`` / ``edge_index`` / ``y``。
            model:   冻结源域 GNN（``eval()`` 模式），用于提取低频信号。
            aligner: 输入维度对齐器（如正交投影），将目标域 x 投影到源域维度。
            return_signals: 是否同时返回 ``PromptSignals``（供下游对比损失 / 分析使用）。

        Returns:
            (p_homo, p_hete) 或 (p_homo, p_hete, signals)
        """
        model.eval()
        aligner.eval()

        # 1) 估计图同配率（few-shot 跨域下噪声可能较大；优先使用 graph_family 覆盖）
        homophily = estimate_edge_homophily(data.edge_index, data.y)
        if self.graph_family is not None:
            is_homophilic_graph = self.graph_family == "homophilic"
        else:
            is_homophilic_graph = homophily >= self.homophily_cfg.threshold

        # 2) 提取双频信号
        x_aligned = aligner(data.x)                       # [N, source_dim]
        h_low = model(x_aligned, data.edge_index)         # [N, hidden_dim]  低频
        h_high = self._build_residual(x_aligned, h_low)   # [N, hidden_dim]  高频残差
        h_aug = torch.cat([h_low, h_high], dim=-1)        # [N, 2 * hidden_dim]

        print(
            f"[PromptGenerator] edge homophily = {homophily:.4f} | "
            f"graph_family = {'homophilic' if is_homophilic_graph else 'heterophilic'} "
            f"{'(forced)' if self.graph_family is not None else '(auto)'}"
        )

        # 3) 同配提示：球形 K-Means（任何数据集都用）
        print(f"[PromptGenerator] spherical K-Means → P_homo (K1={self.k_homo})")
        p_homo = self._spherical_kmeans(h_low, self.k_homo)  # [K1, hidden_dim]

        # 4) 异配提示：按图族分支（idea 阶段一硬约束）
        if is_homophilic_graph:
            print(f"[PromptGenerator] basic K-Means → P_hete (K2={self.k_hete})  [homophilic graph]")
            p_hete = self._basic_kmeans(h_aug)                # [K2, 2 * hidden_dim]
            hetero_clusterer_used = "kmeans"
        else:
            print(
                f"[PromptGenerator] {self.hetero_clusterer.upper()} → P_hete "
                f"(K2={self.k_hete})  [heterophilic graph, K-Means forbidden]"
            )
            p_hete = self._nonconvex_clustering(h_aug)        # [K2, 2 * hidden_dim]
            hetero_clusterer_used = self.hetero_clusterer

        # 5) 统一维度（拼接到 [K1+K2, D]）
        target_dim = p_homo.size(-1)
        p_hete = self._align_prompt_dim(p_hete, target_dim)

        # 6) 包装为可训练参数
        p_homo = nn.Parameter(p_homo.to(self.device), requires_grad=True)
        p_hete = nn.Parameter(p_hete.to(self.device), requires_grad=True)

        if not return_signals:
            return p_homo, p_hete

        signals = PromptSignals(
            h_low=h_low,
            h_high=h_high,
            h_aug=h_aug,
            edge_homophily=float(homophily),
            is_homophilic=bool(is_homophilic_graph),
            hetero_clusterer_used=hetero_clusterer_used,
            z_target=None,
            pool_size=0,
        )
        return p_homo, p_hete, signals


# =============================================================================
#  TargetSSLPromptGenerator：目标域无监督嵌入 + 均衡/池内聚类
# =============================================================================


class TargetSSLPromptGenerator:
    """在目标图上跑 GRACE/GAE/BGRL 得到 Z_target，再初始化 P_homo / P_hete。

    流程（与用户选定方案一致）：
        1. 无监督 SSL → Z_target（仅用 x / edge_index，不用标签）
        2. P_homo = balanced spherical KMeans(Z_target)  全图
        3. V_pool = Top-ρ 异配候选（基于特征异配度，无标签）
        4. P_hete = spherical KMeans(Z_target[V_pool])
    """

    def __init__(
        self,
        k_homo: int = 10,
        k_hete: int = 10,
        device: str = "cpu",
        ssl_method: SSLMethod = "grace",
        ssl_epochs: int = 200,
        ssl_lr: float = 0.01,
        ssl_hidden_dim: Optional[int] = None,
        ssl_out_dim: Optional[int] = None,
        rho: float = 0.15,
        feat_drop: float = 0.2,
        edge_drop: float = 0.2,
        random_state: int = 42,
        graph_family: Optional[str] = None,
        homophily_threshold: float = 0.5,
    ) -> None:
        self.k_homo = int(k_homo)
        self.k_hete = int(k_hete)
        self.device = torch.device(device)
        self.ssl_method = ssl_method
        self.ssl_epochs = int(ssl_epochs)
        self.ssl_lr = float(ssl_lr)
        self.ssl_hidden_dim = ssl_hidden_dim
        self.ssl_out_dim = ssl_out_dim
        self.rho = float(rho)
        self.feat_drop = float(feat_drop)
        self.edge_drop = float(edge_drop)
        self.random_state = int(random_state)
        self.graph_family = graph_family
        self.homophily_threshold = float(homophily_threshold)
        self._legacy = PromptGenerator(
            k_homo=k_homo,
            k_hete=k_hete,
            device=device,
            random_state=random_state,
            graph_family=graph_family,
            homophily_threshold=homophily_threshold,
        )

    def generate(
        self,
        data,
        model: Optional[nn.Module] = None,
        aligner: Optional[nn.Module] = None,
        return_signals: bool = False,
    ):
        del model, aligner

        homophily = estimate_edge_homophily(data.edge_index, data.y)
        if self.graph_family is not None:
            is_homophilic_graph = self.graph_family == "homophilic"
        else:
            is_homophilic_graph = homophily >= self.homophily_threshold

        hidden = self.ssl_hidden_dim if self.ssl_hidden_dim is not None else 128
        out_dim = self.ssl_out_dim if self.ssl_out_dim is not None else hidden

        z_target, ssl_used = fit_target_ssl_embeddings(
            data,
            method=self.ssl_method,
            hidden_dim=hidden,
            out_dim=out_dim,
            epochs=self.ssl_epochs,
            lr=self.ssl_lr,
            feat_drop=self.feat_drop,
            edge_drop=self.edge_drop,
            device=self.device,
            verbose=True,
        )

        print(
            f"[TargetSSLPromptGenerator] edge homophily = {homophily:.4f} | "
            f"graph_family = {'homophilic' if is_homophilic_graph else 'heterophilic'}"
        )
        print(
            f"[TargetSSLPromptGenerator] balanced spherical K-Means on Z_target "
            f"→ P_homo (K1={self.k_homo})"
        )
        with torch.no_grad():
            p_homo = self._legacy._balanced_spherical_kmeans(z_target.detach(), self.k_homo)

        local_h = compute_local_heterophily(
            data.edge_index,
            data.num_nodes,
            y=None,
            x=data.x.to(z_target.device),
        )
        pool_mask, pool_indices = select_top_rho_pool(local_h, self.rho)
        pool_size = int(pool_indices.numel())
        print(
            f"[TargetSSLPromptGenerator] V_pool = {pool_size} nodes (rho={self.rho:.3f}, "
            f"feature heterophily, no labels)"
        )

        z_pool = z_target[pool_mask]
        k_hete_eff = min(self.k_hete, max(1, z_pool.size(0)))
        if k_hete_eff < self.k_hete:
            print(
                f"[TargetSSLPromptGenerator] k_hete adjusted {self.k_hete} → {k_hete_eff} "
                f"(pool size)"
            )
        print(
            f"[TargetSSLPromptGenerator] spherical K-Means on Z_target[V_pool] "
            f"→ P_hete (K2={k_hete_eff})"
        )
        with torch.no_grad():
            p_hete = self._legacy._spherical_kmeans(z_pool.detach(), k_hete_eff)

        target_dim = p_homo.size(-1)
        p_hete = PromptGenerator._align_prompt_dim(p_hete, target_dim)
        p_homo = nn.Parameter(p_homo.to(self.device), requires_grad=True)
        p_hete = nn.Parameter(p_hete.to(self.device), requires_grad=True)

        if not return_signals:
            return p_homo, p_hete

        h_low = z_target
        h_high = z_target.new_zeros(z_target.size(0), 0)
        h_aug = z_target
        signals = PromptSignals(
            h_low=h_low,
            h_high=h_high,
            h_aug=h_aug,
            edge_homophily=float(homophily),
            is_homophilic=bool(is_homophilic_graph),
            hetero_clusterer_used=f"target_ssl_{ssl_used}",
            z_target=z_target,
            pool_size=pool_size,
        )
        return p_homo, p_hete, signals
