# Graph Prompt Learning for Cross-Domain Heterophilic Node Classification

> **维护约定**：本 README 是项目 **idea、架构、实验协议与实现路线图** 的**唯一权威文档**。讨论中达成的设计共识须先写进本文，再改代码。  
> **配套文档**：  
> - [`docs/fewshot_experiment_progress.md`](docs/fewshot_experiment_progress.md) — 实验数值、问题清单、**实现 Todo / 进度**  
> - [`docs/ablation_matrix.md`](docs/ablation_matrix.md) — 消融矩阵与可复现 CLI  

---

## 〇、核心概念与设计共识【修订 2026-05-20】

### 0.1 提示节点是什么？【与你的原始 idea 对齐】

每个 **同配提示** \(P_{homo}^k\) 与 **异配提示** \(P_{hete}^\ell\) 是图上的 **静态虚拟节点（semantic hub / 槽位）**：

- **init（一次性）**：冻结 GNN 低频/高频信号 → K-Means / GMM 聚类 → 得到 \(K_1+K_2\) 个 **固定特征向量**；
- **train（你的设计）**：**不更新** \(P_{homo}, P_{hete}\) 本身；训练的是 **Gumbel 路由 \(EX\)**、适应分支 adapter、门控与分类头；
- **语义**：\(P_{homo}\) = 低频「同盟」枢纽；\(P_{hete}\) = 高频/残差「跨界/补偿」枢纽（见 idea 原文）；
- **不是** ground-truth 类别中心；**可靠** = 路由清晰 + 拓扑/特征指标达标 + 分类提升，**不是** init 与人类命名 1:1。

> **2026-05-20 文档笔误**：曾写成「可学习语义原型」——那是把 **代码默认**（`--no_freeze_prompts`）误当成了 **设计定义**。你的原始 idea 与 GRAPHITE 式叙事一致：**提示节点 = init 后静态，可导部分主要是连边（路由）**。

**实现偏差**：`train.py` 默认 `freeze_prompts=False`，`p_homo/p_hete` 为 `nn.Parameter` 会随对比损失/CE 漂移。对齐原始 idea 应使用 **`--freeze_prompts`**（主实验与论文表待统一）；可学习 prompt 仅作 **消融**。

### 0.2 连接规则（设计目标）

| 规则 | 说明 |
|------|------|
| 候选池 | 仅 \(V_{pool}\) 内原图节点 **允许** 连提示边（Top-ρ 异配/难节点） |
| **互斥单连** | 每个池内节点 **最多 1 条** prompt 边，**要么** 连某个 \(P_{homo}\)，**要么** 连某个 \(P_{hete}\) |
| 可选扩展 | 增加第 \(K_1+K_2+1\) 类 **「不连 prompt」**，表示路由置信度低时留在原图 |
| 池外节点 | **不连** 任何 prompt（结构硬约束） |

### 0.3 设计目标 vs 当前实现（必读）

| 项目 | 设计目标 | 当前代码（`gumbel_route.py`） | 状态 |
|------|----------|-------------------------------|------|
| 每节点 prompt 连接数 | ≤ 1，homo **XOR** hete | 池内节点默认 **1 homo + 1 hete**（`topk=1` 各一路） | ⚠️ **待改** |
| 互斥竞争 | \(K_1+K_2\) 联合分配 | `EX_homo` 与 `EX_hete` **独立** Gumbel | ⚠️ **待改** |
| Amazon 图族 | **heterophilic**，ρ ≥ 0.5 | profile 仍为 homophilic，ρ=0.2 | ⚠️ **待改** |
| squirrel ρ | ≥ 0.5 | ρ=0.4 | ⚠️ **待改** |
| 异配图 \(P_{homo}\) 语义 | 嵌入 **同盟槽位**，不声称 = 纯净评分类 | 全图均衡 K-Means on \(Z_{target}\) | 📝 叙事已澄清，init 待增强 |
| 异配图 \(P_{hete}\) 语义 | **边界/补偿** 槽位 | 池内 K-Means on \(Z_{target}\) | 📝 叙事已澄清，init 待增强 |
| 提示向量是否更新 | **init 后冻结（静态）** | 默认 `--no_freeze_prompts` 可反传 | ⚠️ **与原始 idea 不符，应用 `--freeze_prompts` 作主设定** |

> **审计脚本**：`test/test_prompt_topology_audit.py`（init 阶段拓扑 + 特征同配分解，**不依赖** 分类训练）。

---

## 一、研究动机 (Motivation)

在图表示学习的 **跨域迁移** 中，GNN 与图提示方法常依赖 **同配假设**：相连节点更可能同类。真实图大量 **异配**（尤其共购、超链接网络），原边消息传递易 **过平滑** 或 **跨类噪声扩散**。

| 工作 | 启发 |
|------|------|
| **GRAPHITE (ICLR 2026)** | 特征提示节点 **重构拓扑**，提升可消息传递下的有效同配性 |
| **GP2F (ICLR 2026)** | **双分支**：冻结分支保留源域先验；适应分支学习目标域 |

**本工作核心**：跨域 Few-shot 下，用 **静态提示节点（init 聚类）+ 可导 Gumbel 路由 + 异配感知 MP + 双分支融合**，使高异配目标图也能稳定分类。

**不声称**：无监督 init 即在 Amazon 上恢复「纯净评分类语义簇」；该点见 [§3.4](#34-异配图上的提示语义与-cora-区分修订-2026-05-20)。

---

## 二、整体流水线（五阶段）

```text
目标图 G = (V, E, X, Y)     【Y 仅用于 CE / 评估；init 默认不用 Y】
  │
  ├─[阶段一] 提示初始化（静态）→ P_homo ∈ R^{K1×d}, P_hete ∈ R^{K2×d}
  │            输出：固定虚拟节点特征 +（可选）SSL/双频中间信号
  │            预期：粗粒度「同盟槽位 / 补偿槽位」，非最终语义
  │
  ├─[阶段二] Top-ρ 候选池 V_pool + 互斥 Gumbel 路由 → 增强图 G'
  │            预期：池内节点唯一归属某一 prompt；homo/hete 分工可观测
  │            【现码】双路独立 EX，每节点可能 homo+hete 各连一条 → 待改
  │
  ├─[阶段三] 适应分支：Prompt-Aware MP on (N + K1 + K2) 节点
  │            预期：高异配节点弱化有害原边、强化所选 prompt 边
  │
  ├─[阶段四] 冻结分支 H_frozen ∥ H_adapted[:N] → 门控 H_final
  │            预期：源域先验 + 目标域结构适配互补
  │
  └─[阶段五] L = L_CE + λ1·L_sparse + λ2·L_consist + λ3^h·L_c^h + λ3^e·L_c^e
               预期：CE 监督类边界；对比损失塑造原型；一致性稳定双分支
```

---

## 三、阶段一：提示原型初始化

**模块**：`prompts/cluster_generator.py`、`prompts/target_ssl.py`

**本阶段做什么**：在目标图上得到 \(P_{homo}\)、\(P_{hete}\) 的 **初始中心**（聚类质心），作为后续 MP 的 **静态** 虚拟节点特征。

**本阶段不做什么**：不决定最终连边（阶段二）；不用 few-shot 训练标签做 init（避免泄漏）。

### 3.1 路径 A：`dual_freq`（消融 / 异配 hete 备选）

| 步骤 | 输入 | 操作 | 输出 |
|------|------|------|------|
| 1 | \(X\)，冻结源域 GNN | 对齐 → 低频 \(H_{low}\) | 平滑结构信号 |
| 2 | \(X - H_{low}\) | 高频残差 \(H_{high}\) | 异配/个体偏移 |
| 3 | \(H_{low}\) | 球形 K-Means → **\(P_{homo}\)** | \(K_1\) 个同盟原型 |
| 4 | \([H_{low}; H_{high}]\) | 同配图：K-Means；**异配图：GMM/谱聚类** → **\(P_{hete}\)** | \(K_2\) 个补偿原型 |

**预期效果**：

- **Cora 等同配图**：\(P_{hete}\) 捕获跨主题/残差模式，数量可少（\(K_2=1\) profile）。
- **Amazon / squirrel**：残差大 → \(P_{hete}\) 更贴 **「与原图低频平滑方向不一致」** 的模式，**比** 仅在 \(Z_{pool}\) 上 K-Means **更适合** 作 hete init 对照（待消融）。

### 3.2 路径 B：`target_ssl`（跨域 Few-shot **主推**）

| 步骤 | 输入 | 操作 | 输出 |
|------|------|------|------|
| 1 | \(X, E\) | GRACE / GAE / **BGRL**（大图） | \(Z_{target} \in \mathbb{R}^{N \times d}\) |
| 2 | \(Z_{target}\) | 均衡球形 K-Means（\(K_1\)） | **\(P_{homo}\)** |
| 3 | 局部异配度 | Top-ρ → **\(V_{pool}\)**（默认 **特征异配**，无标签） | 候选连接节点 |
| 4 | \(Z_{target}[V_{pool}]\) | 球形 K-Means（\(K_2\)） | **\(P_{hete}\)** |

**预期效果**：

- \(Z_{target}\) 编码共购/共引 **共现** 与文本相似；
- \(P_{homo}\)：全图嵌入的 **\(K_1\) 个软分区中心**（容量常取类数，**不承诺** = 标签类）；
- \(P_{hete}\)：难区域子空间上的 **补偿中心**。

**已知局限（实验 + 审计）**：

- Amazon 上 \(P_{homo}\) **不是** 5 个纯净评分档簇；跨档共购使 **「共现相似 ≠ 评分同类」**。
- 仅特征异配 Top-ρ **不能** 很好覆盖 **label-异配边端点**（审计 Het-recall ~16% @ ρ=0.15）。
- init 后 **不保证** 高特征同配 lift（审计 ΔFH_all 常为负）。

**训练期（设计）**：**冻结** \(P_{homo}, P_{hete}\)，只优化路由 \(EX\)、adapter、门控、分类头（`--freeze_prompts`）。  
**训练期（现码默认）**：`--no_freeze_prompts` 允许向量漂移——与原始 idea **不一致**，论文主表应改回冻结并做消融对比。

### 3.3 原型「可靠性」如何度量（init 阶段）

| 指标 | 含义 | 用途 |
|------|------|------|
| 簇内平均余弦 | \(Z\) 或残差空间紧密度 | init 质量 |
| Silhouette | 聚类可分性 | 对比 homo/hete 路径 |
| **Oracle** 簇-标签 NMI | 仅分析，不用 train | 说明与评分档错位程度 |
| 拓扑审计 | bridge rate、P-H、ΔFH | 见 [§8.5](#85-拓扑与提示审计指标修订-2026-05-20) |

### 3.4 异配图上的提示语义（与 Cora 区分）【修订 2026-05-20】

| | **Cora（整体同配）** | **Amazon-ratings / squirrel（高异配）** |
|--|---------------------|----------------------------------------|
| 图结构 | Edge H 高（~0.81） | Amazon Edge H ~**0.38**；squirrel 更低 |
| \(V_{pool}\) 规模 ρ | **小**（~0.15） | **大**（设计目标 **≥ 0.5**） |
| \(P_{homo}\) | 主题/类 **同盟枢纽**（近似合理） | **嵌入同盟槽位**；不声称 = 评分类 |
| \(P_{hete}\) | 少量 **跨界隔离** | **边界/跨模式补偿**；非可解释「跨档桥梁」 |
| 成功标准 | P-H↑、精度↑ | **homo/hete 分工** + bridge on \(E_{het}\)↑ + 精度↑ |

**Amazon 数据事实（Platonov et al., ICLR 2023）**：共购边连接不同评分档很常见 → 不能把「邻居 = 同类」当作默认。  

**squirrel**：维基超链接、类别为流量档；**强异配** + 高密度；宜用 **大池 + 互斥路由**；注意 duplicate 节点问题（应用 filtered 版做严肃结论）。

---

## 四、阶段二：可导拓扑重构（路由）

**模块**：`prompts/gumbel_route.py`

### 4.1 候选池 \(V_{pool}\)

| 步骤 | 做什么 | 预期效果 |
|------|--------|----------|
| 1 | 算每个节点局部异配度 \(h_i\) | 识别「邻居结构/特征与己不一致」的难节点 |
| 2 | Top-ρ 取 \(h_i\) 最大子集 | 仅难节点可连 prompt，控制规模 |
| 3 | 池选信号 | **设计（异配图）**：优先 **标签异配度**（oracle 或结构代理）；**现码 target_ssl 默认**：特征异配、无标签 |

**ρ 设计目标（`DATASET_PROFILES`，待代码同步）**：

| 图族 | 数据集示例 | 目标 ρ |
|------|------------|--------|
| 同配/隐含异配 | Cora, CiteSeer, PubMed | 0.15 |
| **高异配** | **Amazon-ratings**, **squirrel**, Actor | **≥ 0.50** |
| 极端异配 | Minesweeper | 0.35–0.40（网格结构另议） |

### 4.2 互斥路由（设计目标）【修订 2026-05-20】

**做什么**：对每个 \(u \in V_{pool}\)，在 \(\{P_{homo}^k\}_{k=1}^{K_1} \cup \{P_{hete}^\ell\}_{\ell=1}^{K_2} \cup \{\varnothing\}\}\) 上 **一次** Gumbel-Softmax，得到 **唯一** 连接或空。

**预期效果**：

| 效果 | 说明 |
|------|------|
| 角色分工 | 节点要么走 **同盟**（homo），要么走 **补偿/跨界**（hete） |
| 可解释 | MP 中每个节点只受 **一种** prompt 语义影响 |
| 指标 | P-H(homo) 与 P-H(hete) **可分化**；Shared-hete bridge on label-异配边 **可上升** |
| 「愿意连」 | 路由熵低、max prob 高；或主动选 \(\varnothing\) |

**推荐实现方案（路线图，未落地）**：

- **方案 A（推荐）**：联合 logits 维度 \(K_1+K_2(+1)\)，单次 Gumbel；
- **方案 B**：先 homo/hete 门控，再在选中分支内选槽位。

**当前实现（偏差）**：

- 独立 `EX_homo`、`EX_hete`；`build_prompt_edge_index` 对池内节点 **各建** homo 与 hete 边；
- 导致审计中 homo ≈ hete、双 prompt 度 = 2。

### 4.3 对比损失与稀疏

| 项 | 做什么 | 预期效果 |
|----|--------|----------|
| \(L_{sparse}\) | 鼓励 EX 稀疏 | 接近 one-hot 路由 |
| \(L_{contrast}^h, L_{contrast}^e\) | 同 prompt 槽位上的节点嵌入拉近 | 塑造 **路由/池内表示** 与槽位对齐；静态 prompt 时不改 \(P\) 本身 |

---

## 五、阶段三：异配感知消息传递

**模块**：`models/prompt_conv.py`、`models/dual_branch.py`

| 步骤 | 做什么 | 预期效果 |
|------|--------|----------|
| 构图 | 原边 `edge_type=0` + prompt 边 `edge_type=1` | 增强图 \(N+K_1+K_2\) |
| \(\alpha, \beta\) | 对高 \(h_i\) 节点：削弱原边、加强 prompt 边 | 减轻 **有害异配邻居** 影响 |
| PromptBranchEncoder | 在增强图上卷积 | 从所选原型吸收 **同盟或补偿** 信号 |

**预期（互斥路由后）**：连 \(P_{homo}\) 的节点主要获得 **类内汇聚**；连 \(P_{hete}\) 的获得 **跨模式隔离/绕行**。

---

## 六、阶段四：双分支门控融合

| 分支 | 做什么 | 预期效果 |
|------|--------|----------|
| 冻结分支 | 源域预训练 GCN on 原图 | 通用特征先验 |
| 适应分支 | 阶段三 on 增强图 | 目标域结构适配 |
| 门控 \(g\) | \(H_{final}=(1-g)H_{frozen}+g H_{adapted}\) | 渐进信任适应分支（cosine \(g: 0.2\to0.65\)） |

---

## 七、阶段五：多任务损失

\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{sparse} + \lambda_2 \mathcal{L}_{consist} + \lambda_3^{homo}\mathcal{L}_{contrast}^{homo} + \lambda_3^{hete}\mathcal{L}_{contrast}^{hete}
\]

| 项 | 仅作用于 | 预期效果 |
|----|----------|----------|
| \(\mathcal{L}_{CE}\) | `train_mask`（few-shot） | 类边界监督 |
| \(\mathcal{L}_{consist}\) | 全图 logits（现实现） | 双分支预测一致；\(\lambda_2\) 按图族缩放 |
| \(\mathcal{L}_{contrast}\) | 池内 + 分路 | 连到同一静态槽位的节点表示簇内紧、簇间分 |

**动态 \(\lambda_2\)**：profile `homophilic` ×0.25，`heterophilic` ×2.0（Amazon 应按 **heterophilic** 档，待 profile 修正）。

**训练稳定性**：`loss_warmup_epochs=50` 线性升高辅助损失；`consist_max=1.0` 裁剪。

---

## 八、数据集与评估

### 8.1 数据集分族（修订 ρ / family 目标）

| 数据集 | 节点量级 | Edge H（参考） | 设计 family | 目标 ρ | \(K_1/K_2\)（profile） |
|--------|----------|----------------|-------------|--------|------------------------|
| Cora | 2.7K | ~0.81 | homophilic | 0.15 | 7 / 1 |
| CiteSeer, PubMed | — | 高 | homophilic | 0.15 | 类数 / 1 |
| **Amazon-ratings** | 24.5K | **~0.38** | **heterophilic** | **≥0.50** | 5 / 2 |
| Flickr | 大 | 中 | homophilic | 0.20 | 7 / 2 |
| squirrel | 5.2K | 低 | heterophilic | **≥0.50** | 5 / 5 |
| Actor, Minesweeper | — | 低/特殊 | heterophilic | 0.35–0.40 | 见 `load_data.py` |

### 8.2 各数据集上 prompt 应连哪些节点（设计意图）

**共通**：只有 \(V_{pool}\) 内节点可有 prompt 边；**互斥** 连一个 homo **或** 一个 hete。

**Cora**

- **\(P_{hete}\)**：label-异配边端点、跨学科/综述类、局部异配高节点 → **跨界隔离**。
- **\(P_{homo}\)**：同领域但未充分互引、需 **语义填补** 的节点。

**Amazon-ratings**

- **\(P_{hete}\)**：跨评分档共购、邻域标签混杂、**hub 商品** → **补偿/隔离**（非「纯净跨档簇」）。
- **\(P_{homo}\)**：嵌入上接近某 **同盟槽位**、需强化 **同档信号** 的节点（**不假设** init 簇 = 评分类）。
- **ρ ≥ 50%**：异配是 **主流病态**，小池无法覆盖多数跨档边。

**squirrel**

- **\(P_{hete}\)**：跨流量档超链接、高度数 hub、异配邻居占主导 → **超级枢纽/通道**。
- **\(P_{homo}\)**：同档长尾页、需 **类内聚拢** 的节点。
- **ρ ≥ 50%**；优先 **filtered** 数据版本。

### 8.3 分类主指标

| 指标 | 用途 |
|------|------|
| **ES Test Acc** | 论文主表（val best checkpoint） |
| macro-F1 / balanced acc | Minesweeper 等等价类基线对比 |

### 8.4 提示与拓扑分析指标

| 指标 | 含义 |
|------|------|
| Edge H | 原图标签同配率 |
| P-H / P-E | 节点标签 vs 所连 prompt 多数伪标签一致率 |
| Het-endpoint recall | label-异配边端点落在 \(V_{pool}\) 比例 |
| Shared-\(P_{hete}\) bridge | label-异配边两端连 **同一** hete 原型比例 |
| ΔFH_all | 加 prompt 边后 **特征同配** 变化（投影空间） |
| 路由熵 / 空类占比 | 「愿意连」与分工清晰度 |

### 8.5 拓扑与提示审计指标【修订 2026-05-20】

```bash
python test/test_prompt_topology_audit.py \
  --dataset Cora \
  --prompt_init target_ssl \
  --pretrained_dir ./pretrained_gnns \
  --runs 3 --compare_random 5
```

**用途**：在 **分类训练之前** 判断 init+路由是否实现设计分工；**不能** 用精度单独证明拓扑成功。

**Cora 5-shot / 1-shot 已有结果摘要**（10-run / 3-run，现码双连）：

| 设定 | ES Test ≈ |
|------|-----------|
| 5-shot | **~55%**（> GraphTOP ~51%） |
| 1-shot | **~35%** |

**Amazon 探路**：ES Test ~**23%**；拓扑审计 **未达** ΔFH>0、bridge 极低 → **prompt 未实现设计语义**（在改互斥/大池前 **不夸大** 叙事）。

---

## 九、代码结构

```text
MyIdea/
├── train.py
├── load_data.py                  # DATASET_PROFILES（ρ/family 待与 §8.1 同步）
├── loss.py
├── prompts/
│   ├── cluster_generator.py
│   ├── target_ssl.py
│   ├── gumbel_route.py           # ⚠️ 互斥路由待改
│   └── routing_debug.py
├── models/
├── docs/
│   ├── fewshot_experiment_progress.md   # 实验数值 + 实现 Todo
│   └── ablation_matrix.md
└── test/
    ├── test_prompt_homophily.py
    ├── test_prompt_topology_audit.py    # 拓扑审计【新增】
    ├── test_pure_dual_branch.py
    └── test_frozen_gnn_baseline.py
```

---

## 十、实现进度与路线图【修订 2026-05-20】

### 10.1 已完成（✅）

| ID | 内容 |
|----|------|
| M1 | 五阶段核心模块（数据、SSL/dual_freq init、双 EX 路由、损失、双分支、早停） |
| M2 | `target_ssl` 主推 init；PubMed→Cora 5-shot **~55%** ES Test（10-run） |
| M3 | Minesweeper 10-run 探路；Amazon bgrl 探路 ~23% |
| M4 | `test_prompt_topology_audit.py` + 互斥路由 **设计共识** 写入 README |
| M5 | 文档：`fewshot_experiment_progress.md`、`ablation_matrix.md` |

### 10.2 进行中 / 待办（按优先级）

| 优先级 | ID | 任务 | 预期验收 |
|--------|-----|------|----------|
| **P0** | T1 | **README / idea 对齐**（本节） | 设计 vs 实现表清晰 |
| **P0** | T2 | **互斥单连路由**（\(K_1+K_2(+1)\) 联合 Gumbel） | 每节点 ≤1 prompt 边；homo/hete 指标分化 |
| **P0** | T3 | 互斥前后 **Cora + Amazon audit** 对照 | bridge、P-H、ΔFH 表 |
| **P1** | T4 | `load_data`：Amazon→heterophilic，ρ=0.5；squirrel ρ=0.5 | profile 与 §8.1 一致 |
| **P1** | T5 | 异配图 **池选**：结构/标签异配代理（无泄漏） | Het-recall ↑ |
| **P1** | T6 | Amazon：hete init 消融 `target_ssl` vs `dual_freq` 残差 | ES Test / 拓扑至少一项 ↑ |
| **P2** | T7 | 1-shot / 3-shot 曲线（Cora 已有 1-shot ~35%） | 论文表 |
| **P2** | T8 | Minesweeper macro-F1 | 公平对比 |
| **P2** | T9 | Amazon 10-run（**T2–T6 后再跑**） | 均值/方差可报 |
| **P3** | T10 | `--anneal_tau`；按 Edge H 自适应 λ2 | 稳定性 |

> **详细任务表、负责人字段、完成勾选** 见 [`docs/fewshot_experiment_progress.md` §4](docs/fewshot_experiment_progress.md)。

---

## 十一、实验命令（Few-shot）

**公共约定**：`--shots 5 --val_per_class 30 --epochs 200 --early_stop_patience 40 --early_stop_min_epochs 20 --runs 10 --seed 42 --no_routing_debug`

### 11.1 Cora（主行）

```bash
python train.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --val_per_class 30 --epochs 200 \
  --early_stop_patience 40 --early_stop_min_epochs 20 \
  --checkpoint_dir ./checkpoints \
  --lr 0.005 --prompt_lr_scale 0.25 --weight_decay 5e-4 \
  --lambda1 0.01 --lambda2 0.05 --lambda3 0.1 \
  --loss_warmup_epochs 50 --consist_max 1.0 \
  --gate_start 0.2 --gate_end 0.65 --gate_warmup_epochs 150 \
  --grad_clip 1.0 --runs 10 --seed 42 --no_routing_debug \
  --prompt_init target_ssl --ssl_method grace --ssl_epochs 200 --ssl_lr 0.01 \
  --no_freeze_prompts --k_homo 10 --k_hete 10
```

### 11.2 拓扑审计（init 阶段）

```bash
python test/test_prompt_topology_audit.py \
  --dataset Cora --prompt_init target_ssl \
  --pretrained_dir ./pretrained_gnns --runs 3 --compare_random 5

python test/test_prompt_topology_audit.py \
  --dataset Amazon-ratings --prompt_init target_ssl \
  --ssl_method bgrl --ssl_epochs 100 --runs 3 --compare_random 5
```

### 11.3 Minesweeper / Amazon 训练

见 [`docs/fewshot_experiment_progress.md`](docs/fewshot_experiment_progress.md) 命令备忘；Amazon **建议在 T2–T6 完成后再 10-run**。

### 11.4 消融矩阵

见 [`docs/ablation_matrix.md`](docs/ablation_matrix.md)。

---

## 十二、依赖与环境

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Python 3.8+；`torch`、`torch-geometric`、`numpy`、`scikit-learn`。

---

## 十三、变更日志

| 日期 | 变更摘要 |
|------|----------|
| 2026-05-18 | 五阶段架构、8 数据集、消融脚本 |
| 2026-05-19 | Few-shot 协议；`target_ssl`；Cora/Minesweeper/Amazon 实验结果；`fewshot_experiment_progress.md` |
| **2026-05-20** | **【Idea 大修】** **互斥单连**、异配图 ρ≥0.5、设计 vs 实现偏差表、拓扑审计、路线图；**更正**：提示节点为 **init 后静态**（非「可学习原型」）；可学习仅作代码默认/消融 |

---

*实验数值以 `docs/fewshot_experiment_progress.md` 为准；idea 以本 README 为准。*
