# Graph Prompt Learning for Cross-Domain Heterophilic Node Classification

> **维护约定**：本 README 是项目 idea 与实验协议的**唯一权威文档**。若在后续开发或讨论中修改了动机、架构、损失、数据集或实验流程，**必须同步更新本文件**。
>
> **配套文档**（详细数值与待办）：
> - [`docs/fewshot_experiment_progress.md`](docs/fewshot_experiment_progress.md) — Few-shot 实验进度、问题清单、命令备忘  
> - [`docs/ablation_matrix.md`](docs/ablation_matrix.md) — 消融矩阵与可复现 CLI  

---

## 一、研究动机 (Motivation)

在图表示学习的**跨域迁移**中，现有 GNN 与图提示方法通常强烈依赖**同配假设 (Homophily Assumption)**：相连节点更可能同类。然而真实场景存在大量**异配图 (Heterophilic Graphs)**，强行套用同配假设会导致过平滑与特征坍缩。

本框架受两篇工作启发：

| 工作 | 启发 |
|------|------|
| **GRAPHITE (ICLR 2026)** | 引入「特征提示节点」重构拓扑，提升输入图的结构/特征同配性，使异配图经提示重构后能满足下游 GNN 的同配假设 |
| **GP2F (ICLR 2026)** | **双分支架构**：冻结分支保留源域通用先验；适应分支学习目标域独有分布 |

**核心目标**：在跨域、异配场景下，通过可学习的图提示拓扑 + 异配感知消息传递 + 双分支融合，实现稳健的节点分类。

### 【修订 2026-05-19】论文实验设定：Few-shot 为主协议

当前领域工作（含 GraphTOP / ProNoG 等）主表普遍采用 **1-shot / 3-shot / 5-shot** 节点分类。本仓库**论文主线**对齐该设定：

| 项目 | 约定 |
|------|------|
| 训练标签 | 每类 **K** 个节点（`--shots K`，当前开发以 **K=5** 为主） |
| 验证 | 每类 **30** 个节点（`--val_per_class 30`） |
| 测试 | 其余节点 |
| 重复 | **`--runs 10`**，`seed + i` |
| 主指标 | **ES Test Acc**（val 最优 checkpoint 恢复后 test；**勿用**未早停时的 Final Test @ epoch 200） |
| 源域预训练 | `PubMed` + `pretrained_gnns/PubMed_SimGRACE_GCN_1.pth` |

---

## 二、整体流水线（五阶段）

```text
目标图 G
  │
  ├─[阶段一] 提示初始化 → P_homo (K1), P_hete (K2)
  │            ├─ dual_freq：源域冻结 GNN 双频聚类（PromptGenerator）
  │            └─ target_ssl【论文主推】：目标域 SSL 嵌入 Z_target + 聚类（TargetSSLPromptGenerator）
  │
  ├─[阶段二] Top-ρ 候选池 + Gumbel-Softmax → EX_homo, EX_hete → 增强图 G'
  │
  ├─[阶段三] 适应分支 Prompt-Biased MP（(N+K) 节点）
  │
  ├─[阶段四] 冻结分支 H_frozen ∥ 适应分支 H_adapted[:N] → 门控融合 H_final
  │
  └─[阶段五] 多任务损失 L = L_CE + λ1·L_sparse + λ2·L_consist + λ3·L_contrastive
```

---

## 三、阶段一：提示初始化

**模块**：`prompts/cluster_generator.py`（`PromptGenerator` / `TargetSSLPromptGenerator`）、`prompts/target_ssl.py`（GRACE / GAE / BGRL）

### 3.1 路径 A：`dual_freq`（代码默认，消融对照）

| 提示类型 | 数量 | 信号来源 | 聚类策略 |
|----------|------|----------|----------|
| **同配提示** \(P_{homo}\) | \(K_1\) | 冻结 GNN 低频表征 \(H_{homo}\) | **球形 K-Means** |
| **异配提示** \(P_{hete}\) | \(K_2\) | 高频残差 \(H_{hete}^{res}=X_{aligned}-H_{homo}\) 等 | 同配族：K-Means / GMM；异配族：**GMM / 谱聚类**（`--hetero_clusterer`） |

### 3.2 路径 B：`target_ssl`【修订 2026-05-19 · 论文主推】

跨域 Few-shot 下，源域冻结 GNN 的特征与目标域分布错位较大。改为在**目标图仅用 `x` + `edge_index` 无监督学习**嵌入 \(Z_{target}\)，再聚类得到提示：

| 步骤 | 说明 |
|------|------|
| SSL | `prompts/target_ssl.py`：`grace` / `gae` / `bgrl`（大图推荐 **bgrl**，避免 GRACE 全图 \(O(N^2)\) 对比） |
| \(P_{homo}\) | 对 \(Z_{target}\) 全图 **均衡球形 K-Means**（`K1`） |
| \(V_{pool}\) | Top-ρ **特征异配**节点（**默认不用标签**，避免 5-shot 泄漏） |
| \(P_{hete}\) | 在 \(Z_{target}[V_{pool}]\) 上球形 K-Means（`K2`） |

**约束**：\(P_{homo}, P_{hete}\) 默认包装为 `nn.Parameter`（`--no_freeze_prompts`，默认开启），参与反传；消融可 `--freeze_prompts` 只训路由。

> **注意**：`--hetero_clusterer` **仅对 `dual_freq` 有效**；`target_ssl` 路径不使用 GMM/谱聚类。

**高频残差**（`dual_freq`）：\(H_{hete}^{res} = X_{aligned} - H_{homo}\)。严格高通图滤波预留于 `load_data.normalize_high_freq_adj`。

---

## 四、阶段二：基于 Gumbel-Softmax 的可导拓扑重构

**模块**：`prompts/gumbel_route.py`

### 4.1 Top-ρ 异配候选池 \(V_{pool}\)

1. 计算原图 \(N\) 个节点的**局部异配度**（优先：邻居异类比例；备选：特征异配；`target_ssl` 池选默认**特征异配、无标签**）。
2. 选取异配度最高的 **Top-ρ**（默认按 `load_data.DATASET_PROFILES` 的 `default_rho`，可用 `--rho` 覆盖）。
3. **仅** \(V_{pool}\) 内节点与提示节点建立可学习连边。

### 4.2 Logits 与稀疏邻接

- **Gumbel-Softmax** 得到 **`EX_homo`**、**`EX_hete`** 两路稀疏矩阵。
- 【修订 2026-05-19】实验默认 **τ=1.0 固定**（未开 `--anneal_tau`）；后续可试退火到 0.1。

### 4.3 对比损失

**Prompt-Edge Contrastive Loss**（`loss.prompt_edge_contrastive_loss`）：homo / hete **分路**计算（`lambda3_homo` / `lambda3_hete`）。大图建议 `--contrastive_sample_size 512`。

---

## 五、阶段三：适应分支的异配度感知消息传递

**模块**：`models/prompt_conv.py` → `PromptAwareGNNConv`

- 图规模 **\((N+K)\)**；`edge_type`：0=原图边，1=提示边。
- 对异配度高的节点：弱化原图异配邻居权重 \(\alpha\)，强化提示边权重 \(\beta\)。

**适应分支**：`models/dual_branch.py` → `PromptBranchEncoder` + `GP2FAdapter`。

---

## 六、阶段四：双分支维度对齐与动态门控融合

**模块**：`models/dual_branch.py` → `DualBranchGNN`

| 分支 | 输入 | 输出 |
|------|------|------|
| **冻结分支** | 原图 \((N)\) + 预训练源域 GNN | \(H_{frozen}\) |
| **适应分支** | 增强图 \((N+K)\) + Prompt-Aware MP | \(H_{adapted}^{full}\) → 截断为 \(H_{adapted}[:N]\) |

**门控融合**：\(H_{final} = (1-g)\cdot H_{frozen} + g\cdot H_{adapted}\)

- 默认 **`gate_start=0.2`，`gate_end=0.65`**（`GateSchedule`，cosine 退火 + 可学习偏移）。【修订】README 早期写法 `gate_end=0.8` 已改为与 `train.py` 及主实验一致。
- 预训练：`pretrained_gnns/` + `models/base_gnn.load_pretrained_backbone`。

---

## 七、阶段五：图特异性动态多任务损失

**模块**：`loss.py` → `compute_total_loss`

\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{sparse} + \lambda_2 \mathcal{L}_{consist} + \lambda_3^{homo}\mathcal{L}_{contrast}^{homo} + \lambda_3^{hete}\mathcal{L}_{contrast}^{hete}
\]

| 项 | 默认权重（主实验） | 说明 |
|----|-------------------|------|
| \(\mathcal{L}_{CE}\) | — | 仅 **train_mask** 上交叉熵 |
| \(\mathcal{L}_{sparse}\) | `λ1=0.01` | `EX.mean()`，鼓励稀疏路由 |
| \(\mathcal{L}_{consist}\) | `λ2=0.05` × 图族乘子 | 对称 KL；`consist_max=1.0` 可裁剪 |
| \(\mathcal{L}_{contrast}\) | `λ3=0.1`（分两路） | 前 `loss_warmup_epochs=50` 线性升温 |

`dynamic_lambda2`：profile 为 **homophilic** 时 ×0.25，**heterophilic** 时 ×2.0。

### 【修订 2026-05-19】按真实 Edge H 审视 profile

`DATASET_PROFILES.family` 用于 λ2 等动态权重，但**真实边同配率**可能与 profile 不一致（例：**Amazon-ratings** profile=`homophilic`，实测 **Edge H≈0.38**）。后续拟增加按 `estimate_edge_homophily` 自适应 λ2，或 ablation 固定 heterophilic 档。

---

## 八、多场景数据集（8 Benchmarks）

### 8.1 极端异配与多部图

**数据集**：`Minesweeper`，`Actor`，`squirrel`

| 提示 | 拓扑角色 |
|------|----------|
| \(P_{hete}\) | 孤岛直连 / 跨域枢纽（见 idea 叙事） |
| \(P_{homo}\) | 长尾同盟 |

**Profile 默认**：`rho≈0.35–0.40`，`k_homo/k_hete` 见 `load_data.py`。

### 8.2 强同配与隐含异配

**数据集**：`Cora`，`CiteSeer`，`PubMed`，`Amazon-ratings`，`Flickr`

**【修订 2026-05-19】Amazon-ratings**：profile 标 homophilic，但边标签同配率可很低（~0.38）；提示模块在该集上 **P-H≈Edge H**，分类 ES Test ~23%（探路），与「拉高同配」叙事需分开汇报。

### 8.3 数据集别名

| 别名 | 规范名称 |
|------|----------|
| `Amazon` | `Amazon-ratings` |
| `SQUIRREL-F`, `Squirrel` | `squirrel` |
| `ACTOR` | `Actor` |

### 8.4 评估指标

| 指标 | 含义 |
|------|------|
| **ES Test Acc** | 论文主表：早停 + best checkpoint |
| **Edge Homophily** | 原图真实边同标签比例 |
| **P-H / P-E** | `prompt_edge_homophily`：提示 homo/hete 路由的「标签–伪标签」一致率 |
| **Δ = P-H − Edge H** | 提示边相对原边的同配提升（**非**全图 Edge H 改变） |

**【修订 2026-05-19 · 实验结论】**

| 现象 | 说明 |
|------|------|
| P-H 提升因数据集而异 | Minesweeper：Δ≈**+0.09**；Amazon：Δ≈**+0.01** |
| P-H ≈ P-E 常见 | homo / hete 两路在指标上**尚未分化**，叙事上「同盟 vs 桥梁」待加强 |
| P-H ≠ 分类性能 | Minesweeper raw acc 可 **低于** 多数类基线（~80%），需报 **macro-F1** |
| Finetune 难改 P-H | 训练中 P-H 常几乎不变，结构主要由 **SSL + 初始聚类/路由** 决定 |

---

## 九、代码结构

```text
MyIdea/
├── train.py                      # 端到端 Few-shot 训练（主入口，默认早停）
├── load_data.py                  # 8 数据集 + DATASET_PROFILES + 5-shot 划分
├── loss.py                       # 多任务损失 + prompt_edge_homophily
├── prompts/
│   ├── cluster_generator.py      # PromptGenerator + TargetSSLPromptGenerator
│   ├── target_ssl.py             # GRACE / GAE / BGRL（目标域无监督）
│   ├── gumbel_route.py           # Top-ρ + GumbelRouter
│   └── routing_debug.py          # 路由调试 dump
├── models/
│   ├── base_gnn.py
│   ├── dual_branch.py
│   ├── prompt_conv.py
│   └── gp2f_adapter.py
├── pretrained_gnns/
├── checkpoints/                  # 早停 best 权重（默认 ./checkpoints/seed_{seed}.pt）
├── docs/
│   ├── fewshot_experiment_progress.md
│   └── ablation_matrix.md
└── test/
    ├── test_prompt_homophily.py
    ├── test_pure_dual_branch.py
    └── test_frozen_gnn_baseline.py
```

### 实现进度（截至 2026-05-19）

| # | 模块 / 能力 | 状态 |
|---|-------------|------|
| 1–10 | 核心模块（数据、提示、路由、损失、双分支、消融脚本） | ✅ 见变更日志 2026-05-18 |
| 11 | **`target_ssl` 提示初始化**（`TargetSSLPromptGenerator` + `target_ssl.py`） | ✅ |
| 12 | **Few-shot 早停 + checkpoint**，主指标 **ES Test** | ✅ |
| 13 | **`--use_profile_k`**、大图 `contrastive_sample_size` | ✅ |
| 14 | 实验进度文档 `docs/fewshot_experiment_progress.md` | ✅ |

### Few-shot 实验进度摘要（5-shot，PubMed 源域）

> 完整表格、问题清单与命令见 [`docs/fewshot_experiment_progress.md`](docs/fewshot_experiment_progress.md)。

| 目标域 | 配置要点 | ES Test（当前） | 文献对照 |
|--------|----------|-----------------|----------|
| **Cora** | `target_ssl`+GRACE 200ep, K=10/10, **10-run** | **55.49 ± —**（10-run 均值） | GraphTOP ≈ **51.26%** |
| **Minesweeper** | `target_ssl`+GRACE 200ep, profile K=2/2, **10-run** | **59.65 ± 11.77%** | GraphTOP **61.25±5.08**；ProNoG **63.03±2.74** |
| **Amazon-ratings** | `target_ssl`+bgrl, profile K=5/2, **1-run 探路** | **~23%** | 待填 GraphTOP 格 |

**待办（P0）**：1/3-shot 曲线；Minesweeper **macro-F1**；Amazon **10-run**；Cora/Minesweeper 消融矩阵（`ablation_matrix.md`）。

---

## 十、独立消融测试规范

### 10.1 提示模块剥离 — `test/test_prompt_homophily.py`

验证阶段一、二对 **P-H / Edge H** 的影响（无双分支分类器）。

```bash
python test/test_prompt_homophily.py --dataset Minesweeper --rho 0.35 --runs 3
python test/test_prompt_homophily.py --dataset Amazon-ratings --rho 0.20 --runs 3
```

### 10.2 双分支剥离 — `test/test_pure_dual_branch.py`

```bash
python test/test_pure_dual_branch.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --epochs 200 --runs 10
```

### 10.3 冻结 GNN 线性探针 — `test/test_frozen_gnn_baseline.py`

```bash
python test/test_frozen_gnn_baseline.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --epochs 200 --runs 10
```

---

## 十一、实验命令（Few-shot 主协议）

**公共约定**：`--shots 5 --val_per_class 30 --epochs 200 --early_stop_patience 40 --early_stop_min_epochs 20 --runs 10 --seed 42 --no_routing_debug`

### 11.1 完整模型 — Cora（论文主行）

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

### 11.2 完整模型 — Minesweeper

```bash
python train.py \
  --source_dataset PubMed --target_dataset Minesweeper \
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
  --no_freeze_prompts --use_profile_k
```

### 11.3 完整模型 — Amazon-ratings（大图：bgrl + 对比采样）

```bash
python train.py \
  --source_dataset PubMed --target_dataset Amazon-ratings \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --val_per_class 30 --epochs 200 \
  --early_stop_patience 40 --early_stop_min_epochs 20 \
  --checkpoint_dir ./checkpoints \
  --lr 0.005 --prompt_lr_scale 0.1 --weight_decay 5e-4 \
  --lambda1 0.01 --lambda2 0.03 --lambda3 0.05 \
  --loss_warmup_epochs 50 --consist_max 0.5 \
  --gate_start 0.2 --gate_end 0.55 --gate_warmup_epochs 150 \
  --grad_clip 1.0 --runs 10 --seed 42 --no_routing_debug \
  --prompt_init target_ssl --ssl_method bgrl --ssl_epochs 100 --ssl_lr 0.01 \
  --no_freeze_prompts --use_profile_k \
  --contrastive_sample_size 512
```

### 11.4 消融对照：`dual_freq`（代码默认 init）

```bash
python train.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --val_per_class 30 --epochs 200 \
  --early_stop_patience 40 --early_stop_min_epochs 20 \
  --runs 10 --seed 42 --no_routing_debug \
  --prompt_init dual_freq --use_profile_k
```

### 11.5 实验矩阵与消融 ID

见 [`docs/ablation_matrix.md`](docs/ablation_matrix.md)（M0 GraphTOP 文献格、M1–M4、异配图 H0–H3）。

| 设置 | 脚本 |
|------|------|
| prompts ❌ dual-branch ❌ | `test/test_frozen_gnn_baseline.py` |
| prompts ❌ dual-branch ✅ | `test/test_pure_dual_branch.py` |
| prompts ✅ dual-branch ❌ | `test/test_prompt_homophily.py` |
| prompts ✅ dual-branch ✅ | `train.py`（`--prompt_init target_ssl`） |

---

## 十二、依赖与环境

- Python 3.8+
- 安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- 主要包：`torch`、`torch-geometric`、`numpy`、`scikit-learn`
- 数据首次下载至 `--data_root`（默认 `./data`）

---

## 十三、变更日志（Idea / README / 代码）

| 日期 | 变更摘要 |
|------|----------|
| 2026-05-18 | 重写 README：五阶段架构、8 数据集、Top-ρ、`EX_homo`/`EX_hete`、消融脚本；核心模块 1–10 ✅ |
| 2026-05-18 | 新增 `test/test_frozen_gnn_baseline.py` |
| **2026-05-19** | **【实验协议】** 明确 Few-shot（1/3/5-shot）为论文主线；主指标 **ES Test**；默认 **早停** + checkpoint（`--early_stop_patience 40`） |
| **2026-05-19** | **【Idea/实现】** 阶段一增加 **`target_ssl`** 路径（`TargetSSLPromptGenerator` + `prompts/target_ssl.py`），作为跨域 Few-shot **主推** init；`dual_freq` 保留为消融 |
| **2026-05-19** | **【实验结果】** PubMed→Cora 10-run ES Test **55.49%**（> GraphTOP ~51.26%）；Minesweeper 10-run **59.65±11.77%**（对照 GraphTOP 61.25±5.08）；Amazon 探路 ~23% |
| **2026-05-19** | **【实验结论写入 README §8.4】** P-H 提升因数据集而异；P-H≈P-E 未分化；Amazon profile 与 Edge H 错位；Minesweeper 需 macro-F1 |
| **2026-05-19** | **【文档】** 新增 `docs/fewshot_experiment_progress.md`；实验命令与 `ablation_matrix.md` 对齐；修正 `gate_end` 文档为 **0.65** |
| **2026-05-19** | **【待办】** 1/3-shot；Minesweeper macro-F1；Amazon 10-run；按 Edge H 自适应 λ2；`--anneal_tau` |

---

*若在对话中修改了 idea，请更新上文对应章节（标注【修订 YYYY-MM-DD】），并在第十三节追加记录。详细实验数值以 `docs/fewshot_experiment_progress.md` 为准。*
