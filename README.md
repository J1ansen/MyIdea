# Graph Prompt Learning for Cross-Domain Heterophilic Node Classification

> **维护约定**：本 README 是项目 idea 与实验协议的**唯一权威文档**。若在后续开发或讨论中修改了动机、架构、损失、数据集或实验流程，**必须同步更新本文件**。

---

## 一、研究动机 (Motivation)

在图表示学习的**跨域迁移**中，现有 GNN 与图提示方法通常强烈依赖**同配假设 (Homophily Assumption)**：相连节点更可能同类。然而真实场景存在大量**异配图 (Heterophilic Graphs)**，强行套用同配假设会导致过平滑与特征坍缩。

本框架受两篇工作启发：

| 工作 | 启发 |
|------|------|
| **GRAPHITE (ICLR 2026)** | 引入「特征提示节点」重构拓扑，提升输入图的结构/特征同配性，使异配图经提示重构后能满足下游 GNN 的同配假设 |
| **GP2F (ICLR 2026)** | **双分支架构**：冻结分支保留源域通用先验；适应分支学习目标域独有分布 |

**核心目标**：在跨域、异配场景下，通过可学习的图提示拓扑 + 异配感知消息传递 + 双分支融合，实现稳健的节点分类。

---

## 二、整体流水线（五阶段）

```text
目标图 G
  │
  ├─[阶段一] 双频提示初始化 → P_homo (K1), P_hete (K2)
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

## 三、阶段一：双频提示初始化与自适应语义簇

**模块**：`prompts/cluster_generator.py` → `PromptGenerator`

| 提示类型 | 数量 | 信号来源 | 聚类策略 |
|----------|------|----------|----------|
| **同配提示** \(P_{homo}\) | \(K_1\) | 冻结 GNN 低频表征 \(H_{homo}\) | **球形 K-Means**（任意数据集均使用） |
| **异配提示** \(P_{hete}\) | \(K_2\) | 高频残差 + 低频拼接 \(H_{aug}=[H_{homo}; H_{hete}^{res}]\) | **同配图**（Cora 等）：基础 K-Means / GMM；**异配图**（Minesweeper 等）：**禁止普通 K-Means**，使用 **GMM** 或 **谱聚类** |

**约束**：\(K_1 + K_2\) 个提示原型须包裹为 `nn.Parameter`，参与反向传播。

**高频残差**（当前实现）：\(H_{hete}^{res} = X_{aligned} - H_{homo}\)（维度对齐后相减）。严格高通图滤波预留于 `load_data.normalize_high_freq_adj`。

---

## 四、阶段二：基于 Gumbel-Softmax 的可导拓扑重构

**模块**：`prompts/gumbel_route.py`

### 4.1 Top-ρ 异配候选池 \(V_{pool}\)

1. 计算原图 \(N\) 个节点的**局部异配度**（优先：邻居异类比例；备选：1 − 邻居特征余弦相似度）。
2. 选取异配度最高的 **Top-ρ**（默认 ρ=0.3，可按数据集 profile 调整）节点构成待连接池 \(V_{pool}\)。
3. **仅** \(V_{pool}\) 内节点与提示节点建立可学习连边；池外节点对应 `EX` 行为 0。

### 4.2 Logits 与稀疏邻接

- 计算候选节点与 \(K\) 个提示节点的连接意愿 **Logits**。
- **Gumbel-Softmax** 得到稀疏矩阵 **\(E_X\)**（分 **`EX_homo`**、**`EX_hete`** 两路）。
- 同配/异配提示在拓扑上的物理意义见**第七节**。

### 4.3 对比损失

**Prompt-Edge Contrastive Loss**（`loss.prompt_edge_contrastive_loss`）：若原图节点 \(u,v\) 连向同一提示节点则拉近表征，否则推远。用于优化 Logits / 路由参数。

> **实现状态**：Top-ρ 与 `EX_homo`/`EX_hete` 分离已在 `test/test_prompt_homophily.py` 与 `train.py` 主训练链落地；`train.py` 支持 `--rho` 覆盖数据集默认候选池比例。

---

## 五、阶段三：适应分支的异配度感知消息传递

**模块**：`models/prompt_conv.py` → `PromptAwareGNNConv`（继承 `MessagePassing`，**禁止**直接用普通 PyG 卷积替代）

- 原图节点与提示节点在节点维拼接，图规模为 **\((N+K)\)**。
- 在 `message` 聚合中引入**边类型门控**（`edge_type`: 0=原图边，1=提示边）：
  - 对异配度高的节点：**弱化**与原图异配邻居的聚合权重 \(\alpha\)；
  - **强化**经 \(E_X\) 来自提示节点的权重 \(\beta\)。

**适应分支编码器**：`models/dual_branch.py` → `PromptBranchEncoder` + `GP2FAdapter`。

**设计意图（叙事）**：低参数 \(W_{adapter}\) / Mini-GNN 产生目标域提示流 \(M_{prompt}\)，与原图流 \(M_{real}\) 门控融合为 \(H_{adapted}\)（实现上通过 `prompt_branch` + `adapter` 统一表达）。

---

## 六、阶段四：双分支维度对齐与动态门控融合

**模块**：`models/dual_branch.py` → `DualBranchGNN`

| 分支 | 输入 | 输出 |
|------|------|------|
| **冻结分支** | 原图 \((N)\) + 预训练源域 GNN | \(H_{frozen} \in \mathbb{R}^{N \times D}\) |
| **适应分支** | 增强图 \((N+K)\) + Prompt-Aware MP | \(H_{adapted}^{full} \in \mathbb{R}^{(N+K) \times D}\) |

**维度对齐**：

- 源/目标特征维不匹配时，使用**正交投影**（`OrthogonalProjection` / `InputAligner`），避免普通随机 `Linear` 造成方差损失。
- 适应分支输出须**显式截断**：`H_adapted = H_adapted_full[:N, :]`（提示节点不参与分类）。

**门控融合**：

\[
H_{final} = (1 - g) \cdot H_{frozen} + g \cdot H_{adapted}, \quad g \in [0,1]
\]

- 初始化应**偏向冻结分支**（如 \(g \approx 0.2\)，即 0.8 frozen / 0.2 adapted）。
- 随 **Epoch 平滑增大** \(g\)，逐步信任适应分支：`GateSchedule(start=0.2, end=0.8, mode="cosine")` 作为基础退火，叠加小幅可学习偏移。

**预训练权重**：`pretrained_gnns/` + `models/base_gnn.load_pretrained_backbone`。

---

## 七、阶段五：图特异性动态多任务损失

**模块**：`loss.py`

\[
\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{sparse} + \lambda_2 \mathcal{L}_{consist} + \lambda_3 \mathcal{L}_{contrastive}
\]

| 项 | 含义 | 动态约束 |
|----|------|----------|
| \(\mathcal{L}_{CE}\) | 节点分类交叉熵 | — |
| \(\mathcal{L}_{sparse}\) | 提示边稀疏（如 `EX.mean()`） | — |
| \(\mathcal{L}_{consist}\) | 双分支输出分布一致性（对称 KL） | **同配图** \(\lambda_2\) 较小；**异配图** \(\lambda_2\) **显著增大**，用源域先验压制异配噪声 |
| \(\mathcal{L}_{contrastive}\) | Prompt-Edge 对比损失 | 后续可拆分为 homo / hete 两路 |

`dynamic_lambda2(is_homophilic)` 已实现；`train.py` 已统一调用 `compute_total_loss`，避免训练脚本中重复拼装各项损失。

---

## 八、多场景数据集的拓扑语义映射（8 Benchmarks）

实验在以下 8 个数据集上进行。编写数据加载、邻接重构与评估脚本时，应在注释中体现 \(P_{homo}\) / \(P_{hete}\) 的物理角色。

### 8.1 极端异配与多部图 (Extreme Heterophily & Multipartite)

**数据集**：`Minesweeper`，`Actor`，`squirrel`（论文记法 SQUIRREL-F）

| 提示 | 拓扑角色 | 典型语义 |
|------|----------|----------|
| \(P_{hete}\) | **孤岛直连通道** | Minesweeper：地雷节点打破网格隔离，形成「危险模式」逐层传递；Squirrel：连接跨领域高流量页，防特征被长尾稀释；Actor：连接跨流派「万金油」配角，隔离共演噪声 |
| \(P_{homo}\) | **长尾同盟** | 将边缘、特征相似但无物理连边的节点在输入层聚拢 |

**默认 Top-ρ**：0.35–0.40（见 `load_data.py` → `DATASET_PROFILES`）

### 8.2 强同配与隐含异配 (Strong Homophily & Implicit Bipartite)

**数据集**：`Cora`，`CiteSeer`，`PubMed`，`Amazon-ratings`，`Flickr`

| 提示 | 拓扑角色 | 典型语义 |
|------|----------|----------|
| \(P_{homo}\) | **语义填补器** | 引文网：未互引的同领域论文；Flickr：视觉相似但未元数据连边的图像 |
| \(P_{hete}\) | **功能跨界桥梁** | Amazon：互补共买（电脑-鼠标）的跨品类隔离聚合；引文网：跨学科综述的隔离，缓解过平滑 |

**默认 Top-ρ**：0.15–0.20

### 8.3 数据集别名

| 别名 | 规范名称 |
|------|----------|
| `Amazon` | `Amazon-ratings` |
| `SQUIRREL-F`, `Squirrel` | `squirrel` |
| `ACTOR` | `Actor` |

### 8.4 评估指标（除全局 Homophily 外）

- **结构同配性 (Edge Homophily)**：相连节点同标签边的比例。
- **特征同配性 (Feature Homophily)**：相连节点特征余弦相似度均值。
- **Prompt-Edge Homophily**（建议）：对每个提示节点统计连接原图节点的多数类伪标签，计算连边标签一致比例；**分 \(P_{homo}\) / \(P_{hete}\) 汇报**。

---

## 九、代码结构

```text
MyIdea/
├── train.py                      # 端到端 5-shot 训练（主入口）
├── requirements.txt              # Python 依赖
├── load_data.py                  # 数据集加载与 Few-shot 划分
├── loss.py                       # 对比 / 一致性 / 总损失工具
├── prompts/
│   ├── cluster_generator.py      # 阶段一：PromptGenerator
│   └── gumbel_route.py           # 阶段二：Top-ρ + GumbelRouter
├── models/
│   ├── base_gnn.py               # 预训练 GCN 骨架
│   ├── dual_branch.py            # 阶段三–四：DualBranchGNN
│   ├── prompt_conv.py            # PromptAwareGNNConv
│   └── gp2f_adapter.py           # 低秩 Adapter
├── pretrained_gnns/              # 源域预训练权重 (.pth)
└── test/
    ├── test_prompt_homophily.py  # 消融①：仅提示 + 拓扑（阶段 1–2）
    └── test_pure_dual_branch.py    # 消融②：无双分支提示的纯双分支 baseline（阶段 3–4）
```

### 实现进度（随开发更新）

> 开发阶段：**核心模块 10/10 已完成**（实验矩阵中的「仅冻结 GNN」baseline 仍可选后续补充）

| # | 模块 / 能力 | 状态 | 关键产物 |
|---|------|------|----------|
| 1 | **`load_data.py`**：8 数据集统一加载 + `DATASET_PROFILES` + 5-shot 划分 + 跨域入口 | ✅ | `load_dataset`, `build_few_shot_masks`, `load_cross_domain`, `DatasetProfile`, `estimate_edge_homophily` |
| 2 | **`prompts/cluster_generator.py`**：双频提示初始化（球形 K-Means / 基础 K-Means / GMM / 谱聚类） | ✅ | `PromptGenerator(graph_family=...)`, `PromptSignals` |
| 3 | **`prompts/gumbel_route.py`**：Top-ρ + `EX_homo`/`EX_hete` + 向量化提示边构建 | ✅ | `PromptRouter`, `PromptRoutingOutput`, `build_prompt_edge_index` |
| 4 | **`loss.py`**：homo/hete 分路对比损失 + `prompt_edge_homophily` 指标 + 统一入口 | ✅ | `LossConfig`, `LossOutputs`, `compute_total_loss`, `prompt_edge_homophily` |
| 5 | **`models/prompt_conv.py`**：异配感知 MP，α/β + γ 差异化公式 + 节点级 `local_heterophily` 接入 | ✅ | `PromptAwareGNNConv`, `ORIG_EDGE`, `PROMPT_EDGE` |
| 6 | **`models/dual_branch.py`**：双分支 + 正交投影 + **门控 Epoch 退火** | ✅ | `GateSchedule`, `DualBranchGNN(gate_schedule=...)`, `return_branch_logits`, `node_heterophily` 贯通 |
| 7 | **`train.py`**：接入 Top-ρ + `EX_homo/EX_hete` + 跨域 CLI + 5-shot 协议 + 门控 schedule | ✅ | `--source_dataset`, `--target_dataset`, `--rho`, 调用 `PromptRouter` / `compute_total_loss` |
| 8 | 消融脚本迁移：`test_prompt_homophily.py` / `test_pure_dual_branch.py` → 统一 `load_data` | ✅ | `PromptRouter` / `load_dataset` / `load_cross_domain` / `prompt_edge_homophily` |
| 9 | 清理过时脚本：`pure_dual_branch.py` / `global_router.py` / `prototype_generator.py` / `test_multi_datasets.py` / `test_prompt_module.py` | ✅ | 自仓库移除，主链路仅保留 `train.py` + `test/*` |
| 10 | 更新 README 变更日志 + 实验命令同步 | ✅ | `requirements.txt`、三脚本统一 CLI 与 8 数据集协议 |

---

## 十、独立消融测试规范

### 10.1 提示模块剥离 — `test/test_prompt_homophily.py`

**目的**：验证阶段一、二是否提升图的同配性（不启用双分支分类器）。

**流程**：提示初始化 → Top-ρ → Gumbel 连边 → 对比提示边加入前后的 Edge / Feature Homophily。

```bash
# 单数据集
python test/test_prompt_homophily.py --dataset Cora --runs 3

# 指定 Top-ρ
python test/test_prompt_homophily.py --dataset Minesweeper --rho 0.35 --runs 3

# 多数据集 / 全部 8 个 benchmark
python test/test_prompt_homophily.py --datasets Cora Minesweeper Actor --runs 3
python test/test_prompt_homophily.py --all --runs 3 --data_root ./data
```

### 10.2 双分支模块剥离 — `test/test_pure_dual_branch.py`

**目的**：验证阶段三、四（维度对齐 + 融合）的跨域 baseline（**无提示模块**）。

```bash
python test/test_pure_dual_branch.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --shots 5 --epochs 100 --runs 5
```

记录 **Final Test Acc** 作为完整模型（prompts ✅ + dual-branch ✅）的对照 baseline。

---

## 十一、实验命令（当前可运行）

### 11.1 完整模型（提示 + 双分支）

在目标域上 5-shot 微调；默认加载 `pretrained_gnns/PubMed_SimGRACE_GCN_1.pth`（源域 PubMed）：

```bash
python train.py \
  --source_dataset PubMed --target_dataset Cora \
  --pretrained_name PubMed_SimGRACE_GCN_1.pth \
  --k_homo 10 --k_hete 10 \
  --shots 5 --epochs 200 --lr 0.01 \
  --lambda1 1.0 --lambda2 1.0 --lambda3 1.0 \
  --runs 5
```

常用参数：`--rho`，`--tau`，`--hard_route`，`--topk_prompts`，`--gate_start`，`--gate_end`，`--gate_warmup_epochs`，`--hetero_clusterer {gmm,spectral}`，`--homophily_threshold 0.5`。

### 11.2 建议的跨域实验矩阵（协议目标）

| 设置 | 脚本 | 说明 |
|------|------|------|
| prompts ❌ dual-branch ❌ | 待恢复 baseline 脚本 | 仅线性探针 / 冻结 GNN |
| prompts ❌ dual-branch ✅ | `test/test_pure_dual_branch.py` | 纯双分支（无提示） |
| prompts ✅ dual-branch ❌ | `test/test_prompt_homophily.py` | 仅拓扑同配性 |
| prompts ✅ dual-branch ✅ | `train.py` | 完整框架 |

**目标跨域对**（示例）：`PubMed → Amazon-ratings`，`PubMed → Cora`；通过 `--source_dataset` / `--target_dataset` 在 `train.py` 与 `test_pure_dual_branch.py` 中统一指定。

---

## 十二、依赖与环境

- Python 3.8+
- 安装依赖（建议在项目根目录创建虚拟环境后执行）：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- 主要包：`torch`、`torch-geometric`、`numpy`、`scikit-learn`（GMM / 谱聚类）
- PyG 数据集首次运行会自动下载到 `--data_root`（默认 `./data`）

---

## 十三、变更日志（Idea / README / 代码）

| 日期 | 变更摘要 |
|------|----------|
| 2026-05-18 | 重写 README：五阶段架构、8 数据集拓扑语义、Top-ρ、`EX_homo`/`EX_hete`、消融脚本与实现状态表；实验命令对齐现有 `train.py` / `test_*` |
| 2026-05-18 | **核心模块改造 1/10**：`load_data.py` 重写——8 数据集统一加载、`DATASET_PROFILES`（family / default_rho / homo&hete 拓扑语义）、5-shot `build_few_shot_masks`、跨域 `load_cross_domain` |
| 2026-05-18 | **核心模块改造 2/10**：`prompts/cluster_generator.py` 重写——严格区分同配族（基础 K-Means）与异配族（GMM / 谱聚类）；新增 `PromptSignals` 中间产物；可由 `graph_family` 外部强制图族判定 |
| 2026-05-18 | **核心模块改造 3/10**：`prompts/gumbel_route.py` 重写——新增 `PromptRouter` 一站式封装 + `PromptRoutingOutput`；公共 `build_prompt_edge_index`（向量化、含双向边）；`from_dataset_profile` 接入 `default_rho` |
| 2026-05-18 | **核心模块改造 4/10**：`loss.py` 重写——`LossConfig` / `LossOutputs`；homo / hete 分路对比损失；修复软 EX 的 same-prompt 判定 bug；新增 `prompt_edge_homophily` 指标；对比损失支持池内采样 |
| 2026-05-18 | **核心模块改造 5/10**：`models/prompt_conv.py` 重写——明确 α/β + γ 差异化公式；支持外部传入节点级 `node_heterophily`（与 `gumbel_route` 共享）；修复 root_lin 维度一致性 |
| 2026-05-18 | **核心模块改造 6/10**：`models/dual_branch.py` 增强——新增 `GateSchedule` 的 epoch 退火门控、冻结分支强制 eval、适应分支接入 `node_heterophily`，并支持返回冻结/适应两支 logits 供统一损失使用 |
| 2026-05-18 | **核心模块改造 7/10**：`train.py` 主链路接入 `PromptRouter`、Top-ρ、`EX_homo/EX_hete` 分路、跨域 CLI、`GateSchedule` 与 `compute_total_loss`，训练日志新增 gate / pool / 分路 contrastive 指标 |
| 2026-05-18 | **核心模块改造 8/10**：消融脚本迁移——`test_prompt_homophily.py` / `test_pure_dual_branch.py` 统一使用 `load_data` + `PromptRouter`；提示同配性消融新增 `prompt_edge_homophily` 汇报 |
| 2026-05-18 | **核心模块改造 9/10**：移除过时脚本 `pure_dual_branch.py`、`global_router.py`、`prototype_generator.py`、`test_multi_datasets.py`、`test_prompt_module.py` |
| 2026-05-18 | **核心模块改造 10/10**：新增 `requirements.txt`；README 实验命令、实现进度表与三脚本 CLI 对齐 |

---

*若在对话中修改了 idea，请更新上文对应章节，并在第十三节追加一条变更记录。*
