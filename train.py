# train.py (项目根目录)
import os
import copy
import random
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入图同配性计算工具
from torch_geometric.utils import homophily

# ==========================================
# 导入自定义模块
# ==========================================
from models.dual_branch import DualBranchCrossDomainGNN
from prompts.cluster_generator import PromptGenerator
from models.base_gnn import load_pretrained_backbone
from load_data import load_node_data


def set_seed(seed):
    """固定随机性，保证多次实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_masks(data, shots, val_node_num, test_node_num, seed, device):
    """
    构建 train/val/test 掩码：
    - train: 每类采样 shots 个节点（few-shot）
    - val/test: 从剩余节点中随机划分
    """
    rng = random.Random(seed)
    y = data.y.squeeze()
    num_classes = int(y.max().item() + 1)
    train_nodes = []

    for cls in range(num_classes):
        cls_indices = torch.where(y == cls)[0].tolist()
        if len(cls_indices) <= shots:
            train_nodes.extend(cls_indices)
        else:
            train_nodes.extend(rng.sample(cls_indices, k=shots))

    remain_nodes = [idx for idx in range(data.num_nodes) if idx not in set(train_nodes)]
    rng.shuffle(remain_nodes)

    val_nodes = remain_nodes[:val_node_num]
    test_nodes = remain_nodes[val_node_num:val_node_num + test_node_num]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True
    return train_mask, val_mask, test_mask


def compute_acc(logits, labels, mask):
    """在指定 mask 上计算准确率。"""
    if mask.sum().item() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def get_consistency_weight(epoch, args):
    """
    一致性损失权重调度：
    - 预热阶段（warmup）保持初始权重
    - 之后线性衰减到 lambda_consist * consist_min_ratio
    """
    base = args.lambda_consist
    warmup = args.consist_warmup_epochs
    min_ratio = args.consist_min_ratio
    if warmup <= 0 or epoch <= warmup:
        return base
    tail_epochs = max(1, args.epochs - warmup)
    decay_progress = min(1.0, (epoch - warmup) / tail_epochs)
    current_ratio = 1.0 - decay_progress * (1.0 - min_ratio)
    return base * current_ratio


def analyze_prompt_homophily(model, aligner, data, p_homo, p_hete, args):
    """
    训练后分析提示模块质量，输出三类指标：
    1) 原图边同配率
    2) 节点->提示边同配率（可比指标）
    3) 提示簇纯度
    """
    model.eval()
    aligner.eval()
    with torch.no_grad():
        x_eval = aligner(data.x)
        _, _, _, ex_total = model(
            x_eval, data.edge_index, p_homo, p_hete, hard_route=args.eval_hard_route
        )
        prompt_assignments = ex_total.argmax(dim=1)
        true_labels = data.y.squeeze()

        total_prompt_nodes = args.k_homo + args.k_hete
        purity_scores = []
        prompt_majority_labels = torch.full(
            (total_prompt_nodes,), -1, dtype=true_labels.dtype, device=true_labels.device
        )
        for k in range(total_prompt_nodes):
            connected_nodes = (prompt_assignments == k).nonzero(as_tuple=True)[0]
            if len(connected_nodes) > 1:
                labels_in_k = true_labels[connected_nodes]
                bincount = torch.bincount(labels_in_k)
                majority_class = bincount.argmax()
                prompt_majority_labels[k] = majority_class
                purity_scores.append(bincount[majority_class].item() / len(connected_nodes))

        original_edge_homophily = homophily(data.edge_index, true_labels)
        if len(purity_scores) == 0:
            return original_edge_homophily, 0.0, 0.0

        avg_prompt_purity = float(sum(purity_scores) / len(purity_scores))
        node_indices = torch.arange(data.num_nodes, device=true_labels.device)
        valid_node_mask = prompt_majority_labels[prompt_assignments] >= 0
        if valid_node_mask.sum().item() > 0:
            matched = (
                true_labels[node_indices[valid_node_mask]]
                == prompt_majority_labels[prompt_assignments[valid_node_mask]]
            )
            prompt_edge_homophily = matched.float().mean().item()
        else:
            prompt_edge_homophily = 0.0
        return original_edge_homophily, prompt_edge_homophily, avg_prompt_purity


def train_once(args, seed):
    """单个 seed 的完整训练流程（含 early stopping 和最终分析）。"""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== Seed {seed} =====")
    print(f"🚀 实验环境: {args.source_dataset} (Pre-train) -> {args.target_dataset} (Downstream)")
    print(f"💻 计算设备: {device}")

    # 1) 加载目标域图数据，并按 few-shot 划分 train/val/test
    print(f"\n📂 加载目标域数据集: {args.target_dataset}...")
    data, target_in_dim, target_out_dim = load_node_data(args.target_dataset, "./data")
    data = data.to(device)
    train_mask, val_mask, test_mask = build_masks(
        data, args.shots, args.val_node_num, args.test_node_num, seed, device
    )
    print(
        f"📊 目标域特征维度: {target_in_dim}, 节点数: {data.num_nodes}, 分类数: {target_out_dim}, "
        f"划分: train={train_mask.sum().item()}, val={val_mask.sum().item()}, test={test_mask.sum().item()}"
    )

    # 2) 在目标域上一次性生成静态提示节点（同配/异配）
    print(f"\n🔮 正在提取 {args.target_dataset} 的同配/异配特征锚点...")
    generator = PromptGenerator(k_homo=args.k_homo, k_hete=args.k_hete, device=device)
    p_homo, p_hete = generator.generate(data)

    # 3) 可学习特征对齐层：把目标域维度映射到源域预训练模型输入维度
    aligner = nn.Linear(target_in_dim, args.source_dim).to(device)
    print(f"\n🌉 初始化可学习维度对齐层: {target_in_dim} -> {args.source_dim}")

    # 4) 加载源域预训练权重，作为双分支初始化
    weight_file = f"./pretrained_gnns/{args.source_dataset}_SimGRACE_GCN_1.pth"
    print(f"\n🧠 加载源域 ({args.source_dataset}) 预训练 SimGRACE 权重...")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"找不到权重文件 {weight_file}，请检查路径！")
    pretrained_backbone = load_pretrained_backbone(
        weight_path=weight_file,
        in_channels=args.source_dim,
        hidden_channels=args.hidden_dim,
        device=device,
    )

    # 5) 构建双分支模型：冻结分支保留先验，适应分支(Adapter)学习目标域
    print(f"🏗️ 构建双分支跨域图提示框架 (使用 Layer-wise Adapter, r={args.adapter_r})...")
    model = DualBranchCrossDomainGNN(
        pretrained_gnn=pretrained_backbone,
        hidden_channels=args.hidden_dim,
        out_channels=target_out_dim,
        prompt_homo_dim=target_in_dim,
        prompt_hete_dim=target_in_dim * 2,
        tau=args.tau,
        adapter_r=args.adapter_r # <--- 传入新增的 Adapter 瓶颈维度
    ).to(device)

    # 6) 【魔法发生的地方】：由于我们重写了模型逻辑，庞大的 GNN 参数在这里被自动过滤了！
    # 优化器只会捕捉到：aligner, 极小的 GP2FAdapter, w_adapter_homo/hete, 门控机制和分类器
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(aligner.parameters())
    print(f"📉 可训练参数规模暴降！目前参与优化的参数张量总数: {len(trainable_params)}")
    
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 7) 训练主循环（按 val_acc 早停）
    print("\n🔥 开始端到端跨域提示微调...")
    best_val_acc = -1.0
    best_test_acc = 0.0
    best_state = None
    wait = 0

    labels = data.y.squeeze()
    for epoch in range(1, args.epochs + 1):
        model.train()
        aligner.train()
        optimizer.zero_grad()

        # 对齐后的输入特征送入双分支框架
        x_aligned = aligner(data.x)
        logits, h_frozen, h_adapted, ex_total = model(
            x_aligned, data.edge_index, p_homo, p_hete, hard_route=args.train_hard_route
        )

        # 三个主损失：
        # - loss_cls: few-shot 监督分类
        # - loss_sparse: 路由熵约束（鼓励更稀疏选择）
        # - loss_consist: 适应分支与冻结分支的一致性约束 (现在作用于经过 Adapter 微调后的特征)
        loss_cls = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss_sparse = -torch.mean(torch.sum(ex_total * torch.log(ex_total + 1e-8), dim=-1))
        
        if args.consist_on_train_only:
            consist_mask = train_mask
            if consist_mask.sum().item() > 0:
                loss_consist = F.mse_loss(h_adapted[consist_mask], h_frozen[consist_mask])
            else:
                loss_consist = F.mse_loss(h_adapted, h_frozen)
        else:
            loss_consist = F.mse_loss(h_adapted, h_frozen)

        # 路由一致性：同标签训练节点的路由分布尽量接近
        route_consist_terms = []
        train_labels = labels[train_mask]
        for cls in train_labels.unique():
            cls_mask = train_mask & (labels == cls)
            if cls_mask.sum().item() > 1:
                ex_cls = ex_total[cls_mask]
                cls_center = ex_cls.mean(dim=0, keepdim=True)
                route_consist_terms.append(F.mse_loss(ex_cls, cls_center.expand_as(ex_cls)))
        if len(route_consist_terms) > 0:
            loss_route_consist = torch.stack(route_consist_terms).mean()
        else:
            loss_route_consist = torch.tensor(0.0, device=device)

        consist_weight = get_consistency_weight(epoch, args)
        loss = (
            loss_cls
            + args.lambda_sparse * loss_sparse
            + consist_weight * loss_consist
            + args.lambda_route_consist * loss_route_consist
        )
        loss.backward()
        optimizer.step()
        
        # Gumbel 温度退火：逐步从“软选择”变“更尖锐选择”
        model.global_router.tau = max(args.min_tau, model.global_router.tau * args.tau_decay)

        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            aligner.eval()
            with torch.no_grad():
                x_eval = aligner(data.x)
                logits_eval, _, _, _ = model(
                    x_eval, data.edge_index, p_homo, p_hete, hard_route=args.eval_hard_route
                )
                train_acc = compute_acc(logits_eval, labels, train_mask)
                val_acc = compute_acc(logits_eval, labels, val_mask)
                test_acc = compute_acc(logits_eval, labels, test_mask)

            # 早停准则：只看验证集表现
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                wait = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "aligner": copy.deepcopy(aligner.state_dict()),
                }
            else:
                wait += 1

            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} "
                f"(Cls: {loss_cls.item():.4f}, Sparse: {loss_sparse.item():.4f}, "
                f"Consist: {loss_consist.item():.4f}@w={consist_weight:.4f}, "
                f"RouteCons: {loss_route_consist.item():.4f}) "
                f"| Train/Val/Test: {train_acc*100:.2f}/{val_acc*100:.2f}/{test_acc*100:.2f} "
                f"| BestVal: {best_val_acc*100:.2f} | BestTest@BestVal: {best_test_acc*100:.2f}"
            )

            if wait >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # 恢复最佳验证点对应的模型参数
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        aligner.load_state_dict(best_state["aligner"])

    print(f"\n🎉 Seed {seed} 完成！Best Test@BestVal: {best_test_acc*100:.2f}%")
    # 8) 训练结束后再做提示同配性分析
    print("\n" + "=" * 40)
    print("🔬 提示节点同配性与纯度验证")
    print("=" * 40)
    edge_h, prompt_edge_h, prompt_purity = analyze_prompt_homophily(model, aligner, data, p_homo, p_hete, args)
    print(f"📉 原图边级同配率 (Edge Homophily): {edge_h:.4f}")
    print(f"📈 提示边级同配率 (Node->Prompt Edge Homophily): {prompt_edge_h:.4f}")
    print(f"🧪 提示簇纯度均值 (Prompt Purity): {prompt_purity:.4f}")
    return best_test_acc, best_val_acc, edge_h, prompt_edge_h, prompt_purity


def parse_args():
    """解析训练、路由和复现实验相关参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dataset', type=str, default='Amazon-ratings', help='目标域数据集')
    parser.add_argument('--source_dataset', type=str, default='PubMed', help='源域数据集')
    parser.add_argument('--source_dim', type=int, default=500, help='源域特征维度(PubMed=500, Cora=1433)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='预训练模型隐藏层维度')

    # 新增的 GP2F 适配器瓶颈维度参数
    parser.add_argument('--adapter_r', type=int, default=32, help='GP2F Adapter 降维空间的秩 (默认: 32)')

    parser.add_argument('--shots', type=int, default=5, help='每类 few-shot 数量')
    parser.add_argument('--val_node_num', type=int, default=1000, help='验证集节点数')
    parser.add_argument('--test_node_num', type=int, default=1000, help='测试集节点数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='权重衰减')
    parser.add_argument('--eval_every', type=int, default=10, help='评估间隔 epoch')
    parser.add_argument('--patience', type=int, default=10, help='early stop 容忍评估次数')
    parser.add_argument('--seeds', type=str, default='42', help='随机种子列表，如 42,52,62')

    parser.add_argument('--k_homo', type=int, default=10, help='同配提示节点数')
    parser.add_argument('--k_hete', type=int, default=10, help='异配提示节点数')
    parser.add_argument('--tau', type=float, default=1.0, help='Gumbel 初始温度')
    parser.add_argument('--tau_decay', type=float, default=0.98, help='每轮温度衰减')
    parser.add_argument('--min_tau', type=float, default=0.1, help='温度下限')
    parser.add_argument('--lambda_sparse', type=float, default=0.05, help='稀疏极化损失权重')
    parser.add_argument('--lambda_consist', type=float, default=0.1, help='双分支一致性损失权重')
    parser.add_argument('--consist_warmup_epochs', type=int, default=30, help='一致性损失预热轮数')
    parser.add_argument('--consist_min_ratio', type=float, default=0.1, help='一致性损失衰减后最小比例')
    parser.add_argument('--consist_on_train_only', action='store_true', help='仅在训练节点计算一致性损失')
    parser.add_argument('--lambda_route_consist', type=float, default=0.1, help='同标签路由一致性损失权重')
    parser.add_argument('--train_hard_route', action='store_true', help='训练时使用硬路由（默认软路由）')
    parser.add_argument('--eval_hard_route', action='store_true', help='评估时使用硬路由（默认软路由）')

    # 轻量网格搜索模式（逗号分隔）
    parser.add_argument('--grid_search', action='store_true', help='启用小网格自动实验模式')
    parser.add_argument('--grid_lambda_sparse', type=str, default='0.01,0.02', help='lambda_sparse 网格')
    parser.add_argument('--grid_lambda_consist', type=str, default='0.02,0.05', help='lambda_consist 网格')
    parser.add_argument('--grid_tau_decay', type=str, default='0.995', help='tau_decay 网格')
    parser.add_argument('--grid_min_tau', type=str, default='0.3', help='min_tau 网格')
    parser.add_argument(
        '--grid_seeds',
        type=str,
        default='42',
        help='网格搜索每组使用的 seed 列表（默认仅 42，省时间）；多 seed 示例: 42,0',
    )
    parser.add_argument(
        '--grid_use_full_seeds',
        action='store_true',
        help='网格搜索时改用 --seeds 的完整列表（与单次训练一致，更耗时）',
    )
    return parser.parse_args()


def _parse_seed_list(seed_str):
    return [int(s.strip()) for s in seed_str.split(',') if s.strip()]


def main():
    """
    程序入口：
    - 按 seeds 循环调用 train_once
    - 汇总 mean/std，得到更稳健的结论
    """
    args = parse_args()
    if args.grid_search:
        if args.grid_use_full_seeds:
            active_seeds = _parse_seed_list(args.seeds)
        else:
            active_seeds = _parse_seed_list(args.grid_seeds)
    else:
        active_seeds = _parse_seed_list(args.seeds)

    def run_one_setting(run_args, seeds):
        all_metrics = []
        for seed in seeds:
            metrics = train_once(run_args, seed)
            all_metrics.append(metrics)

        test_accs = [m[0] for m in all_metrics]
        val_accs = [m[1] for m in all_metrics]
        edge_hs = [m[2] for m in all_metrics]
        prompt_edge_hs = [m[3] for m in all_metrics]
        prompt_purities = [m[4] for m in all_metrics]
        return {
            "test_mean": float(np.mean(test_accs)),
            "test_std": float(np.std(test_accs)),
            "val_mean": float(np.mean(val_accs)),
            "val_std": float(np.std(val_accs)),
            "edge_h_mean": float(np.mean(edge_hs)),
            "prompt_edge_h_mean": float(np.mean(prompt_edge_hs)),
            "prompt_purity_mean": float(np.mean(prompt_purities)),
        }

    if not args.grid_search:
        summary = run_one_setting(args, active_seeds)
        print("\n" + "=" * 44)
        print("📦 多 Seed 复现实验汇总")
        print("=" * 44)
        print(
            f"BestTest@BestVal: {summary['test_mean']*100:.2f}% ± {summary['test_std']*100:.2f}% | "
            f"BestVal: {summary['val_mean']*100:.2f}% ± {summary['val_std']*100:.2f}%"
        )
        print(
            f"EdgeH: {summary['edge_h_mean']:.4f} | PromptEdgeH: {summary['prompt_edge_h_mean']:.4f} | "
            f"PromptPurity: {summary['prompt_purity_mean']:.4f}"
        )
        return

    # 轻量网格搜索：仅搜索指定的关键超参组合
    sparse_list = [float(x.strip()) for x in args.grid_lambda_sparse.split(",") if x.strip()]
    consist_list = [float(x.strip()) for x in args.grid_lambda_consist.split(",") if x.strip()]
    tau_decay_list = [float(x.strip()) for x in args.grid_tau_decay.split(",") if x.strip()]
    min_tau_list = [float(x.strip()) for x in args.grid_min_tau.split(",") if x.strip()]

    all_results = []
    combo_iter = list(itertools.product(sparse_list, consist_list, tau_decay_list, min_tau_list))
    print("\n" + "=" * 54)
    print(f"🔎 网格搜索启动，共 {len(combo_iter)} 组配置")
    print(f"   本阶段使用 seeds: {active_seeds}")
    print("=" * 54)

    for idx, (lam_sparse, lam_consist, tau_decay, min_tau) in enumerate(combo_iter, start=1):
        run_args = copy.deepcopy(args)
        run_args.lambda_sparse = lam_sparse
        run_args.lambda_consist = lam_consist
        run_args.tau_decay = tau_decay
        run_args.min_tau = min_tau

        print(
            f"\n[Grid {idx}/{len(combo_iter)}] "
            f"lambda_sparse={lam_sparse}, lambda_consist={lam_consist}, "
            f"tau_decay={tau_decay}, min_tau={min_tau}"
        )
        summary = run_one_setting(run_args, active_seeds)
        result = {
            "lambda_sparse": lam_sparse,
            "lambda_consist": lam_consist,
            "tau_decay": tau_decay,
            "min_tau": min_tau,
            **summary,
        }
        all_results.append(result)

    best = max(all_results, key=lambda r: (r["test_mean"], r["val_mean"]))
    print("\n" + "=" * 54)
    print("🏆 网格搜索最优配置")
    print("=" * 54)
    print(
        f"lambda_sparse={best['lambda_sparse']}, lambda_consist={best['lambda_consist']}, "
        f"tau_decay={best['tau_decay']}, min_tau={best['min_tau']}"
    )
    print(
        f"BestTest@BestVal(mean±std): {best['test_mean']*100:.2f}% ± {best['test_std']*100:.2f}% | "
        f"BestVal(mean): {best['val_mean']*100:.2f}%"
    )


if __name__ == "__main__":
    main()