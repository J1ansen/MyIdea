Motivation:
在图表示学习领域，跨域问题是一个重要研究方向，但现有的GNN以及图提示方法往往基于同配假设，即默认相连的节点更加相似（更可能属于同一类），然而在实际研究中，还存在着大量异配图，即相连的节点可能属于不同类别，如果此时仍然默认同配假设就会导致GNN性能下降。
受到最新论文GRAPHITE（ICLR2026）的启发，我们想要通过引入特征提示节点提升输入（异配）图的同配性来适配GNN模型，并吸收GP2F（ICLR2026）的双分支架构(冻结分支保留源域知识，适应分支学习目标域知识)来作为解决跨域问题的核心。

本文的idea如下：
提示初始化：
将目标图输入冻结GNN,得到低频信号H_homo对表征进行k-means聚类得到K1个同配提示节点P_homo;
计算图的高频信号H_hete（残差）,拼接低频和高频信号得到H_aug,对H_aug进行聚类得到K2个异配提示节点P_hete;

GumbelSoftmax连边:
计算原图节点和提示节点之间的连接意愿分数Logits,进行Gumbel-softmax转换生成EX_homo和EX_hete矩阵;

双分支消息传递:
原图输入冻结分支得到包含源域通用先验的节点表征H_frozen(N维),提示图输入适应分支,得到H_adapted(N+K维),由于提示节点不参与最终分类,直接舍弃K维尾缀对齐了冻结分支的维度。
通过门控机制自适应融合冻结分支H_frozen和适应分支H_adapted的特征。

适应分支设计:
设计一个低参数的权重矩阵W_adapter,原图节点会沿着EX拼接矩阵(EX_homo和EX_hete)的提示边有选择性地聚合来自提示节点的特征信号,生成目标域独有的提示信息流M_prompt(N+K维),再采用一个门控向量,缩放提示信息,自适应融合原图信息M_real和提示信息M_prompt最终生成H_adapted。


实验一：PubMed->Amazon-rating
a. prompts❌ dual-branch❌
python train_baseline.py --source_dataset PubMed --target_dataset Amazon-ratings
b. prompts❌ dual-branch✅
python train_adapter_only.py --source_dataset PubMed --target_dataset Amazon-ratings --adapter_r 32
c. prompts✅ dual-branch✅

提交：
git status
git add .
git commit -m "feat: 完成了特征提示+双分支的初步实现"
git push origin main