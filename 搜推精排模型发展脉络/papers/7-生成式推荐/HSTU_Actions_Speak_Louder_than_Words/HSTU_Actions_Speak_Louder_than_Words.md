---
paper_id: "[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)"
title: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
authors: "Jiaqi Zhai, Lucy Liao, Xing Liu, Yanbin Gu, Boming Yin, Cathy Wu, Alex Beutel, Jian Li, Alek Kolcz, Jongsoo Park, Maxim Naumov, Wenlin Chen"
institution: "Meta"
pushlication: "ICML 2024 2024-02-26"
tags:
  - 精排论文
  - HSTU
  - 生成式推荐
  - Transformer
  - Scaling-Law
  - 序列建模
  - 万亿参数
  - 统一建模
quality_score: "10/10"
link:
  - "[PDF](https://arxiv.org/pdf/2402.17152)"
  - "[arXiv](https://arxiv.org/abs/2402.17152)"
  - "[GitHub](https://github.com/meta-recsys/generative-recommenders)"
date: "2024-02-26"
---

## 一、研究背景与动机

### 1.1 领域现状

2024 年，NLP 领域的 GPT 系列已经证明了"统一序列建模 + Scaling"的范式威力。然而推荐系统仍然停留在"特征工程 + 浅层模型"的范式中——DLRM 依赖大量人工特征工程，特征类型极其异构（稀疏类别特征、稠密数值特征、交叉特征等），模型无法端到端学习数据中的全部信息，参数的增加也不能自动带来性能的提升。

### 1.2 现有方法的局限性

**异构特征范式的不可扩展性**：传统 DLRM 的特征由不同的工程管道生产和维护，增加更多特征需要更多的工程投入。这种范式天然不具备 Scaling Law——更多参数不等于更好性能。

**推荐数据的独特挑战**：将推荐问题转化为序列建模面临三个独特挑战。第一，高基数和动态变化的词汇表——推荐系统的"词汇"（物料池）通常包含数十亿 Item，且不断有新 Item 加入和旧 Item 过期。第二，天文级别的计算需求——Meta 每天需要处理的推荐"token"比 GPT-3 训练用到的 token 还多几个数量级。第三，非平稳流式数据——用户兴趣和物料分布不断漂移。

### 1.3 本文解决方案概述

HSTU 将推荐问题重新定义为**序列转导任务**（Sequential Transduction），提出了一种专门为推荐数据设计的新架构——Hierarchical Sequential Transduction Units。HSTU 相比标准 Transformer 在推荐场景下训练和推理效率提升了 10x-1000x，在 Meta 的 Instagram 和 Facebook 平台上实现了万亿参数级部署，带来了 12.4%+ 的核心指标提升。

## 二、解决方案

### 2.1 核心思想

GRs（Generative Recommenders）的核心思想是将推荐系统中异构的特征和行为统一编码为时间序列，然后用一个统一的序列模型进行建模。这与 NLP 中"一切皆 token"的范式一脉相承——论文直接用标题宣示了这一观点："Actions Speak Louder than Words"（行动比词语更响亮），暗示用户行为序列比手工特征更有价值。

### 2.2 整体架构

#### 模块1：统一的序列化表示

选择时间跨度最长的用户行为序列作为主时间序列，按时间顺序记录用户交互的 Item 特征，每个位置包含交互的 ItemID、交互时间戳和行为类型（如点击、购买、收藏等）。

对于稀疏类别特征（如人口统计信息），采用压缩策略——在一段时间内只保留最早的特征值，以辅助序列的形式合并到主时间序列中。对于稠密数值特征（如 CTR 统计值），论文大胆地抛弃了这些手工特征，转而让模型直接从原始行为数据中学习。

#### 模块2：召回与排序的统一重定义

**召回任务**被定义为学习概率分布 $p(x_{i+1}|u_i)$，其中 $u_i$ 是第 $i$ 个时刻的用户表征。线上召回时，用 next predict 位置的 embedding 作为 user embedding，与 item embedding 做 ANN 检索。

**排序任务**通过交错排列 Item 和 Action 来实现 target-aware 建模，将排序表述为 $p(a_{i+1}|\Phi_0, a_0, \Phi_1, a_1, ..., \Phi_{i+1})$。候选 Item 的计算在所有候选间共享，实现一次推理完成全部候选打分。

#### 模块3：HSTU 架构

HSTU 堆叠多个层，每个层由三个子层组成：

**Pointwise 投影层**：对输入序列 $X$ 通过单层 MLP $f_1$ 和 SiLU 激活函数进行非线性变换，拆分为四个分量 $U(X), V(X), Q(X), K(X)$。与标准 Transformer 的三个投影相比，多了一个 $U(X)$ 用于门控。

**Pointwise 空间聚合层**：核心公式为 $A(X)V(X) = \phi_2(Q(X)K(X)^T + r^{bp,t})V(X)$，其中 $r^{bp,t}$ 是同时编码位置和时间的相对注意力偏置。关键设计是**不对注意力权重做 softmax 归一化**——保留原始的兴趣强度信息，使模型能区分"只有一个强兴趣"和"有很多弱兴趣"的情况。

**Pointwise 转换层**：$Y(X) = f_2(\text{Norm}(A(X)V(X)) \odot U(X))$，其中 $U(X)$ 起门控作用，部分替代了标准 Transformer 中 FFN 的功能，同时减少了参数量和计算量。

#### 模块4：算法优化

**Stochastic Length（随机序列长度）**：训练时对序列长度进行随机采样，既是正则化手段，也能显著减少训练计算量。

**M-FALCON**：通过对 Key-Value Cache 进行低秩压缩来加速在线推理，在几乎不损失精度的情况下大幅减少推理的内存和计算开销。

#### 模块5：线上服务架构

排序阶段采用"候选折叠"策略：将多个候选 Item 分批放入序列尾部，用户历史行为序列的计算在所有候选间共享，只有候选 Item 之间的注意力被断开。一次前向推理即可完成所有候选的打分。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| MovieLens-20M | 2000万交互 | 公开基准 |
| Amazon Reviews | 多品类 | 公开基准 |
| Meta 内部数据集 | 万亿级 token/天 | Scaling Law 验证 + 在线部署 |

### 3.2 实验设置

#### 3.2.1 基线方法

- SASRec、BERT4Rec（序列推荐 SOTA）
- DLRM、DCN V2（工业推荐 SOTA）
- 标准 Transformer（直接应用于推荐）

#### 3.2.2 评估指标

- **NE（Normalized Entropy）**：归一化交叉熵
- **HR@K、NDCG@K**：召回和排序指标
- **在线核心业务指标**

### 3.3 实验结果与分析

#### 公开数据集

在 MovieLens-20M 和 Amazon Reviews 上，HSTU 在召回和排序任务上均优于 SASRec、BERT4Rec 等主流序列推荐模型。

#### Scaling Law 验证

在 Meta 内部数据集上，模型参数量从数百万扩展到万亿级别，FLOP/example 从 $10^8$ 扩展到超过 $10^{12}$（接近 GPT-3 的计算规模）。HSTU 展现出清晰的 Scaling Law——NE 随计算量的增加持续下降，且在整个两个数量级的范围内没有出现饱和。作为对比，DLRM 和 DCN V2 在计算量增加到一定程度后出现明显的性能饱和。

Scaling 的有效性来源于三个维度的协同扩展：层数（捕获更深层的序列依赖）、序列长度（利用更长的用户历史）、以及 Embedding 维度和 Attention Head 数量。

### 消融实验

去掉门控机制（$U(X)$）、去掉时间编码、或用 softmax 替代 pointwise 注意力，都会导致性能下降。验证了各组件的贡献。

### 在线 A/B 测试

HSTU 在 Meta 的 Instagram Reels、Facebook Reels 等多个核心推荐场景上线，带来了 **12.4%+** 的核心指标提升。在 Meta 这种已经高度优化的系统上，1% 的提升都极为困难，12.4% 意味着 GRs+HSTU 框架相对于传统 DLRM 框架实现了质的飞跃。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议探索更长的用户序列建模、更高效的在线更新机制、以及将 GRs 框架扩展到更多推荐场景。

### 4.2 基于分析的未来方向

1. **方向1：与 DLRM 交叉特征的融合**
   - 动机：HSTU 完全抛弃了手工特征，但某些场景中交叉特征仍有不可替代的价值
   - 可能的方法：混合式架构（如 MTGR 的方案）
   - 预期成果：兼具 GR 的 Scaling 能力和 DLRM 的特征工程优势
   - 挑战：如何在统一框架中平衡两种信息源

2. **方向2：多模态序列建模**
   - 动机：用户行为不仅包含点击/购买，还包含浏览时长、滑动速度等连续信号
   - 可能的方法：将多模态信号编码为不同类型的 token
   - 预期成果：更丰富的用户兴趣表征

3. **方向3：实时在线学习**
   - 动机：推荐数据是非平稳流式数据，模型需要快速适应分布漂移
   - 可能的方法：增量式 KV Cache 更新 + 在线微调
   - 预期成果：更快的模型响应速度

### 4.3 改进建议

1. **改进1：引入 softmax 的可选模式**
   - 当前问题：完全去掉 softmax 可能在某些场景下损失注意力的归一化优势
   - 改进方案：可学习的温度参数控制归一化程度
   - 预期效果：自适应地在不同场景中选择最优的注意力模式

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**10/10** - HSTU 是推荐系统领域的里程碑式工作，首次将推荐重新定义为生成式序列转导任务，在万亿参数规模工业化部署并验证了 Scaling Law。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 10/10 | 将推荐重新定义为生成式序列转导任务，提出统一建模框架，是推荐系统领域的范式级创新 |
| 技术质量 | 9/10 | HSTU 架构设计精巧（无 softmax 注意力、门控替代 FFN、时间偏置），算法优化有深度 |
| 实验充分性 | 10/10 | 公开数据集 + 超大规模内部数据集 + Scaling Law 实验 + 多场景在线 A/B 测试 |
| 写作质量 | 9/10 | 论文结构清晰，动机阐述充分，标题和叙事节奏都很好 |
| 实用性 | 10/10 | 万亿参数级工业部署，12.4%+ 在线提升，Meta 开源实现 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- 无 softmax 的注意力设计——保留兴趣强度的绝对信息，区分"一个强兴趣"和"多个弱兴趣"
- 门控机制 $U(X)$ 替代 FFN——减少参数量同时增强表达能力
- 候选折叠策略——一次推理完成所有候选打分，推理效率远超逐候选方式
- Stochastic Length——同时起到正则化和计算节省的双重作用

#### 5.2.2 需要深入理解的部分

- 无 softmax 注意力在什么条件下优于 softmax？是否存在序列长度的临界点？
- 万亿参数中稀疏参数和 Dense 参数的比例是多少？
- M-FALCON 的低秩压缩在多大程度上损失了信息？

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wukong_Towards_a_Scaling_Law_for_Large_Scale_Recommendation|Wukong]] - 从特征交互角度验证推荐 Scaling Law
- [[MTGR_Multi_Task_Generative_Recommender|MTGR]] - 美团将 HSTU 与 DLRM 交叉特征融合的工业落地

### 6.2 背景相关
- [[DIN_Deep_Interest_Network|DIN]] - 用户兴趣建模的起点，target attention 机制
- [[SIM_Search_based_User_Interest_Modeling|SIM]] - 长序列建模，candidate-specific 检索
- SASRec / BERT4Rec - 序列推荐的 Transformer 基线

### 6.3 后续工作
- [[RankMixer_Scaling_Up_Ranking_Models_in_Industrial_Recommenders|RankMixer]] - 硬件感知的推荐 Scaling
- [[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans]] - 统一 Transformer 精排
- OneRec（快手）- 另一种生成式推荐的工业落地

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2402.17152)
- [GitHub 开源实现](https://github.com/meta-recsys/generative-recommenders)
- [知乎解读：Meta 万亿参数生成式推荐](https://zhuanlan.zhihu.com/p/684478048)

> [!tip] 关键启示
> HSTU 为推荐系统指明了一条新的发展路径：从"特征工程+浅层模型"到"统一序列建模+Scaling"。这与 NLP 从特征工程时代到预训练大模型时代的范式转变如出一辙。用户行为序列比手工特征更有价值——只要模型足够大、数据足够多，模型可以从原始行为中学到一切。

> [!warning] 注意事项
> - HSTU 完全抛弃了交叉特征，在某些低点击率高复购率的场景（如外卖）中可能不是最优选择（MTGR 的实验证实了这一点）
> - 万亿参数的训练和推理成本极高，只有 Meta 级别的公司才有资源复现
> - 论文未详细公开 Scaling Law 的具体数据点和拟合公式

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是推荐系统领域近年来最重要的论文之一，开启了"生成式推荐"的新范式。无论是否直接采用 HSTU 架构，其"统一序列建模 + Scaling"的思想对所有推荐系统从业者都有深远影响。
