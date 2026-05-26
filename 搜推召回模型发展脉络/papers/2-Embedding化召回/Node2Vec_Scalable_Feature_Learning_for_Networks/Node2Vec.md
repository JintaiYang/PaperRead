---
paper_id: "[arXiv:1607.00653](https://arxiv.org/abs/1607.00653)"
title: "node2vec: Scalable Feature Learning for Networks"
authors: "Aditya Grover, Jure Leskovec"
institution: "Stanford University"
pushlication: "KDD 2016"
tags:
  - 召回论文
  - Node2Vec
  - 图Embedding
  - 随机游走
  - 网络表示学习
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/1607.00653)"
  - "[Github](https://github.com/aditya-grover/node2vec)"
date: "2016-07-03"
paper_info: "[[Node2Vec]]"
---

## 一、研究背景与动机

### 1.1 领域现状

网络分析中的核心任务之一是学习节点的低维表示（embedding），以便用于下游任务如节点分类、链接预测和社区检测。2014 年 DeepWalk 首次将 Word2Vec 思想引入图结构：在图上做随机游走生成节点序列，然后用 Skip-gram 训练节点 embedding。DeepWalk 验证了这一范式的有效性，但其采用的是均匀随机游走，缺乏对游走策略的控制。

### 1.2 现有方法的局限性

网络中的节点关系存在两种基本结构属性：同质性（homophily）——相连的节点倾向于属于同一社区，应有相似的 embedding；结构等价性（structural equivalence）——在网络中扮演相似结构角色的节点（如不同社区的"枢纽节点"）也应有相似的 embedding。DeepWalk 的均匀随机游走无法在这两种属性之间灵活权衡，而传统的 BFS（广度优先）和 DFS（深度优先）策略各自只能捕获其中一种。

### 1.3 本文解决方案概述

Node2Vec 提出了一种带偏置的随机游走策略，通过两个超参数 $p$（return parameter）和 $q$（in-out parameter）控制游走在 BFS 和 DFS 之间的偏好，从而灵活地在同质性和结构等价性之间权衡。

## 二、解决方案

### 2.1 核心思想

Node2Vec 的核心创新在于将随机游走的策略参数化。传统 BFS 倾向于探索局部邻域（捕获同质性），DFS 倾向于探索远距离结构（捕获结构等价性）。通过 $p$ 和 $q$ 两个参数，可以在两个极端之间连续地插值，生成不同性质的节点序列，进而学习到不同侧重的 embedding。

### 2.2 整体架构

**偏置随机游走**：假设当前游走从节点 $t$ 到达节点 $v$，下一步从 $v$ 跳转到邻居 $x$ 的非归一化转移概率为：

$$\alpha_{pq}(t, x) = \begin{cases} \frac{1}{p} & \text{if } d_{tx} = 0 \\ 1 & \text{if } d_{tx} = 1 \\ \frac{1}{q} & \text{if } d_{tx} = 2 \end{cases}$$

其中 $d_{tx}$ 是节点 $t$ 和 $x$ 之间的最短路径距离。$p$ 控制返回前一节点的概率（越小越倾向于回头，类似 BFS），$q$ 控制向远处探索的概率（越小越倾向于深度探索，类似 DFS）。

**训练流程**：

1. 对图中每个节点，执行 $r$ 次偏置随机游走，每次长度为 $l$
2. 将所有游走序列作为"句子"，用 Skip-gram + Negative Sampling 训练
3. 得到每个节点的 $d$ 维 embedding 向量

**优化目标**：最大化对数似然：

$$\max_f \sum_{u \in V} \left[ -\log Z_u + \sum_{n_i \in N_S(u)} f(n_i) \cdot f(u) \right]$$

其中 $N_S(u)$ 是通过游走策略 $S$ 得到的节点 $u$ 的邻域，$f(u)$ 是节点 $u$ 的 embedding。

## 三、实验结果

### 3.1 数据集

| 数据集 | 节点数 | 边数 | 任务 |
|--------|--------|------|------|
| BlogCatalog | 10,312 | 333,983 | 多标签分类 |
| PPI (Protein-Protein Interaction) | 3,890 | 76,584 | 多标签分类 |
| Wikipedia | 4,777 | 184,812 | 多标签分类 |
| Facebook | 4,039 | 88,234 | 链接预测 |
| arXiv ASTRO-PH | 18,772 | 198,110 | 链接预测 |

### 3.2 实验设置

#### 3.2.1 基线方法

DeepWalk、LINE、Spectral Clustering、以及基于手工特征的方法（Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment）。

#### 3.2.2 评估指标

节点分类使用 Macro-F1 和 Micro-F1；链接预测使用 AUC。

### 3.3 实验结果与分析

Node2Vec 在多标签分类和链接预测任务上均优于 DeepWalk 和 LINE。在 BlogCatalog 数据集上，Node2Vec 相比 DeepWalk Micro-F1 提升约 2-3%。在链接预测任务上，Node2Vec 在 Facebook 数据集上 AUC 达到 0.9680，显著优于 DeepWalk 的 0.9680 和 LINE 的 0.9490。

关键发现：$p$ 和 $q$ 的最优值因任务而异。在同质性较强的网络（如社交网络）中，较小的 $p$（BFS 偏向）效果更好；在结构等价性重要的任务中，较小的 $q$（DFS 偏向）效果更好。

### 消融实验

论文系统性地分析了 $p$ 和 $q$ 对下游任务的影响：当 $p=1, q=1$ 时退化为 DeepWalk（均匀随机游走），验证了偏置游走的有效性。

## 四、未来工作建议

### 4.1 作者建议的未来工作

将 Node2Vec 扩展到有向图、异构图和动态图。探索基于注意力机制的邻域聚合方法替代随机游走。

### 4.2 基于分析的未来方向

1. **方向1：应用于推荐场景的用户-物品二部图**
   - 动机：推荐系统天然具有图结构（用户-物品交互图）
   - 可能的方法：在二部图上执行 Node2Vec，学习用户和物品 embedding
   - 预期成果：融入图结构信息的 embedding 质量更高
   - 挑战：二部图的游走策略需要特殊设计

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.0/10** - Node2Vec 在 DeepWalk 基础上做了一个精妙的改进，通过参数化的偏置游走策略实现了对 embedding 性质的灵活控制。这一思想后来被广泛用于推荐系统中的图 embedding 学习。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 偏置游走策略设计精巧，BFS/DFS 权衡的 insight 深刻 |
| 技术质量 | 8/10 | 理论分析到位，参数化设计优雅 |
| 实验充分性 | 8/10 | 多数据集、多任务验证，参数敏感性分析全面 |
| 写作质量 | 8/10 | 动机阐述清晰，同质性 vs 结构等价性的分析引人入胜 |
| 实用性 | 8/10 | 在推荐、社交网络分析等领域广泛应用 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

$p$ 和 $q$ 的物理含义非常直观：$p$ 控制"回头看"的概率，$q$ 控制"向远处走"的概率。这种设计使得用户可以根据任务需求灵活调整 embedding 的性质。

#### 5.2.2 需要深入理解的部分

同质性 vs 结构等价性是网络分析中的核心二元对立。理解这一概念有助于在实际应用中选择合适的图 embedding 方法。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- DeepWalk (2014) - Node2Vec 的前身，均匀随机游走 + Word2Vec
- [[EGES|EGES]] - 阿里巴巴在 DeepWalk/Node2Vec 基础上融入 side information

### 6.2 背景相关
- [[Item2Vec|Item2Vec]] - 同样基于 Word2Vec 框架的推荐 embedding 方法
- LINE (2015) - 另一种图 embedding 方法，保留一阶和二阶邻近

### 6.3 后续工作
- GraphSAGE (2017) - 从游走式方法演进到消息传递式图神经网络
- [[PinSage|PinSage]] - 基于 GraphSAGE 的工业级图召回

## 外部资源

- [Github 代码](https://github.com/aditya-grover/node2vec)
- [arXiv 论文](https://arxiv.org/abs/1607.00653)

> [!tip] 关键启示
> Node2Vec 的核心贡献是证明了随机游走策略的选择对 embedding 质量至关重要。不同的游走策略捕获不同的网络属性，这一思想后来深刻影响了图神经网络的邻域采样设计。

> [!warning] 注意事项
> - $p$ 和 $q$ 的最优值需要针对具体任务调参
> - 随机游走方法的时间复杂度与游走次数和长度成正比，在超大规模图上可能成为瓶颈
> - 无法处理动态图和新增节点（transductive 方法）

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。理解网络表示学习的关键论文，为后续的图神经网络方法奠定了概念基础。
