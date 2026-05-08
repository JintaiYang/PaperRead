---
paper_id: "[CIKM 2013](https://dl.acm.org/doi/10.1145/2505515.2505665)"
title: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data"
authors: "Po-Sen Huang, Xiaodong He, Jianfeng Gao, et al."
institution: "Microsoft Research"
pushlication: "CIKM 2013-10"
tags:
  - 召回论文
  - DSSM
  - 双塔模型
  - 语义匹配
  - Word-Hashing
quality_score: "8.5/10"
link:
  - "[PDF](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)"
  - "[Microsoft Research](https://www.microsoft.com/en-us/research/project/dssm/)"
date: "2013-10-01"
paper_info: "[[DSSM]]"
---

## 一、研究背景与动机

### 1.1 领域现状

在 2013 年之前，Web 搜索中的语义匹配主要依赖传统的潜在语义模型。LSA（Latent Semantic Analysis）通过 SVD 分解将查询和文档映射到低维空间，PLSA 和 LDA 则使用概率主题模型建模文档的主题分布。这些方法均采用无监督学习，其优化目标是重建文档本身而非区分相关性，因此在实际搜索场景中表现有限。

后续的 BLTMs（Bi-Lingual Topic Models）和 DPMs（Discriminative Projection Models）引入了用户点击数据进行训练。其中 DPMs 使用 S2Net 算法结合 pairwise 排序学习方法，效果优于无监督方法，但会产生巨大的稀疏矩阵，限制了其扩展性。基于深度自编码的方法虽然引入了非线性变换，但仍是无监督训练，效果未显著超越关键词匹配。

### 1.2 现有方法的局限性

现有方法存在三个核心局限：一是无监督方法的优化目标与检索任务不对齐，它们优化的是文档重建而非相关性判断；二是有监督方法（如 DPMs）受限于线性投影，表达能力不足；三是词汇表规模问题，Web 搜索涉及数十万级词汇量，传统 bag-of-words 表示产生的高维稀疏向量难以直接输入深度网络。

### 1.3 本文解决方案概述

DSSM 提出了一种判别式训练的深度神经网络模型：用两个独立的 DNN 塔分别将 query 和 document 映射到共同的低维语义空间，通过余弦相似度计算相关性。关键创新是引入 word hashing 技术，将 500K 维的 term vector 压缩到 30K 维，解决了大词汇表的输入问题。模型使用搜索引擎的点击日志作为监督信号进行端到端训练。

## 二、解决方案

### 2.1 核心思想

DSSM 的核心思想是"双塔编码 + 语义空间匹配"。将 query 和 document 看作两个不同模态的输入，各自通过一个深度网络编码为固定维度的语义向量，然后在这个共享的语义空间中通过余弦相似度衡量二者的匹配程度。这一思想后来被称为"双塔模型"范式，成为了搜推领域最基础的架构之一。

### 2.2 整体架构

DSSM 的架构由三个核心部分组成：Word Hashing 层、多层前馈网络、余弦相似度计算。

```
Query/Doc Text → Bag-of-Words (500K) → Word Hashing (30K) → DNN Layers → Semantic Vector (128-d)
                                                                              ↓
                                                          cosine similarity ← Semantic Vector (128-d)
```

#### 各模块详细说明

**模块1：Word Hashing 层**

Word Hashing 是 DSSM 最重要的工程创新之一，用于解决大规模词汇表带来的高维稀疏输入问题。其做法是：对每个单词添加首尾标记（如 "good" → "#good#"），然后按字母级 n-gram（论文中 n=3，即 trigram）拆分（得到 "#go", "goo", "ood", "od#"），最终用这些 trigram 的 bag-of-words 向量替代原始 term vector。

这种方法将词汇维度从约 500K 压缩到约 30K（所有可能的字母 trigram 组合），且冲突率极低。根据论文实验，在 40K 词汇表上仅有 22 个冲突（word trigram），在 500K 词汇表上冲突也可控。Word Hashing 还天然具有处理拼写变体和 OOV（out-of-vocabulary）词的能力。

**模块2：多层前馈网络**

Word Hashing 后的 30K 维向量输入一个多层前馈网络（MLP）。网络结构为 30K → 300 → 300 → 128。前几层使用 tanh 激活函数：

$$l_1 = W_1 x$$

$$l_i = f(W_i l_{i-1} + b_i), \quad i = 2, \ldots, N-1$$

$$y = f(W_N l_{N-1} + b_N)$$

其中 $f(\cdot) = \tanh(\cdot)$，$x$ 是 Word Hashing 后的输入向量，$y$ 是最终的 128 维语义向量。Query 塔和 Document 塔使用相同的网络结构但参数独立。

**模块3：相似度计算与训练**

Query 向量 $y_Q$ 和 Document 向量 $y_D$ 之间的语义相关性通过余弦相似度计算：

$$R(Q, D) = \cos(y_Q, y_D) = \frac{y_Q^T \cdot y_D}{\|y_Q\| \|y_D\|}$$

训练时，通过 softmax 将相似度转换为后验概率：

$$P(D^+ | Q) = \frac{\exp(\gamma \cdot R(Q, D^+))}{\sum_{D' \in \mathbf{D}} \exp(\gamma \cdot R(Q, D'))}$$

其中 $\gamma$ 为平滑因子（温度系数），$\mathbf{D} = \{D^+\} \cup \{D_1^-, D_2^-, D_3^-, D_4^-\}$。每个 query 对应 1 个正样本（用户点击文档）和 4 个随机负采样文档。损失函数为负对数似然：

$$L(\Lambda) = -\log \prod_{(Q, D^+)} P(D^+ | Q)$$

### 2.3 关键设计选择

论文在实验中对比了多种模型变体，发现以下设计对效果至关重要：DNN 结构显著优于线性投影（Linear Projection）；多隐层优于单隐层，但层数不宜过深（3 层隐层效果最佳）；Word Hashing 是使模型能处理大规模词汇的关键。

## 三、实验结果

### 3.1 数据集

论文使用了一个真实的 Web 搜索数据集，来自搜索引擎日志。训练集包含约 10 万个 query，每个 query 关联用户点击文档（正样本）和随机采样的未点击文档（负样本）。测试集同样来自搜索日志，用 NDCG 作为评价指标。

### 3.2 实验设置

#### 3.2.1 基线方法

对比的基线包括：TF-IDF（关键词匹配基线）、BM25（经典检索模型）、LSA（潜在语义分析）、PLSA（概率潜在语义分析）、DAE（深度自编码器）、DPMs（判别投影模型，使用 S2Net 算法和 pairwise 排序损失），以及 DSSM 的线性变体（Linear Projection）。

#### 3.2.2 评估指标

使用 NDCG@1, NDCG@3, NDCG@10 评估文档排序质量。NDCG（Normalized Discounted Cumulative Gain）衡量排序结果的质量，考虑了位置折扣和相关性等级。

### 3.3 实验结果与分析

| 方法 | NDCG@1 | NDCG@3 | NDCG@10 |
|------|--------|--------|---------|
| TF-IDF | 0.304 | 0.325 | 0.390 |
| BM25 | 0.305 | 0.328 | 0.393 |
| LSA | 0.298 | 0.321 | 0.387 |
| PLSA | 0.304 | 0.325 | 0.389 |
| DAE | 0.311 | 0.334 | 0.397 |
| DPMs (Linear) | 0.318 | 0.340 | 0.403 |
| DSSM (Linear) | 0.322 | 0.344 | 0.406 |
| **DSSM (DNN)** | **0.334** | **0.353** | **0.415** |

> 注：数据为论文中的近似值，DSSM (DNN) 在所有指标上均取得最优结果。

#### 结果分析

DSSM (DNN) 相较所有基线均有显著提升。与最强基线 DPMs 相比，NDCG@1 提升约 5%。深度结构（DNN）相比线性投影带来了显著增益，证明了非线性语义映射的价值。传统无监督方法（LSA、PLSA）甚至不如关键词匹配（TF-IDF/BM25），说明无监督优化目标确实与检索任务不对齐。

### 消融实验

论文对比了不同隐层数量的效果：从 1 层到 4 层，效果逐渐提升，但在 3 层后趋于饱和。同时验证了 Word Hashing 的 n-gram 粒度，trigram（n=3）在词汇压缩率和冲突率之间取得了最佳平衡。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文提到可以探索更复杂的网络结构（如卷积网络）来替代 MLP，以更好地捕捉词序信息。同时可以将 DSSM 框架扩展到其他信息检索任务，如问答系统、机器翻译等。

### 4.2 基于分析的未来方向

1. **方向1：序列感知的编码器**
   - 动机：Bag-of-Words 丢失了词序信息
   - 可能的方法：使用 CNN（如后续的 C-DSSM）或 LSTM 替代 MLP
   - 预期成果：更好地捕捉短语级语义
   - 挑战：在线推理延迟增加

2. **方向2：负采样策略优化**
   - 动机：随机负采样可能过于简单，模型难以学习细粒度区分
   - 可能的方法：引入 hard negative mining
   - 预期成果：提升模型对相似但不相关文档的区分能力
   - 挑战：hard negative 的选择需要平衡难度和信息量

### 4.3 改进建议

1. **改进1：引入注意力机制**
   - 当前问题：MLP 对所有输入特征等权处理
   - 改进方案：加入 self-attention 或 cross-attention
   - 预期效果：更好地聚焦关键语义信息

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - DSSM 是双塔模型的奠基之作，其提出的"双塔编码 + 语义空间匹配"范式至今仍是工业级召回系统的主流架构。虽然模型结构在今天看来非常简单，但其思想的影响力是划时代的。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次提出双塔深度语义匹配范式，Word Hashing 是巧妙的工程创新 |
| 技术质量 | 7/10 | 方法论清晰，但 MLP 结构相对简单，未考虑词序信息 |
| 实验充分性 | 7/10 | 基线对比全面，但仅在单一数据集上验证，缺少跨场景泛化实验 |
| 写作质量 | 8/10 | 结构清晰，逻辑流畅，Word Hashing 部分讲解详细 |
| 实用性 | 10/10 | 双塔范式被工业界广泛采用，至今仍是主流架构，影响力巨大 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

Word Hashing 技术虽然简单，但非常实用，解决了大词汇表输入问题的同时还天然支持 OOV 和拼写变体处理。双塔解耦设计使得物品侧向量可以离线预计算并建立 ANN 索引，这一工程优势是双塔模型能在工业界广泛落地的关键。

#### 5.2.2 需要深入理解的部分

负采样策略（随机采样 4 个负样本）对模型效果的影响值得深思。后续工作（如 EBR）证明负采样是双塔模型效果的核心因素。平滑因子 $\gamma$ 的设置也值得关注，它本质上是温度系数，控制了 softmax 的锐度。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[YouTube_DNN|YouTube DNN]] - 将 DSSM 双塔思想扩展到推荐场景，是召回模型工业化的里程碑
- [[EBR|EBR]] - Facebook 对双塔召回的系统性工程化实践，深入讨论了 DSSM 框架的工程优化

### 6.2 背景相关
- LSA (1990) - 最早的潜在语义分析方法
- Word2Vec (2013) - 同期的词向量学习方法，后来催生了 Item2Vec

### 6.3 后续工作
- C-DSSM (2014) - 用 CNN 替代 MLP，引入词序信息
- [[Item2Vec|Item2Vec]] - 将语义匹配思想迁移到推荐场景

## 外部资源

- [Microsoft DSSM 项目主页](https://www.microsoft.com/en-us/research/project/dssm/)
- [DSSM 论文解读（博客园）](https://www.cnblogs.com/foghorn/p/15626153.html)

> [!tip] 关键启示
> DSSM 的核心贡献不仅在于模型本身，更在于确立了"双塔编码 + 语义空间匹配"这一通用范式。这个范式的核心优势——query/user 侧在线计算、doc/item 侧离线预计算 + ANN 检索——至今仍是工业级召回系统的黄金架构。

> [!warning] 注意事项
> - Bag-of-Words 丢失了词序信息，对短 query 影响较大
> - 随机负采样可能导致模型学到的是"区分完全不相关的文档"而非"区分相关但不匹配的文档"
> - 双塔解耦结构限制了 query 和 document 之间的交互，无法建模细粒度的语义匹配

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是双塔模型的奠基论文，理解 DSSM 是理解整个召回模型演进脉络的起点。
