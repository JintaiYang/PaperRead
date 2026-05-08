---
title: "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall"
short_name: "MIND"
year: 2019
venue: "CIKM 2019"
authors: "Chao Li, Zhiyuan Liu, Mengmeng Wu, et al."
affiliation: "阿里巴巴"
direction: "多兴趣召回"
tags:
  - 召回论文
  - MIND
  - 多兴趣
  - 胶囊网络
  - 动态路由
  - 论文笔记
paper_info: "[[MIND]]"
quality_score: "8.5/10"
---

# MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

> **Chao Li, Zhiyuan Liu, Mengmeng Wu, et al.** | 阿里巴巴 | CIKM 2019

## 一、研究背景与动机

### 1.1 领域现状

双塔模型（如 YouTube DNN、DSSM）是工业召回的主流方案，但它们将用户表示为单一向量（single vector）。对于兴趣多样化的用户（如同时喜欢运动、电子产品、书籍的用户），单一向量难以准确刻画所有兴趣方向，导致召回结果倾向于用户的"平均"兴趣。

### 1.2 现有方法的局限性

YouTube DNN 用 average pooling 聚合行为序列为一个向量，本质上是对所有行为的"均值"表示。DIN 虽然引入注意力机制区分不同行为的重要性，但需要 target item 作为 query，是精排模型，无法用于召回阶段。问题的根源在于：一个向量无法表达用户多面的兴趣。

### 1.3 本文解决方案概述

MIND 提出用多个兴趣向量表示用户，每个向量刻画一个兴趣方向。核心技术是借鉴胶囊网络的动态路由（Dynamic Routing）机制，将行为序列自动聚类为 K 个兴趣簇，每个簇生成一个兴趣向量。同时提出 label-aware attention 训练技巧，使多兴趣向量在训练时能高效学习。

## 二、解决方案

### 2.1 核心思想

一个用户 → K 个兴趣向量。每个兴趣向量对应用户的一个兴趣维度。检索时，每个兴趣向量独立做 ANN 检索，最终合并 Top-N 结果。

### 2.2 整体架构

**Multi-Interest Extractor Layer（多兴趣提取层）**：

输入用户行为序列的 item embedding $(e_1, e_2, \ldots, e_n)$，通过 B2I（Behavior-to-Interest）动态路由生成 K 个兴趣胶囊向量 $(\mu_1, \mu_2, \ldots, \mu_K)$。

动态路由过程：
1. 初始化路由系数 $b_{ij} = 0$
2. 迭代 T 次（通常 T=3）：
   - $c_{ij} = \text{softmax}(b_{ij})$（对每个行为 $i$，计算其分配到各兴趣 $j$ 的概率）
   - $s_j = \sum_i c_{ij} \cdot W_j e_i$（加权聚合）
   - $\mu_j = \text{squash}(s_j)$（squash 激活）
   - $b_{ij} \leftarrow b_{ij} + \mu_j^T W_j e_i$（更新路由系数）

直觉理解：相似的行为会被路由到同一个兴趣胶囊，不同类型的行为被分到不同胶囊，从而实现行为的自动聚类。

**Label-Aware Attention**：训练时有 target item，用 target item 的 embedding 对 K 个兴趣向量做 attention，选出与 target 最相关的兴趣向量来计算 loss。这比随机选一个或平均效果好得多。

**物品塔**：标准 embedding lookup。

**Serving**：每个兴趣向量独立做 ANN 检索，取各路 top-N 合并去重。

### 2.3 训练

Sampled softmax loss。Label-aware attention 确保训练梯度只回传到最相关的兴趣向量。

## 三、实验结果

### 3.1 数据集

Amazon Books 和淘宝天猫工业数据集。评估指标：Recall@N、HitRate@N。

### 3.2 实验结果与分析

在 Amazon Books 上 hitrate@50 相比 YouTube DNN 提升 ~15%，相比 SDM 提升 ~8%。K 的选择：K=4~8 在大多场景最优，K 太大反而过拟合。动态路由 vs. Self-Attention 聚类：动态路由略优。在天猫线上 A/B 测试中，CTR 提升显著（具体数值未公开），已部署于天猫移动端首页推荐。

## 四、未来工作建议

作者提出可探索的方向：动态确定 K 值（根据用户行为丰富度自适应选择兴趣数量），以及将多兴趣建模推广到精排阶段。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - MIND 开创了"多兴趣召回"方向，是推荐系统召回领域的里程碑论文。胶囊网络动态路由的引入既有理论美感，又有工程实用价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次提出多兴趣向量表示用户，开创全新召回范式 |
| 技术质量 | 8/10 | 动态路由机制设计巧妙，label-aware attention 实用 |
| 实验充分性 | 8/10 | 公开数据集 + 工业数据集 + 在线 A/B |
| 写作质量 | 8/10 | 结构清晰，动机阐述充分 |
| 实用性 | 9/10 | 已在天猫部署，后续引发大量跟进工作（ComiRec、SINE 等） |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- [[YouTube_DNN|YouTube DNN (2016)]] - 单向量用户表示的代表，MIND 解决其局限性
- DIN (2018) - target-aware attention 思想的来源

### 6.2 同方向后续
- [[ComiRec|ComiRec (2020)]] - MIND 的改进，增加可控性和多样性调节
- [[SINE|SINE (2021)]] - 稀疏兴趣网络，解决 MIND 兴趣数固定的问题

### 6.3 方法来源
- Sabour et al. "Dynamic Routing Between Capsules" (2017) - 胶囊网络原始论文

## 外部资源

- [arXiv](https://arxiv.org/abs/1904.08030)
- [知乎解读](https://zhuanlan.zhihu.com/p/262638999)

> [!tip] 关键启示
> "用多个向量表示用户"这一思路看似简单，但它彻底改变了召回系统的用户表示范式。后续的 ComiRec、SINE、PinnerFormer 等都沿用了这一思路。在工程上，多向量检索带来的索引开销是可接受的（K 通常只有 4~8）。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。开创多兴趣召回方向，影响深远。
