---
title: "Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations"
short_name: "Deep Retrieval"
year: 2020
venue: "arXiv 2020"
authors: "Weihao Gao, Xiangjun Fan, Jiankai Sun, et al."
affiliation: "Google + 字节跳动"
direction: "树索引召回"
tags:
  - 召回论文
  - Deep Retrieval
  - 路径索引
  - EM算法
  - 论文笔记
paper_info: "[[Deep_Retrieval]]"
quality_score: "8/10"
---

# Deep Retrieval: An End-to-End Learnable Structure Model for Large-Scale Recommendations

> **Weihao Gao, Xiangjun Fan, Jiankai Sun, et al.** | Google + 字节跳动 | arXiv 2020

## 一、研究背景与动机

### 1.1 领域现状

大规模推荐系统的召回阶段主要有两种范式：embedding + ANN（如双塔 + Faiss）和 tree-based（如 TDM）。Embedding + ANN 方案成熟但受限于 ANN 的近似性和 embedding 空间的表达能力；TDM 需要预定义树结构，树的质量直接影响效果。

### 1.2 现有方法的局限性

TDM 的核心限制是"一个物品只能在树的一个叶节点"，这意味着一个物品只有一条检索路径。但现实中，一个商品可能属于多个类别、适合多种用户场景，强制映射到一个叶节点会丢失信息。此外，TDM 的树结构需要预训练或启发式构建，不能端到端学习。

### 1.3 本文解决方案概述

Deep Retrieval（DR）提出用 K 路 D 层的路径结构（而非树结构）索引物品。每个物品可以被分配到多条路径，突破了 TDM 的"一物品一路径"限制。路径-物品分配通过 EM 算法端到端学习。

## 二、解决方案

### 2.1 核心思想

将物品索引建模为 D 层，每层有 K 个节点的路径结构。一个物品由一条长度为 D 的路径 $(c_1, c_2, \ldots, c_D)$ 表示，其中 $c_d \in \{1, \ldots, K\}$。整个索引空间大小为 $K^D$（远大于物品数量），每个物品可被分配到 J 条路径（多路径）。

### 2.2 模型结构

**路径生成模型**：给定用户特征 $x$，用一个 DNN 逐层预测路径中每层的节点：

$$P(c_1, c_2, \ldots, c_D | x) = \prod_{d=1}^{D} P(c_d | c_1, \ldots, c_{d-1}, x)$$

每层的条件概率通过 softmax 计算。这本质上是一个自回归的离散选择模型。

**路径-物品映射**：一个离线表，记录每条路径对应的物品集合。

### 2.3 训练：EM 算法

**E 步**：固定路径生成模型，更新物品到路径的分配。对每个物品，选择使其被检索概率最大的 J 条路径。

**M 步**：固定路径-物品分配，训练路径生成模型。对于正样本 (user, item)，该 item 对应的所有路径都作为正例，训练模型生成这些路径。

E/M 交替迭代直到收敛。

### 2.4 Serving

给定用户 $x$，用 beam search 在路径结构上找到 top-B 条路径，查表得到候选物品集合。

## 三、实验结果

### 3.1 数据集

YouTube 推荐系统的真实数据。公开数据集：MovieLens-20M。

### 3.2 实验结果与分析

在 YouTube 数据上，Deep Retrieval 召回质量与双塔 + ANN 方案持平或略优，但提供了不同类型的召回结果（两路互补）。多路径（J>1）比单路径（J=1）效果显著提升约 +10%，验证了"一物品多路径"的价值。在线部署后，与双塔方案融合使用效果最佳。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8/10** - Deep Retrieval 是对 TDM 的重要改进，"多路径 + EM 学习"的思路新颖。在 YouTube 的实际部署验证了方法的工业可行性。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 路径结构替代树结构，多路径 + EM 学习是新颖的设计 |
| 技术质量 | 8/10 | EM 算法框架严谨，beam search 检索方案实用 |
| 实验充分性 | 7/10 | YouTube 部署可信，但公开数据集实验有限 |
| 写作质量 | 7/10 | 结构清晰但部分细节不够详细 |
| 实用性 | 8/10 | 在 YouTube 落地，证明工业可行 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- [[TDM|TDM (2018)]] - 树索引召回的开创者，Deep Retrieval 解决其限制

### 6.2 方法联系
- Beam Search - Serving 时的检索算法
- EM Algorithm - 路径分配的优化框架

### 6.3 互补方案
- [[EBR|EBR (2020)]] - Embedding + ANN 的代表，与 DR 互补使用

## 外部资源

- [arXiv](https://arxiv.org/abs/2007.07203)

> [!tip] 关键启示
> Deep Retrieval 展示了结构化索引在大规模召回中的潜力。与 embedding + ANN 不同，结构化索引将检索过程建模为离散选择序列，提供了另一种视角。实践中，DR 和双塔通常互补使用——两路召回的物品重叠率不高，融合后效果更好。

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。理解结构化索引召回的新范式。
