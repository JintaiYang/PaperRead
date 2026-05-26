---
paper_id: "[CIKM 2019](https://dl.acm.org/doi/10.1145/3357384.3357818)"
title: "SDM: Sequential Deep Matching Model for Online Large-scale Recommender System"
authors: "Fuyu Lv, Taiwei Jin, Changlong Yu, et al."
institution: "阿里巴巴"
pushlication: "CIKM 2019"
tags:
  - 召回论文
  - SDM
  - 序列建模
  - 双塔
  - LSTM
  - Multi-head-Attention
quality_score: "7.5/10"
link:
  - "[PDF](https://dl.acm.org/doi/pdf/10.1145/3357384.3357818)"
  - "[ACM](https://dl.acm.org/doi/10.1145/3357384.3357818)"
date: "2019-11-03"
paper_info: "[[SDM]]"
---

## 一、研究背景与动机

### 1.1 领域现状

YouTube DNN 确立了双塔召回的基本框架，但其用户塔使用简单的 average pooling 处理行为序列，丢失了丰富的时序信息。用户的兴趣具有时间动态性——短期兴趣（session 内浏览行为）和长期兴趣（历史偏好）对下一次交互有不同的影响。

### 1.2 现有方法的局限性

YouTube DNN 的 average pooling 将所有历史行为等权平均，无法区分近期行为和远期行为的重要性差异。DIN 引入了注意力机制，但它是精排模型（需要 target item 触发注意力），不适用于召回阶段（无法预知 target item）。

### 1.3 本文解决方案概述

SDM 在双塔框架的用户塔中引入序列建模：用 LSTM 编码 session 内的短期行为序列，用 Multi-head Self-Attention 聚合多个 session 的长期偏好，最终融合短期和长期兴趣生成用户 embedding。

## 二、解决方案

### 2.1 核心思想

将用户兴趣分解为短期意图（当前 session 的行为模式）和长期偏好（历史跨 session 的稳定偏好），分别用不同的网络结构建模，再通过门控机制融合。

### 2.2 整体架构

**短期兴趣建模**：当前 session 的行为序列 $(e_1, e_2, \ldots, e_t)$ 输入 LSTM，取最后一个隐状态 $h_t$ 作为短期兴趣向量。再通过 Attention 对 LSTM 各步输出加权聚合，得到更精细的短期向量。

**长期兴趣建模**：提取最近 $M$ 个 session 的 session embedding（每个 session 通过 average pooling 得到一个向量），使用 Multi-head Self-Attention 聚合为长期兴趣向量 $p_u$。

**兴趣融合**：通过门控网络融合短期兴趣 $s_u$ 和长期兴趣 $p_u$：

$$o_u = \text{Gate}(s_u, p_u) = g \odot s_u + (1-g) \odot p_u$$

其中 $g = \sigma(W_g [s_u; p_u; s_u - p_u; s_u \odot p_u])$ 是门控向量。

**物品塔**：标准的 embedding + DNN 物品编码器。

**训练**：与 YouTube DNN 类似，使用 sampled softmax 训练。Serving 时用户塔输出的融合向量做 ANN 检索。

## 三、实验结果

### 3.1 数据集

淘宝生产环境数据。离线评估使用 hitrate@50 和 NDCG@50，在线通过 A/B 测试评估。

### 3.2 实验结果与分析

在离线评估中，SDM 相比 YouTube DNN hitrate@50 提升约 +8%，相比纯 LSTM 方法提升约 +4%。融合短期和长期兴趣比仅使用其中一个效果好约 +5%。在线 A/B 测试中，CTR 和 GMV 均有显著提升。

门控权重分析显示：在用户处于活跃浏览状态时，短期兴趣权重更高；在用户行为稀疏时，长期兴趣权重更高。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - SDM 是"序列增强双塔"方向的代表作，系统性地将 LSTM 和 Attention 引入召回的用户塔，展示了序列建模对召回效果的价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 长短期兴趣融合思路合理，但各组件（LSTM、Attention、Gate）均非新提出 |
| 技术质量 | 8/10 | 框架设计完整，门控融合机制设计合理 |
| 实验充分性 | 7/10 | 有在线 A/B 测试，但公开数据集验证有限 |
| 写作质量 | 7/10 | 结构清晰但部分细节不够详细 |
| 实用性 | 8/10 | 直接可用于工业召回系统的用户塔升级 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[YouTube_DNN|YouTube DNN]] - SDM 的基础框架，SDM 改进了其用户塔
- [[MIND|MIND]] - 同样改进用户表示，但方向不同（多兴趣 vs 序列建模）

### 6.2 后续工作
- SASRec (2018) - 用纯 Self-Attention 做序列推荐
- BERT4Rec (2019) - 双向 Transformer 做序列推荐

## 外部资源

- [ACM Digital Library](https://dl.acm.org/doi/10.1145/3357384.3357818)

> [!tip] 关键启示
> SDM 展示了在双塔框架内增强用户表示的一种通用思路：分别建模不同时间尺度的兴趣，再通过门控融合。这一框架可以灵活替换各组件（LSTM→Transformer，Gate→Attention 等）。

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。理解序列增强双塔的典型方案。
