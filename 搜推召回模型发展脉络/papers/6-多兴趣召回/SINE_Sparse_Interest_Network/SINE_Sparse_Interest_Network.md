---
title: "Sparse-Interest Network for Sequential Recommendation"
short_name: "SINE"
year: 2021
venue: "WSDM 2021"
authors: "Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, et al."
affiliation: "阿里巴巴"
direction: "多兴趣召回"
tags:
  - 召回论文
  - SINE
  - 稀疏兴趣
  - 多兴趣
  - Concept Routing
  - 论文笔记
paper_info: "[[SINE]]"
quality_score: "7.5/10"
---

# SINE: Sparse-Interest Network for Sequential Recommendation

> **Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, et al.** | 阿里巴巴 | WSDM 2021

## 一、研究背景与动机

### 1.1 领域现状

MIND 和 ComiRec 开创了多兴趣召回方向，分别用胶囊路由和 Self-Attention 从行为序列中提取 K 个兴趣向量。但这些方法有两个共同问题：K 值固定（所有用户都产生相同数量的兴趣向量），且兴趣是直接从行为序列"提取"的，缺乏全局的兴趣语义先验。

### 1.2 现有方法的局限性

MIND/ComiRec 的兴趣提取完全依赖当前 session 的行为序列，无法利用全局的兴趣分布信息。此外，固定的 K 值不合理：行为丰富的用户可能需要更多兴趣向量，行为稀疏的用户可能只需要 1-2 个。K 过大会导致稀疏用户产生冗余的噪声兴趣向量。

### 1.3 本文解决方案概述

SINE 提出"稀疏兴趣网络"：预定义一个大的全局兴趣概念池（Concept Pool，规模 $N \gg K$），对每个用户，从概念池中稀疏地激活 $K$ 个最相关的概念，然后基于这些激活的概念生成兴趣向量。关键创新是引入 Concept Routing 机制实现稀疏激活。

## 二、解决方案

### 2.1 核心思想

用全局共享的概念池（可学习的 embedding 矩阵，每行代表一个兴趣概念）提供兴趣的语义先验。每个用户只激活其中少量概念（稀疏），避免冗余兴趣表示。

### 2.2 整体架构

**Concept Pool**：$C \in \mathbb{R}^{N \times d}$，N 个全局兴趣概念向量，通过训练学习。$N$ 通常取几百到几千。

**Sparse Interest Extraction**：
1. 对用户行为序列 $H = (e_1, \ldots, e_n)$，计算每个行为与所有概念的注意力分数
2. 对每个概念，聚合相关行为得到一个分数，选择 top-K 个分数最高的概念（稀疏激活）
3. 对每个激活的概念，用其对应的注意力权重聚合行为序列，生成兴趣向量

**Intent Prediction**：为了在 serving 时从 K 个兴趣向量中选择最终的用户表示（没有 target item），SINE 训练一个 intent predictor 预测用户下一步最可能的兴趣方向，用预测结果加权 K 个兴趣向量。

### 2.3 训练

Sampled softmax loss。Concept pool 和 intent predictor 端到端训练。

## 三、实验结果

### 3.1 数据集

Amazon Books、Amazon CDs、MovieLens-1M、Taobao。

### 3.2 实验结果与分析

SINE 在 Recall@20 上相比 MIND 提升约 +3-5%，相比 ComiRec 提升约 +2-3%。概念池大小 N=500-1000 时效果最佳。稀疏激活比全量激活效果好，验证了稀疏性的价值。Intent predictor 比 label-aware attention 在线上 serving 时更实用（不需要 target item）。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - SINE 是多兴趣召回方向的合理改进，概念池和稀疏激活的思路有价值，但整体创新增量相对有限。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 概念池 + 稀疏激活是有意义的改进 |
| 技术质量 | 8/10 | 框架设计完整，intent predictor 实用 |
| 实验充分性 | 8/10 | 多数据集对比充分 |
| 写作质量 | 7/10 | 结构清晰 |
| 实用性 | 7/10 | 概念池增加了额外的存储和计算开销 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- [[MIND|MIND (2019)]] - 多兴趣召回开创者
- [[ComiRec|ComiRec (2020)]] - Self-Attention 多兴趣方案

### 6.2 技术来源
- Mixture of Experts (MoE) - 概念池 + 稀疏激活的思想与 MoE 类似

## 外部资源

- [arXiv](https://arxiv.org/abs/2102.09267)
- [GitHub](https://github.com/qiaoyu-tan/SINE)
- [知乎解读](https://zhuanlan.zhihu.com/p/658272630)

> [!tip] 关键启示
> SINE 的概念池思想类似于 Mixture of Experts 中的"专家池"：全局共享的专家/概念，每次只稀疏激活少量。这种"大池子 + 稀疏选择"的模式在后续的大模型（如 MoE-based LLM）中得到了更广泛的应用。

> [!success] 推荐指数
> ⭐⭐⭐ 选读。多兴趣方向的改进，了解稀疏激活思路。
