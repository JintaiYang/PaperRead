---
title: "Recommender Systems with Generative Retrieval"
short_name: "TIGER"
year: 2023
venue: "NeurIPS 2023"
authors: "Shashank Rajput, Nikhil Mehta, Anima Singh, et al."
affiliation: "Google"
direction: "生成式召回"
tags:
  - 召回论文
  - TIGER
  - 生成式召回
  - Semantic ID
  - RQ-VAE
  - 论文笔记
paper_info: "[[TIGER]]"
quality_score: "9/10"
---

# TIGER: Recommender Systems with Generative Retrieval

> **Shashank Rajput, Nikhil Mehta, Anima Singh, et al.** | Google | NeurIPS 2023

## 一、研究背景与动机

### 1.1 领域现状

推荐系统的召回阶段主流范式是 embedding + ANN：将用户和物品编码为连续向量，通过近似最近邻检索获取候选。这一范式自 DSSM (2013) 以来主导了十年，期间的改进主要集中在 embedding 质量（双塔精细化、多兴趣等）和 ANN 效率。

### 1.2 现有方法的局限性

Embedding + ANN 存在本质限制：ANN 的检索结果是近似的，检索过程与模型训练解耦（embedding 空间和 ANN 索引分开优化），物品的 ID 是原子化的随机编号（没有语义信息）。受 NLP 领域生成式检索（Generative Retrieval）的启发，是否可以让推荐模型直接"生成"目标物品的标识符？

### 1.3 本文解决方案概述

TIGER（Transformer Index for GEnerative Recommenders）提出了推荐系统的生成式召回范式：为每个物品分配有语义含义的 Semantic ID（通过 RQ-VAE 量化物品内容特征获得），然后训练一个 Seq-to-Seq Transformer 自回归生成用户下一个交互物品的 Semantic ID。

## 二、解决方案

### 2.1 核心思想

将推荐召回重新定义为"序列到序列的生成任务"：输入是用户历史交互物品的 Semantic ID 序列，输出是下一个推荐物品的 Semantic ID。

### 2.2 Semantic ID 生成

**RQ-VAE（Residual-Quantized Variational Autoencoder）**：

1. 用预训练的内容编码器（如 BERT）将物品的文本/图像特征编码为连续向量
2. 用 RQ-VAE 将连续向量量化为离散码字序列 $(c_1, c_2, \ldots, c_m)$
3. 每个 $c_i$ 来自不同的 codebook，组合起来就是物品的 Semantic ID

RQ（Residual Quantization）的优势：层次化编码。$c_1$ 捕获粗粒度语义（品类），$c_2$ 捕获中粒度语义（子类），$c_3$ 捕获细粒度语义（具体属性）。语义相似的物品会共享前缀码字。

### 2.3 生成式检索模型

Encoder-Decoder Transformer（类似 T5 架构）：

**Encoder**：输入用户历史交互物品的 Semantic ID 序列（将多个物品的码字 flatten 为一个长序列），编码用户兴趣。

**Decoder**：自回归生成下一个物品的 Semantic ID $(c_1, c_2, \ldots, c_m)$，每步预测一个码字。

**Beam Search**：serving 时用 beam search 生成多个候选 Semantic ID，映射回物品。

### 2.4 训练

Teacher forcing + cross-entropy loss。在每个码字位置上计算分类损失。

## 三、实验结果

### 3.1 数据集

Amazon Beauty、Amazon Sports、Amazon Toys。

### 3.2 实验结果与分析

TIGER 在 Recall@10/NDCG@10 上全面超越双塔 + ANN 方案和序列推荐方案（SASRec、BERT4Rec），提升约 +10-25%。Semantic ID 比随机 ID 效果好约 +15%，验证了语义量化的价值。RQ-VAE 的 codebook 层数和大小对效果有显著影响。TIGER 在冷启动物品上表现尤其好（Semantic ID 天然包含内容语义信息）。

## 四、未来工作建议

Semantic ID 的生成方式（RQ-VAE vs. 其他量化方法）、更高效的生成式检索（减少 beam search 开销）、融合协同过滤信号到 Semantic ID 中。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9/10** - TIGER 是推荐系统生成式召回方向的奠基论文，Semantic ID + 自回归生成的范式极具创新性和影响力。它开启了一个全新的研究方向。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 10/10 | 开创推荐领域的生成式召回范式，Semantic ID 概念极具创新性 |
| 技术质量 | 8/10 | RQ-VAE + Transformer 框架清晰，但有些设计选择缺乏深入分析 |
| 实验充分性 | 7/10 | 公开数据集效果好，但缺乏工业级验证 |
| 写作质量 | 9/10 | 论文写作清晰，概念阐述到位 |
| 实用性 | 8/10 | 启发了大量后续工作，但自身的工业落地案例有限 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 方法来源
- DSI (Differentiable Search Index, Tay et al., 2022) - NLP 领域的生成式检索
- RQ-VAE (Zeghidour et al., 2021) - 残差量化变分自编码器

### 6.2 同方向后续
- [[HSTU|HSTU (2024)]] - Meta 的万亿参数生成式推荐，验证 scaling law
- [[OneRec|OneRec (2025)]] - 快手的生成式推荐，首次在工业界超越级联架构
- [[MTGR|MTGR (2025)]] - 美团的生成式推荐，融合 HSTU 和 DLRM

### 6.3 前驱概念
- [[TDM|TDM (2018)]] - 树索引召回，结构化检索的早期尝试
- [[Deep_Retrieval|Deep Retrieval (2020)]] - 路径索引，离散化检索的另一种方案

## 外部资源

- [arXiv](https://arxiv.org/abs/2305.05065)
- [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)
- [GitHub](https://github.com/baiyimeng/TIGER)
- [知乎解读](https://zhuanlan.zhihu.com/p/1938973235350861588)

> [!tip] 关键启示
> TIGER 的核心贡献是将推荐召回从"检索"范式转向"生成"范式。Semantic ID 赋予物品有语义结构的离散标识，使得 Transformer 可以像生成自然语言一样"生成"推荐结果。这一范式转换可能是推荐系统继 embedding + ANN 之后的下一个重大演进方向。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。生成式召回的奠基之作，开启全新范式。
