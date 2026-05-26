---
title: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"
short_name: "OneRec"
year: 2025
venue: "arXiv 2025"
authors: "Jiaxin Deng, Shiyao Wang, Yuchen Jiang, et al."
affiliation: "快手"
direction: "生成式召回"
tags:
  - 召回论文
  - OneRec
  - 生成式推荐
  - 统一召排
  - MoE
  - DPO
  - 论文笔记
paper_info: "[[OneRec]]"
quality_score: "9/10"
---

# OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment

> **Jiaxin Deng, Shiyao Wang, Yuchen Jiang, et al.** | 快手 | arXiv 2025

## 一、研究背景与动机

### 1.1 领域现状

传统推荐系统采用多阶段级联架构：召回 → 粗排 → 精排 → 重排。每个阶段独立优化，存在信息损失和目标不一致的问题。HSTU (Meta, 2024) 证明了生成式推荐的可行性和 Scaling Law，但 HSTU 主要聚焦于模型架构设计，缺乏对整体推荐流程统一化的完整方案。

### 1.2 现有方法的局限性

级联架构的核心问题：1）召回和排序目标不一致（召回优化覆盖率，排序优化 CTR/GMV）；2）前一阶段的错误会传递到后续阶段（召回漏掉的好物品排序阶段无法弥补）；3）多阶段维护的工程复杂度高。HSTU 虽然提出了统一架构，但在工业实践中尚未完全超越精心设计的级联系统。

### 1.3 本文解决方案概述

OneRec 是首个在工业界全面超越级联推荐架构的端到端生成式推荐模型。核心创新包括：Encoder-Decoder 架构 + Sparse MoE 扩展模型容量、Session-wise 生成替代逐点预测、以及从 LLM 领域引入的 Iterative Preference Alignment（IPA，类似 DPO）。

## 二、解决方案

### 2.1 核心思想

将整个推荐流程（召回+排序）统一为一个 Seq-to-Seq 生成任务。Encoder 编码用户历史行为，Decoder 直接生成一个 session 的推荐结果序列。

### 2.2 Encoder-Decoder 架构

**Encoder**：处理用户历史行为序列，采用 Sparse MoE 层扩展模型容量。MoE 使得模型可以在不成比例增加计算量的情况下大幅增加参数量。OneRec-1B 的 MoE 配置：8 个专家，每次激活 2 个。

**Decoder**：自回归地生成一个 session 的推荐物品序列。与 TIGER 不同，OneRec 不生成 Semantic ID，而是直接预测物品 ID（通过 item embedding 的最近邻查找）。

### 2.3 Session-wise Generation

传统方法逐点预测（对每个候选物品独立打分），OneRec 以 session 为单位生成推荐结果。这使得模型可以隐式地考虑推荐结果的整体质量（多样性、连贯性等）。

### 2.4 Iterative Preference Alignment (IPA)

借鉴 LLM 中的 DPO/RLHF：
1. 先用预训练的 OneRec 通过 beam search 生成多组推荐结果
2. 用一个 Reward Model（RM）对这些结果打分
3. 选得分最高和最低的结果构建偏好对
4. 用 DPO 损失微调模型
5. 迭代上述过程

IPA 使得模型从"预测用户会点什么"进化到"生成用户最满意的推荐列表"。

## 三、实验结果

### 3.1 数据集

快手 App 全量在线数据。

### 3.2 实验结果与分析

在快手 App 的在线 A/B 测试中：OneRec-1B + IPA 使总观看时长提升 +1.68%，平均观看时长提升 +6.56%。这是首次端到端生成式模型在大规模工业场景中全面超越精心设计的级联系统。模型运营成本降低至原系统的 10.6%（统一架构省去了多阶段的独立维护）。MoE 的效果：1B 参数 MoE 模型优于 300M 的 dense 模型，验证了 MoE 在推荐中的价值。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9/10** - OneRec 是生成式推荐方向的工业化里程碑。它首次在大规模工业场景中证明了端到端生成式推荐可以全面超越级联架构，IPA 是将 LLM 对齐技术迁移到推荐系统的开创性探索。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | Session-wise 生成 + IPA 偏好对齐是重要创新 |
| 技术质量 | 9/10 | MoE 扩展、IPA 训练流程设计严谨 |
| 实验充分性 | 9/10 | 快手 App 大规模在线验证，效果数据详实 |
| 写作质量 | 8/10 | 结构清晰，但部分实验细节不够详细 |
| 实用性 | 10/10 | 已在快手全量部署，运营成本大幅降低 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- [[HSTU|HSTU (2024)]] - 生成式推荐架构 + Scaling Law 的先驱
- [[TIGER|TIGER (2023)]] - Semantic ID + 生成式召回的学术探索

### 6.2 技术来源
- MoE (Mixture of Experts) - 稀疏专家模型扩展容量
- DPO (Rafailov et al., 2023) - Direct Preference Optimization，IPA 的技术基础

### 6.3 同方向
- [[MTGR|MTGR (2025)]] - 美团的生成式推荐方案

## 外部资源

- [arXiv](https://arxiv.org/abs/2502.18965)
- [知乎解读](https://zhuanlan.zhihu.com/p/1984017206233817839)
- [腾讯云解读](https://cloud.tencent.com/developer/article/2532909)

> [!tip] 关键启示
> OneRec 最大的意义是证明了"端到端生成式推荐可以取代级联架构"这一猜想。IPA（偏好对齐）的引入更是将推荐系统的优化目标从"预测用户行为"提升到"生成用户满意的推荐结果"——这是一个根本性的范式转变。运营成本降低至 10.6% 也说明统一架构在工程效率上的巨大优势。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。工业级生成式推荐的标杆，统一召排的落地验证。
