---
title: "Embedding-based Retrieval in Facebook Search"
short_name: "EBR"
year: 2020
venue: "KDD 2020"
authors: "Jui-Ting Huang, Ashish Sharma, Shuying Sun, et al."
affiliation: "Facebook"
direction: "双塔精细化"
tags:
  - 召回论文
  - EBR
  - 双塔
  - 负采样
  - Faiss
  - 工程化
  - 论文笔记
paper_info: "[[EBR]]"
quality_score: "9/10"
---

# EBR: Embedding-based Retrieval in Facebook Search

> **Jui-Ting Huang, Ashish Sharma, Shuying Sun, et al.** | Facebook | KDD 2020

## 一、研究背景与动机

### 1.1 领域现状

搜索系统传统上依赖倒排索引 + Boolean 匹配（关键词精确匹配），这在 Facebook 社交搜索场景下问题尤其突出——用户搜索的往往是人名、群组名等实体，而非网页关键词。Embedding-based Retrieval（EBR）已在通用 Web 搜索中有探索，但如何在社交搜索这种独特场景中系统性落地，尚无成熟的工业实践报告。

### 1.2 现有方法的局限性

Boolean 匹配无法处理语义匹配（如搜"球鞋"应能召回"Air Jordan"），也无法利用用户个性化信息。现有的 embedding 方法论文多聚焦模型设计，缺乏对工程落地全链路的系统性讨论：负样本构建、特征工程、serving 架构、与倒排索引的融合、ANN 检索的工程细节。

### 1.3 本文解决方案概述

EBR 是一篇工业系统论文，系统性地讨论了在 Facebook Search 中部署 embedding-based retrieval 的全流程：统一 embedding 框架（双塔）、训练数据构造（hard negative mining + 混合负采样）、特征工程、ANN serving（基于 Faiss）、与传统倒排索引的混合检索策略，以及全链路优化经验。

## 二、解决方案

### 2.1 核心思想

不追求模型创新，而是将双塔模型工程化做到极致。论文的核心贡献在于系统性的工程经验：如何选负样本、如何设计特征、如何调 ANN、如何与已有系统融合。

### 2.2 统一 Embedding 框架

**模型结构**：标准双塔。Query 塔：query 文本 + 搜索者特征（位置、社交关系等）。Document 塔：实体文本 + 实体属性特征。两塔输出 embedding，内积作为相似度分数。

**训练**：用点击 (query, document) 对作为正样本，损失函数为 triplet loss 或 cross-entropy。

### 2.3 负采样策略（核心贡献）

**Random Negatives**：从全库随机采样，易于区分，模型快速收敛但区分度不够。

**Hard Negatives**：从模型前一轮检索结果中未被点击的 document 作为难负样本。这迫使模型学习更精细的区分能力。

**混合采样**：最终方案 = random negatives + hard negatives（不同比例混合），在保证训练稳定性的同时提升区分精度。论文发现 hard negatives 比例约 10%~30% 时效果最佳。

### 2.4 特征工程

对搜索场景，除文本特征外，还引入了社交特征（搜索者与被搜实体的社交距离）、地理特征、实体类型等结构化特征。论文发现：文本特征提供基础语义匹配能力，社交/位置特征提供个性化能力，两者互补。

### 2.5 Serving 架构

基于 Faiss 的 ANN 检索。embedding 定期离线生成并加载到 ANN 索引。在线请求：query 塔实时计算 → Faiss ANN 检索 → 与倒排索引结果合并 → 进入排序。

### 2.6 与倒排索引融合

embedding 召回和传统 Boolean 召回是互补的：Boolean 擅长精确匹配，embedding 擅长语义匹配。最终方案是两路并行召回，合并后统一排序。

## 三、实验结果

### 3.1 数据集

Facebook Search 全量线上数据。

### 3.2 实验结果与分析

在 Facebook Search 上部署 EBR 后，搜索成功率（search success rate）提升约 +2%（绝对值），这在 Facebook 搜索这种成熟系统上是非常显著的提升。Embedding 召回路与传统倒排索引路互补，合并后比单路效果好。Hard negative mining 比纯 random negative 在 recall@10 上提升约 +5%。社交特征和位置特征各带来约 +1% 的增量。

## 四、未来工作建议

作者提到可继续探索的方向：在线学习 embedding（而非离线批量更新）、多模态 embedding（图片、视频实体的检索）、更高效的 ANN 索引结构。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9/10** - EBR 是工业界 embedding 召回的"教科书"级论文。它不追求模型创新，而是全面系统地分享了 Facebook 在搜索召回中的工程经验，对工业实践有极高参考价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 模型本身无创新，贡献在于系统性的工程方法论 |
| 技术质量 | 9/10 | 全链路覆盖，每个环节都有深入分析 |
| 实验充分性 | 9/10 | 大规模线上验证，丰富的消融实验 |
| 写作质量 | 9/10 | 工程细节翔实，条理清晰 |
| 实用性 | 10/10 | 直接可用的工业落地指南，hard negative mining 等技巧广泛被采用 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DSSM|DSSM (2013)]] - 双塔模型的鼻祖
- [[YouTube_DNN|YouTube DNN (2016)]] - 推荐场景的双塔先驱
- [[Airbnb_Embedding|Airbnb Embedding (2018)]] - 另一篇优秀的工业 embedding 实践

### 6.2 技术组件
- Faiss (2017, Facebook) - EBR 使用的 ANN 检索库

## 外部资源

- [arXiv](https://arxiv.org/abs/2006.11632)
- [知乎精读](https://zhuanlan.zhihu.com/p/395200364)

> [!tip] 关键启示
> EBR 最大的启示是：在工业系统中，模型本身往往不是最重要的环节。负样本构建策略、特征工程、serving 架构、与已有系统的融合——这些"工程"环节往往决定了 embedding 召回能否真正落地并发挥效果。Hard negative mining 已成为双塔训练的标配技巧。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。做召回系统的工程师必看的实践指南。
