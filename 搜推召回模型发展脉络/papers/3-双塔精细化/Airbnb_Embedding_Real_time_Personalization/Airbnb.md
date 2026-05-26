---
paper_id: "[KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3219885)"
title: "Real-time Personalization using Embeddings for Search Ranking at Airbnb"
authors: "Mihajlo Grbovic, Haibin Cheng"
institution: "Airbnb"
pushlication: "KDD 2018"
tags:
  - 召回论文
  - Airbnb-Embedding
  - Embedding
  - 搜索排序
  - 工程化
  - 负采样
quality_score: "9.0/10"
link:
  - "[PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885)"
  - "[ACM](https://dl.acm.org/doi/10.1145/3219819.3219885)"
date: "2018-07-19"
paper_info: "[[Airbnb_Embedding]]"
---

## 一、研究背景与动机

### 1.1 领域现状

Airbnb 是全球最大的短租住宿平台，搜索和推荐面临独特挑战：用户搜索通常有明确的目的地和日期，且一次旅行通常只预订一个 listing；预订行为极度稀疏（相比点击和浏览）；listing 具有强烈的地域属性。2018 年之前，Airbnb 的搜索排序主要依赖手工特征工程和 GBDT 模型。

### 1.2 现有方法的局限性

直接将 Word2Vec/Item2Vec 应用到 Airbnb 场景存在两个问题：一是标准的 Skip-gram 无法区分"浏览 session"和"预订 session"的重要性差异——预订是最终业务目标，应被特殊对待；二是标准的随机负采样忽略了地域信息——同城 listing 之间的区分比跨城 listing 更有价值。

### 1.3 本文解决方案概述

论文提出了两种 embedding 方法：Listing Embedding（短期 session 级别）和 User-Type & Listing-Type Embedding（长期用户级别）。核心创新包括：在 session 中加入预订 listing 作为全局上下文、同城负采样策略、以及基于用户和 listing 类型的跨 session embedding。

## 二、解决方案

### 2.1 核心思想

将业务目标（预订行为）和领域知识（地域信息）深度融入 embedding 学习过程，而非使用通用的 Word2Vec 方案。这体现了一个重要理念：好的 embedding 不仅需要好的算法，更需要与业务场景深度结合的训练策略。

### 2.2 整体架构

**Listing Embedding**：基于用户点击 session 训练。对每个 session $s = (l_1, l_2, \ldots, l_M)$，使用 Skip-gram 优化目标，做三个关键修改：

1. **Booked Listing 作为全局上下文**：如果 session 以预订结束，将预订的 listing $l_{\text{booked}}$ 作为全局上下文加入每个 sliding window 的正样本对中，使整个 session 的 embedding 学习都"锚定"在最终预订目标上。

2. **同城负采样**：除了标准的随机负采样外，额外从与中心 listing 同城的 listing 中采样 hard negative，迫使模型学习同城内更细粒度的区分。

3. **适应性训练**：对冷启动 listing，通过已有同类 listing 的平均 embedding 初始化。

修改后的优化目标：

$$\mathcal{L} = \sum_{(l, c) \in \mathcal{D}_p} \log \sigma(v_c^T v_l) + \sum_{(l, c) \in \mathcal{D}_n} \log \sigma(-v_c^T v_l) + \log \sigma(v_{l_b}^T v_l) + \sum_{(l, m_n) \in \mathcal{D}_{m_n}} \log \sigma(-v_{m_n}^T v_l)$$

其中 $\mathcal{D}_p$ 是正样本对，$\mathcal{D}_n$ 是随机负样本，$l_b$ 是预订 listing（全局上下文），$\mathcal{D}_{m_n}$ 是同城负样本。

**User-Type & Listing-Type Embedding**：为解决预订数据稀疏问题，将用户和 listing 分别归类（按国家、设备类型、是否首单等维度），在类型级别学习 embedding。通过将用户预订序列转化为 (user_type, listing_type) 交替序列，用 Word2Vec 学习跨 session 的长期偏好。

## 三、实验结果

### 3.1 实验设置

在 Airbnb 生产环境中进行在线 A/B 测试，主要指标为预订转化率和搜索排名质量。

### 3.2 实验结果与分析

Listing Embedding 加入搜索排序模型后，预订量提升了 **+3.75%**。User-Type & Listing-Type Embedding 进一步带来 **+1.63%** 的预订量提升。同城负采样和全局上下文的消融实验表明两者都带来了显著的增量收益。

定性分析展示了同城 listing 在 embedding 空间中按价格区间、房型等维度自然聚类，证明了 embedding 的语义质量。

## 四、未来工作建议

### 4.1 作者建议的未来工作

探索更复杂的序列模型（如 RNN）替代 Word2Vec，以更好地捕获 session 内的浏览意图变化。

### 4.2 基于分析的未来方向

1. **方向1：端到端的 Embedding + 排序联合训练**
   - 动机：当前 embedding 和排序模型是分开训练的
   - 可能的方法：将 embedding 作为特征输入排序模型，反向传播梯度更新 embedding
   - 预期成果：embedding 更直接地服务于最终排序目标

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - 这是将通用 embedding 方法与具体业务场景深度结合的标杆论文。论文展示了如何通过领域知识（地域、预订目标）系统性地改进 Item2Vec，获得显著的业务提升。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 全局上下文和同城负采样都是精巧的业务驱动创新 |
| 技术质量 | 9/10 | 设计决策有清晰的业务动机，实现细节完整 |
| 实验充分性 | 9/10 | 大规模 A/B 测试验证，定性和定量分析均充分 |
| 写作质量 | 9/10 | 工业论文的典范，每个设计选择都有业务动机支撑 |
| 实用性 | 10/10 | 直接可复制到各类电商/内容推荐场景 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

"全局上下文"（booked listing as global context）是这篇论文最精妙的设计。它将业务目标（预订）直接融入了 embedding 的学习过程，使得所有点击行为都"围绕"最终转化目标组织。同城负采样则体现了"hard negative 比 random negative 更有价值"的洞察。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Item2Vec|Item2Vec]] - Airbnb Embedding 的基础框架
- [[EBR|EBR]] - Facebook 的 hard negative mining 策略是 Airbnb 同城负采样的思想延续

### 6.2 背景相关
- Word2Vec (2013) - 底层训练框架
- [[DSSM|DSSM]] - 双塔范式的鼻祖

### 6.3 后续工作
- [[EGES|EGES]] - 阿里巴巴类似地将 side information 融入图 embedding

## 外部资源

- [ACM Digital Library](https://dl.acm.org/doi/10.1145/3219819.3219885)

> [!tip] 关键启示
> Airbnb Embedding 的核心教训：通用的 embedding 方法（Word2Vec/Item2Vec）在实际业务中需要与领域知识深度结合。全局上下文（预订目标）和 hard negative（同城 listing）这两个看似简单的修改，带来了巨大的业务价值。

> [!warning] 注意事项
> - 同城负采样依赖于地域信息的可用性，不一定适用于所有场景
> - User-Type 的归类方式需要领域专家参与设计
> - 两阶段 embedding（listing-level + type-level）增加了系统复杂度

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是 embedding 工程化落地的最佳案例论文，展示了如何将领域知识系统性地融入 embedding 学习。
