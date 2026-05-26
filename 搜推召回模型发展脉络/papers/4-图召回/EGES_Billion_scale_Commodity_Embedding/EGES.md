---
paper_id: "[arXiv:1803.02349](https://arxiv.org/abs/1803.02349)"
title: "Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba"
authors: "Jizhe Wang, Pipei Huang, Huan Zhao, et al."
institution: "阿里巴巴"
pushlication: "KDD 2018"
tags:
  - 召回论文
  - EGES
  - 图Embedding
  - Side-Information
  - 冷启动
  - 淘宝
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/1803.02349)"
  - "[arXiv](https://arxiv.org/abs/1803.02349)"
date: "2018-03-06"
paper_info: "[[EGES]]"
---

## 一、研究背景与动机

### 1.1 领域现状

淘宝手机客户端的推荐系统（"猜你喜欢"）服务数亿用户和数十亿商品。召回阶段需要从海量商品中高效检索用户可能感兴趣的候选。2018 年之前，淘宝的召回主要依赖 ItemCF 和简单的协同过滤方法。DeepWalk 和 Node2Vec 等方法提供了图上的 embedding 学习思路，但存在冷启动问题——新商品没有足够的行为数据来学习高质量 embedding。

### 1.2 现有方法的局限性

DeepWalk/Node2Vec 等方法仅利用图的拓扑结构（用户-商品交互关系）来学习 embedding，完全忽略了商品自身的属性信息（品类、品牌、价格等）。这导致两个问题：一是冷启动商品无法获得有意义的 embedding；二是属性相似但交互模式不同的商品无法被关联。

### 1.3 本文解决方案概述

EGES（Enhanced Graph Embedding with Side Information）在基于随机游走的图 embedding 方法上，融入商品的多种属性作为 side information。每种属性对应一个独立的 embedding 空间，通过可学习的注意力权重加权聚合为最终的商品 embedding。这样即使是新商品，也可以通过其属性 embedding 获得合理的初始表示。

## 二、解决方案

### 2.1 核心思想

将图 embedding 的学习从"仅依赖拓扑结构"扩展到"拓扑结构 + 属性信息"。核心是设计一个加权聚合机制，自动学习每种属性对最终 embedding 的贡献权重。

### 2.2 整体架构

**图构建**：基于用户行为日志构建商品-商品交互图。如果两个商品在同一用户 session 内被连续浏览/点击，则建立边，边权为共现频次。

**随机游走**：在图上执行 DeepWalk 风格的随机游走生成商品序列。

**Side Information 融合**：每个商品 $v$ 有 $n$ 种 side information（品类、品牌、价格区间、店铺等），每种对应一个 embedding。最终商品 embedding 为加权平均：

$$\mathbf{H}_v = \frac{\sum_{s=0}^{n} e^{a_v^s} \cdot \mathbf{W}_s[v]}{\sum_{s=0}^{n} e^{a_v^s}}$$

其中 $\mathbf{W}_0[v]$ 是商品自身的 embedding，$\mathbf{W}_s[v]$ 是第 $s$ 种属性的 embedding，$a_v^s$ 是可学习的注意力权重（softmax 归一化）。

**训练**：使用 Skip-gram + Negative Sampling，以聚合后的 $\mathbf{H}_v$ 作为商品表示参与优化。

**冷启动处理**：新商品没有图结构信息（$\mathbf{W}_0[v]$ 无法训练），但其属性 embedding（品类、品牌等）已经从其他同属性商品的训练中获得。因此新商品可以直接用属性 embedding 的加权平均作为初始表示。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| 淘宝行为日志 | ~10 亿商品, ~100 亿行为 | 真实生产数据 |

### 3.2 实验结果与分析

在淘宝手猜场景的 A/B 测试中，EGES 相比 DeepWalk 基线提升了 CTR **+10%** 以上。冷启动商品的推荐效果提升尤为显著。注意力权重可视化显示，不同品类的商品对各属性的依赖程度不同：服装类商品更依赖品牌和款式，电子产品更依赖品类和价格。

## 四、未来工作建议

### 4.1 基于分析的未来方向

1. **方向1：从浅层图 Embedding 到 GNN**
   - 动机：随机游走仅捕获图的局部结构
   - 可能的方法：用 GCN/GraphSAGE 替代 DeepWalk
   - 预期成果：更好地聚合多跳邻域信息

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - EGES 优雅地解决了图 embedding 的冷启动问题，其 side information 融合机制被后续大量工作采用。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | Side information 加权融合思路清晰，注意力权重设计合理 |
| 技术质量 | 8/10 | 方法简洁有效，工程实现可行 |
| 实验充分性 | 8/10 | 十亿级真实场景验证，A/B 测试令人信服 |
| 写作质量 | 8/10 | 系统清晰，动机和方法阐述流畅 |
| 实用性 | 9/10 | 直接应用于淘宝，且方法易于复制 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

注意力权重的可解释性非常好——可以直观地看到不同品类的商品依赖哪些属性，这对线上调试和优化很有价值。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- DeepWalk (2014) - EGES 的基础框架
- [[Node2Vec|Node2Vec]] - 另一种图上随机游走 embedding 方法
- [[Airbnb_Embedding|Airbnb Embedding]] - 类似地将业务知识融入 embedding 学习

### 6.2 背景相关
- [[Item2Vec|Item2Vec]] - 行为序列上的 embedding 学习

### 6.3 后续工作
- [[PinSage|PinSage]] - 从随机游走演进到 GNN
- [[LightGCN|LightGCN]] - 简化 GCN 用于推荐

## 外部资源

- [arXiv 论文](https://arxiv.org/abs/1803.02349)

> [!tip] 关键启示
> Side Information 的融入是解决冷启动问题的关键思路。EGES 的加权注意力机制不仅有效，而且可解释，这在工业实践中非常重要。

> [!warning] 注意事项
> - 属性 embedding 的维度和数量需要根据场景调整
> - 注意力权重可能被高频属性（如通用品类）主导
> - 基于 DeepWalk 的训练方式在超大图上可能需要分布式方案

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。图 embedding + side information 融合的经典方案，对理解工业级图召回非常有帮助。
