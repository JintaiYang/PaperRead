---
paper_id: "[arXiv:2304.00902v2](https://arxiv.org/abs/2304.00902)"
title: "FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction"
authors: "Kelong Mao, Jieming Zhu, Liangcai Su, et al."
institution: "Tsinghua University / Huawei Noah's Ark Lab"
pushlication: "AAAI 2023 2023-04-03"
tags:
  - 精排论文
  - FinalMLP
  - 双流MLP
  - 特征选择
  - CTR预估
  - 门控聚合
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2304.00902)"
  - "[Github](https://github.com/reczoo/FuxiCTR)"
date: "2023-04-03"
---

## 一、研究背景与动机

### 1.1 领域现状

经过多年研究，CTR 预估领域积累了大量显式特征交叉网络（DCN、xDeepFM、AutoInt 等）。一个共识是：显式交叉网络对 CTR 预估至关重要，它们与 DNN 互补形成双流结构。然而 FinalMLP 对这一共识提出了挑战。

### 1.2 现有方法的局限性

论文通过大规模实验发现，很多看似精巧的显式交叉网络（DCN V2、xDeepFM 等）在公平比较下，并不总是优于精心调优的纯 MLP 双流模型。这表明：一方面，现有 benchmark 的不公平比较（超参数调优不充分、DNN baseline 太弱）掩盖了真相；另一方面，双流结构的"互补性"可能更多来自于特征选择的多样性（两个流看到不同的特征子集），而非显式交叉本身。

### 1.3 本文解决方案概述

FinalMLP 提出了一个增强的双流 MLP 模型：两个 MLP 流接收不同的特征子集（通过 feature selection gates），加上流间交互（stream-level interaction）和门控聚合（gated aggregation），在不使用任何显式交叉网络的情况下达到了与 SOTA 交叉网络可比甚至更优的效果。

## 二、解决方案

### 2.1 核心思想

FinalMLP 的核心论点是：双流 CTR 模型的成功并非来自显式交叉网络本身，而是来自三个被忽视的设计要素：特征选择的多样性（两个流看到不同特征）、流间信息交换（inter-stream interaction）、以及门控聚合（而非简单拼接）。只要这三个要素到位，纯 MLP 双流也能匹敌复杂的交叉网络。

### 2.2 整体架构

FinalMLP 的架构为：

$$\hat{y} = \sigma\left(\text{Gate}(\mathbf{h}_1, \mathbf{h}_2)\right)$$

其中 $\mathbf{h}_1, \mathbf{h}_2$ 分别是两个 MLP 流的输出。

#### 各模块详细说明

**模块1：Feature Selection Gates**

- **功能**：为每个 MLP 流选择不同的特征子集
- **机制**：每个流有独立的 gate 网络，输出软选择权重
- **关键意义**：让两个流"看到"不同的特征视角，增加互补性

**模块2：双流 MLP**

- 两个独立的多层 MLP，分别处理各自选择的特征
- 每个 MLP 可以有不同的深度和宽度配置

**模块3：Stream-level Interaction**

- **功能**：两个流之间的信息交换
- **实现**：Bilinear fusion 或简单拼接后通过 FC 层

**模块4：Gated Aggregation**

- **功能**：门控融合两个流的输出
- **公式**：$\text{Gate}(\mathbf{h}_1, \mathbf{h}_2) = \mathbf{w}_1^T \mathbf{h}_1 + \mathbf{w}_2^T \mathbf{h}_2 + b$
- 其中权重可以是输入自适应的

## 三、实验结果

### 3.1 数据集

| 数据集 | 数据类型 |
|--------|----------|
| Criteo | 广告点击 |
| Avazu | 移动广告 |
| MovieLens-1M | 电影评分 |
| Frappe | 应用推荐 |

### 3.2 实验设置

#### 3.2.1 基线方法

- DNN、DeepFM、DCN V2、xDeepFM、AutoInt、FiBiNET、FINAL、AFN+

#### 3.3.2 评估指标

- **AUC**、**Logloss**

### 3.3 实验结果与分析

FinalMLP 在 4 个数据集上与 FINAL、DCN V2 等 SOTA 交叉网络持平或更优。特别是在 Criteo 上，FinalMLP 超越了所有显式交叉网络，说明精心设计的纯 MLP 双流确实能匹敌复杂交叉。

#### 结果分析

最重要的发现是：feature selection gate 对性能影响最大。当两个流使用相同特征时，FinalMLP 退化为普通双流 MLP，性能显著下降。这验证了论文的核心论点——互补性来自特征选择多样性。

### 消融实验

#### 消融结果和分析

- **去掉 Feature Selection**：AUC 显著下降（最关键组件）
- **去掉 Stream Interaction**：小幅下降
- **简单拼接 vs 门控聚合**：门控聚合优于简单拼接
- **单流 vs 双流**：双流显著优于单流

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议重新审视显式交叉网络的价值，以及探索更高效的特征选择策略。

### 4.2 基于分析的未来方向

1. **方向1：多流架构**
   - 动机：如果两个流的特征选择带来了互补性，那更多流是否更好？
   - 可能的方法：3-4 个流，每个流选择不同特征子集
   - 挑战：参数量和计算量的增加

### 4.3 改进建议

1. **改进1：动态流数量**
   - 当前问题：固定为双流
   - 改进方案：根据特征空间大小自动决定流数量

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - FinalMLP 对 CTR 领域的一个重要假设（"显式交叉必不可少"）提出了有力质疑，其发现对指导模型设计有重要价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 挑战显式交叉的必要性是重要的学术贡献 |
| 技术质量 | 7/10 | 消融实验设计科学，但方法本身较简单 |
| 实验充分性 | 8/10 | 四个数据集，公平对比，消融充分 |
| 写作质量 | 8/10 | 论点清晰，论证有力 |
| 实用性 | 8/10 | 纯 MLP 架构更简单、更易部署 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- 特征选择的多样性可能比交叉方式本身更重要
- 门控聚合优于简单拼接
- 提醒研究者注意 benchmark 的公平性

#### 5.2.2 需要深入理解的部分

- 在工业级大规模场景中，显式交叉的价值是否更明显？
- Feature selection gate 学到了什么模式？是否可以用领域知识替代？

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[FINAL|FINAL]] - 同一团队的工作，提供了统一的交叉框架
- [[DCN_V2|DCN V2]] - 被 FinalMLP 挑战的显式交叉方法

### 6.2 背景相关
- [[Wide_and_Deep|Wide & Deep]] - 双流架构的起源
- [[DeepFM|DeepFM]] - 经典双流模型

### 6.3 后续工作
- [[DCN_V3|DCN V3]] - 回应了 FinalMLP 的挑战

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2304.00902)
- [FuxiCTR 开源框架](https://github.com/reczoo/FuxiCTR)

> [!tip] 关键启示
> 显式特征交叉未必是 CTR 预估成功的关键——特征选择的多样性、流间交互、门控聚合这些"基础设施"可能同样或更加重要。这提醒我们在追求更复杂的交叉结构之前，先确保基础设计到位。

> [!warning] 注意事项
> - FinalMLP 的结论基于公开数据集，在工业级特征规模下显式交叉的价值可能更大
> - 特征选择 gate 需要额外的超参数调优
> - 论文未排除显式交叉在特定场景下的必要性

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。FinalMLP 是一篇重要的"反思"论文，其发现对 CTR 模型设计的指导意义不亚于提出一个新模型。特别值得工业从业者关注。
