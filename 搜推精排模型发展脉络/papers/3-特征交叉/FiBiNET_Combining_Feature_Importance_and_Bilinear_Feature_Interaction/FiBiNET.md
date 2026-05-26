---
paper_id: "[arXiv:1905.09433](https://arxiv.org/abs/1905.09433)"
title: "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction"
authors: "Tongwen Huang, Zhiqi Zhang, Junlin Zhang"
institution: "Sina Weibo"
pushlication: "RecSys 2019 2019-05-23"
tags:
  - 精排论文
  - FiBiNET
  - SENet
  - Bilinear交互
  - 特征重要性
  - CTR预估
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/1905.09433)"
date: "2019-05-23"
---

## 一、研究背景与动机

### 1.1 领域现状

2019 年，CTR 预估模型在特征交叉方面已有丰富的工作（FM、DCN、DeepFM、xDeepFM 等），但普遍存在两个被忽视的问题：一是所有特征被等同对待，未考虑不同特征对预测的重要性差异（feature importance）；二是 FM 系列的内积交互方式过于简单，缺乏更灵活的交互建模能力。

### 1.2 现有方法的局限性

FM 及其变体使用内积 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 建模特征交互，这种交互是对称且固定的——无法根据不同特征对调整交互方式。另外，不同特征域（如"用户性别"和"商品价格"）对 CTR 的贡献差异巨大，但现有模型在 Embedding 后直接进行交叉，未做重要性区分。

### 1.3 本文解决方案概述

FiBiNET 提出两个改进：一是引入 SENET（Squeeze-and-Excitation Network）机制动态学习每个特征的重要性权重，在交叉之前先对特征 Embedding 进行加权；二是用 Bilinear 交互替代内积，通过额外的可学习矩阵 $\mathbf{W}$ 增强交互的表达能力。

## 二、解决方案

### 2.1 核心思想

FiBiNET 的思路是"先精选、再交叉"：在特征交叉之前，先通过 SENET 机制识别哪些特征更重要并动态加权，然后用 Bilinear 函数（而非简单内积）进行更灵活的交互。这两个改进分别解决了"交叉什么"和"怎么交叉"两个问题。

### 2.2 整体架构

![[image-20190131-163202.png|800]]

> 图1：FiBiNET 整体架构。原始 Embedding 经过 SENET 层产生加权 Embedding，然后分别用原始和加权 Embedding 进行 Bilinear 交叉，最后与 DNN 拼接预测。

#### 各模块详细说明

**模块1：SENET 层（特征重要性学习）**

- **功能**：动态学习每个特征的重要性权重
- **处理流程**：
  1. **Squeeze**：对每个特征的 Embedding 向量做 mean pooling，得到 $\mathbf{z} \in \mathbb{R}^m$
  2. **Excitation**：通过两层 FC 网络学习权重：$\mathbf{a} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{z}))$，其中 $\mathbf{W}_1 \in \mathbb{R}^{m/r \times m}$，$\mathbf{W}_2 \in \mathbb{R}^{m \times m/r}$，$r$ 是缩减比
  3. **Re-weight**：$\hat{\mathbf{e}}_i = a_i \cdot \mathbf{e}_i$

- **关键意义**：SENET 借鉴自 CV 领域的 SENet（Squeeze-and-Excitation Network），首次将"通道注意力"的思想应用于推荐系统的特征重要性学习

![[image-20181025003821109.png|800]]

> 图2：SENET 层的详细结构。通过 Squeeze-Excitation 机制为每个特征分配动态权重。

**模块2：Bilinear 交互层**

- **功能**：用 bilinear 函数替代内积进行更灵活的特征交互
- **三种 Bilinear 变体**：
  - **Field-All**：所有特征对共享一个矩阵 $\mathbf{W}$：$f_{ij} = \mathbf{e}_i \cdot \mathbf{W} \cdot \mathbf{e}_j$
  - **Field-Each**：每个特征有独立的矩阵：$f_{ij} = \mathbf{e}_i \cdot \mathbf{W}_i \cdot \mathbf{e}_j$
  - **Field-Interaction**：每个特征对有独立的矩阵：$f_{ij} = \mathbf{e}_i \cdot \mathbf{W}_{ij} \cdot \mathbf{e}_j$

- **与内积的关系**：当 $\mathbf{W} = \mathbf{I}$ 时退化为内积，bilinear 是内积的推广

![[image-20181025110512959.png|800]]

> 图3：Bilinear 交互层的三种变体。从左到右参数量递增，交互灵活性递增。

**模块3：组合与预测**

- 原始 Embedding 的 Bilinear 交互 + SENET 加权 Embedding 的 Bilinear 交互 + 原始 Embedding → 拼接 → DNN → sigmoid

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 数据类型 |
|--------|--------|----------|
| Criteo | 4500万 | 广告点击 |
| Avazu | 4000万 | 移动广告 |

### 3.2 实验设置

#### 3.2.1 基线方法

- LR、FM、FFM、FNN、PNN
- DeepFM、xDeepFM、DCN

#### 3.3.2 评估指标

- **AUC**、**Logloss**

### 3.3 实验结果与分析

| 方法 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|----------------|-----------|---------------|
| DeepFM | 0.8007 | 0.4493 | 0.7812 | 0.3741 |
| xDeepFM | 0.8012 | 0.4488 | 0.7818 | 0.3735 |
| DCN | 0.7990 | 0.4510 | 0.7816 | 0.3737 |
| **FiBiNET** | **0.8021** | **0.4481** | **0.7823** | **0.3730** |

#### 结果分析

FiBiNET 在两个数据集上均优于所有基线，Criteo AUC 达到 0.8021。SENET 和 Bilinear 各贡献了约一半的提升，说明特征重要性学习和灵活交互机制都是有效的改进。

### 消融实验

#### 消融结果和分析

- **去掉 SENET**：AUC 下降约 0.05%，说明特征重要性加权有正面贡献
- **Bilinear vs Inner Product**：Bilinear（Field-Each）优于内积约 0.08% AUC
- **三种 Bilinear 变体**：Field-Each 是性能与参数量的最佳平衡

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议探索更高效的特征重要性建模方式和 Bilinear 交互的优化。

### 4.2 基于分析的未来方向

1. **方向1：将 SENET 扩展为多级重要性**
   - 动机：当前 SENET 对每个特征给出一个全局权重，未考虑不同交互场景下的差异
   - 可能的方法：条件化的 SENET，根据 target item 动态调整特征权重
   - 预期成果：更精细的重要性建模

### 4.3 改进建议

1. **改进1：轻量化 Bilinear**
   - 当前问题：Field-Interaction 的参数量为 $O(m^2 d^2)$
   - 改进方案：低秩分解 Bilinear 矩阵
   - 预期效果：在保持灵活性的同时降低参数量

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - FiBiNET 提出了两个实用的改进（SENET + Bilinear），尤其是将 CV 领域的通道注意力引入 CTR 预估的思路有启发价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | SENET 借鉴自 CV 不算原创，Bilinear 是内积的自然推广 |
| 技术质量 | 7/10 | 方法直觉清晰，但理论分析较少 |
| 实验充分性 | 7/10 | 两个数据集，消融完整，但缺乏工业验证 |
| 写作质量 | 8/10 | 结构清晰，图表直观 |
| 实用性 | 8/10 | SENET 机制易于嵌入现有模型，Bilinear 交互增加的参数可控 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- SENET 特征重要性学习可作为通用模块插入任何 CTR 模型
- Bilinear 交互提供了"在内积和全连接之间"的灵活交互方式
- Field-Each 是参数和性能的甜蜜点

#### 5.2.2 需要深入理解的部分

- SENET 学到的特征权重是否与领域知识一致？
- Bilinear 矩阵 $\mathbf{W}$ 在训练后的结构（是否低秩、是否稀疏）

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DeepFM|DeepFM]] - FiBiNET 的主要改进对象，内积交互被 Bilinear 替代
- [[FINAL|FINAL]] - 也探索了特征交互的多种变体

### 6.2 背景相关
- Hu et al. "Squeeze-and-Excitation Networks" - CV 领域的 SENet 原始论文

### 6.3 后续工作
- FiBiNET++ - 微博后续改进版本
- [[FINAL|FINAL]] - 统一分析了各种特征交互方式

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1905.09433)

> [!tip] 关键启示
> 特征交叉不仅关乎"怎么交叉"（Bilinear vs 内积），更关乎"交叉什么"（SENET 特征重要性筛选）。先筛选再交叉的两步策略比直接交叉所有特征更有效。

> [!warning] 注意事项
> - SENET 引入了额外的 Squeeze-Excitation 计算，在特征数很多时增加延迟
> - Bilinear 矩阵的 Field-Interaction 变体参数量较大
> - 论文未提供工业级部署的效果验证

> [!success] 推荐指数
> ⭐⭐⭐ 选择性阅读。FiBiNET 的 SENET 特征重要性机制是值得了解的思路，Bilinear 交互也是内积的有价值推广，但整体创新幅度一般。建议重点理解 SENET 的设计思路和 Bilinear 的三种变体。
