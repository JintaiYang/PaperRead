---
paper_id: "[arXiv:1803.05170](https://arxiv.org/abs/1803.05170)"
title: "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems"
authors: "Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, et al."
institution: "University of Science and Technology of China / Microsoft Research Asia"
pushlication: "KDD 2018 2018-03-14"
tags:
  - 精排论文
  - xDeepFM
  - CIN
  - Vector-wise交叉
  - 特征交叉
  - CTR预估
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/1803.05170)"
  - "[Github](https://github.com/Leavingseason/xDeepFM)"
date: "2018-03-14"
---

## 一、研究背景与动机

### 1.1 领域现状

2018 年，CTR 预估领域的特征交叉研究已形成两条技术路线：bit-wise 交叉（如 DCN 的 Cross Network，在特征向量各维度独立进行交叉）和 vector-wise 交叉（如 FM，在整个 Embedding 向量层面进行交互）。DCN 虽然能显式建模高阶交叉，但其交叉是 bit-wise 的——即最终的交叉项是 Embedding 各维度的混合乘积，而非保持 Embedding 的 vector-wise 语义。

### 1.2 现有方法的局限性

论文指出 DCN 的 Cross Network 存在一个被忽视的问题：虽然形式上每层建模了新的交叉，但由于权重是向量（秩1投影），每层输出实际上是 $\mathbf{x}_0$ 的标量倍数，即 $\mathbf{x}_{l+1} = \alpha \cdot \mathbf{x}_0 + \mathbf{x}_l$。这意味着 Cross Network 实际上建模的是一种非常特殊的交叉形式，而非通用的多项式交叉。

另外，DeepFM 的 FM 部分只能建模二阶 vector-wise 交互，无法扩展到更高阶。DNN 虽能隐式学习高阶交互，但其建模是 bit-wise 的，且缺乏显式性和可解释性。

### 1.3 本文解决方案概述

xDeepFM 提出了 CIN（Compressed Interaction Network），在 vector-wise 层面显式建模任意阶特征交叉。CIN 每层通过 Hadamard 积 + 压缩操作生成新的特征图（feature map），第 $k$ 层包含所有 $k+1$ 阶的 vector-wise 交叉项。CIN 与 DNN 并行组成 xDeepFM。

## 二、解决方案

### 2.1 核心思想

CIN 的设计灵感来自 CNN：在 CNN 中，特征图（feature map）在层间通过卷积核进行信息交换；在 CIN 中，"特征图"是 Embedding 矩阵的行（每行对应一个特征的 Embedding），层间通过 Hadamard 积产生新的特征交叉图。关键区别在于 CIN 保持了 vector-wise 的语义——每个交叉项仍然是一个完整的 $D$ 维向量，而非打碎后的标量混合。

### 2.2 整体架构

xDeepFM 由三个并行组件组成：线性部分（Linear）、CIN 部分（显式 vector-wise 交叉）、DNN 部分（隐式交叉）。

$$\hat{y} = \sigma\left(\mathbf{w}_{linear}^T \mathbf{a} + p_{CIN}^+ + p_{DNN}^+ + b\right)$$

#### 各模块详细说明

**模块1：CIN（Compressed Interaction Network）**

- **功能**：显式建模 vector-wise 的高阶特征交叉
- **输入**：初始特征矩阵 $\mathbf{X}^0 \in \mathbb{R}^{m \times D}$，其中 $m$ 是特征数，$D$ 是 Embedding 维度
- **第 $k$ 层计算**：

$$\mathbf{X}^k_{h,*} = \sum_{i=1}^{H_{k-1}} \sum_{j=1}^{m} W^{k,h}_{ij} (\mathbf{X}^{k-1}_{i,*} \circ \mathbf{X}^0_{j,*})$$

其中 $\circ$ 是 Hadamard 积（element-wise 乘法），$W^{k,h}_{ij}$ 是可学习权重，$H_k$ 是第 $k$ 层的特征图数量。

- **关键性质**：第 $k$ 层的每个特征图是第 $k-1$ 层所有特征图与原始特征矩阵 $\mathbf{X}^0$ 的 Hadamard 交叉的加权和。由于 $\mathbf{X}^0$ 保持不变且每层引入一次乘积，第 $k$ 层包含了所有 $k+1$ 阶 vector-wise 交叉项
- **输出**：每层的特征图通过 sum pooling 压缩为标量，所有层的输出拼接后通过线性层
- **参数量**：$O(\sum_{k=1}^L H_{k-1} \times m \times H_k)$，通过控制 $H_k$ 可以权衡表达力与参数量

**模块2：DNN 部分**

- **功能**：隐式学习高阶非线性交互（bit-wise）
- **结构**：标准的 Embedding + 多层全连接网络

**模块3：Linear 部分**

- **功能**：捕获一阶特征贡献
- **结构**：标准线性层

### 方法架构图

```
原始特征 → Embedding → X⁰ (m × D)
                  ↓                  ↓
            CIN Layer 1          DNN Layer 1
            (X⁰ ⊙ X⁰)            (FC + ReLU)
                  ↓                  ↓
            CIN Layer 2          DNN Layer 2
            (X¹ ⊙ X⁰)            (FC + ReLU)
                  ↓                  ↓
            Sum Pooling          Last Hidden
                  ↓                  ↓
            ←── Concatenate + Linear ──→
                        ↓
                      σ(ŷ)
```

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 数据类型 |
|--------|--------|----------|----------|
| Criteo | 4500万 | 39维 | 广告点击 |
| Dianping | 约340万 | 餐厅推荐 | 点评推荐 |
| Bing News | 约100万 | 新闻推荐 | 点击日志 |

### 3.2 实验设置

#### 3.2.1 基线方法

- LR、FM、DNN、PNN（IPNN/OPNN/PIN）
- Wide & Deep、DCN、DeepFM

#### 3.3.2 评估指标

- **AUC**：ROC-AUC
- **Logloss**：对数损失

### 3.3 实验结果与分析

| 方法 | Criteo AUC | Criteo Logloss | Dianping AUC | Bing News AUC |
|------|-----------|----------------|-------------|---------------|
| LR | 0.7862 | 0.4632 | 0.7664 | -- |
| FM | 0.7890 | 0.4603 | 0.7709 | -- |
| DNN | 0.7915 | 0.4576 | 0.7754 | -- |
| DCN | 0.7921 | 0.4572 | 0.7759 | -- |
| DeepFM | 0.7921 | 0.4573 | 0.7756 | -- |
| **xDeepFM** | **0.7928** | **0.4565** | **0.7773** | -- |

#### 结果分析

xDeepFM 在三个数据集上均优于 DCN 和 DeepFM，但提升幅度较小（Criteo AUC +0.07%）。相比 DCN，xDeepFM 的优势在 Dianping 上更明显（+0.14%），说明 vector-wise 交叉在特征异质性较强的场景中更有效。

### 消融实验

#### 消融结果和分析

- **CIN-only vs DNN-only**：CIN-only 在 Criteo 上接近 DNN 但不如 DNN，说明显式 vector-wise 交叉和隐式 bit-wise 学习各有优势
- **CIN 深度**：3 层 CIN 效果最佳，进一步加深改善有限
- **CIN vs Cross Network**：CIN 优于 DCN 的 Cross Network，验证了 vector-wise 优于 bit-wise 的论点

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议探索 CIN 的计算优化（因为 CIN 的计算复杂度较高），以及将 CIN 应用于更多推荐场景。

### 4.2 基于分析的未来方向

1. **方向1：CIN 的计算效率优化**
   - 动机：CIN 每层需要计算 $H_{k-1} \times m$ 次 Hadamard 积，计算量随层数和特征数增长较快
   - 可能的方法：稀疏化、低秩近似、或采用更高效的交叉操作
   - 预期成果：在保持 vector-wise 交叉能力的同时降低计算开销
   - 挑战：如何在压缩中保留关键交叉信息

### 4.3 改进建议

1. **改进1：动态特征图数量**
   - 当前问题：每层的特征图数量 $H_k$ 是预设的超参数
   - 改进方案：根据信息量自动调整每层特征图数量
   - 预期效果：更灵活的容量分配

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - xDeepFM 在理论上提出了 vector-wise 交叉的重要概念，区分了 bit-wise 和 vector-wise 两种交叉范式，但 CIN 的计算复杂度限制了其工业应用。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 首次明确提出 vector-wise 交叉，CIN 的设计与 CNN 类比巧妙 |
| 技术质量 | 7/10 | 理论分析充分，但 CIN 计算开销大，工业落地困难 |
| 实验充分性 | 8/10 | 三个数据集，多基线对比，消融完整 |
| 写作质量 | 7/10 | 论文较长，CIN 的数学表述有些复杂 |
| 实用性 | 6/10 | CIN 计算复杂度较高，在大规模工业场景中较难直接使用 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Bit-wise vs Vector-wise 交叉的区分是重要的概念贡献
- CIN 与 CNN 的类比（特征图 ↔ Embedding 行）提供了新的理解视角
- 对 DCN Cross Network 的"秩1限制"分析深入

#### 5.2.2 需要深入理解的部分

- CIN 的参数量和计算量具体有多大？与 DNN 相比如何？
- Vector-wise 交叉在什么场景下明显优于 bit-wise？

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DCN|DCN]] - xDeepFM 的主要对比对象，bit-wise 显式交叉
- [[DeepFM|DeepFM]] - FM 的二阶 vector-wise 交叉，xDeepFM 将其推广到任意阶

### 6.2 背景相关
- [[Wide_and_Deep|Wide & Deep]] - 双流架构的起源

### 6.3 后续工作
- [[DCN_V2|DCN V2]] - 通过升秩改进了 DCN 的 bit-wise 交叉能力
- [[FINAL|FINAL]] - 重新审视了 bilinear 交叉的设计
- [[FinalMLP|FinalMLP]] - 挑战了显式交叉的必要性

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1803.05170)
- [GitHub 官方代码](https://github.com/Leavingseason/xDeepFM)

> [!tip] 关键启示
> 特征交叉存在 bit-wise 和 vector-wise 两种范式——CIN 在 vector-wise 层面显式建模高阶交叉，保持了 Embedding 的语义完整性。这一概念区分为后续特征交叉研究提供了重要的理论基础。

> [!warning] 注意事项
> - CIN 的计算复杂度较高（$O(mHD)$每层），在大规模特征场景中可能成为瓶颈
> - 实际效果提升相对 DCN/DeepFM 不大，性价比存疑
> - CIN 的超参数（每层特征图数量）调优较为困难

> [!success] 推荐指数
> ⭐⭐⭐ 选择性阅读。xDeepFM 的理论贡献（vector-wise 交叉概念）值得了解，但 CIN 本身在工业实践中的应用有限。建议重点理解其对 DCN 的分析和 bit-wise vs vector-wise 的区分，模型细节可快速浏览。
