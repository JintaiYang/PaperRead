---
paper_id: "[arXiv:1708.05123](https://arxiv.org/abs/1708.05123)"
title: "Deep & Cross Network for Ad Click Predictions"
authors: "Ruoxi Wang, Bin Fu, Gang Fu, et al."
institution: "Google / Stanford University"
pushlication: "ADKDD 2017 2017-08-15"
tags:
  - 精排论文
  - DCN
  - Cross-Network
  - 特征交叉
  - CTR预估
  - 显式交叉
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/1708.05123)"
date: "2017-08-15"
---

## 一、研究背景与动机

### 1.1 领域现状

CTR 预估中，特征交叉是提升模型表达能力的关键。Wide & Deep 通过手动定义的交叉特征实现"记忆"，DeepFM 通过 FM 自动建模二阶交叉。然而，这些方法在高阶交叉方面仍有局限：Wide & Deep 的交叉阶数取决于人工定义，FM 固定为二阶，DNN 虽然理论上能学习任意阶交互，但隐式学习的效率低、可解释性差。

### 1.2 现有方法的局限性

DNN 学习特征交叉是隐式的，需要大量参数和数据才能逼近特定的交叉模式——而很多 CTR 场景中有效的交叉模式本质上是简单的乘性关系（如 $x_i \cdot x_j \cdot x_k$），DNN 的加性结构（线性变换 + 激活）在建模乘性关系时效率低下。残差网络虽能将表征阶数从 $O(1)$ 提升到 $O(L)$，但仍不是乘性的。论文希望设计一种网络结构，能以参数高效的方式自动且显式地学习有界阶数的特征交叉。

### 1.3 本文解决方案概述

DCN 提出了 Cross Network，通过特殊的层间递推实现显式的特征交叉。Cross Network 的每一层在当前表征上施加一次与原始输入 $\mathbf{x}_0$ 的外积操作，从而在第 $l$ 层自动包含所有不超过 $l+1$ 阶的交叉项。Cross Network 与 DNN 并行组成 DCN，前者负责显式交叉，后者负责隐式非线性学习。

## 二、解决方案

### 2.1 核心思想

Cross Network 的设计灵感来自一个关键观察：如果每层的输出都是原始输入 $\mathbf{x}_0$ 与当前输入 $\mathbf{x}_l$ 的外积的函数，那么经过 $L$ 层后，网络就自动包含了所有不超过 $L+1$ 阶的交叉项。而且每层只需一个权重向量 $\mathbf{w}_l \in \mathbb{R}^d$ 和一个偏置 $\mathbf{b}_l$，参数量仅为 $O(d \times L)$，远小于 DNN 的 $O(d^2 \times L)$。

### 2.2 整体架构

![[deep_cross_network_narrow.eps|800]]

> 图1：DCN 整体架构。左侧 Cross Network 显式建模高阶特征交叉，右侧 Deep Network 隐式学习非线性交互，两者共享底部 Embedding 层，输出拼接后经 sigmoid 预测。

#### 各模块详细说明

**模块1：Embedding 和 Stacking 层**

- **功能**：将稀疏特征映射为稠密向量并与连续特征拼接
- **输出**：$\mathbf{x}_0 = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_k, x_{dense}] \in \mathbb{R}^d$

**模块2：Cross Network**

- **功能**：显式学习有界阶数的特征交叉
- **递推公式**：

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$

其中 $\mathbf{x}_0 \mathbf{x}_l^T$ 是外积（rank-1 矩阵），$\mathbf{w}_l$ 将其投影回 $d$ 维向量。残差连接 $+ \mathbf{x}_l$ 保证至少保留前一层的信息。

展开后可以证明，第 $l$ 层的输出 $\mathbf{x}_l$ 是 $\mathbf{x}_0$ 各分量的多项式，最高阶数为 $l+1$。例如 2 层 Cross Network 可以表达 $x_i \cdot x_j \cdot x_k$ 这样的三阶交叉。

- **参数量**：$L$ 层 Cross Network 仅需 $2 \times d \times L$ 个参数（每层一个 $\mathbf{w}_l$ 和一个 $\mathbf{b}_l$）
- **关键性质**：每层的输出始终与 $\mathbf{x}_0$ 同维，不会引发维度膨胀

![[cross_type_x0.eps|800]]

> 图2：Cross Network 的交叉类型可视化。每一层都通过与 $\mathbf{x}_0$ 的外积引入新的交叉项。

**模块3：Deep Network**

- **功能**：隐式学习复杂非线性交互
- **结构**：标准的多层全连接网络
- **公式**：$\mathbf{h}_{l+1} = \text{ReLU}(\mathbf{W}_l \mathbf{h}_l + \mathbf{b}_l)$

**模块4：Combination Layer**

- **功能**：拼接 Cross Network 和 Deep Network 的输出，通过 sigmoid 输出 CTR 预测

$$p = \sigma\left(\mathbf{w}_{logits}^T [\mathbf{x}_{L_{cross}}, \mathbf{h}_{L_{deep}}] + b\right)$$

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 数据类型 |
|--------|--------|----------|----------|
| Criteo | 4500万 | 39维（13连续+26类别） | 广告点击 |

### 3.2 实验设置

#### 3.2.1 基线方法

- DNN：纯深度网络
- LR：逻辑回归
- FM：因子分解机
- DCN（本文方法）

#### 3.3.2 评估指标

- **Logloss**：对数损失，值越小越好

### 3.3 实验结果与分析

论文的核心实验发现：DCN 在参数量接近 DNN 的情况下，Logloss 更低。Cross Network 的 6 层配置在 Criteo 上实现了最优的 logloss，且参数量仅为 DNN 的约 40%。

![[logloss_vs_crosslayers.eps|800]]

> 图3：Cross Network 层数对 Logloss 的影响。随着层数增加，Logloss 先降后升，6 层是最优选择。

#### 结果分析

Cross Network 的参数效率优势明显：它用 $O(dL)$ 的参数量就能建模 $L+1$ 阶交叉，而 DNN 需要 $O(d^2L)$ 的参数来隐式近似同样的交叉。论文还验证了 Cross Network 和 Deep Network 的互补性——单独使用 Cross Network 效果不如 DCN，说明显式交叉和隐式非线性学习缺一不可。

### 消融实验

#### 消融结果和分析

- 单独 Cross Network 的表现优于 LR 和 FM，但不如 DNN——说明纯多项式交叉缺乏非线性激活的灵活性
- DCN（Cross + Deep）优于两者各自，验证了互补性
- Cross Network 层数从 1 到 6 层持续改善，6 层以上改善微弱

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议探索更高效的交叉方式以及在其他任务上的应用。

### 4.2 基于分析的未来方向

1. **方向1：提升 Cross Network 的表达能力**
   - 动机：Cross Network 每层的投影矩阵实际上是秩1的（$\mathbf{x}_0 \mathbf{w}_l^T$），表达能力受限
   - 可能的方法：用矩阵替代向量（即 DCN V2 的 Cross Network with matrix）
   - 预期成果：更丰富的交叉模式
   - 挑战：参数量从 $O(dL)$ 增加到 $O(d^2L)$

### 4.3 改进建议

1. **改进1：Mix-rank Cross Network**
   - 当前问题：$\mathbf{w}_l$ 是向量，投影是秩1的
   - 改进方案：使用低秩矩阵 $\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$（即 DCN V2 的方案）
   - 预期效果：在可控参数增加下提升交叉能力

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.0/10** - DCN 首次提出了 Cross Network 的概念，用极简的参数化实现了显式有界阶数的特征交叉，开创了一条重要的研究路线。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | Cross Network 的设计非常巧妙，用外积递推实现显式交叉是全新思路 |
| 技术质量 | 7/10 | 理论证明了多项式阶数，但秩1限制影响实际表达能力 |
| 实验充分性 | 6/10 | 仅 Criteo 一个公开数据集，基线较少 |
| 写作质量 | 8/10 | Google 论文风格，简洁清晰，数学推导完整 |
| 实用性 | 8/10 | 参数高效，易于实现，但秩1限制在实际应用中可能需要配合更深的 DNN |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Cross Network 的 $O(dL)$ 参数量极其经济，为后续改进留下巨大空间
- 显式交叉的可解释性优于 DNN 隐式交叉
- 残差连接保证了低阶信息的保留

#### 5.2.2 需要深入理解的部分

- 秩1限制具体如何影响交叉的表达能力？DCN V2 通过升秩解决了什么问题？
- Cross Network 的梯度流特性：每层都有到 $\mathbf{x}_0$ 的直接连接，是否有类似 ResNet 的梯度优势？

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wide_and_Deep|Wide & Deep]] - DCN 可视为 Wide & Deep 的改进，用 Cross Network 替代人工交叉
- [[DeepFM|DeepFM]] - 同时期的另一条自动交叉路线（FM-based）

### 6.2 背景相关
- [[GBDT_LR|GBDT+LR]] - 特征交叉自动化的开端

### 6.3 后续工作
- [[DCN_V2|DCN V2]] - 升秩改进，Cross Network with Matrix
- [[DCN_V3|DCN V3]] - 进一步引入 Exponential Cross Network
- [[xDeepFM|xDeepFM]] - 另一条显式高阶交叉路线（CIN）

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1708.05123)
- [Google Research Blog](https://ai.googleblog.com/2017/08/deep-cross-network.html)

> [!tip] 关键启示
> Cross Network 用极简的递推公式（$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$）实现了自动的显式高阶特征交叉，参数量仅 $O(dL)$，开创了"显式交叉网络"这一重要研究方向。

> [!warning] 注意事项
> - Cross Network 每层的交叉投影是秩1的，表达能力受限（DCN V2 已修复）
> - 论文实验规模较小，仅在 Criteo 上验证，工业场景的效果未充分展示
> - 交叉阶数与层数绑定，增加阶数必须增加层数

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。DCN 是显式特征交叉网络的开山之作，理解 Cross Network 的设计原理和局限性是理解 DCN V2、V3 以及整个特征交叉网络演进的基础。
