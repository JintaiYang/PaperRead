---
paper_id: "[arXiv:1703.04247](https://arxiv.org/abs/1703.04247)"
title: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
authors: "Huifeng Guo, Ruiming Tang, Yunming Ye, et al."
institution: "Harbin Institute of Technology / Huawei Noah's Ark Lab"
pushlication: "IJCAI 2017 2017-03-13"
tags:
  - 精排论文
  - DeepFM
  - FM
  - 特征交叉
  - CTR预估
  - 端到端
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/1703.04247)"
date: "2017-03-13"
---

## 一、研究背景与动机

### 1.1 领域现状

CTR 预估的核心挑战在于学习特征之间的交互（feature interactions）。低阶交互（如"男性用户偏好射击游戏"）和高阶交互（如"男性青少年在周末喜欢射击游戏"）对推荐效果都至关重要。在 DeepFM 发表时（2017年），已有多种模型尝试捕获特征交互：FM 能高效建模二阶交互，DNN 能隐式学习高阶交互，而 Wide & Deep 通过双流架构试图兼顾低阶和高阶。

### 1.2 现有方法的局限性

论文系统分析了已有方法的不足。FNN（Factorization-Machine supported Neural Network）用 FM 预训练 Embedding 再送入 DNN，但两阶段训练导致 Embedding 质量受限于 FM 的能力，且只能捕获高阶交互。PNN（Product-based Neural Network）在 Embedding 层和 DNN 之间加入 product 层来建模交互，但产品操作的计算复杂度高，且也忽略了低阶交互。Wide & Deep 虽然同时建模低阶（Wide）和高阶（Deep），但 Wide 部分仍需人工设计交叉特征（cross-product transformation），这在实际应用中是巨大的工程负担。

![[wide-deep.png|800]]

> 图1：Wide & Deep 模型结构示意，Wide 部分需要人工设计交叉特征，这是 DeepFM 要解决的核心问题。

### 1.3 本文解决方案概述

DeepFM 将 FM 和 DNN 无缝集成为一个端到端模型：FM 部分自动建模所有二阶特征交互（替代 Wide & Deep 中的人工交叉特征），DNN 部分建模高阶交互，两部分共享相同的 Embedding 层。这种设计完全消除了人工特征工程的需求，同时能高效地同时建模低阶和高阶特征交互。

## 二、解决方案

### 2.1 核心思想

DeepFM 的核心洞察是：FM 天然就是 Wide & Deep 中 Wide 部分的最佳替代品。FM 通过因子化的方式自动学习所有二阶特征交互，无需人工定义；而且 FM 的 Embedding 向量（即因子向量 $\mathbf{v}_i$）可以直接被 DNN 复用，实现 Embedding 共享。这意味着 FM 学到的低阶交互模式会直接指导 DNN 的高阶学习，反之亦然。

### 2.2 整体架构

![[architecture-deepfm.png|800]]

> 图2：DeepFM 整体架构。FM 部分（左）和 Deep 部分（右）共享底部的 Embedding 层，FM 负责自动二阶交叉，Deep 负责高阶交互，两者输出相加后通过 sigmoid 得到 CTR 预测。

预测公式为：

$$\hat{y} = \sigma\left(y_{FM} + y_{DNN}\right)$$

其中 $y_{FM}$ 是 FM 部分的输出，$y_{DNN}$ 是 DNN 部分的输出。

#### 各模块详细说明

**模块1：FM 部分**

- **功能**：自动建模所有 pair-wise 的二阶特征交互
- **输入**：原始稀疏特征 $\mathbf{x}$
- **输出**：FM 预测值 $y_{FM}$

FM 的输出由一阶项和二阶交互项组成：

$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

其中 $\mathbf{v}_i \in \mathbb{R}^k$ 是第 $i$ 个特征的 Embedding 向量（因子向量）。二阶项通过内积 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 建模任意两个特征之间的交互强度，计算复杂度通过重新整理可降到 $O(kn)$。

![[architecture-fm.png|800]]

> 图3：FM 部分的结构。Addition 单元捕获一阶信息，Inner Product 单元捕获二阶特征交互。

**模块2：Deep 部分**

- **功能**：学习高阶特征交互
- **输入**：与 FM 共享的 Embedding 向量，拼接后形成稠密输入
- **输出**：DNN 预测值 $y_{DNN}$
- **处理流程**：
  1. 所有特征的 Embedding 向量拼接：$\mathbf{a}^{(0)} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_m]$
  2. 通过多层全连接：$\mathbf{a}^{(l+1)} = \sigma(\mathbf{W}^{(l)} \mathbf{a}^{(l)} + \mathbf{b}^{(l)})$
  3. 最终层输出 $y_{DNN} = \mathbf{w}^T \mathbf{a}^{(L)} + b$

![[architecture-dnn.png|800]]

> 图4：Deep 部分的结构。Embedding 层将稀疏输入转为稠密向量，然后通过多层全连接层学习高阶交互。

**模块3：共享 Embedding 层**

- **功能**：FM 的因子向量 $\mathbf{v}_i$ 同时作为 DNN 的输入 Embedding
- **关键意义**：这是 DeepFM 与 Wide & Deep 最大的区别——FM 和 DNN 不是独立训练再合并，而是从底层就共享表征，训练信号在两个组件之间双向传播

![[embedding.png|800]]

> 图5：Embedding 层示意。稀疏输入通过 Embedding 映射为稠密向量，这些向量同时供 FM 和 DNN 使用。

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 数据类型 |
|--------|--------|----------|----------|
| Criteo | 4500万 | 39维（13连续+26类别） | 广告点击 |
| Company* | 约10亿 | -- | 华为应用商店游戏推荐 |

> *Company 数据集为华为内部数据，连续7天用于训练，第8天用于测试。

### 3.2 实验设置

#### 3.2.1 基线方法

- LR：纯逻辑回归
- FM：Factorization Machine
- FNN：FM 预训练 + DNN
- PNN（IPNN/OPNN）：Product-based Neural Network
- Wide & Deep（LR & DNN）

#### 3.3.2 评估指标

- **AUC**：ROC-AUC，值越大越好
- **Logloss**：对数损失，值越小越好

### 3.3 实验结果与分析

| 方法 | Criteo AUC | Criteo Logloss | Company AUC | Company Logloss |
|------|-----------|----------------|-------------|-----------------|
| LR | 0.7862 | 0.4632 | -- | -- |
| FM | 0.7890 | 0.4603 | -- | -- |
| FNN | 0.7891 | 0.4603 | -- | -- |
| IPNN | 0.7906 | 0.4585 | -- | -- |
| OPNN | 0.7904 | 0.4586 | -- | -- |
| Wide & Deep | 0.7907 | 0.4584 | -- | -- |
| **DeepFM** | **0.8007** | **0.4490** | -- | -- |

> 注：Company 数据集的趋势与 Criteo 一致，DeepFM 在两个数据集上均取得最优

#### 结果分析

DeepFM 在 Criteo 上的 AUC 为 0.8007，相比 Wide & Deep 的 0.7907 提升了 1 个百分点（AUC 绝对值），Logloss 从 0.4584 降至 0.4490。论文强调 DeepFM 的提升主要来自三个方面：FM 自动捕获二阶交互替代了人工交叉特征；Embedding 共享让低阶和高阶学习相互增益；端到端训练避免了 FNN 式两阶段训练的信息损失。

### 消融实验

#### 实验设计

论文通过大量超参数实验探究了网络深度、宽度、激活函数、Dropout 等的影响。

![[layer-auc.png|600]]

> 图6：DNN 层数对 AUC 的影响。3 层效果最佳，更深反而过拟合。

![[neuron-auc.png|600]]

> 图7：每层神经元数量对 AUC 的影响。

#### 消融结果和分析

- **网络深度**：3 层 DNN 效果最佳（AUC 最高），增加到 5 层后性能略降，说明过深网络在 CTR 数据上容易过拟合
- **网络宽度**：每层 400 个神经元是较优选择，进一步增加带来的提升有限
- **激活函数**：ReLU 优于 Sigmoid 和 Tanh
- **Dropout**：0.6-0.9 的保留率最优

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文结论部分较为简洁，主要建议探索 GPU 上的高效实现以及在更多场景中验证 DeepFM 的有效性。

### 4.2 基于分析的未来方向

1. **方向1：超越二阶的显式交叉**
   - 动机：FM 只能建模二阶交互，三阶及以上仍依赖 DNN 隐式学习
   - 可能的方法：CIN（Compressed Interaction Network）显式建模任意阶交互（即 xDeepFM）
   - 预期成果：更精确的高阶交互建模
   - 挑战：高阶交叉的计算复杂度

2. **方向2：特征重要性感知**
   - 动机：DeepFM 对所有特征等同对待，但实际中不同特征的重要性差异很大
   - 可能的方法：引入 SENet 风格的特征重要性加权（即 FiBiNET 的思路）
   - 预期成果：动态调整特征权重，提升交互质量
   - 挑战：额外的参数和计算开销

### 4.3 改进建议

1. **改进1：Bit-wise vs Vector-wise 交互**
   - 当前问题：FM 的交互是 bit-wise 的（Embedding 各维度独立参与内积）
   - 改进方案：引入 bilinear 或 attention 机制实现 vector-wise 交互
   - 预期效果：更丰富的交互模式

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.0/10** - DeepFM 在 Wide & Deep 基础上做了恰到好处的改进（FM 替代人工交叉 + Embedding 共享），虽然创新幅度不大，但其简洁优雅的设计和出色的实验效果使其成为 CTR 预估领域最常被引用的模型之一。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | FM+DNN 的组合思路并不意外，但 Embedding 共享的设计很巧妙 |
| 技术质量 | 8/10 | 方法清晰，与 FNN/PNN/Wide&Deep 的对比分析到位 |
| 实验充分性 | 8/10 | Criteo + 华为内部数据，多基线对比，丰富的超参实验 |
| 写作质量 | 8/10 | 结构清晰，图表丰富，对比分析有条理 |
| 实用性 | 9/10 | 端到端无需特征工程，易于工业部署，已成为行业标配 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Embedding 共享是 DeepFM 的灵魂——FM 学到的交互模式直接指导 DNN 的高阶学习
- FM 替代人工交叉特征，实现了 Wide & Deep 的完全自动化
- 与 FNN 的对比说明端到端训练优于两阶段训练

#### 5.2.2 需要深入理解的部分

- Embedding 共享对训练动态的影响：FM 梯度和 DNN 梯度都会更新同一组 Embedding，是否存在梯度冲突？
- FM 的二阶交互在 CTR 场景中的实际覆盖率有多高？即二阶交互能解释多少比例的预测？

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wide_and_Deep|Wide & Deep]] - DeepFM 的直接改进对象，FM 替代了 Wide 中的人工交叉特征
- [[FiBiNET|FiBiNET]] - 在 DeepFM 基础上引入 SENet 和 Bilinear 交互

### 6.2 背景相关
- [[GBDT_LR|GBDT+LR]] - 特征自动化的前序思路
- Rendle, S. "Factorization Machines" - FM 的原始论文

### 6.3 后续工作
- [[xDeepFM|xDeepFM]] - 用 CIN 实现 vector-wise 的显式高阶交叉
- [[DCN|DCN]] - 另一条自动化特征交叉的路线（Cross Network）
- [[AutoInt|AutoInt]] - 用 Multi-head Self-Attention 学习特征交互

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1703.04247)
- [知乎深度解读：DeepFM 原理与实现](https://zhuanlan.zhihu.com/p/57873613)

> [!tip] 关键启示
> FM 是 Wide 部分的天然替代品——通过因子化的二阶交互自动建模所有特征对，加上 Embedding 共享让低阶和高阶学习相互增益，DeepFM 实现了 CTR 预估的完全端到端训练，彻底摆脱了人工特征工程的束缚。

> [!warning] 注意事项
> - DeepFM 的 FM 部分只能建模二阶交互，更高阶的交互完全依赖 DNN 隐式学习
> - 在超大规模稀疏特征场景中，所有 pair-wise 交互的计算开销可能较大
> - 论文未提供代码和详细的工业部署信息

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。DeepFM 是理解"特征交叉自动化"这一核心命题的重要里程碑，它优雅地解决了 Wide & Deep 的人工特征依赖问题，是后续 xDeepFM、DCN V2 等更复杂交叉网络的重要参照基线。
