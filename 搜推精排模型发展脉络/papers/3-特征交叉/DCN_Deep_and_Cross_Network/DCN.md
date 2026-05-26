---
paper_id: "[arXiv:1708.05123](https://arxiv.org/abs/1708.05123)"
title: "Deep & Cross Network for Ad Click Predictions"
authors: "Ruoxi Wang, Bin Fu, Gang Fu, Mingliang Wang"
institution: "Google / Stanford University"
publication: "ADKDD 2017 2017-08-15"
tags:
  - 精排论文
  - DCN
  - Cross-Network
  - 特征交叉
  - CTR预估
  - 显式交叉
  - 参数高效
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/1708.05123)"
  - "[arXiv Source](https://arxiv.org/e-print/1708.05123)"
date: "2017-08-15"
---

## 一、研究背景与动机

### 1.1 领域现状

Web 规模的推荐系统面临的核心挑战之一是如何有效地从海量稀疏且大多数为离散的特征中识别出有效的特征交叉。在 CTR 预估领域，特征交叉的质量直接决定了模型的表达能力。传统的线性模型（如 LR）依赖手工构造的交叉特征实现"记忆"，但人工交叉的阶数受限于工程人员的领域知识和计算资源。DNN 虽然理论上能通过多层非线性变换隐式地学习任意阶特征交互，但其学习效率和可解释性都存在不足——网络需要巨大的参数量才能逼近简单的乘性关系。

论文明确指出了两个关键需求：（1）避免 task-specific 的特征工程；（2）以一种高效且可扩展的方式显式建模特征交叉。

### 1.2 现有方法的局限性

论文分析了 DNN 在特征交叉建模上的固有缺陷。DNN 的每一层本质上是线性变换加非线性激活：$\mathbf{h}_{l+1} = \sigma(\mathbf{W}_l \mathbf{h}_l + \mathbf{b}_l)$。这种加性结构在逼近乘性交叉模式（如 $x_i \cdot x_j \cdot x_k$）时效率低下，需要大量参数和训练数据。残差网络（ResNets）通过跨层连接将表征阶数从 $O(1)$ 提升到 $O(L)$（$L$ 为层数），但其本质仍是加性的，无法高效地实现显式的乘性交叉。

此外，FM 虽然能自动建模特征交叉，但被限制在二阶。更高阶的 FM 扩展在参数量和计算复杂度上面临指数增长。Wide & Deep 的 Wide 部分仍需手动设计交叉特征，无法自动学习。

### 1.3 本文解决方案概述

DCN 提出了一种新的网络结构——Cross Network，通过特殊的递推公式在每一层自动施加一次与原始输入的外积操作，从而在第 $l$ 层自动包含所有不超过 $l+1$ 阶的特征交叉项。Cross Network 与 DNN 并行组成 DCN 的双塔结构：Cross Network 负责以极高的参数效率（$O(dL)$）学习显式的有界阶特征交叉，DNN 负责补充隐式的非线性学习能力。两者共享底部的 Embedding + Stacking 层，顶部拼接后通过 logits 层输出预测。

## 二、解决方案

### 2.1 核心思想

Cross Network 的核心洞察是：如果第 $l+1$ 层的输出被设计为 $\mathbf{x}_0$ 与 $\mathbf{x}_l$ 的外积的线性投影加上残差连接，那么 $l$ 层 Cross Network 的输出就是 $\mathbf{x}_0$ 各分量的 $(l+1)$ 阶多项式。论文通过 Theorem 1 严格证明了这一点。

关键在于，这个多项式的系数不是独立参数化的，而是由 $O(dL)$ 个参数共同决定的。这意味着 Cross Network 实现了一种参数共享的多项式拟合——它并不完全覆盖 $(l+1)$ 阶多项式的所有项（那需要 $O(d^{l+1})$ 个独立系数），而是在一个受限但有效的子空间内建模交叉。

### 2.2 整体架构

![[deep_cross_network.png|800]]

> 图1：DCN 整体架构。底部 Embedding & Stacking 层将稀疏特征和稠密特征统一为向量 $\mathbf{x}_0$，左侧 Cross Network 通过多层递推实现显式特征交叉，右侧 Deep Network 通过多层全连接实现隐式非线性学习，顶部 Combination Output Layer 将两路输出拼接后通过 sigmoid 预测点击概率。

#### 各模块详细说明

**模块1：Embedding and Stacking 层**

- **功能**：将异构的输入特征（稀疏类别特征 + 连续数值特征）统一为一个稠密向量
- **处理流程**：对每个类别特征 $i$，学习一个 embedding 向量 $\mathbf{x}_{embed,i} \in \mathbb{R}^{n_e}$；然后将所有 embedding 与归一化后的连续特征拼接：

$$\mathbf{x}_0 = [\mathbf{x}_{embed,1}^T, \dots, \mathbf{x}_{embed,k}^T, x_{dense}^T] \in \mathbb{R}^d$$

论文在 Criteo 数据集上使用 6 维的 embedding（$n_e = 6$），最终 $d = 26 \times 6 + 13 = 169$。

**模块2：Cross Network**

- **功能**：以参数高效的方式显式学习有界阶数的特征交叉
- **递推公式**：

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$$

其中 $\mathbf{w}_l, \mathbf{b}_l \in \mathbb{R}^d$ 是第 $l$ 层的参数。$\mathbf{x}_0 \mathbf{x}_l^T$ 计算了原始输入与当前层表征的外积（本质上是 rank-1 矩阵），$\mathbf{w}_l$ 将这个外积投影回 $d$ 维空间。残差连接 $+ \mathbf{x}_l$ 保证了至少保留前一层的所有信息。

![[cross_layer.png|800]]

> 图2：一层 Cross Layer 的计算过程可视化。输入 $\mathbf{x}_l$ 与 $\mathbf{x}_0$ 做外积后由 $\mathbf{w}_l$ 投影，加上偏置和残差得到 $\mathbf{x}_{l+1}$。

- **关键性质**：
  - 第 $l$ 层输出中包含所有不超过 $l+1$ 阶的交叉项（Theorem 1）
  - 每层输出维度始终为 $d$，不会引发维度膨胀
  - 总参数量：$L_c$ 层 Cross Network 仅需 $d \times L_c \times 2$ 个参数
  - 时间和空间复杂度均为 $O(d \times L_c)$
  - 实际计算时无需显式构造 $\mathbf{x}_0 \mathbf{x}_l^T$ 这个 $d \times d$ 矩阵——先计算 $\mathbf{x}_l^T \mathbf{w}_l$（标量），再乘以 $\mathbf{x}_0$

**模块3：Deep Network**

- **功能**：学习高度非线性的特征交互
- **结构**：标准的多层全连接网络，使用 ReLU 激活
- **公式**：$\mathbf{h}_{l+1} = \text{ReLU}(\mathbf{W}_l \mathbf{h}_l + \mathbf{b}_l)$
- **参数量**：$d \times m + m + (L_d - 1) \times (m \times m + m)$，其中 $m$ 为隐层宽度

**模块4：Combination Output Layer**

- **功能**：融合 Cross Network 和 Deep Network 的输出，产生最终预测
- **公式**：

$$p = \sigma\left(\mathbf{w}_{logits}^T [\mathbf{x}_{L_c}, \mathbf{h}_{L_d}] + b_{logits}\right)$$

损失函数为带 L2 正则化的 logloss：

$$\text{loss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right] + \lambda \sum_l \|\mathbf{w}_l\|^2$$

### 2.3 Cross Network 的理论分析

**Theorem 1（多项式逼近）**：设 $L_c$ 层 Cross Network 的第 $i$ 个输出为 $g_l(x_0)$。则 $g_l$ 是关于 $\mathbf{x}_0$ 的多项式，阶数为 $l+1$。具体形式为：

$$g_l(\mathbf{x}_0) = \sum_{\alpha} c_\alpha (\mathbf{w}_0, \dots, \mathbf{w}_l, \mathbf{b}_0, \dots, \mathbf{b}_l) \cdot x_0^{[\alpha_1]} x_0^{[\alpha_2]} \cdots x_0^{[\alpha_{l+1}]}$$

这里 $\alpha$ 遍历所有不超过 $l+1$ 阶的多重指标，而系数 $c_\alpha$ 由网络的 $O(dL_c)$ 个参数共同确定。

论文还分析了 Cross Network 与高阶 FM 的关系：$O(d)$ 的 Cross Network 参数子集足以重构整个 FM 模型。这意味着 Cross Network 的能力至少不弱于 FM。

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 特征类型 | 用途 |
|--------|--------|----------|----------|------|
| Criteo Display Ads | 4500万行 | 39维（13连续+26类别） | 广告点击 | CTR 主实验 |
| Forest Covertype | 581,012 | 54维（10连续+44二值） | 森林覆盖类型 | 非CTR验证 |
| Higgs | 1100万 | 28维（21低级+7高级） | 粒子物理 | 非CTR验证 |

### 3.2 实验设置

#### 3.2.1 基线方法

论文对比了以下方法：

- **LR（Logistic Regression）**：线性基线
- **FM（Factorization Machines）**：自动二阶交叉
- **DNN（Deep Neural Network）**：纯深度网络基线
- **DC（Deep Crossing）**：残差网络结构
- **DCN（本文方法）**：Cross Network + Deep Network

#### 3.2.2 评估指标

- **Logloss（对数损失）**：主要评估指标。论文强调在大规模 CTR 场景下，0.001 量级的 logloss 改善在实际系统中可能带来显著的收入提升

#### 3.2.3 训练细节

- 优化器：Adam
- Batch size：512
- Batch normalization 应用于 Deep Network
- Gradient clip norm：100
- L2 正则化
- Early stopping 基于验证集

论文在 Criteo 上搜索了以下超参数：hidden layers 数量（2-5）、hidden layer size（32-1024）、cross layers 数量（1-6）、学习率、正则系数。最终 DCN 使用 6 层 cross layer + 2 层 1024 宽的 deep layer。

### 3.3 实验结果与分析

#### CTR 预测结果（Criteo）

| 方法 | Logloss | 参数量 |
|------|---------|--------|
| LR | 0.4474 | -- |
| FM | 0.4464 | -- |
| DC (Deep Crossing) | 0.4425 | -- |
| DNN (best) | 0.4428 | 约 530K |
| **DCN (best)** | **0.4419** | 约 320K |

DCN 相比 DNN 将 logloss 降低了 0.0009，同时参数量减少约 40%。与传统方法相比，DCN 相对 LR 提升 0.0055，相对 FM 提升 0.0045。

![[logloss_vs_crosslayers.png|800]]

> 图3：Cross layer 数量对 Logloss 的影响。图中展示了不同 cross layer 数量（1-6层）在不同 DNN 配置下的 logloss 曲线。随着 cross layer 增加，logloss 整体呈下降趋势，验证了更高阶交叉对模型的正向贡献。最优配置为 6 层 cross layer。

#### 非 CTR 任务结果

| 数据集 | DCN 准确率 | DNN 准确率 | 提升 |
|--------|-----------|-----------|------|
| Forest Covertype | 0.9740 | 0.9725 | +0.0015 |
| Higgs | 0.7347 (logloss: 0.4494) | -- | -- |

非 CTR 任务的结果表明 Cross Network 对特征交叉的建模能力不限于广告场景，在分类任务中同样有效。

#### 结果分析

DCN 的优势体现在两个维度：（1）在相同参数预算下，DCN 的 logloss 低于 DNN，说明 Cross Network 以更高效的方式捕获了 DNN 需要大量参数才能隐式学习的交叉模式；（2）DCN 使用约 DNN 60% 的参数量就达到了同等或更优的效果，参数效率显著提高。

论文还发现 Cross Network 的层数存在最优点——超过 6 层后改善趋于平缓，这与多项式阶数的实际有效性有关（过高阶的交叉项在真实数据中可能贡献有限或引入噪声）。

### 3.4 消融实验

论文通过对比不同组件组合来验证各模块的贡献：

| 配置 | Logloss | 说明 |
|------|---------|------|
| Cross Network only | 0.4447 | 优于 LR(0.4474) 和 FM(0.4464)，说明显式高阶交叉有效 |
| Deep Network only (DNN) | 0.4428 | 纯深度网络基线 |
| DCN (Cross + Deep) | 0.4419 | 两者互补，最优结果 |

消融结果表明：（1）单独的 Cross Network 已经超越了 LR 和 FM，验证了自动高阶显式交叉的有效性；（2）但单独 Cross Network 不如 DNN，说明纯多项式交叉（缺乏非线性激活）的灵活性不足；（3）Cross + Deep 的组合效果最优，两种学习方式存在明确的互补关系。

论文还分析了 Cross Network 参数数量与 DNN 参数数量的关系。在 Criteo 上，6 层 Cross Network 仅需 $169 \times 6 \times 2 = 2028$ 个参数，相比 DNN 的 1024×1024 量级的权重矩阵，效率差异高达数百倍。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中提到了几个方向：（1）探索更多的 Cross Network 变体以进一步提升表达能力；（2）将 DCN 应用到更多预测任务中；（3）研究 Cross Network 的理论性质，如泛化能力和收敛特性。

### 4.2 基于分析的未来方向

1. **方向1：提升 Cross Network 的秩**
   - 动机：当前 Cross Network 每层的交叉投影本质上是秩1的（$\mathbf{x}_0 \mathbf{w}_l^T$ 是 rank-1 矩阵），这严重限制了每层能捕获的交叉模式的多样性
   - 可能的方法：用全秩矩阵 $\mathbf{W}_l \in \mathbb{R}^{d \times d}$ 替代向量 $\mathbf{w}_l$，即 $\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$（DCN V2 的方案）
   - 预期成果：在 $O(d^2 L)$ 参数下实现更丰富的交叉模式
   - 挑战：参数量增长需要通过低秩分解（mixture of experts）控制

2. **方向2：自适应交叉阶数**
   - 动机：当前交叉阶数与层数绑定，不同特征对可能需要不同的交叉阶数
   - 可能的方法：引入门控机制控制每层的交叉强度
   - 预期成果：更灵活的交叉阶数分配

### 4.3 改进建议

1. **改进1：引入特征选择机制**
   - 当前问题：Cross Network 对所有特征维度施加同等的交叉，但实际中不同特征对的交叉价值差异很大
   - 改进方案：在外积前增加 attention 或 gating 机制，选择性地增强有价值的交叉
   - 预期效果：减少无效交叉的噪声，提升模型精度

2. **改进2：Bit-wise 到 Vector-wise 的交叉**
   - 当前问题：DCN 在 embedding 拼接后的 bit-wise 层面做交叉，丢失了特征域的语义边界
   - 改进方案：在 feature-wise 或 vector-wise 层面设计交叉操作
   - 预期效果：更具语义的交叉模式

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.0/10** - DCN 首次将"显式有界阶特征交叉"这一概念用极简且优雅的数学形式实现，以 $O(dL)$ 的参数量完成了 DNN 需要 $O(d^2 L)$ 才能隐式逼近的交叉建模，开创了一条重要的研究路线。其影响力体现在后续 DCN V2、DCN V3、CIN (xDeepFM) 等一系列工作都以 Cross Network 的思想为基础进行扩展。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | Cross Network 的递推设计巧妙地在每层引入一次外积交叉，是全新的网络设计范式 |
| 技术质量 | 7/10 | Theorem 1 的多项式阶数证明严谨，但 rank-1 约束是一个较强的限制 |
| 实验充分性 | 6/10 | 仅 Criteo 一个公开 CTR 数据集作为主实验，基线方法偏少，缺少在线 A/B 测试结果 |
| 写作质量 | 8/10 | Google 论文风格，结构清晰，数学推导完整，但论文偏短缺少细节讨论 |
| 实用性 | 8/10 | 参数高效、实现简单（核心代码仅数行），工业界广泛采用 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Cross Network 的 $O(dL)$ 参数量极其经济——6 层 Cross Network 在 $d=169$ 时仅需 2028 个参数，而等效能力的 DNN 需要百万量级参数
- 递推公式中 $\mathbf{x}_l^T \mathbf{w}_l$ 先计算得到标量再乘以 $\mathbf{x}_0$ 的技巧避免了显式构造外积矩阵，将计算复杂度从 $O(d^2)$ 降为 $O(d)$
- Cross Network 与 FM 的理论联系：$O(d)$ 参数子集可以重构 FM，证明了 Cross Network 的表达能力下界

#### 5.2.2 需要深入理解的部分

- Rank-1 限制的实际影响：每层只有一个 $\mathbf{w}_l$ 向量来选择交叉方向，这意味着每层只能捕获一种"交叉模式"。DCN V2 通过升秩和 mixture of experts 解决了这一限制
- Cross Network 的梯度流：由于残差连接和与 $\mathbf{x}_0$ 的直接相乘，梯度可以高效地流回底层，这与 ResNet 类似
- 参数共享的多项式逼近 vs 完全自由的多项式逼近之间的 trade-off：DCN 选择了前者以换取参数效率，但可能遗漏了某些重要的高阶交叉模式

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wide_and_Deep|Wide & Deep]] - DCN 可视为 Wide & Deep 的改进，用 Cross Network 替代人工交叉的 Wide 部分
- [[DeepFM|DeepFM]] - 同时期的另一条自动交叉路线，用 FM 建模二阶交叉，与 DCN 的高阶显式交叉形成互补思路
- [[Deep_Crossing|Deep Crossing (DC)]] - 微软 2016 年的残差网络方案，论文中的对比基线之一

### 6.2 背景相关
- [[FM|Factorization Machines]] - 二阶特征交叉的经典方法，Cross Network 的表达能力严格覆盖 FM
- [[ResNets|残差网络 (ResNets)]] - 残差连接的灵感来源，Cross Network 的 $+\mathbf{x}_l$ 项保证了低阶信息的传递

### 6.3 后续工作
- [[DCN_V2|DCN V2 (2020)]] - 将 rank-1 升级为全秩矩阵 + mixture of low-rank experts，大幅提升了交叉的表达能力
- [[DCN_V3|DCN V3]] - 进一步引入 Exponential Cross Network，不再受限于多项式形式
- [[xDeepFM|xDeepFM (CIN)]] - 另一条显式高阶交叉路线，在 vector-wise 层面做交叉
- [[AutoInt_Automatic_Feature_Interaction_Learning|AutoInt]] - 用 Multi-head Self-Attention 自动学习特征交互

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1708.05123)
- [TensorFlow 官方实现](https://github.com/tensorflow/models/tree/master/official/recommendation/ranking)

> [!tip] 关键启示
> Cross Network 用极简递推公式 $\mathbf{x}_{l+1} = \mathbf{x}_0 \mathbf{x}_l^T \mathbf{w}_l + \mathbf{b}_l + \mathbf{x}_l$ 实现了自动的显式高阶特征交叉。其核心价值不仅在于参数效率（$O(dL)$ vs DNN 的 $O(d^2L)$），更在于建立了一种新的设计范式——将特征交叉从"隐式学习"转变为"显式建模"，为后续 DCN V2/V3、xDeepFM、AutoInt 等工作奠定了理论基础。

> [!warning] 注意事项
> - Cross Network 每层的交叉投影是 rank-1 的，表达能力受限——每层只能捕获一种交叉方向。DCN V2 已通过全秩矩阵修复
> - 论文实验仅在 Criteo 一个公开 CTR 数据集上验证，缺少在线实验和多场景验证
> - 交叉阶数与层数强绑定，增加阶数必须增加层数，缺乏灵活性
> - 参数共享的多项式不能覆盖所有 $(l+1)$ 阶交叉项，某些重要模式可能被遗漏

> [!success] 推荐指数
> ⭐⭐⭐⭐ 强烈推荐阅读。DCN 是显式特征交叉网络的开山之作，理解其设计动机（为什么要显式交叉）、数学性质（多项式阶数与 rank-1 约束）和工程实现（外积计算的优化技巧）是深入理解整个特征交叉网络演进路线的必要基础。
