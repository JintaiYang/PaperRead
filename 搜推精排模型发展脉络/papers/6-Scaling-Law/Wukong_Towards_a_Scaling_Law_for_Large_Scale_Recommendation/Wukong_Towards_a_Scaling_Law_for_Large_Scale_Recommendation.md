---
paper_id: "[arXiv:2403.02545](https://arxiv.org/abs/2403.02545)"
title: "Wukong: Towards a Scaling Law for Large-Scale Recommendation"
authors: "Buyun Zhang*, Liang Luo*, Yuxin Chen*, et al."
institution: "Meta AI"
pushlication: "ICML 2024, 2024-03-04"
tags:
  - Scaling-Law
  - 推荐系统
  - Factorization-Machine
  - 特征交互
  - Dense-Scaling
  - 大规模推荐
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2403.02545)"
date: "2024-03-04"
---

## 一、研究背景与动机

### 1.1 领域现状

大语言模型（LLM）领域已经建立了清晰的 Scaling Law：模型质量随计算量的增加呈幂律提升，这一规律指导了 GPT-4、PaLM 等模型的成功开发。然而，在深度学习推荐系统（DLRS）领域，尽管推荐模型消耗了全球数据中心超过一半的 AI 训练周期和超过 70% 的 AI 推理周期，却一直缺乏类似的 Scaling Law 来指导模型的规模化发展。

现有的推荐系统模型（如 DLRM、DCNv2、AutoInt+、MaskNet 等）主要通过增加 Embedding 表的大小来扩展模型容量，即所谓的"稀疏扩展"（Sparse Scaling）。这种方式虽然增加了模型参数量，但并未显著增加计算量（FLOPs），因为 Embedding 查找本质上是 O(1) 操作。

### 1.2 现有方法的局限性

论文指出现有方法存在三个核心问题：

1. **稀疏扩展的收益递减**：当 Embedding 维度超过一定阈值后，继续增大 Embedding 表带来的质量提升迅速衰减。实验表明，在 Criteo 数据集上，当 Embedding 维度从 64 增加到 128 时有明显提升，但继续增大到 256、512 时提升微乎其微。

2. **Dense 部分缺乏有效的扩展方式**：现有模型的 Dense 部分（如 MLP、Cross Network）在增大规模时往往出现训练不稳定（Loss 爆炸）或质量下降的问题。例如 DCNv2 在 GFLOP 从 3 增加到 85 时，模型质量反而持续下降。

3. **缺乏统一的 Scaling Law**：不同模型架构在扩展时表现差异巨大，没有一个统一的框架来预测"增加多少计算量能带来多少质量提升"。

### 1.3 本文解决方案概述

本文提出了 Wukong（悟空）架构，一种专为推荐系统设计的、能够展现 Scaling Law 的网络结构。其核心思想是通过堆叠 Factorization Machine Block（FMB）和 Linear Compression Block（LCB）构成的 Interaction Stack，以二进制指数方式高效捕获高阶特征交互，同时通过金字塔形的压缩结构控制计算复杂度。实验证明 Wukong 在超过 2 个数量级的计算范围内（>100 GFLOP）展现出稳定的幂律 Scaling Law。

## 二、解决方案

### 2.1 核心思想

Wukong 的核心洞察是：推荐系统的质量提升关键在于高效捕获高阶特征交互。传统 FM 只能捕获 2 阶交互，而 Wukong 通过堆叠 FM 层并结合残差连接，使得第 $i$ 层能够捕获 1 到 $2^i$ 阶的所有交互——这是一种"二进制指数"增长模式。

具体来说，第 1 层 FMB 输入包含 1 阶信息 $X^1$，输出 2 阶交互 $X^2$；第 2 层通过残差连接接收 $X^1 + X^2$，其点积运算 $(X^1 + X^2)(X^1 + X^2)^T$ 产生 1 到 4 阶交互；以此类推，$l$ 层 Wukong 能捕获 1 到 $2^l$ 阶交互。这比 Transformer 类方法（如 AutoInt+）更高效——后者每层只产生奇数阶交互（1, 3, 5, ...），$l$ 层只能到 $3^l$ 阶但缺少偶数阶。

### 2.2 整体架构

Wukong 的整体架构由三部分组成：Embedding Layer → Interaction Stack → MLP Head。

![[arch.png|697]]

> 图1：Wukong 整体架构。左侧展示了完整的网络结构，右侧详细展示了 Factorization Machine Block（FMB）的内部结构。Interaction Stack 由多层 FMB + LCB 组成，呈金字塔形逐层压缩特征数量。

#### 各模块详细说明

**模块1：Embedding Layer**
- **功能**：将稀疏特征（categorical features）转换为稠密向量表示
- **输入**：$n$ 个稀疏特征，每个特征通过 Embedding 表查找得到 $d$ 维向量
- **输出**：$X_0 \in \mathbb{R}^{n \times d}$，即 $n$ 个 $d$ 维 Embedding 向量的矩阵
- **关键设计**：所有模型统一使用 128 维 Embedding，通过 Bottom MLP 将连续特征也映射到相同维度

**模块2：Interaction Stack（核心创新）**

Interaction Stack 是 Wukong 的核心，由 $l$ 层堆叠而成，每层包含一个 FMB 和一个 LCB，通过残差连接组合：

$$X_{i+1} = \text{LN}(\text{concat}(\text{FMB}_i(X_i), \text{LCB}_i(X_i)) + X_i)$$

其中 LN 为 Layer Normalization。

**Factorization Machine Block (FMB)**：
- **功能**：捕获特征间的交互信息
- **输入**：$X_i \in \mathbb{R}^{n_i \times d}$（$n_i$ 个特征，每个 $d$ 维）
- **处理流程**：
  1. 通过 MLP 将输入投影为 $k$ 组表示：$U = \text{MLP}(X_i) \in \mathbb{R}^{n_i \times kd}$，reshape 为 $U \in \mathbb{R}^{k \times n_i \times d}$
  2. 对每组计算点积交互：$Z_j = U_j U_j^T Y_j$，其中 $Y_j$ 也由 MLP 投影得到
  3. 输出 $n_F$ 个特征向量（$n_F < n_i$，实现压缩）
- **输出**：$\text{FMB}(X_i) \in \mathbb{R}^{n_F \times d}$
- **关键优化**：使用 Optimized FM 将复杂度从 $O(n^2 d)$ 降低到 $O(nkd)$

![[dot-compress.png|800]]

> 图2：Optimized FM 的计算优化。通过将 $XX^T Y$ 分解为 $(XY^T) \cdot X$ 的形式（利用 $k \ll n$ 的性质），将二次复杂度降为线性复杂度。

**Linear Compression Block (LCB)**：
- **功能**：线性压缩特征数量，保留低阶信息
- **输入**：$X_i \in \mathbb{R}^{n_i \times d}$
- **处理**：通过线性投影将 $n_i$ 个特征压缩为 $n_L$ 个：$\text{LCB}(X_i) = W X_i$，其中 $W \in \mathbb{R}^{n_L \times n_i}$
- **输出**：$\text{LCB}(X_i) \in \mathbb{R}^{n_L \times d}$
- **设计意图**：与 FMB 的非线性交互互补，提供直接的线性信息通路

每层输出的特征数为 $n_{i+1} = n_F + n_L + n_i$（concat 后加残差），但由于 $n_F, n_L < n_i$，整体呈金字塔形逐层收缩。

**模块3：MLP Head**
- **功能**：将 Interaction Stack 的输出映射为最终预测
- **输入**：Interaction Stack 最后一层的输出，flatten 为一维向量
- **处理**：多层全连接网络（典型配置：3 层，每层 2048-16384 单元）
- **输出**：标量预测值（CTR 预估）

### 2.3 数学形式化

Wukong 的高阶交互捕获能力可以形式化为：$l$ 层 Wukong 最小化的目标为：

$$\min \sum_{i, j \in S} \left( r_{ij} - \sum_{k \in \{1, 2, ..., 2^{l-1}\}} X^k {X^k}^T \right)$$

其中 $r_{ij}$ 是用户 $i$ 对物品 $j$ 的评分，$X^k$ 表示包含第 $k$ 阶信息的表示。相比之下，传统 FM 只能最小化 $r_{ij} - X^1 {X^1}^T$（仅 2 阶交互）。

### 2.4 与 Transformer 的关键区别

Wukong 的结构虽然与 Transformer 相似（都是堆叠的注意力/交互层），但有两个关键区别：

1. **Bit-wise MLP vs. Position-wise FFN**：Wukong 使用 bit-wise MLP（对 flatten 后的所有特征维度操作），而 Transformer 使用 position-wise FFN（对每个 token 独立操作）。这使得 Wukong 能为异构特征学习不同的投影矩阵，更适合推荐场景中特征类型多样的情况。

2. **金字塔形 vs. 均匀形**：Wukong 逐层压缩特征数量（金字塔形），而 Transformer 保持每层维度不变。这使得 Wukong 能排除不必要的计算，在相同 FLOPs 下获得更好的质量。

实验验证：将 Wukong 的组件逐步应用到 AutoInt+ 上，V=FFN→V=MLP 提升 LogLoss 0.34%，加入 Layer MLP 提升 0.65%，组合金字塔形状后达到 0.57% 提升且节省 90% FLOPs。

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征数 | 类型 | 来源 |
|--------|--------|--------|------|------|
| Frappe | 288K | 10 | App 推荐 | 公开 |
| MicroVideo | 12.7M | 9 | 短视频推荐 | 公开 |
| MovieLens | 2M | 3 | 电影推荐 | 公开 |
| KuaiVideo | 12.5M | 15 | 短视频推荐 | 公开 |
| TaobaoAds | 26M | 23 | 广告点击 | 公开 |
| Criteo TB | 4B | 39 | 广告点击 | 公开 |
| Internal | 146B | 720 | 推荐 | Meta 内部 |

### 3.2 实验设置

#### 3.2.1 基线方法

论文对比了 7 个代表性的推荐模型：
- **DLRM**：Meta 的经典推荐模型，使用 Bottom MLP + Dot Product Interaction + Top MLP
- **DCNv2**：Cross Network v2，使用显式的交叉层捕获特征交互
- **AutoInt+**：基于 Multi-head Self-Attention 的特征交互模型
- **AFN+**：使用对数神经元的自适应因子分解网络
- **MaskNet**：使用 Instance-guided Mask 的特征交互模型
- **FinalMLP**：双流 MLP 架构，使用 Feature Selection 机制
- **xDeepFM**：使用 Compressed Interaction Network (CIN) 的模型

#### 3.2.2 评估指标

- **AUC**（Area Under ROC Curve）：公开数据集上的主要指标
- **Normalized Entropy (NE)**：内部数据集上的主要指标，即 LogLoss 除以数据集熵的归一化值
- **Relative LogLoss (RLL)**：Scaling 实验中使用，相对于 DLRM baseline 的 LogLoss 改进百分比

#### 3.2.3 实验细节

所有模型使用统一的超参数搜索空间，进行了超过 3000 次实验运行。使用 Adam 优化器（Dense 部分）和 Rowwise AdaGrad（Sparse 部分），全局 batch size 为 131,072，Embedding 维度统一为 128，使用 FP32 精度训练。

### 3.3 实验结果与分析

#### 公开数据集结果

| 方法 | Frappe (AUC) | MicroVideo (AUC) | MovieLens (AUC) | KuaiVideo (AUC) | TaobaoAds (AUC) | Criteo TB (AUC) |
|------|-------------|-----------------|----------------|----------------|----------------|----------------|
| AFN+ | 0.9872 | 0.6862 | 0.9700 | 0.6672 | 0.6413 | 0.8098 |
| AutoInt+ | 0.9860 | 0.6862 | 0.9694 | 0.6680 | 0.6413 | 0.8098 |
| DCNv2 | 0.9873 | 0.6862 | 0.9700 | 0.6672 | 0.6413 | 0.8098 |
| DLRM | 0.9849 | 0.6856 | 0.9693 | 0.6668 | 0.6399 | 0.8093 |
| FinalMLP | 0.9862 | 0.6862 | 0.9697 | 0.6672 | 0.6413 | 0.8098 |
| MaskNet | 0.9870 | 0.6862 | 0.9700 | 0.6672 | 0.6413 | 0.8098 |
| xDeepFM | 0.9838 | 0.6862 | 0.9693 | 0.6672 | 0.6413 | 0.8098 |
| **Wukong** | **0.9876** | **0.6870** | **0.9706** | **0.6690** | **0.6420** | **0.8106** |

> Wukong 在所有 6 个公开数据集上均取得最优 AUC，在 Criteo TB 上达到 0.8106，超越所有基线。

#### 内部数据集 Scaling 结果

![[scaling.png|800]]

> 图3：各模型在 Meta 内部 146B 数据集上的 Scaling 曲线。横轴为 GFLOP/example，纵轴为 Relative LogLoss（越低越好）。Wukong 是唯一在整个计算范围内持续改善的模型。

内部数据集上的关键发现：

| 模型 | 最大 GFLOP | 最佳 RLL (Task1) | 最佳 RLL (Task2) | Scaling 趋势 |
|------|-----------|-----------------|-----------------|-------------|
| DLRM | 71.23 | -0.37 | -0.35 | 饱和 |
| DCNv2 | 84.71 | -0.43 | -0.45 | 持续下降（变差） |
| AutoInt+ | 68.83 | 0.15 | 0.05 | Loss 爆炸 |
| MaskNet | 64.21 | -0.40 | -0.40 | 饱和 |
| FinalMLP | 58.12 | -0.37 | -0.38 | 饱和 |
| AFN+ | 43.4 | 0.21 | 0.14 | 变差 |
| **Wukong** | **108** | **-0.76** | **-0.76** | **持续改善** |

Wukong 在 108 GFLOP 时达到 -0.76% RLL，远超所有基线。更重要的是，其 Scaling 曲线呈现清晰的幂律关系，没有饱和迹象。

![[scaling-params.png|800]]

> 图4：模型质量 vs. 参数量。Wukong 在相同参数量下质量远优于其他模型，说明其优势来自架构设计而非单纯的参数增加。

### 消融实验

#### 实验设计

消融实验在 Meta 内部数据集上进行，逐步移除 Wukong 的各个组件来验证其贡献。

![[ablation-component.png|800]]

> 图5：组件消融实验。分别移除 FMB、LCB、Layer Norm、Residual Connection 后的性能变化。

#### 消融结果和分析

1. **FMB 的贡献**：移除 FMB 后模型退化为纯线性压缩，质量显著下降，证明非线性特征交互是 Scaling 的关键。

2. **LCB 的贡献**：移除 LCB 后模型仍能 Scale，但效率降低，说明线性通路提供了重要的信息保留。

3. **Residual Connection**：移除残差连接后训练不稳定，证明残差是深层堆叠的必要条件。

4. **Layer Normalization**：移除 LN 后大模型训练出现 Loss 爆炸，证明 LN 对训练稳定性至关重要。

![[ablation-scaling.png|800]]

> 图6：Scaling 消融实验。对比不同组件配置下的 Scaling 曲线，完整 Wukong 展现最优的 Scaling 行为。

#### Beyond MLP 分析

![[beyond-mlp.png|800]]

> 图7：Wukong 的 Interaction Stack 相比单纯增大 MLP 的优势。在相同 FLOPs 下，Wukong 的质量远优于简单加宽/加深 MLP，证明了结构化特征交互的价值。

### Scaling Law 拟合

![[scaling-log.png|800]]

> 图8：对数坐标下的 Scaling Law 拟合。Wukong 的质量-计算量关系在对数坐标下呈线性，符合幂律 $\text{NE} \propto C^{-\alpha}$ 的形式。

![[scaling-wukong.png|800]]

> 图9：Wukong 不同配置的 Scaling 曲线汇总，展示了从 0.53 GFLOP 到 108 GFLOP 的完整 Scaling 行为。

### 数据量 Scaling

![[ne-data.png|800]]

> 图10：模型质量 vs. 训练数据量。类似 LLM 中的发现，大模型更加数据高效——达到相同质量所需的样本数更少。所有 Wukong 模型在 146B 数据上仍未收敛，暗示更大数据集能带来进一步提升。

### 推理效率

![[scaling-qps.png|800]]

> 图11：模型质量 vs. 推理 QPS。Wukong 在相同推理延迟下能达到更好的质量，展示了其在实际部署中的优势。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中指出三个方向：(1) 如何高效地 serving 大规模推荐模型，包括模型压缩、量化、蒸馏等技术；(2) 从理论层面理解为什么 Wukong 能展现 Scaling Law，建立更严格的数学框架；(3) 验证 Wukong 的 Scaling Law 是否能推广到推荐以外的领域（如搜索、广告排序等）。

### 4.2 基于分析的未来方向

1. **方向1：Scaling Law 的理论解释**
   - 动机：论文目前只是经验性地观察到 Scaling Law，缺乏理论解释
   - 可能的方法：从信息论角度分析 FMB 的信息增益，或从统计学习理论角度分析泛化界
   - 预期成果：建立推荐系统的 Scaling Law 理论框架
   - 挑战：推荐系统的异构特征和稀疏交互使得理论分析比 LLM 更复杂

2. **方向2：稀疏-稠密联合 Scaling**
   - 动机：论文主要关注 Dense Scaling，但实际系统中 Sparse 和 Dense 需要协同扩展
   - 可能的方法：设计 Embedding 维度随 Interaction Stack 深度自适应增长的机制
   - 预期成果：找到 Sparse 和 Dense 的最优配比关系
   - 挑战：Sparse 部分的内存和通信开销限制了联合扩展

3. **方向3：高效推理与部署**
   - 动机：108 GFLOP 的模型在实时推荐场景中推理成本过高
   - 可能的方法：知识蒸馏（大 Wukong → 小 Wukong）、动态计算（根据样本难度调整层数）、量化
   - 预期成果：在保持 90%+ 质量的前提下将推理成本降低 5-10x
   - 挑战：推荐模型的蒸馏效果通常不如 NLP 模型

### 4.3 改进建议

1. **改进1：动态层数选择**
   - 当前问题：所有样本使用相同深度的 Interaction Stack，但简单样本可能不需要高阶交互
   - 改进方案：引入 Early Exit 机制，根据中间层的置信度决定是否继续计算
   - 预期效果：平均推理 FLOPs 降低 30-50%，质量损失 < 0.1%

2. **改进2：特征级别的自适应压缩**
   - 当前问题：LCB 对所有特征使用相同的压缩率，但不同特征的重要性差异很大
   - 改进方案：引入 Attention-based 的特征选择机制，让模型学习哪些特征在每层更重要
   - 预期效果：在相同 FLOPs 下提升模型质量

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - 这是推荐系统领域的重要里程碑论文，首次系统性地证明了推荐模型可以像 LLM 一样展现 Scaling Law，并提出了一个简洁有效的架构来实现这一目标。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 首次将 Scaling Law 概念引入推荐系统，FMB 的二进制指数交互思想新颖，但单个组件（FM、残差、LN）都是已有技术 |
| 技术质量 | 9/10 | 实验极其充分（3000+ 次运行），数学分析清晰，架构设计有理论支撑 |
| 实验充分性 | 9/10 | 6 个公开数据集 + 146B 内部数据集，7 个强基线，详尽的消融实验和 Scaling 分析 |
| 写作质量 | 8/10 | 结构清晰，图表丰富，但部分数学推导可以更严谨 |
| 实用性 | 8/10 | 架构简洁易实现，但大规模部署的推理成本是实际应用的主要障碍 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- **二进制指数交互**：$l$ 层捕获 $2^l$ 阶交互的设计思想，可以推广到其他需要高阶交互的场景
- **Optimized FM**：$O(nkd)$ 的高效实现，通过 $k \ll n$ 的投影将二次复杂度降为线性
- **金字塔形压缩**：逐层减少特征数量的设计，在保持信息的同时控制计算量
- **Dense Scaling 范式**：相比传统的 Sparse Scaling（增大 Embedding 表），Dense Scaling（增加计算量）是更有前景的方向

#### 5.2.2 需要深入理解的部分

- FMB 中 MLP 投影的具体实现细节：如何将 $n \times d$ 的输入投影为 $k$ 组，以及 $k$ 的选择对性能的影响
- Scaling Law 的拟合参数：幂律指数 $\alpha$ 的具体值及其物理含义
- 内部数据集 146B 样本的特征工程：720 个特征如何组织和处理

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DLRM]] - Meta 的经典推荐模型，Wukong 的 baseline 之一
- [[DCNv2]] - Cross Network v2，显式特征交叉的代表方法
- [[AutoInt]] - 基于 Self-Attention 的特征交互，Wukong 与之对比分析最多

### 6.2 背景相关
- [[Scaling Laws for Neural Language Models]] - Kaplan et al. 2020，LLM Scaling Law 的奠基工作
- [[Chinchilla]] - Hoffmann et al. 2022，计算最优的 LLM Scaling
- [[Factorization Machines]] - Rendle 2010，FM 的原始论文，Wukong 的理论基础

### 6.3 后续工作
- 大规模推荐模型的高效推理和部署
- 推荐系统 Scaling Law 的理论解释
- 稀疏-稠密联合 Scaling 的研究

## 外部资源

- [ICML 2024 论文页面](https://icml.cc/virtual/2024/poster/34567)
- [arXiv 预印本](https://arxiv.org/abs/2403.02545)

> [!tip] 关键启示
> 推荐系统的质量提升不应只依赖增大 Embedding 表（稀疏扩展），而应通过增加结构化的特征交互计算（稠密扩展）来实现 Scaling Law。Wukong 用简洁的 FM 堆叠 + 金字塔压缩证明了这一点。

> [!warning] 注意事项
> - 论文的 Scaling Law 主要在 Meta 内部数据集上验证，公开数据集规模不足以展现完整的 Scaling 行为
> - 108 GFLOP 的模型在实时推荐场景中的推理成本可能过高，需要配合模型压缩技术
> - 论文未讨论 Wukong 在多任务学习场景下的表现，而实际推荐系统通常是多任务的

> [!success] 推荐指数
> ⭐⭐⭐⭐☆ 强烈推荐阅读！这是推荐系统 Scaling Law 方向的开创性工作，对理解"如何有效扩展推荐模型"提供了重要的实证和架构指导。对于从事大规模推荐系统研发的工程师和研究者，这篇论文提供了明确的技术路线图。
