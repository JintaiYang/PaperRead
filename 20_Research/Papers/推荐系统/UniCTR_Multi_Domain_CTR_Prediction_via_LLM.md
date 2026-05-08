---
paper_id: "[arXiv:2312.10743](https://arxiv.org/abs/2312.10743)"
title: "A Unified Framework for Multi-Domain CTR Prediction via Large Language Models"
authors: "Zichuan Fu, Xiangyang Li, Chuhan Wu, et al."
institution: "Huawei Noah's Ark Lab, City University of Hong Kong"
pushlication: "ACM TOIS 2024, 2023-12-17"
tags:
  - 多域CTR预测
  - 大语言模型
  - 零样本预测
  - 域特定网络
  - 推荐系统
  - 跷跷板现象
  - Prompt工程
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2312.10743)"
date: "2025-05-06"
---

## 一、研究背景与动机

### 1.1 领域现状

CTR（Click-Through Rate）预测是在线推荐平台的核心任务，用于估计用户点击广告或商品的概率。随着商业平台的数字化扩展，一个平台往往同时提供在线购物、外卖配送、打车出行、视频娱乐等多种服务，每种服务构成一个独立的"域"（domain）。因此，推荐系统需要在多个域上同时做出准确的 CTR 预测，这就是多域 CTR（Multi-Domain CTR, MDCTR）预测问题。

现有的 MDCTR 方法主要分为两类：一类是以 MMOE、PLE 为代表的多专家门控模型，通过多个专家网络和门控机制来平衡域间共性与差异；另一类是以 STAR 为代表的星型拓扑模型，通过共享网络与域特定网络的逐元素乘法来融合信息。这些方法已成为工业界的主流方案，但仍存在根本性的局限。

### 1.2 现有方法的局限性

论文明确指出现有 MDCTR 系统面临三个核心挑战：

**第一，数据稀疏导致的跷跷板现象（Seesaw Phenomenon）**。在实际业务中，不同域的数据量差异巨大——例如电商购物域可能有数百万样本，而礼品卡域可能只有十几万。模型在联合训练时容易被数据丰富的域主导，导致数据稀疏域的性能显著下降。这种"此消彼长"的现象被称为跷跷板效应。

**第二，可扩展性受限**。MMOE 和 PLE 的组件紧密耦合，新增一个域就需要从头重建和训练整个模型。STAR 虽然支持新增域，但其设计要求每个域特定网络必须与骨干网络具有完全相同的结构和尺寸，随着域数增加，模型参数量成倍膨胀。

**第三，泛化能力弱，无法做零样本预测**。传统方法将所有特征编码为离散 ID（one-hot 向量），这个过程丢失了特征的语义信息。例如，"电脑"和"键盘"在 ID 空间中是完全无关的两个向量，但在语义空间中它们高度相关。当一个全新的域出现时，由于没有该域的训练数据，传统模型完全无法做出有效预测。

![Comparison](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/Comparison.png)

> 图1：传统多域 CTR 模型（左）与 Uni-CTR（右）的架构对比。左侧传统方法将特征编码为离散 ID，通过 embedding 层映射为稠密向量，丢失了语义信息；右侧 Uni-CTR 将特征转化为自然语言 prompt，利用 LLM 的语义理解能力保留完整信息。

### 1.3 本文解决方案概述

针对以上三个挑战，论文提出 Uni-CTR 框架，核心设计理念是：**用自然语言作为跨域的通用信息载体，利用 LLM 的预训练世界知识作为域间的"桥梁"**。具体包含四个关键设计：

- **Prompt-based Semantic Modeling**：将用户-商品特征转化为结构化的自然语言 prompt，保留完整语义信息
- **LLM Backbone**：利用预训练 LLM 编码 prompt，其各层表示捕获不同粒度的跨域共性
- **Domain-Specific Networks (DSN)**：通过 Ladder Network 从 LLM 不同层提取信息，建模各域的细粒度特征
- **Masked Loss Strategy**：通过梯度 mask 实现 DSN 之间的完全解耦，使每个 DSN 可独立插拔

## 二、解决方案

### 2.1 核心思想

Uni-CTR 的核心洞察是：**不同域的用户-商品交互本质上都可以用自然语言描述，而 LLM 在大规模预训练中已经积累了丰富的世界知识和语义理解能力，这些知识天然地构成了跨域的"公共知识库"**。

举个直觉性的例子：如果一个用户在电商域购买了运动鞋，LLM 能够理解"运动鞋"与"健身"、"跑步"等概念的语义关联，从而在视频推荐域中推断该用户可能对运动类视频感兴趣——即使视频域没有该用户的任何历史数据。这种跨域的语义推理能力是传统 ID-based 方法完全不具备的。

### 2.2 整体架构

![Uni-CTR Architecture](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/Uni-CTR.png)

> 图2：Uni-CTR 完整架构图。整个框架分为四个核心部分：(1) 底部的 Prompt 输入层，将域/用户/商品特征转化为文本序列；(2) 中间的 LLM Backbone（多层 Transformer），生成层级化的语义表示 $h_0, h_1, \ldots, h_L$；(3) 右侧的多个 Domain-Specific Networks，每个 DSN 通过 Ladder Network 从 LLM 不同层提取信息，经过 Gate Network 融合后由 Tower Network 输出预测；(4) 左侧的 General Network，直接使用 LLM 最后一层表示进行零样本预测。Masked Loss 确保每个样本只更新对应域的 DSN 参数。

![Uni-CTR DSN Detail](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/Uni-CTR3.png)

> 图3：Uni-CTR 的 DSN 内部结构细节图。展示了 Ladder Network 如何通过快捷连接从 LLM 的第 $\phi, 2\phi, \ldots, F\phi$ 层逐级提取中间表示，以及 Gate Network 如何将 Ladder 输出与 LLM 最终层表示进行 attention pooling 融合。

Uni-CTR 的数据流如下：输入特征首先通过 Prompt 模板转化为文本序列 $x_{\text{text}}$，然后送入 LLM Backbone 进行编码，产生 $L+1$ 层表示 $\boldsymbol{H} = \{\boldsymbol{h}_0, \boldsymbol{h}_1, \ldots, \boldsymbol{h}_L\}$。这些多层表示同时被送入对应域的 DSN（通过 Ladder Network 每隔 $\phi$ 层提取一次）和 General Network（仅使用 $\boldsymbol{h}_L$）。最终预测由对应域的 DSN Tower Network 输出（已知域）或 General Network 输出（未知域）。

#### 模块1：Prompt-based Semantic Modeling

![Prompt Template](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/Prompt.png)

> 图4：Prompt 模板设计。将三类特征（Domain Context、User Information、Product Information）整合为一条完整的自然语言序列。用户的点击历史不是简单列出商品 ID，而是替换为商品的文本描述（标题），从而保留了丰富的语义信息。

**功能**：将异构的用户-商品特征统一转化为 LLM 可理解的文本序列。

传统方法的输入是 one-hot 编码后的稀疏向量，例如 Occupation=Doctor 被编码为 $[1,0,0,\ldots,0]$，完全丢失了"医生"这个词的语义。Uni-CTR 的 prompt 设计包含三部分信息：

- **Domain Context ($d$)**：在 prompt 开头显式标注域名称（如 "Fashion:"），让 LLM 理解当前预测的业务场景
- **User Information ($u$)**：包含用户 ID 和点击历史。关键设计是将点击历史中的商品 ID 替换为对应的商品文本描述（从商品数据库中检索），例如将 "product_12345" 替换为 "Nike Air Max Running Shoes"
- **Product Information ($p$)**：包含当前候选商品的 ID、标题、品牌、价格等完整描述

最终的 prompt 模板格式为：

$$x_{\text{text}} = \text{[Domain]: The user ID is user\_[ID], who clicked `[Title1]' and `[Title2]' recently. The current product is product\_[ID], title is [Name], brand is [Brand], price is [Price].}$$

这种设计的优势在于：LLM 可以直接理解 "Nike Air Max Running Shoes" 与 "Adidas Gym Shorts" 之间的语义关联（都是运动相关），而传统 ID 编码无法捕获这种关系。

#### 模块2：LLM Backbone

**功能**：对 prompt 文本进行多层编码，生成层级化的语义表示，捕获从表面词汇到深层语义的多粒度信息。

Prompt 文本首先经过 Tokenizer 转化为 token 序列，然后通过 embedding 层得到初始表示 $\boldsymbol{h}_0$，再逐层通过 $L$ 层 Transformer：

$$\boldsymbol{h}_l = \text{Transformer}_l(\boldsymbol{h}_{l-1}), \quad l \in \{1, 2, \ldots, L\}$$

论文收集所有层的表示：$\boldsymbol{H} = \{\boldsymbol{h}_0, \boldsymbol{h}_1, \ldots, \boldsymbol{h}_L\}$。这里的关键洞察来自 NLP 领域的研究：LLM 的不同层捕获不同粒度的语义——底层学习表面短语级特征（如词汇共现），高层理解更复杂的语义概念（如意图、偏好）。因此，DSN 需要从多个层提取信息，而非仅使用最后一层。

论文默认使用 **Sheared-LLaMA**（1.3B 参数，24 层 Transformer）作为 backbone，并通过 **LoRA**（rank=8, alpha=32）进行高效微调，避免全量微调的巨大计算开销。

#### 模块3：Domain-Specific Network (DSN)

**功能**：为每个域建模细粒度的域特定特征。每个域对应一个独立的 DSN，DSN 之间完全解耦，支持灵活增删。

每个 DSN 由三个子模块组成：

**（1）Ladder Network（梯子网络）**

Ladder Network 通过快捷连接从 LLM 的不同层"侧向"提取中间表示。设置频率超参 $\phi$（论文中称为 "Freq"），每隔 $\phi$ 层部署一个 ladder。对于 24 层的 LLM 和 $\phi=6$，共有 $F = L/\phi = 4$ 个 ladder。

第 $f$ 个 ladder 的计算方式为：

$$\boldsymbol{lad}_{f} = \begin{cases} Ladder_{1}(\boldsymbol{h}_\phi) & \text{if } f = 1 \\ Ladder_{f}(\boldsymbol{h}_{f \cdot \phi} + \boldsymbol{lad}_{f-1}) & \text{if } f \in \{2, \ldots, F\} \end{cases}$$

每个 $Ladder_f$ 是一个小型 Transformer encoder block。注意第一个 ladder 只接收 LLM 第 $\phi$ 层的表示，而后续 ladder 同时接收 LLM 对应层的表示和前一个 ladder 的输出（残差连接），形成逐层精炼的信息流。

**（2）Gate Network（门控网络）**

Gate Network 的作用是动态平衡域特定特征（来自 Ladder）和跨域共性特征（来自 LLM 最后一层）。具体做法是将两者拼接后通过 attention pooling 计算自适应权重：

$$\boldsymbol{O} = \text{concat}(\boldsymbol{h}_L, \boldsymbol{lad}_{F})$$

$$\boldsymbol{score} = \text{tanh}(\boldsymbol{W}_k \boldsymbol{O}) \boldsymbol{W}_q$$

$$\boldsymbol{A} = \text{softmax}(\boldsymbol{score})$$

$$\boldsymbol{R}^{d_m} = \boldsymbol{A}^T \boldsymbol{O}$$

其中 $\boldsymbol{W}_k$ 和 $\boldsymbol{W}_q$ 是可学习参数。$\boldsymbol{R}^{d_m}$ 是融合了共性和特性的压缩表示，包含了该域预测所需的全部信息。

**（3）Tower Network（塔网络）**

Tower Network 是一个 3 层 MLP（512→256→128），将 Gate 输出映射为最终的 CTR 预测值：

$$\hat{y}^{d_m} = \text{MLP}(\boldsymbol{R}^{d_m}; \boldsymbol{W}_\gamma^{d_m}, \boldsymbol{b}_\gamma^{d_m})$$

#### 模块4：General Network

**功能**：学习所有域的共性模式，支持对未见域的零样本预测。

General Network 的结构非常简单——直接使用 LLM 最后一层表示 $\boldsymbol{h}_L$ 通过一个 MLP 进行预测：

$$\hat{y}^G = \text{MLP}(\boldsymbol{h}_L; \boldsymbol{W}_\sigma^G, \boldsymbol{b}_\sigma^G)$$

它不使用任何 Ladder Network，因此不包含域特定信息，只建模跨域共性。当遇到完全未见过的新域时，由于没有对应的 DSN，系统直接使用 General Network 的输出作为预测结果。General Network 在训练时接收所有域的样本，因此能够学到域无关的通用预测模式。

#### Masked Loss Strategy（核心设计）

**功能**：确保各 DSN 之间的梯度完全解耦，使每个 DSN 可独立插拔而不影响其他 DSN。

训练时，对于来自域 $d_m$ 的样本，通过一个 mask 向量，只保留对应域 DSN 的预测参与 loss 计算：

$$\boldsymbol{mask}^{d_m} = [I(d_1=d_m), I(d_2=d_m), \ldots, I(d_M=d_m)]$$

总损失由两部分组成：

$$\mathcal{L} = \mathcal{L}^D + \mathcal{L}^G = \ell(\hat{y}^{d_m}, y) + \ell(\hat{y}^G, y)$$

**梯度解耦的关键性质**：对于域 $d_n$（$n \neq m$）的 DSN 参数 $\boldsymbol{\theta}_{\text{DSN}}^{d_n}$，由于 mask 的作用：

$$\nabla_{\boldsymbol{\theta}_{\text{DSN}}^{d_n}} \mathcal{L} = 0 \quad \text{when } n \neq m$$

这意味着域 $d_m$ 的样本只会更新三部分参数：(1) DSN $d_m$ 的参数，(2) LLM backbone 的参数，(3) General Network 的参数。其他所有 DSN 的参数完全不受影响。这个设计带来两个关键优势：**缓解跷跷板问题**（每个 DSN 只被自己域的数据更新）和**真正的可插拔性**（新增域只需添加一个新 DSN 并训练，冻结其他所有参数即可）。

## 三、实验结果

### 3.1 数据集

论文使用 **Amazon Review Data (2018)** 数据集，选取 5 个商品类别作为不同域。评分 >3 为正样本，≤3 为负样本，数据划分为 80%/10%/10%。

| 域 | 用户数 | 商品数 | 样本数 | 特点 |
|------|--------|--------|---------|------|
| Fashion | 749,233 | 186,189 | 883,636 | 数据中等，商品多样性高 |
| Digital Music | 127,174 | 66,010 | 1,584,082 | 数据丰富，用于可扩展性测试 |
| Musical Instruments | 903,060 | 112,132 | 1,512,530 | 数据丰富，用户基数大 |
| Gift Cards | 128,873 | 1,547 | 147,194 | **数据极稀疏**，商品种类仅 1547 |
| All Beauty | 319,335 | 32,486 | 371,345 | 用于零样本测试（训练时不可见） |

值得注意的是，论文排除了 Ali-CCP 和 Ali-Mama 等常用数据集，原因是这些数据集的特征已被匿名化为纯 ID，没有文本语义信息，不适合 LLM-based 方法。

### 3.2 实验设置

**Uni-CTR 配置**：LLM backbone 为 Sheared-LLaMA（1.3B 参数，24 层 Transformer）。每个 DSN 包含 4 个 ladder 层（小型 Transformer encoder block），Tower 网络为 3 层 MLP（512×256×128）。训练使用 8 张 Tesla V100 GPU，batch size=128，AdamW 优化器配合 CyclicLR 调度（学习率范围 $[1\times10^{-6}, 8\times10^{-5}]$），LoRA（rank=8, alpha=32）加速 LLM 微调，dropout=0.3 + L2 正则化防止过拟合。

**基线配置**：所有基线模型的 MLP 隐藏层统一为 512×256×128。多域模型中，MMOE 使用 3 个专家（维度 1024×1024），PLE 使用 1 个共享专家 + 2 个域特定专家，STAR 的辅助网络和星型拓扑网络维度为 1024×512×256。

#### 3.2.1 基线方法

- **单域模型（8个）**：PNN、DCN、DeepFM、xDeepFM、DIEN、AutoInt、FiBiNET、IntTower
- **多域模型（6个）**：Shared Bottom、MMOE、PLE、STAR、SAR-Net、DFFM

#### 3.2.2 评估指标

- **AUC**：ROC 曲线下面积，衡量模型区分正负样本的整体能力
- **RelaImpr**：相对改进率，$\text{RelaImpr} = \left(\frac{\text{AUC}_{model} - 0.5}{\text{AUC}_{base} - 0.5} - 1\right) \times 100\%$

在 CTR 领域，AUC 提升 0.001 即被认为是显著改进。

### 3.3 主实验结果与分析

| 类别 | 方法 | Fashion AUC | M.I. AUC | Gift Cards AUC |
|------|------|-------------|----------|----------------|
| 单域 | PNN | 0.6979 | 0.6859 | 0.5959 |
| 单域 | DCN | 0.6985 | 0.6893 | 0.6126 |
| 单域 | DeepFM | 0.6982 | 0.6880 | 0.5937 |
| 单域 | xDeepFM | **0.7031** | 0.6892 | 0.6121 |
| 单域 | DIEN | 0.6995 | 0.6881 | 0.6105 |
| 单域 | AutoInt | 0.7003 | 0.6888 | 0.5976 |
| 单域 | FiBiNET | 0.6770 | 0.6878 | 0.6120 |
| 单域 | IntTower | 0.6988 | 0.6888 | 0.6100 |
| 多域 | Shared Bottom | 0.6946 | 0.6875 | 0.5907 |
| 多域 | MMOE | 0.6907 | 0.6857 | 0.6104 |
| 多域 | PLE | 0.6842 | 0.6813 | **0.6375** |
| 多域 | STAR | 0.6874 | 0.6831 | 0.6242 |
| 多域 | SAR-Net | 0.6824 | 0.6763 | 0.6055 |
| 多域 | DFFM | 0.6973 | 0.6856 | 0.6324 |
| **LLM多域** | **Uni-CTR** | **0.7523** | **0.7569** | **0.7246** |

> 注：M.I. = Musical Instruments。粗体为各列最优（Uni-CTR 行除外）。所有 Uni-CTR 结果通过 $p < 0.05$ 显著性检验。

**关键发现**：

- **Uni-CTR 全面碾压所有基线**：在三个域上 AUC 绝对值提升分别为 +0.0492（Fashion）、+0.0676（Musical Instruments）、+0.0871（Gift Cards），远超 CTR 领域 0.001 的显著性阈值。
- **数据越稀疏，LLM 优势越大**：Gift Cards 域仅有 14.7 万样本且商品种类极少（1547），传统模型最优仅 0.6375，而 Uni-CTR 达到 0.7246，RelaImpr 高达 63.35%。
- **传统多域模型存在明显跷跷板**：在数据丰富的 Fashion 域，单域模型（xDeepFM 0.7031）反而优于多数多域模型（PLE 0.6842），说明传统多域方法在联合训练时牺牲了数据丰富域的性能。Uni-CTR 通过 Masked Loss 解耦，在所有域上都取得最优。

### 3.4 零样本预测（RQ2）

实验设置：使用 Fashion、Musical Instruments、Gift Cards 三个域训练，在完全未见过的 **All Beauty** 域上做零样本预测。Uni-CTR 使用 General Network 预测（不使用任何 DSN）。

![Zero-shot Prediction](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/zeroshot.png)

> 图5：零样本预测性能对比。单域模型（蓝色）AUC 约 0.50-0.51，几乎等于随机猜测；传统多域模型（橙色）约 0.55-0.58，有一定泛化能力但有限；Uni-CTR（红色）超过 0.64，领先传统多域模型 6+ 个百分点。

- 单域模型的 AUC 约 0.50-0.51，说明它们完全无法泛化到新域
- 传统多域模型（MMOE、PLE、STAR 等）的 AUC 约 0.55-0.58，联合训练确实学到了一些跨域共性，但提升有限
- Uni-CTR 的 AUC 超过 0.64，相对最优传统多域模型提升超过 **6 个百分点**。这归因于 LLM 的预训练世界知识能够将不同域的商品映射到统一语义空间，即使未见过 All Beauty 域的数据，LLM 也"知道"美妆产品的语义含义

### 3.5 Scaling Law 分析（RQ3）

![Scaling Law](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/scale.png)

> 图6：不同规模 LLM backbone 的性能对比（三个域的 AUC）。从左到右依次为 TinyBERT（14M）、BERT（110M）、DeBERTa-V3-Large（340M）、Sheared-LLaMA（1.3B），性能随模型规模持续提升。

| LLM Backbone | 参数量 | Fashion AUC | Musical Instruments AUC | Gift Cards AUC |
|---|---|---|---|---|
| TinyBERT | 14M | ~0.69 | ~0.69 | ~0.65 |
| BERT | 110M | ~0.72 | ~0.72 | ~0.69 |
| DeBERTa-V3-Large | 340M | ~0.74 | ~0.74 | ~0.71 |
| Sheared-LLaMA | 1.3B | **0.7523** | **0.7569** | **0.7246** |

Scaling law 在 MDCTR 任务中同样成立。值得注意的是，基于 BERT（仅 110M 参数）的 Uni-CTR 已经超越了所有传统多域模型，说明 LLM 的语义理解能力（而非单纯参数量）是性能提升的关键。这为工业部署提供了灵活选择：对延迟敏感的场景可以使用 BERT 级别的 backbone，仍能获得显著收益。

### 3.6 可扩展性验证（RQ4）

冻结已在 Fashion、Musical Instruments、Gift Cards 三个域上训练好的 Uni-CTR（LLM backbone + 3 个 DSN 全部冻结），仅新增一个 DSN 在 Digital Music 域上微调：

| 类别 | 方法 | 可扩展性 | Digital Music AUC | RelaImpr |
|------|------|---------|-------------------|----------|
| 单域 | xDeepFM | -- | 0.5957 | 19.12% |
| 多域 | STAR | 支持 | 0.6038 | 9.83% |
| 多域 | MMOE/PLE | 不支持 | -- | -- |
| **LLM多域** | **Uni-CTR** | **支持** | **0.6140** | -- |

仅训练一个新 DSN（参数量远小于 LLM backbone），Uni-CTR 就比完全重训的单域最优模型（xDeepFM）高出 19.12%，比同样支持扩展的 STAR 高出 9.83%。MMOE 和 PLE 由于组件紧密耦合，无法在不重训整个模型的情况下新增域。

### 3.7 可视化分析（RQ5）

#### LLM 不同层的表示分布（训练前）

![tSNE Layer 0](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/tSNE-LLM-non-trained-layer0.png)

> 图7(a)：LLM 第 0 层（embedding 层）的 t-SNE 可视化（训练前）。三种颜色代表三个域（Fashion、Musical Instruments、Gift Cards），表示完全混合在一起，说明底层捕获的是跨域共性特征。

![tSNE Layer 2](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/tSNE-LLM-non-trained-layer2.png)

> 图7(b)：LLM 第 2 层的 t-SNE 可视化（训练前）。不同域的表示开始出现轻微分离趋势，说明中间层开始捕获粗粒度的域特征。

![tSNE Layer 4](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/tSNE-LLM-non-trained-layer4.png)

> 图7(c)：LLM 第 4 层的 t-SNE 可视化（训练前）。不同域的表示进一步分离，形成了较为明显的聚类结构，说明高层逐渐编码了域特定的语义信息。

这三张图清晰展示了 LLM 不同层的功能分工——底层编码跨域共性（所有域混合），高层逐渐分离出域特定特征。这正是 Ladder Network 需要从多层提取信息的原因：底层提供共性基础，高层提供域区分信号。

#### DSN 训练前后的表示对比

![DSN Untrained](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/dsn-untrained.png)

> 图8(a)：未训练 DSN 的 tower network 倒数第二层表示。三个域的样本完全混合，DSN 无法区分不同域。

![DSN Trained](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/dsn-trained.png)

> 图8(b)：训练后 DSN 的表示。三个域的样本清晰分离为三个独立聚类，验证了 DSN 成功学到了域特定的细粒度特征。

### 3.8 消融实验（RQ6）

#### Prompt 语义信息的影响

使用 DeBERTa-V3-Large 作为 backbone，对比不同 prompt 格式：

| 输入格式 | Fashion | Musical Instruments | Gift Cards | 说明 |
|----------|---------|---------------------|------------|------|
| Full Prompt | 0.7047 | 0.7008 | 0.6825 | 完整语义信息（域名+用户历史+商品描述） |
| Only Feature ID + Name | 0.6960 | 0.6960 | 0.6749 | 仅特征名和 ID 拼接 |
| Only Feature ID | 0.6951 | 0.6835 | 0.6605 | 仅保留特征 ID 值 |

语义信息越丰富，性能越好。在数据稀疏的 Gift Cards 域，Full Prompt vs Only ID 差距高达 0.022，说明 LLM 的优势不仅在于模型容量大，更在于能理解文本语义。

#### Ladder Network 和 LLM Backbone 的影响

![Ablation](UniCTR_Multi_Domain_CTR_Prediction_via_LLM/images/Ablation-Block.png)

> 图9：消融实验中各模块的贡献对比。去掉 Ladder Network 或 LLM Backbone 后，性能均显著下降，尤其在数据稀疏的 Gift Cards 域。

| 模型变体 | Fashion | Musical Instruments | Gift Cards |
|----------|---------|---------------------|------------|
| Uni-CTR（完整） | 0.7391 | 0.7395 | 0.7073 |
| w/o Ladder | 0.7084 | 0.6975 | 0.6723 |
| w/o LLM（替换为 DNN+ID） | 0.6954 | 0.6923 | 0.6100 |
| MMOE（扩大到 340M） | 0.7038 | 0.7005 | 0.6712 |
| STAR（扩大到 340M） | 0.7107 | 0.7016 | 0.6775 |

- **去掉 Ladder Network**：性能下降 0.03-0.04，说明仅使用 LLM 最后一层表示是不够的，多层语义信息的逐级提取对域特定建模至关重要
- **去掉 LLM**（替换为同层数 DNN + 传统 ID 输入）：性能大幅下降，Gift Cards 上从 0.7073 暴跌到 0.6100（下降 0.097）
- **参数量对齐实验**：将 MMOE 和 STAR 的参数量扩大到 340M，性能仍远不如 Uni-CTR，排除了"性能提升仅来自参数量增加"的假设

### 3.9 工业实验

在大规模工业推荐系统上的验证（百万级日志数据，2 个业务域）：

| 类别 | 方法 | Domain 0 AUC | Domain 1 AUC |
|------|------|-------------|-------------|
| 单域最优 | AutoInt / FiBiNET | 0.6788 | 0.6146 |
| 多域 | MMOE | 0.7045 | 0.6640 |
| 多域 | PLE | 0.7019 | 0.6706 |
| 多域 | STAR | 0.7000 | 0.6638 |
| **LLM多域** | **Uni-CTR** | **0.7387** | **0.6881** |

相对最优多域基线（PLE）的 RelaImpr：Domain 0 为 16.72%，Domain 1 为 10.26%。推理加速方案：ONNX 导出 + TensorRT FP16 量化，batch size=32、序列长度=256 时，单样本约 **2ms**（V100 GPU），量化后 AUC 损失 <0.01，满足工业在线 rank 阶段的延迟要求。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中提到，未来将继续研究多域 CTR 预测中输入模态的增强——即探索如何将图像、视频等多模态信息融入 Uni-CTR 框架，进一步丰富语义表示。

### 4.2 基于分析的未来方向

1. **方向1：多模态输入扩展**
   - 动机：当前 Uni-CTR 仅使用文本特征，商品图片（如服装外观）等信息未被利用
   - 可能的方法：将多模态 LLM（如 LLaVA、Qwen-VL）作为 backbone，在 prompt 中嵌入图像 token；或使用 CLIP 提取视觉特征后拼接到文本表示中
   - 预期成果：在视觉驱动的域（如 Fashion、All Beauty）获得额外提升
   - 挑战：多模态输入会显著增加序列长度和推理延迟，需要更激进的压缩策略

2. **方向2：轻量化 LLM backbone 探索**
   - 动机：1.3B 参数的 LLM 在工业部署中仍有延迟和成本压力，多 GPU 推理的资源开销较大
   - 可能的方法：使用知识蒸馏将大 LLM 的跨域知识迁移到小模型；探索 MoE 架构的 LLM，仅激活部分专家
   - 预期成果：在保持 90%+ 性能的前提下将推理延迟降低到 <1ms
   - 挑战：蒸馏过程中如何保留跨域共性知识而不退化为传统 embedding

3. **方向3：动态 Prompt 优化与自适应**
   - 动机：当前 prompt 模板是固定的，但不同域、不同用户可能需要不同的信息组织方式
   - 可能的方法：引入可学习的 soft prompt prefix，或使用强化学习自动选择最优的特征组合和排列顺序
   - 预期成果：进一步提升各域的预测精度，尤其是特征异构性大的域
   - 挑战：可学习 prompt 的搜索空间大，训练不稳定

### 4.3 改进建议

1. **改进1：引入用户行为序列的时序建模**
   - 当前问题：Uni-CTR 的 prompt 中用户点击历史仅列出最近几个商品标题，缺乏时序关系建模（如兴趣演化、周期性行为）
   - 改进方案：在 DSN 的 Ladder Network 中加入时序注意力模块，或在 prompt 中加入时间戳信息让 LLM 理解行为顺序
   - 预期效果：更好地捕获用户兴趣演化，提升对时间敏感域（如新闻、短视频）的预测能力

2. **改进2：DSN 结构自适应**
   - 当前问题：所有域的 DSN 使用相同的 ladder 层数（4 层）和结构，但不同域的数据量和复杂度差异很大（Gift Cards 仅 14.7 万样本 vs Musical Instruments 151 万样本）
   - 改进方案：引入 NAS 或自适应机制，让每个域根据数据量和特征复杂度自动选择最优的 DSN 深度和宽度
   - 预期效果：数据稀疏域使用更简单的 DSN 避免过拟合，复杂域使用更深的 DSN 提升表达能力

3. **改进3：跨域知识迁移的显式建模**
   - 当前问题：域间知识迁移完全依赖 LLM 的隐式表示，缺乏对"哪些域之间更相关"的显式建模
   - 改进方案：在 Gate Network 中引入域间相似度矩阵，允许相关域的 DSN 之间进行受控的信息交换（同时保持 Masked Loss 的解耦性质）
   - 预期效果：更精准的跨域知识迁移，尤其对语义相近的域（如 Fashion 和 All Beauty）

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分

**8.0/10** — 这是首个将 LLM 作为 backbone 应用于多域 CTR 预测的工作，思路清晰、架构设计合理，实验全面且有工业验证。Masked Loss 实现 DSN 解耦的设计简洁优雅，零样本预测能力是重要的实用贡献。主要不足在于公开数据集仅用了 Amazon Review（域间差异相对较小），且论文篇幅较长、Related Work 占比过大。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 首次将 LLM 作为 MDCTR 的 backbone，用自然语言统一跨域特征编码的思路新颖。Masked Loss 解耦策略和 Ladder Network 从多层提取信息的设计有独创性。但各个组件（LoRA、Ladder Side Network、Attention Pooling）本身不是新技术。 |
| 技术质量 | 8/10 | 架构设计合理，各模块分工明确。Masked Loss 的数学推导严谨，证明了 DSN 解耦的性质。LoRA 微调 + ONNX/TensorRT 推理加速的工程方案完整。但 Ladder Network 的频率超参 $\phi$ 的选择缺乏理论指导。 |
| 实验充分性 | 8/10 | 6 个 RQ 覆盖了性能、零样本、scaling law、可扩展性、可视化、消融等多个维度。有工业数据集验证和推理延迟分析。但公开数据集仅用了 Amazon Review 一个来源，且 Ali-CCP 等主流 CTR 数据集因 ID 匿名化被排除，限制了结论的普适性。 |
| 写作质量 | 7/10 | 结构清晰，公式推导完整，图表质量高。但论文较长（Related Work 约占 3 页），部分内容有重复描述。 |
| 实用性 | 9/10 | 直接解决了工业界 MDCTR 的三大痛点（跷跷板、可扩展性、冷启动）。推理延迟 2ms 满足在线要求。DSN 可插拔设计对业务快速迭代非常友好。已有工业验证。 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

**Masked Loss Strategy** 是本文最精巧的设计——通过简单的 indicator mask 实现了 DSN 之间的完全梯度解耦，使得每个 DSN 可以独立训练、独立部署、独立更新。这个思想对所有需要模块化设计的多任务/多域系统都有借鉴意义——不需要复杂的梯度隔离机制，一个 mask 就够了。

**LLM 多层表示的利用**：不同于大多数 LLM 应用只使用最后一层输出，Uni-CTR 通过 Ladder Network 从多层提取信息。t-SNE 可视化清晰展示了不同层的功能分工（底层共性→高层域特定），为 Ladder 的设计提供了直觉支撑。

**General Network 的零样本能力**：一个简单的 MLP 接在 LLM 最后一层上，就能实现有效的零样本预测。这说明 LLM 的预训练知识已经将不同域的商品映射到了有意义的统一语义空间。

#### 5.2.2 需要深入理解的部分

- Ladder Network 的频率超参 $\phi$ 如何影响性能？论文使用 24 层 LLM + 4 个 ladder（即 $\phi=6$），但没有详细的 $\phi$ 敏感性分析
- 在工业实验中，2ms 的延迟是 batch=32 的平均值，单条推理的延迟可能更高。实际部署中如何处理 batch 组装的延迟？
- 论文实验中域数量较少（3-5 个），在数十个域的超大规模场景下的表现有待验证

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关

- [[STAR_One_Model_to_Serve_All|STAR]] — 星型拓扑多域模型，支持域扩展但要求 DSN 与 backbone 结构一致，Uni-CTR 的主要对比对象
- [[PLE_Progressive_Layered_Extraction|PLE]] — 渐进式分层提取，用共享专家+域特定专家解决跷跷板问题，但组件紧耦合不可扩展
- [[CTRL_LLM_CTR|CTRL]] — 首个将预训练语言模型用于 CTR 预测的工作，但仅处理单域场景

### 6.2 背景相关

- [[MMOE_Multi_gate_Mixture_of_Experts|MMOE]] — 多门混合专家模型，多域/多任务建模的里程碑，Uni-CTR 的重要基线
- [[LoRA_Low_Rank_Adaptation|LoRA]] — 低秩适配方法，Uni-CTR 用于高效微调 LLM backbone
- [[Ladder_Side_Tuning|LST]] — Ladder Side Tuning，Uni-CTR 的 DSN 设计灵感来源
- [[DeepFM|DeepFM]] — FM+DNN 并行架构，经典单域 CTR 模型

### 6.3 后续工作

- [[PepNet|PEPNet]] — 参数个性化网络，另一种解决多域跷跷板问题的方案
- [[Efficient_LLM_Recommendation|高效 LLM 推荐]] — 探索更轻量的 LLM 架构用于在线推荐

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2312.10743)
- [ACM TOIS 期刊](https://dl.acm.org/journal/tois)
- [Sheared-LLaMA](https://github.com/princeton-nlp/LLM-Shearing) — 论文使用的 LLM backbone
- [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html) — 实验使用的公开数据集

> [!tip] 关键启示
> 用自然语言作为跨域的通用信息载体，结合 LLM 的预训练世界知识，可以从根本上解决多域 CTR 中的数据稀疏和冷启动问题。Masked Loss 实现的 DSN 解耦设计使系统具备真正的"即插即用"可扩展性——新增域只需训练一个轻量 DSN，无需动已有模型的任何参数。

> [!warning] 注意事项
> - Uni-CTR 强依赖文本语义特征，对于纯 ID 特征的匿名数据集（如 Ali-CCP、Ali-Mama）不适用，这限制了其在某些工业场景的直接应用
> - 1.3B LLM backbone 的训练成本较高（8×V100），小团队部署有门槛；但 110M 的 BERT 版本已能超越传统模型
> - 推理延迟 2ms/sample 虽满足 rank 阶段要求，但相比传统模型（<0.1ms）仍高出一个数量级，不适用于 pre-rank 或 recall 阶段
> - 论文实验中域数量较少（3-5 个），在数十个域的超大规模场景下的表现有待验证

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐阅读！这是 LLM 应用于多域推荐系统的开创性工作，思路清晰、实验扎实、工业可落地。对于正在探索 LLM+推荐系统结合的团队，这篇论文提供了完整的技术方案和工程实践参考。尤其推荐关注 Masked Loss 的解耦设计和 Ladder Network 的多层信息提取思想。

