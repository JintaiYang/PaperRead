---
paper_id: "[arXiv:2602.07520](https://arxiv.org/abs/2602.07520)"
title: "MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization"
authors: "Shanlei Mu, Yuchen Jiang, Shikang Wu, Shiyong Hong, Tianmu Sha, Junjie Zhang, Jie Zhu, Zhe Chen, Zhe Wang, Jingjian Lin"
institution: "ByteDance Search, ByteDance AML"
publication: "KDD 2025 (Submitted), 2026-02"
tags:
  - 多场景学习
  - 多任务学习
  - Tokenization
  - 大规模推荐模型
  - Scaling-Law
  - Prompt
  - 抖音搜索
  - 字节跳动
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2602.07520)"
  - "[HTML](https://arxiv.org/html/2602.07520v2)"
date: "2026-07-10"
---

## 一、研究背景与动机

### 1.1 领域现状

工业级推荐系统通常需要同时处理多个场景（Multi-Scenario Learning, MSL）和多个预测任务（Multi-Task Learning, MTL）。例如抖音搜索需要在单列搜索、双列搜索、内嵌搜索等多场景下，同时预测点击、点赞、收藏等多个用户行为。现有方法大多采用 **shared-specific 架构**——即设计共享组件和场景/任务专属组件的组合：

- **MoE 系方法**（MMoE、PLE、HMoE）：通过共享专家和专属专家的门控混合来建模分布差异
- **动态参数生成方法**（STAR、PEPNet）：利用场景/任务先验信息动态生成专属网络参数

### 1.2 现有方案的两大问题

论文指出当前方法在大规模推荐模型（参数量已达 0.1B~0.5B）背景下存在两个根本性缺陷：

**问题1：Scaling 能力受限（Limitation in Scaling Capability）**

受 LLM 中 Scaling Law 的启发，工业级推荐模型通过堆叠特征交互层来扩展参数规模（如 RankMixer、HSTU、OneTrans 等，参数量已达 0.1B+）。然而现有 MSL/MTL 方法（如 MMoE、PEPNet）通常只作用于中间模块或浅层输出层，未能充分利用深层大参数特征交互模块的能力。场景/任务信号被限制在模型的"浅层"——无法穿透并激活整个参数空间。

**问题2：统一建模困难（Difficulty in Uniformity Modeling）**

多场景学习关注的是**输入分布**的差异（不同场景下用户行为模式不同），而多任务学习关注的是**标签分布**的差异（点击 vs 点赞 vs 收藏的预测目标不同）。传统 shared-specific 结构对这两种差异采用不同的设计思路，难以在统一框架下联合建模场景-任务的复合分布。

### 1.3 本文核心洞察

MDL 从 LLM 的 **Prompting 范式**中获得启发：LLM 通过 prompt token 在 self-attention 中与所有上下文 token 深度交互，从而「激活」模型的海量参数来完成特定下游任务。类似地，MDL 将场景信息和任务信息也 tokenize 为与特征 token 同等地位的 token，使其能够从底层开始、逐层深度参与特征交互，从而：

1. 场景/任务信号不再受限于浅层门控或输出头，可以逐层「prompt」并激活大模型的全部参数空间（解决问题1）
2. 场景 token 和任务 token 在同一空间下统一表示，通过统一的交互协议实现联合建模（解决问题2）

## 二、解决方案

### 2.1 核心思想

MDL 的核心哲学是 **"Tokenize-and-Interact"**：先将特征、场景、任务信息统一 tokenize 到同一语义空间，然后设计三种协同交互机制让这些 token 在多层堆叠中充分交互。

![[MDL_Multi_Distribution_Learner/images/mdl_framework.png|800]]
> 图1：MDL 整体框架。左侧为统一信息 Tokenization 模块（将特征、场景、任务分别转为 token），右侧为 Domain-aware All-Token Interaction 模块（三种交互机制逐层堆叠）。

### 2.2 整体架构

MDL 由两大模块组成：

1. **Unified Information Tokenization**：将异构输入统一转换为三类 token
2. **Domain-aware All-Token Interaction**：通过三种交互机制驱动 token 间的深度信息传递

### 2.3 模块1：Unified Information Tokenization

#### 2.3.1 Feature Tokenization（特征 Token 化）

工业推荐系统的输入特征包括用户特征 $x_u$、物品特征 $x_i$、序列特征 $x_{seq}$ 和交叉特征 $x_{cross}$。首先通过 embedding 层将各类特征转换为 embedding：

$$\mathbf{e}_u, \mathbf{e}_i, \mathbf{e}_{cross} = \text{EmbLayer}(x_u, x_i, x_{cross})$$

序列特征通过专门的序列模块（如 DIN、SIM）处理为 $\mathbf{e}_{seq}$。所有特征拼接为：$\mathbf{e}_{input} = [\mathbf{e}_u; \mathbf{e}_i; \mathbf{e}_{cross}; \mathbf{e}_{seq}]$。

接下来，沿用 RankMixer 的语义分组方案：根据领域知识将特征手动分组为 $N_f$ 个语义一致的 cluster，每个 cluster 的 embedding 通过 Projection 层映射为固定维度的 feature token：

$$\mathbf{t}_j = \text{Proj}(\mathbf{e}_j), \quad \mathbf{t}_j \in \mathbb{R}^{d_f}$$

最终得到 feature tokens $\mathbf{T}_f = [\mathbf{t}_1; \mathbf{t}_2; \cdots; \mathbf{t}_{N_f}] \in \mathbb{R}^{N_f \times d_f}$。

#### 2.3.2 Scenario Tokenization（场景 Token 化）

场景 token 的设计是 MDL 的关键创新之一。与 feature token 类似，场景 token $\mathbf{T}_s \in \mathbb{R}^{(N_s+1) \times d_s}$ 也是固定数量、固定维度的隐表示。其输入由两部分组成：

1. **重要特征的额外 embedding** $\hat{\mathbf{e}}_{imp}$（如 user_id、video_id 的独立 embedding，与 feature token 的 embedding 不共享参数但来自相同原始特征）
2. **场景相关的先验特征 embedding** $\hat{\mathbf{e}}_{spec}$（如场景特有的用户行为序列）

通过 Per-token FFN 将输入映射为场景 token：

$$\mathbf{t}_s = \text{ReLU}(\text{FFN}(\hat{\mathbf{e}}_{imp} \oplus \hat{\mathbf{e}}_{spec}))$$

每个场景 token 使用**独立的 FFN 参数**（Per-token FFN），确保不同场景 token 具有不同的映射函数。

最终得到 $N_s$ 个场景 token（每个对应一个推荐场景）加上 1 个全局场景 token $\mathbf{t}_{s,global}$（学习跨场景共性知识）：

$$\mathbf{T}_s = [\mathbf{t}_{s,1}; \mathbf{t}_{s,2}; \cdots; \mathbf{t}_{s,N_s}; \mathbf{t}_{s,global}]$$

#### 2.3.3 Task Tokenization（任务 Token 化）

类似场景 token，任务 token $\mathbf{T}_t \in \mathbb{R}^{N_t \times d_t}$ 通过输入其他重要特征 embedding 和任务相关特征 embedding，经 Per-token FFN 转换得到。$N_t$ 等于预测任务数量（如 click、like、favorite 各一个 token）。

### 2.4 模块2：Domain-aware All-Token Interaction

三类 token 之间存在三种交互关系，MDL 为每种设计了专门的交互机制。

#### 2.4.1 Feature Token Self-Interaction（特征 token 自交互）

这是大规模推荐模型的基础能力模块。MDL 沿用 RankMixer 的设计，通过 TokenMixing + Per-token FFN 实现特征 token 间的高阶交互：

$$\mathbf{T}_f^{(l+1)} = \text{PertokenFFN}(\text{LN}(\text{TokenMixing}(\mathbf{T}_f^{(l)}) + \mathbf{T}_f^{(l)}))$$

其中 TokenMixing 模块实现 token 间的信息混合（类似 Self-Attention），Per-token FFN 为每个 token 位置提供独立的非线性变换。残差连接和 LayerNorm 保证训练稳定性。这个模块可以被替换为其他特征交互方法（如 Self-Attention、MLP-Mixer）。

#### 2.4.2 Domain-aware Attention（域感知注意力）

这是 MDL 的核心创新——场景/任务 token 与特征 token 之间的交互。设计为**跨注意力（Cross-Attention）的变体**：场景/任务 token 作为 Query，特征 token 作为 Key 和 Value。

以任务 token 为例：

$$\hat{\mathbf{T}}_t^{(l+1)} = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{T}_t^{(l)})(\mathbf{W}_K \mathbf{T}_f^{(l)})^\top}{\sqrt{d}}\right) \mathbf{W}_V \mathbf{T}_f^{(l)}$$

其中 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ 是 Multi-Head Attention 的 QKV 投影矩阵（以 Per-token FFN 形式实现）。

场景 token 类似：

$$\hat{\mathbf{T}}_s^{(l+1)} = \text{softmax}\left(\frac{(\mathbf{W}_Q \mathbf{T}_s^{(l)})(\mathbf{W}_K \mathbf{T}_f^{(l)})^\top}{\sqrt{d}}\right) \mathbf{W}_V \mathbf{T}_f^{(l)}$$

**关键直觉**：不同任务 token 会学会「关注」不同的特征子空间。例如 click token 可能更关注视觉类特征，like token 可能更关注内容质量类特征。由于 tokenization 从底层开始，这种选择性注意力可以逐层叠加、逐层细化。

#### 2.4.3 Domain-fused Module（域融合模块）

场景 token 与任务 token 之间的交互，实现「特定场景下特定任务」的差异化预测。分两步：

**Step 1：实例级场景 token 选择**

对于每个训练/推理实例，根据其所属场景从 $\mathbf{T}_s$ 中选取对应的场景 token（支持一个实例属于多个场景的重叠情况），加上全局场景 token，做 Mean Pooling：

$$\mathbf{t}_{s,avg} = \text{MeanPooling}(\{\mathbf{t}_{s,i}, \mathbf{t}_{s,j}, \mathbf{t}_{s,global}\})$$

**Step 2：信息融合**

将聚合的场景信息直接加到所有任务 token 上：

$$\mathbf{T}_t = \mathbf{T}_t + \mathbf{t}_{s,avg}$$

论文指出这种简单的加法融合在实践中既高效又有效。通过 tokenized 信息融合，场景与任务之间的预测被解耦——支持任意场景-任务组合的灵活建模。

### 2.5 MDL Block 的完整前向传播

一个完整的 MDL Block 中三类 token 的前向传播：

**Feature Token 传播**：
$$\mathbf{T}_f^{(l+1)} = \text{PertokenFFN}(\text{LN}(\text{TokenMixing}(\mathbf{T}_f^{(l)}) + \mathbf{T}_f^{(l)}))$$

**Scenario Token 传播**：
$$\hat{\mathbf{T}}_s^{(l+1)} = \text{DomainAwareAttn}(\mathbf{T}_s^{(l)}, \mathbf{T}_f^{(l+1)}) + \mathbf{T}_s^{(l)}$$
$$\mathbf{T}_s^{(l+1)} = \text{PertokenFFN}(\hat{\mathbf{T}}_s^{(l+1)}) + \hat{\mathbf{T}}_s^{(l+1)}$$

**Task Token 传播**：
$$\hat{\mathbf{T}}_t^{(l+1)} = \text{DomainAwareAttn}(\mathbf{T}_t^{(l)}, \mathbf{T}_f^{(l+1)}) + \mathbf{T}_t^{(l)}$$
$$\tilde{\mathbf{T}}_t^{(l+1)} = \text{DomainFusedModule}(\hat{\mathbf{T}}_t^{(l+1)}, \hat{\mathbf{T}}_s^{(l+1)})$$
$$\mathbf{T}_t^{(l+1)} = \text{PertokenFFN}(\tilde{\mathbf{T}}_t^{(l+1)}) + \tilde{\mathbf{T}}_t^{(l+1)}$$

堆叠 $L$ 层后，最终 task token 通过 Logits Layer 输出各任务预测：

$$\hat{y}_n = \text{LogitsLayer}(\mathbf{t}_{t,n}^{(L)})$$

**梯度流的关键性质**：由于 task token 在最后已通过 Domain-fused Module 聚合了对应场景的 token 信息，特定场景实例的梯度只会更新对应的场景 token——实现了场景之间的自然解耦。且输出 task token 数量与场景数无关（不需要 $N_s \times N_t$ 个输出头），仅需 $N_t$ 个即可实现跨场景差异化预测。

### 2.6 训练目标

标准的多场景多任务 cross-entropy loss：

$$\mathcal{L}_{rec} = \sum_{k=1}^K \sum_{n=1}^N \sum_{i=1}^{|\mathcal{S}^k|} \ell(y_i^k, \hat{y}_i^{n,k})$$

其中 $K$ 为场景数，$N$ 为任务数，$\ell(\cdot)$ 为 cross-entropy loss。

## 三、实验评估

### 3.1 实验设置

**数据集**：来自抖音搜索系统的大规模生产数据集，包含 3 个主要搜索场景（单列搜索、双列搜索、内嵌搜索）和 20+ 预测任务。收集 2 个月的连续用户交互日志，涉及数十亿用户和数亿文档，每个实例包含 500+ 特征。1% 数据作为评测集。

**评估指标**：
- 离线：QAUC（Query-level AUC）—— 对每个 UID-query pair 计算 AUC 后取平均
- 在线：LT30（30 天用户活跃天数）和 Change Query Rate（改 query 率，越低越好）

**模型规模**：所有模型参数量统一控制在约 **0.5B**，通过调整层数和隐维度实现公平对比。

**训练设置**：数百 GPU 混合分布式训练（sparse 部分异步更新 + dense 部分同步更新），sparse 用 Adagrad 优化器，dense 用 RMSProp，batch size = 2048。

### 3.2 主要结果

| Method | Single Click | Single Like | Single Fav | Double Click | Double Like | Double Fav | Inner Click | Inner Like | Inner Fav |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RankMixer | 0.6389 | 0.6610 | 0.6624 | 0.6911 | 0.6621 | 0.6787 | 0.6393 | 0.6786 | 0.6621 |
| SharedBottom | 0.6391 | 0.6611 | 0.6614 | 0.6929 | 0.6629 | 0.6789 | 0.6395 | 0.6794 | 0.6609 |
| MMoE | 0.6394 | 0.6614 | 0.6623 | 0.6931 | 0.6629 | 0.6788 | 0.6397 | 0.6793 | 0.6629 |
| STAR | 0.6389 | 0.6611 | 0.6600 | 0.6930 | 0.6624 | 0.6784 | 0.6393 | 0.6796 | 0.6631 |
| HMoE | 0.6397 | 0.6617 | 0.6623 | 0.6925 | 0.6625 | 0.6790 | 0.6401 | 0.6790 | 0.6634 |
| PEPNet | 0.6396 | 0.6609 | 0.6628 | 0.6928 | 0.6625 | 0.6787 | 0.6403 | 0.6797 | 0.6627 |
| **MDL** | **0.6417** | **0.6632** | **0.6656** | **0.6950** | **0.6671** | **0.6842** | **0.6419** | **0.6820** | **0.6681** |
| Improv. | +0.31% | +0.23% | +0.42% | +0.27% | +0.63% | +0.77% | +0.25% | +0.34% | +0.71% |

**关键发现**：

1. MDL 在所有 3 场景 x 3 任务的 9 个指标上全面且大幅领先所有 baseline。最大提升达 +0.77%（Double-column Fav），这在 0.5B 参数规模的模型上是非常显著的。

2. **数据稀疏场景获益更大**：双列搜索和内嵌搜索（数据更稀疏）的提升幅度大于单列搜索。这表明 token 化的信息传递机制可以更好地缓解数据稀疏问题——通过跨场景共享特征 token 的深层表示。

3. **低正样本率任务获益更大**：Like 和 Fav 任务（正样本率远低于 Click）的提升幅度更大。说明 MDL 的逐层交互可以缓解多任务学习中的 **seesaw effect**（强势任务压制弱势任务），让弱势任务更好地保持自己的分布建模。

4. 所有 shared-specific baseline（SharedBottom、MMoE、STAR、HMoE、PEPNet）相比纯 RankMixer backbone 都有提升，但提升幅度有限（通常 < 0.1%），且不同 baseline 之间差异不大。这说明传统 shared-specific 架构在大参数模型上已接近饱和。

### 3.3 消融实验

| 变体 | Single Delta | Double Delta | Inner Delta |
|------|:-:|:-:|:-:|
| w/o task token | -0.12% | -0.11% | -0.09% |
| w/o task-feature interaction | -0.04% | -0.05% | +0.03% |
| w/o scenario token | -0.17% | -0.16% | -0.15% |
| w/o global scenario token | -0.04% | -0.06% | -0.05% |
| w/o scenario-feature interaction | -0.05% | -0.04% | -0.05% |

**逐条分析**：

- **去掉 task token（-0.12%）**：退化为传统 task-tower 多任务结构，说明 task tokenization 本身带来了显著价值。loss 主要来自 task token 无法逐层与 feature token 深度交互。

- **去掉 task-feature interaction（-0.04%）**：保留 task token 但用 RankMixer 的标准交互替代 Domain-aware Attention。下降较小说明 task token 的存在本身已有价值（提供了任务级别的信息聚合点），但专门的跨注意力进一步提供了差异化特征选择能力。

- **去掉 scenario token（-0.17%）**：最大的单项下降，说明场景 token 化是 MDL 中最关键的组件。退化为传统 scenario-tower 后，场景信号只能在浅层影响模型。

- **去掉 global scenario token（-0.04%~-0.06%）**：全局场景 token 的作用是捕获跨场景共性知识，去掉后各场景 token 之间缺少信息桥梁。

- **去掉 scenario-feature interaction（-0.05%）**：与 task-feature interaction 类似，Domain-aware Attention 的差异化特征选择能力提供了额外增益。

### 3.4 Scaling Law 分析

![[MDL_Multi_Distribution_Learner/images/scaling_params.png|450]]
![[MDL_Multi_Distribution_Learner/images/scaling_flops.png|450]]
> 图2：MDL vs MMoE 在不同参数量/FLOPs 下的 QAUC 增益对比。随着模型规模增大，MDL 的优势持续扩大。

关键观察：

1. MDL 在所有参数规模和 FLOPs 配置下都优于 MMoE
2. **随着模型规模增大，MDL 相对 MMoE 的增益持续增长**——这是论文最重要的发现之一，直接证明了 MDL 能更好地「利用」大参数模型的能力来做多分布学习
3. MMoE 的 scaling 曲线趋于平坦，而 MDL 保持了更陡峭的上升趋势

这验证了论文的核心假设：传统 shared-specific 方法在大模型上的 scaling 能力受限，而 tokenization + 逐层交互的范式能更好地收割 scaling law 的红利。

### 3.5 注意力可视化分析

![[MDL_Multi_Distribution_Learner/images/click_layer1.png|450]]
![[MDL_Multi_Distribution_Learner/images/click_layer2.png|450]]
> 图3：Click task token 在 Layer1 和 Layer2 对 feature tokens 的注意力分布。不同层关注不同的特征子空间。

![[MDL_Multi_Distribution_Learner/images/like_layer1.png|450]]
![[MDL_Multi_Distribution_Learner/images/like_layer2.png|450]]
> 图4：Like task token 在 Layer1 和 Layer2 的注意力分布。与 Click task 有明显差异，验证了任务间的差异化建模。

![[MDL_Multi_Distribution_Learner/images/single_scenario_attn.png|450]]
![[MDL_Multi_Distribution_Learner/images/double_scenario_attn.png|450]]
> 图5：单列搜索 vs 双列搜索场景 token 对 feature tokens 的注意力分布。不同场景关注的特征子空间不同。

**关键发现**：

1. **任务间差异**：Click token 和 Like token 在同一层对 feature tokens 的注意力分布明显不同，说明不同任务学会了选择不同的特征子空间。
2. **层间动态**：同一 task token 在不同层的注意力分布也不同，说明随着逐层信息传递，task token 能动态调整自己的注意力策略。
3. **场景间差异**：单列搜索和双列搜索的场景 token 对 feature tokens 的注意力分布也存在显著差异，验证了场景级别的差异化建模。

### 3.6 在线 A/B 实验

在抖音搜索线上部署一个月的 A/B 测试，baseline 为 RankMixer + MMoE：

| 场景 | Change Query Rate | LT30 |
|------|:-:|:-:|
| ALL | **-0.3267%** | **+0.0626%** |
| 单列搜索 | -0.2678% | +0.0520% |
| 双列搜索 | -0.5079% | +0.0674% |
| 内嵌搜索 | -0.5492% | +0.0630% |

**解读**：

- **LT30 +0.0626%**：30 天用户生命周期指标提升，这是 DAU 增长的核心指标。在抖音搜索的流量规模下，这是非常显著的业务价值。
- **Change Query Rate -0.3267%**：用户主动改 query 的概率下降，说明搜索结果满意度提升——用户不需要反复修改搜索词就能找到满意内容。
- 双列搜索和内嵌搜索（数据更稀疏的场景）获益更大，与离线实验一致。

MDL 已全量部署于抖音搜索，服务数亿用户。

## 四、优势与创新点

1. **首创 Tokenize-and-Interact 范式**：将多场景多任务信息从「辅助信号」提升为与特征平等的「Token」，使其能够从底层开始逐层参与特征交互，充分利用大模型参数。这是对传统 shared-specific 架构的范式性突破。

2. **统一建模 MSL 和 MTL**：通过将 scenario 和 task 信息统一 tokenize 到同一语义空间，MDL 首次实现了在单一框架下同时建模输入分布差异（场景）和标签分布差异（任务），消除了传统方法需要分别设计的割裂感。

3. **Scaling Law 友好**：实验证明随着模型参数/FLOPs 增长，MDL 相比 MMoE 的增益持续增大。这说明 token 化交互能更好地利用模型容量，在大规模推荐系统 scaling 趋势下具有长期价值。

4. **缓解 Seesaw Effect**：在数据稀疏场景（双列搜索、内嵌搜索）和低正样本率任务（like、favorite）上提升更大，说明 tokenized 信息交互能有效缓解多分布学习中的跷跷板效应。

5. **工业级验证**：在抖音搜索上全量部署，服务数亿用户，LT30 +0.0626%、Change Query Rate -0.3267%，且各场景均有显著提升。

## 五、局限性与讨论

1. **对 Backbone 的依赖**：MDL 的特征自交互模块直接复用 RankMixer，论文未探索在其他 backbone（如 Transformer、HSTU）上的适配效果。虽然论文声称该模块可替换，但未提供实证。

2. **Token 数量设计的人工性**：Feature token 的数量 $N_f$ 需要人工按语义分组，scenario/task token 数量等于场景/任务数。论文未讨论当场景数极多（如数十个细粒度场景）时是否需要更灵活的 token 设计。

3. **Domain-fused Module 过于简单**：场景-任务融合仅使用了 MeanPooling + Sum，理论上更复杂的融合策略（如 Gated Fusion、Cross-Attention）可能带来进一步提升。论文承认 "simple pooling is sufficiently effective"，但未做充分对比。

4. **缺乏公开数据集实验**：所有实验均在抖音内部数据上进行，不含任何公开 benchmark（如 AliCCP、KuaiRand），影响了可复现性和外部评估。

5. **在线指标选择的局限**：LT30 和 Change Query Rate 是长期指标，但论文未报告 CTR/CVR 等直接排序指标的在线变化，也未讨论模型上线对推理延迟的影响。

6. **与 Prompt Tuning 的关系未深入**：论文类比 LLM 的 prompting，但 MDL 的 scenario/task token 本质是从输入特征变换而来（非可学习的 soft prompt），与 LLM prompt tuning 存在本质差异，这一点的讨论不够深入。

## 六、与相关工作的关系

| 方法 | 架构范式 | MSL 方式 | MTL 方式 | 与特征交互的深度 | Scaling 友好 |
|------|----------|----------|----------|-----------------|-------------|
| SharedBottom | Shared-Specific | Scenario Tower | Task Tower | 仅影响输出层 | 否 |
| MMoE | MoE | 共享 + 专家路由 | 共享 + 专家路由 | 中间某一层 | 一般 |
| PLE | Hierarchical MoE | 分层专家 | 分层专家 | 中间多层 | 一般 |
| STAR | Star Topology | 中心+场景网络 | - | 影响中间层参数 | 一般 |
| PEPNet | Gate Network | Prior Gate | Prior Gate | 影响 embedding 和隐层 | 一般 |
| HMoE | Hybrid MoE | 隐式+显式 MoE | 隐式+显式 MoE | 专家层 | 一般 |
| **MDL** | **Tokenize-and-Interact** | **Scenario Token** | **Task Token** | **逐层从底到顶** | **是** |

核心对比：传统方法将 scenario/task 信息作为 gate signal 或 routing condition，仅在 1-2 层起作用；MDL 将其 token 化后逐层参与注意力交互，信息渗透到每一层的每个参数。

## 七、个人思考

### 7.1 方法论洞察

MDL 的核心贡献是将「大模型推荐系统如何做多场景多任务」这个工程问题重新定义为「如何让分布信息充分激活大模型参数」。这个视角转换非常有价值——当模型参数达到 0.5B 量级时，传统 MoE/Gate 机制只能作用于局部层，浪费了大量参数的表达潜力。MDL 的 tokenize + layer-wise interaction 方案本质上是一种「分布信息的全模型扩散」机制。

从 LLM 类比的角度，MDL 的 scenario/task token 更像是 prefix token（从数据特征生成）而非 soft prompt（纯可学习向量），这种设计确保了 token 的信息丰富度，但也限制了它的灵活性。

### 7.2 工程部署思考

- **推理开销**：Domain-aware Attention 中 scenario/task token 作为 query 的数量极少（$N_s + N_t$ 约为 20+ 量级），因此 cross-attention 的计算量远小于 feature token self-attention，推理延迟增量可控。
- **训练架构**：论文使用 sparse（Adagrad）+ dense（RMSProp）异步-同步混合训练，数百 GPU 规模，是字节典型的大规模推荐训练框架。
- **灵活性**：由于 scenario token 数量等于场景数，新增场景时需要增加 token 并重新训练，不支持 zero-shot 泛化到新场景。

### 7.3 与 RankMixer 的关系

MDL 以 RankMixer 为 backbone，所有 baseline 也统一集成到 RankMixer 上对比，确保了参数量一致的公平对比。但这也意味着 MDL 的效果高度依赖 RankMixer 的 token mixing 能力——如果 backbone 换成传统 MLP 或 DCN，token 化交互的收益是否同样显著值得验证。

### 7.4 适用性评估

**特别适合**：
- 已经采用大参数推荐模型（0.1B+）的系统，需要 MSL/MTL 且对 scaling 有需求
- 多场景重叠度高（如搜索的单列/双列/内嵌本质是同一份数据的不同分发形式）
- 追求统一框架管理 scenario+task，不想维护大量独立的 scenario-specific 模块

**不太适合**：
- 小模型（参数 < 10M），tokenize 的额外开销占比过大
- 场景间差异极大（如电商+视频+金融跨域），可能需要更强的隔离机制
- 需要快速新增场景且无法全量重训的系统

## 八、总结

### 核心贡献

MDL 提出了一种基于 Tokenization 的多分布学习统一框架，通过 **Unified Information Tokenization + Domain-aware All-Token Interaction** 三层交互机制（Feature Self-Interaction、Domain-aware Attention、Domain-fused Module），让场景和任务信息从底层逐层参与特征交互，充分激活大规模模型的参数空间，实现了 MSL 和 MTL 的统一建模。

### 关键 Takeaways

1. **Tokenize Everything**：将 feature/scenario/task 统一表达为 token 是实现深层信息交互的前提。只有在同一语义空间中，不同类型的信息才能通过注意力机制自然交互。

2. **深度交互 > 浅层调控**：传统 MSL/MTL 方法将场景/任务信息限制在 gate 或输出层，MDL 证明了逐层 cross-attention 交互能更有效地利用大模型参数，且随 scaling 增益递增。

3. **Global Token 很重要**：消融实验显示 global scenario token 贡献约 0.04-0.06% QAUC，说明跨场景共享知识的建模不可忽略。

4. **稀疏场景受益更大**：Token 化信息共享机制天然有利于数据稀疏场景和低频任务，有效缓解多分布学习的 seesaw effect。

5. **工业级验证充分**：抖音搜索全量部署，LT30 +0.0626%（DAU 增长核心指标）、Change Query Rate -0.3267%（用户满意度指标），服务数亿用户。

### 评分理由

- **创新性 8/10**：Tokenize-and-Interact 范式新颖，对 MSL/MTL 问题提出了全新的解决思路
- **技术质量 7.5/10**：模块设计清晰合理，但 Domain-fused Module 过于简单，部分设计缺乏理论分析
- **实验充分性 7/10**：离线消融+在线 AB 完整，但缺公开数据集实验、缺推理延迟分析
- **写作质量 7/10**：结构清晰，但部分公式符号不一致
- **工业价值 9/10**：直接解决大规模推荐系统核心问题，全量部署验证
- **综合 8.0/10**
