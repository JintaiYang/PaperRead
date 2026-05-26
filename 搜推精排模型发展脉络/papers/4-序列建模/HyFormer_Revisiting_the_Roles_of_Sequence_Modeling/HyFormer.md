---
paper_id: "[arXiv:2601.12681](https://arxiv.org/abs/2601.12681)"
title: "HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction"
authors: "Yunwen Huang*, Shiyong Hong*, Xijun Xiao*, Jinqiu Jin*, Xuanyuan Luo, Zhe Wang, Zheng Chai, Shikang Wu, Yuchao Zheng, Jingjian Lin"
institution: "ByteDance AML & ByteDance Search"
publication: "RecSys 2025 (投稿中) [2025-01-23]"
tags:
  - "Hybrid-Transformer"
  - "长序列建模"
  - "特征交互"
  - "CTR预测"
  - "大规模推荐模型"
  - "Scaling-Law"
  - "Query-Decoding"
  - "MLP-Mixer"
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2601.12681)"
date: "2025-01-23"
---

## 一、研究背景与动机

### 1.1 领域现状

工业级大规模推荐模型（LRM）面临的核心挑战在于：如何在严格的延迟和吞吐约束下，联合建模用户的长程行为序列（千级以上）与大量异构非序列特征（用户画像、上下文信号、交叉特征等）。当前主流的工业架构已经收敛到一种"分离式 Scaling 范式"——先用专门的序列 Transformer（如 LONGER、SIM、ETA、TWIN、TransAct）对长行为序列进行编码压缩，再将压缩后的序列 token 与其他异构特征通过 token-mixing 模块（如 RankMixer、Wukong、DCNv2）进行特征交互。这种"先长序列建模，再异构特征交互"的流水线设计已成为工业 LRM 的主导选择。

### 1.2 现有方法的局限性

论文指出这种解耦流水线存在三个根本性限制：

第一，序列压缩阶段的 query 表示过于简化。现有架构中用于聚合长行为序列的 query token 通常仅来自候选 item 相关的有限特征子集，限制了建模长期用户兴趣时可利用的上下文信息量。而直接增加 query token 数量会在 KV-Cache 和 M-Falcon 机制下导致服务效率的明显下降。

第二，序列压缩 token 与异构非序列 token 之间的交互仅发生在模型的后期阶段。跨特征推理被推迟到序列压缩之后，导致不同 token 类型之间的交互浅层且隐式，阻碍了早期层表示从跨域上下文信息中获益。

第三，由于交互模块仅作用于压缩后的序列表示，增加模型容量或序列长度主要改善的是孤立组件，而非联合表示。这导致 Scaling 效率偏低——性能提升相对于额外计算预算的增长速率较慢。

### 1.3 本文解决方案概述

HyFormer 提出了一种统一的混合 Transformer 架构，将长序列建模与特征交互融合到单一骨干网络中。其核心思想是引入一组 Global Tokens 作为长行为序列与异构特征之间的共享语义接口，通过交替执行 Query Decoding（用全局 query 对长序列进行 cross-attention 解码）和 Query Boosting（通过 MLP-Mixer 风格的 token mixing 增强 query 间交互）两个轻量机制，实现序列建模与特征交互之间的双向信息流动。

## 二、解决方案

### 2.1 核心思想

HyFormer 的核心洞察是：传统的"先压缩再交互"范式将表达力和交互灵活性锁死在"晚融合、单向流动"的结构中。HyFormer 将 LRM 建模任务重新定义为一个交替优化过程——在每一层中，先用丰富的全局信息去"询问"长序列（Query Decoding），再通过特征交互来"增强"这些 query 的表达力（Query Boosting），如此反复迭代，使得 query 在逐层深入的过程中携带越来越丰富的语义信息。

用一个直觉性的类比来理解：传统方法相当于"先把一本书压缩成摘要，再拿摘要去和其他信息做对比"；而 HyFormer 相当于"带着越来越多的背景知识反复翻阅原书的不同章节，每次都能读出更深层的含义"。

### 2.2 整体架构

![[hyformer_architecture.png|800]]
> 图1：HyFormer 整体架构。左侧为传统的 LONGER + RankMixer 两阶段流水线，右侧为 HyFormer 的统一架构。HyFormer 将原始序列建模中的"候选 item"扩展为 Global Tokens，通过 Query Decoding 和 Query Boosting 的交替执行实现双向信息流。

HyFormer 的整体架构由以下核心组件构成：

**问题形式化**：给定用户 $u$ 的行为历史 $S = [i^{(u)}_1, \ldots, i^{(u)}_K]$、非序列描述符 $u$（画像、上下文、交叉特征）和候选 item $v$，目标是估计用户与候选 item 交互的概率：

$$P(y = 1 \mid S, u, v) \in [0,1]$$

模型通过最小化标准二元交叉熵损失进行训练：

$$\mathcal{L} = -\frac{1}{|\mathcal{D}|} \sum_{(S, u, v, y) \in \mathcal{D}} \Big[ y \log \hat{y} + (1-y) \log (1 - \hat{y}) \Big]$$

其中 $\hat{y} = f_{\theta}(S, u, v)$ 为模型预测的交互概率。

#### 各模块详细说明

**模块1：Query Generation（查询生成）**

- **功能**：将异构非序列特征转换为用于解码长行为序列的语义 query token
- **输入**：非序列特征向量 $F_1, F_2, \ldots, F_M \in \mathbb{R}^{1 \times D}$ 以及行为序列的池化摘要
- **输出**：$N$ 个 query token $Q \in \mathbb{R}^{N \times D}$

首先，将非序列特征与序列池化摘要拼接形成全局信息：

$$\mathrm{Global \ Info} = \mathrm{Concat}\big(F_{1}, \ldots, F_{M},\; \mathrm{MeanPool}(Seq)\big)$$

然后通过 $N$ 个独立的 FFN 生成 $N$ 个 query token：

$$Q = \big[\mathrm{FFN}_1(\mathrm{Global \ Info}), \ldots, \mathrm{FFN}_N(\mathrm{Global \ Info})\big] \in \mathbb{R}^{N \times D}$$

这里每个 $\mathrm{FFN}_i$ 是一个轻量前馈网络，将全局信息映射到一个 $D$ 维的 query 向量。这种设计使得每个 query 都携带了来自非序列特征和序列全局摘要的信息，相比传统方法仅用候选 item 特征作为 query，信息量更为丰富。

在更深的 HyFormer 层中，query 不再通过 MLP 重新生成，而是直接复用上一层的输出作为新的 query，使得更深层的 cross-attention 能够用语义更丰富的 query 来"询问"长序列。

**模块2：Sequence Representation Encoding（序列表示编码）**

- **功能**：为长行为序列生成逐层的 Key-Value 表示
- **输入**：行为序列 $S$
- **输出**：逐层的 $(K^{(s)}_{l}, V^{(s)}_{l})$ 对

HyFormer 支持三种编码策略，提供不同的容量-效率权衡：

(i) **Full Transformer Encoding**：标准 Transformer 编码器，通过全自注意力捕获细粒度交互和长程依赖：

$$H_{l} = \mathrm{TransformerEnc}_{l}(S)$$

(ii) **LONGER-style Efficient Encoding**：用紧凑短序列对全历史做 cross-attention，将复杂度从 $\mathcal{O}(L_S^2)$ 降至 $\mathcal{O}(L_H L_S)$：

$$H_{l} = \mathrm{CrossAttn}(S_{\text{short}},\; S,\; S)$$

其中 $S_{\text{short}}$ 是长度为 $L_H \ll L_S$ 的紧凑短序列。

(iii) **Decoder-style Lightweight Encoding**：用无注意力的前馈操作变换序列表示，适用于延迟敏感场景：

$$H_{l} = \mathrm{SwiGLU}_{l}(S)$$

无论采用哪种变体，最终都通过线性投影得到逐层的 K/V 状态：

$$K_{l} = H_{l} W^{K}_{l}, \qquad V_{l} = H_{l} W^{V}_{l}$$

K/V 状态在每一层重新计算，使序列特征能够随解码器深度联合演化。

**模块3：Query Decoding（查询解码）**

- **功能**：通过 cross-attention 从长行为序列中提取目标感知信息
- **输入**：query token $Q_{(l)} \in \mathbb{R}^{N \times D}$ 和序列 K/V 表示 $(K_{(l)}, V_{(l)})$
- **输出**：解码后的 query $\tilde{Q}_{(l)}$

$$\tilde{Q}_{(l)} = \mathrm{CrossAttn}\!\left(Q_{(l)},\, K_{(l)},\, V_{(l)}\right)$$

这一步使得全局非序列特征能够直接 attend 到长行为序列，将上下文信号注入到序列感知的 query 表示中。

**模块4：Query Boosting（查询增强）**

- **功能**：增强 query 表示，使其在送入下一层 cross-attention 前获得更丰富的跨 token 交互信息
- **输入**：解码后的 query $\tilde{Q}_{(l)}$ 与非序列特征 token 的拼接
- **输出**：增强后的 query $Q_{\mathrm{boost}}$

首先构建统一的 query 表示：

$$Q = [\tilde{Q}_{(l)}, F_1, \ldots, F_M] \in \mathbb{R}^{T \times D}$$

其中 $T = N + M$。然后应用 MLP-Mixer 风格的 token mixing。具体地，每个 query token $q_t$ 被划分为 $T$ 个通道子空间：

$$q_t = [\, q_t^{(1)} \| q_t^{(2)} \| \cdots \| q_t^{(T)} \,], \quad q_t^{(h)} \in \mathbb{R}^{D/T}$$

对于每个子空间索引 $h$，MLP-Mixer 通过拼接所有 token 位置的对应子空间来聚合信息：

$$\tilde{q}_h = \mathrm{Concat}\big(q_1^{(h)}, q_2^{(h)}, \ldots, q_T^{(h)}\big) \in \mathbb{R}^{D}$$

收集所有混合 token 得到：

$$\hat{Q} = [\tilde{q}_1, \tilde{q}_2, \ldots, \tilde{q}_T] \in \mathbb{R}^{T \times D}$$

再经过逐 token 的 FFN 精炼和残差连接：

$$\widetilde{Q} = \mathrm{PerToken\text{-}FFN}(\hat{Q})$$
$$Q_{\mathrm{boost}} = Q + \widetilde{Q}$$

这种设计使得 query 在每一层都能获得来自其他 query 和非序列特征的交互信息，而计算复杂度保持线性。

**模块5：HyFormer Module（整体堆叠）**

HyFormer 模块通过堆叠多层实现，每层包含一个 Query Decoding 块和一个 Query Boosting 块。在第 $l$ 层：

$$\widehat{Q}^{(l)} = \mathrm{CrossAttn}\big(Q^{(l-1)}, K^{(l)}, V^{(l)}\big)$$

$$\widetilde{Q}^{(l)} = \mathrm{QueryBoost}\big(\mathrm{Concat}(\widehat{Q}^{(l)}, \mathrm{NS \ Tokens})\big)$$

通过多层堆叠，HyFormer 逐步精炼语义 query，使更深层能够用越来越丰富的表示来抽象长序列。顶层 HyFormer 的输出送入下游 MLP 进行最终预测。

**模块6：Multi-Sequence Modeling（多序列建模）**

![[multi_sequence_modeling.png|800]]
> 图2：HyFormer 的多序列建模策略。每个行为序列独立处理，使用专属的 query token 进行 Query Decoding，跨序列交互通过 query 级别的 token mixing 实现。

在工业推荐场景中，用户行为通常组织为多个异构序列（如视频观看序列和商品购买序列）。HyFormer 不采用简单的序列合并（如 MTGR/OneTrans 所做的），而是在每个 HyFormer 块中独立处理每个行为序列。对每个序列构建专属的 query token 集合进行 Query Decoding，保留序列特有的语义，跨序列交互则通过后续的 query 级 token mixing 来处理。

### 2.3 训练与部署优化

论文还介绍了两项工程优化：

**GPU Pooling for Long-Sequence**：利用长序列中真正唯一的特征 ID 数量有限（通常约为总 token 数的 25%）这一稀疏性，在 GPU 上直接重建原始序列特征，减少 Host-to-Device 传输开销和主机内存压力。

**Asynchronous AllReduce**：允许第 $k$ 步的梯度同步与第 $k+1$ 步的前向/反向计算重叠，消除通信气泡。代价是 dense 参数引入一步 staleness（$W_{k} = W_{k-1} + g_{k-1}$），而 sparse 参数可立即更新（$W_{k} = W_{k-1} + g_{k}$）。实验表明这种混合更新策略不影响收敛质量。

## 三、实验结果

### 3.1 数据集

实验在抖音搜索系统的 CTR 预测任务上进行评估，数据集来自连续 70 天的在线用户交互日志，包含 30 亿样本。每个样本包含用户特征、query 特征、文档特征、交叉特征和多个序列特征。

| 数据集 | 样本数 | 序列类型 | 序列长度上限 | 数据来源 |
|--------|--------|----------|-------------|----------|
| 抖音搜索 | 30亿 | 长期序列 | 3000 | 用户长期搜索点击行为 |
| -- | -- | 搜索序列 | 50 | Query Search 过滤的搜索行为 |
| -- | -- | Feed序列 | 50 | Query Search 过滤的 Feed 行为 |

### 3.2 实验设置

#### 3.2.1 基线方法

基线分为两类架构范式：

**传统两阶段模型（BaseArch）**：序列建模和特征交互分离为两个阶段。序列建模使用 LONGER 或 Full Transformer；特征交互使用 RankMixer、Full Transformer 或 Wukong。

**统一架构模型（UniArch）**：序列和非序列特征在单一模型块中联合处理。包括 MTGR 和 OneTrans，前者将所有特征 token 化后用 Transformer 风格骨干联合建模，后者采用金字塔压缩结构。

#### 3.2.2 评估指标

离线评估使用 Query-level AUC（对每个 query 内的样本计算 AUC 后取平均）。同时报告 dense 参数量和训练 FLOPs（batch size = 2048）。

#### 3.2.3 训练细节

- Batch size: 2048
- MLPMixer 输入 token 数统一对齐为 16（13 个非序列 token + 3 个 Global Token，每个序列一个）
- 离线评估采用冷启动训练，在线评估使用 checkpoint 热启动
- 64-GPU 集群训练

### 3.3 实验结果与分析

**主实验结果**：

| 类别 | 序列建模 | 特征交互 | AUC | ΔAUC | Params(×10⁶) | FLOPs(×10¹²) |
|------|----------|----------|-----|------|--------------|--------------|
| BaseArch | LONGER | RankMixer | 0.6478 | -- | 386 | 3.5 |
| BaseArch | LONGER | Full Transformer | 0.6472 | -0.09% | 416 | 6.2 |
| BaseArch | LONGER | Wukong | 0.6465 | -0.20% | 385 | 5.2 |
| BaseArch | Full Transformer | RankMixer | 0.6481 | +0.05% | 388 | 6.6 |
| BaseArch | Full Transformer | Full Transformer | 0.6474 | -0.06% | 418 | 9.3 |
| BaseArch | Full Transformer | Wukong | 0.6468 | -0.15% | 387 | 8.3 |
| UniArch | MTGR/OneTrans (w/ LONGER) | -- | 0.6480 | +0.03% | 406 | 6.6 |
| UniArch | MTGR/OneTrans (w/ Full Transformer) | -- | 0.6483 | +0.08% | 450 | 21.9 |
| **UniArch** | **HyFormer (Ours)** | -- | **0.6489** | **+0.17%** | **418** | **3.9** |

#### 结果分析

从主实验结果可以观察到几个关键发现：

在 BaseArch 组内，特征交互模块的选择对性能影响较大：RankMixer 一致优于 Self-Attention 和 Wukong，这与 RankMixer 论文的结论一致——MLP-Mixer 风格的 token mixing 在工业推荐场景中比 Self-Attention 更适合做特征交互。序列建模方面，Full Transformer 相比 LONGER 带来了一定提升（+0.05%），但代价是 FLOPs 接近翻倍（3.5→6.6）。

BaseArch 中表现较好的组合（Full Transformer + RankMixer，AUC 0.6481）仍低于 HyFormer（0.6489），且 FLOPs 高出近 70%（6.6 vs 3.9）。这表明单向信息流的固有限制无法通过简单增加计算量来弥补。

MTGR/OneTrans 作为统一架构，在使用 Full Transformer 时达到 0.6483，但 FLOPs 高达 21.9×10¹²，是 HyFormer 的 5.6 倍。这主要因为 MTGR 将 Global Tokens 和 Seq Tokens 共同作为 keys，而仅用 Global Tokens 作为 queries，导致 Global Tokens 更容易 attend 到自身而非序列 token。HyFormer 则强制分离信息流：先将具体的序列 item 信息压缩吸收到 Global Tokens 中，再在不同抽象 Global Tokens 之间进行交互。

### 3.4 消融实验

| 配置 | AUC | ΔAUC | Params(×10⁶) | FLOPs(×10¹²) |
|------|-----|------|--------------|--------------|
| **Query 全局上下文消融** | | | | |
| HyFormer | 0.6489 | -- | 418 | 3.9 |
| Query w/o Seq Pooling Tokens | 0.6486 | -0.05% | 415 | 3.9 |
| Query w/o Nonseq and Seq Pooling Tokens | 0.6484 | -0.08% | 414 | 3.8 |
| **Query Boosting 消融** | | | | |
| HyFormer | 0.6489 | -- | 418 | 3.9 |
| HyFormer w/o Global Tokens | 0.6484 | -0.08% | 414 | 3.8 |
| BaseArch w/ Global Tokens | 0.6480 | -0.14% | 505 | 3.6 |
| BaseArch w/o Global Tokens | 0.6478 | -0.17% | 387 | 3.5 |
| **多序列建模消融** | | | | |
| HyFormer | 0.6489 | -- | 418 | 3.9 |
| HyFormer + Merge Seq | 0.6485 | -0.06% | 397 | 3.9 |

消融实验揭示了几个重要发现：

**Query 全局上下文的贡献**：将 query 退化为仅使用原始 target 特征（去掉非序列特征和序列池化 token）导致 -0.08% AUC 下降，说明丰富的 query 信息对后续深层特征交互至关重要。仅去掉跨序列池化 token 也带来 -0.05% 的损失，确认了序列间交互在 HyFormer 结构中的贡献。

**架构设计的贡献**：在 BaseArch 框架下，即使加入 Global Tokens 丰富 query 信息，也仅获得 0.03% 的提升（-0.14% vs -0.17%）。而在 HyFormer 框架中，同样的 query 信息扩展带来了 0.08% 的增益。这说明 HyFormer 的交替堆叠设计能够更充分地利用丰富的 query 信息。

**多序列建模策略**：序列合并（Merge Seq）导致 -0.06% 的 AUC 损失。合并迫使不同序列共享 Global Tokens，忽略了序列的差异性，产生的表示远不如 HyFormer 独立建模各序列时那样具有区分度。

### 3.5 Scaling 分析

![[scaling_params.png|800]]
> 图3(a)：AUC 随参数量的 Scaling 曲线。HyFormer 相比 BaseArch（LONGER + RankMixer）展现出更陡峭的斜率。

![[scaling_flops.png|800]]
> 图3(b)：AUC 随 FLOPs 的 Scaling 曲线。HyFormer 在相同计算预算下获得更高的 AUC，且呈现较强的幂律趋势。

**参数 Scaling**：在 200M 到 1B+ 参数范围内，HyFormer 不仅初始性能优于 BaseArch，还保持了更陡峭的 Scaling 斜率。这表明 HyFormer 中 LONGER 和 RankMixer 交替堆叠所实现的双向信息流，使其在相似参数规模下从增加深度中获得的收益明显大于 BaseArch。

**FLOPs Scaling**：AUC 随 FLOPs 稳步增长，呈现较强的幂律趋势。增加计算资源使模型能够用更丰富的信息处理序列，受益于初始 query 的扩展和 MLP-Mixer 中 query 的反复增强。

**Sparse Dim Scaling**：

| 序列长度 | 架构 | Seq Sparse Dim | AUC | ΔAUC | ΔAUC Gap |
|----------|------|---------------|-----|------|----------|
| 1k | BaseArch | 64 | 0.6478 | -- | -- |
| 1k | BaseArch | 224 | 0.6484 | +0.09% | -- |
| 1k | HyFormer | 64 | 0.6489 | -- | -- |
| 1k | HyFormer | 224 | 0.6497 | +0.12% | +0.03% |
| 3k | BaseArch | 64 | 0.6486 | -- | -- |
| 3k | BaseArch | 224 | 0.6490 | +0.06% | -- |
| 3k | HyFormer | 64 | 0.6499 | -- | -- |
| 3k | HyFormer | 224 | 0.6507 | +0.12% | +0.06% |

无论序列长度如何，丰富序列 side information 对 HyFormer 的收益一致大于 BaseArch。且随着序列变长，HyFormer 相对 BaseArch 的额外增益从 0.03%（1k）扩大到 0.06%（3k），说明 HyFormer 能更充分地利用丰富的序列 K/V 信息。

### 3.6 在线 A/B 测试

| 在线指标 | 增益 |
|----------|------|
| Average Watch Time Per User ↑ | +0.293% |
| Video Finish Play Count Per User ↑ | +1.111% |
| Query Change Rate ↓ | -0.236% |

在抖音搜索平台的大规模在线 A/B 测试中，HyFormer 相比已部署的RankMixer 基线取得了多项指标的正向提升：用户平均观看时长 +0.293%，视频完播数 +1.111%，query 改写率（负向指标）-0.236%。这些结果在服务数十亿用户的生产环境中验证了 HyFormer 的实际价值。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文结论部分提到，HyFormer 的双向信息流设计为未来 LRM 的 scaling 提供了新的上限。作者暗示后续可能在以下方向继续探索：更深层次的统一建模、更大规模的参数扩展、以及将该框架推广到更多业务场景。

### 4.2 基于分析的未来方向

1. **方向1：在线学习与增量更新**
   - 动机：HyFormer 的 Global Token 设计天然适合增量更新场景，因为 query 的语义可以随时间演化
   - 可能的方法：在 Query Generation 阶段引入时间衰减机制，或对 Global Token 做流式更新
   - 预期成果：在保持模型效果的同时降低全量训练频率
   - 挑战：增量更新可能导致 Global Token 语义漂移

2. **方向2：多目标联合优化**
   - 动机：当前论文仅关注 CTR 单目标，但工业系统通常需要多目标（CVR、时长、多样性等）
   - 可能的方法：在 Query Boosting 阶段引入目标感知的 token mixing，不同目标使用不同的 boosting 路径
   - 预期成果：在统一框架内实现多目标的协同优化
   - 挑战：多目标间的冲突可能影响 Global Token 的语义一致性

3. **方向3：将 HyFormer 思想应用于搜索引导排序**
   - 动机：搜索引导（SUG/底纹词/搜索发现）同样面临长序列建模与异构特征交互的问题
   - 可能的方法：将 query 历史序列、点击序列作为多序列输入，用 HyFormer 的独立序列建模+共享 Global Token 交互
   - 预期成果：在搜索引导精排中获得类似的 AUC 提升
   - 挑战：搜索引导场景的序列长度和特征空间与主搜索有差异，需要适配

### 4.3 改进建议

1. **改进1：Query Generation 的动态化**
   - 当前问题：Query Generation 使用固定的 FFN 映射，对不同用户/场景缺乏自适应能力
   - 改进方案：引入 MoE 结构或条件生成机制，根据用户特征动态选择 query 生成路径
   - 预期效果：对长尾用户和冷启动场景可能有更好的适应性

2. **改进2：序列编码的异构化**
   - 当前问题：论文中多序列使用相同的编码策略（LONGER/Full Transformer/SwiGLU），但不同序列的信息密度和长度差异较大
   - 改进方案：为不同序列自适应选择编码策略，如长序列用 LONGER、短序列用 Full Transformer
   - 预期效果：在计算预算不变的情况下获得更好的序列表征

## 五、我的综合评价

### 5.1 价值评分

**7.5/10** - 这是一篇工程导向较强的工业论文，在统一序列建模与特征交互方面提出了实用的解决方案，并在大规模生产环境中得到验证。

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 核心思想（Global Token + 交替 Decoding/Boosting）是对已有组件（LONGER、RankMixer）的有机整合，属于较好的系统级创新，但单个模块的新颖性有限 |
| 技术质量 | 8/10 | 方法论清晰，公式推导完整，消融实验设计合理，能够逐步验证各组件的贡献 |
| 实验充分性 | 7/10 | 在十亿级工业数据集上进行了充分的离线实验和在线 A/B 测试，但缺少公开数据集的对比，可复现性受限 |
| 写作质量 | 7/10 | 整体结构清晰，但部分段落较为冗长，Related Work 中对 MTGR/OneTrans 的讨论略显重复 |
| 实用性 | 9/10 | 已在字节跳动抖音搜索全量部署，服务数十亿用户，实用价值较高 |

### 5.2 重点关注

#### 值得关注的技术点
- Global Token 作为序列建模与特征交互的共享语义接口的设计思想
- 交替式 Query Decoding + Query Boosting 实现双向信息流
- 多序列独立建模 + query 级别交互的策略（优于简单 merge）
- Scaling 分析表明 HyFormer 具有更陡峭的性能曲线

#### 需要深入理解的部分
- Query Generation 中 MeanPool(Seq) 的具体实现细节（是否包含位置编码？）
- MLP-Mixer 中 channel subspace 划分的具体方式（$D/T$ 的维度分配）
- 在线部署中 GPU Pooling 和异步 AllReduce 的工程细节

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[LONGER|LONGER: Scaling Long-Sequence Modeling]] - HyFormer 的序列编码基础，提供 KV-Cache 和 M-Falcon 机制
- [[RankMixer|RankMixer: Scaling Ranking Models]] - HyFormer 的特征交互基础，提供 MLP-Mixer 式 token mixing
- [[MTGR|MTGR: Multi-Token Generative Retrieval]] - 统一架构的对比方法，将所有特征 token 化后用共享 Transformer 编码
- [[OneTrans|OneTrans: Unified Feature Interaction]] - MTGR 的简化版本，使用金字塔压缩结构

### 6.2 背景相关
- [[SIM|SIM: Search-based User Interest Modeling]] - 长序列建模的早期工作
- [[ETA|ETA: Efficient Target Attention]] - 高效目标注意力机制
- [[TWIN|TWIN: Two-Stage Interest Network]] - 两阶段兴趣网络
- [[Wukong|Wukong: Large-Scale Feature Interaction]] - FM 式压缩交互块
- [[HSTU|HSTU: Actions Speak Louder Than Words]] - 统一推荐范式的代表

### 6.3 后续工作
- 暂无已知的直接后续工作

## 外部资源
- [arXiv 论文页面](https://arxiv.org/abs/2601.12681)
- [知乎解读：字节 HyFormer](https://zhuanlan.zhihu.com/p/1997753300553048726)

> [!tip] 关键启示
> 序列建模与特征交互不应是「先后」关系，而应是「交替协同」关系。通过 Global Token 作为共享语义接口，可以在不增加计算开销的前提下实现双向信息流，这一设计思想对搜推系统的模型架构升级具有参考价值。

> [!warning] 注意事项
> - 论文仅在字节内部数据集上验证，缺少公开 benchmark 的对比，泛化性需要进一步验证
> - 在线 A/B 测试的增益中约 1/3 来自序列长度扩展（1k→3k），需区分架构本身的贡献
> - MLP-Mixer 式的 token mixing 在 token 数量较少时（如本文的 16 tokens）效果较好，但 token 数量增大时的表现未知

> [!success] 推荐指数
> ⭐⭐⭐⭐ 对于从事搜推精排方向的研究者，这篇论文提供了一个实用的统一建模框架参考。特别是其「交替优化」的设计思想和多序列独立建模策略，对当前主流的两阶段架构有较好的改进启发。