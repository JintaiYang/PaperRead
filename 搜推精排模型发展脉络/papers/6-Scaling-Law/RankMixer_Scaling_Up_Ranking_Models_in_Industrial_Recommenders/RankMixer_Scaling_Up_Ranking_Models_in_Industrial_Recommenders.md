---
paper_id: "[arXiv:2507.15551](https://arxiv.org/abs/2507.15551)"
title: "RankMixer: Scaling Up Ranking Models in Industrial Recommenders"
authors: "Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, et al."
institution: "ByteDance"
pushlication: "arXiv Preprint 2025-07-21 (v3: 2025-07-26)"
tags:
  - Scaling-Law
  - 推荐系统精排
  - Token-Mixing
  - Sparse-MoE
  - 硬件感知架构
  - 特征交叉
  - Per-token-FFN
quality_score: "9.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2507.15551)"
  - "[arXiv](https://arxiv.org/abs/2507.15551)"
date: "2025-07-21"
---

## 一、研究背景与动机

### 1.1 领域现状

推荐系统精排模型（DLRM）是信息分发的核心环节，其性能直接决定用户体验和商业价值。受 LLM 领域 Scaling Law 成功经验的启发，业界越来越关注如何将精排模型从千万级参数规模扩展到十亿级别。然而，推荐系统与 NLP/CV 任务存在本质差异：在线服务必须严格遵守延迟约束（通常 < 15ms）并支持极高的 QPS，这使得简单地堆叠参数并不可行。

前期的 Scaling Law 研究如 DHEN 和 Wukong 已经在创新结构上取得了一些进展，但这些方法仍然面临计算效率不足的瓶颈。

### 1.2 现有方法的局限性

论文指出了现有推荐系统精排模型在扩展性方面的两个核心困境：

1. **CPU 时代的架构遗产**：传统精排模型（如 DCN、AutoInt、DHEN）的特征交叉模块是为 CPU 设计的，其核心算子是 memory-bound 而非 compute-bound，导致在 GPU 上的 MFU（Model FLOPs Utilization）极低，通常只有个位数百分比（如 4.5%）。这意味着 GPU 的算力被严重浪费。

2. **参数量与计算量的强耦合**：在 CPU 时代的模型中，计算成本与参数量近似成正比，这使得 Scaling Law 提示的"增加参数→提升性能"路径在实际中难以实现。即使 Wukong 展现了较陡的参数 Scaling 曲线，其 FLOPs 增长速度更快，导致在 AUC vs FLOPs 维度上优势大幅缩水。

3. **Self-Attention 在推荐场景的不适配**：Self-Attention 通过 token 内积计算注意力权重，这在 NLP 中有效是因为所有 token 共享统一语义空间。但推荐系统的特征空间天然异构——用户 ID、物品 ID、统计特征等来自完全不同的语义域，包含数亿级别的 ID 空间，计算异构空间之间的内积相似度非常困难。

### 1.3 本文解决方案概述

RankMixer 提出一种硬件感知的精排模型架构，核心思路是通过两个可扩展组件实现参数量与计算量的解耦：Multi-head Token Mixing 用无参数算子实现跨 token 特征交叉，Per-token FFN 为每个 token 分配独立参数以建模不同特征子空间。进一步通过 Sparse MoE 扩展到十亿参数量级，结合 ReLU Routing 和 DTSI（Dense-Training Sparse-Inference）策略解决专家训练不充分和不平衡问题。最终在抖音推荐系统全量部署 1B 参数模型，MFU 从 4.5% 提升至 45%，参数量扩大 70 倍但推理延迟保持不变。

## 二、解决方案

### 2.1 核心思想

RankMixer 的核心设计哲学可以用一句话概括：**让模型设计与硬件特性对齐，使参数增长不再绑定计算增长**。

具体来说，RankMixer 保留了 Transformer 的高并行性框架，但做了两个关键替换：用无参数的 Multi-head Token Mixing 替代二次复杂度的 Self-Attention（解决跨特征信息交换问题），用独立参数的 Per-token FFN 替代共享参数的 FFN（解决异构特征子空间建模问题）。这两个设计使得模型的参数量可以大幅增长而 FLOPs 增长缓慢，同时所有计算都是大型矩阵乘法，天然适合 GPU 的 Tensor Core。

### 2.2 整体架构

RankMixer 的整体流程是：输入特征向量 $\mathbf{e}_{\text{input}}$ 经过语义分组和投影（Tokenization）生成 $T$ 个 feature token，然后通过 $L$ 层 RankMixer Block 迭代精炼，最后通过 mean pooling 输出用于多任务预测的表示。

每个 RankMixer Block 由两个子模块组成，通过残差连接和 LayerNorm 串联：

$$\mathbf{S}_{n-1} = \text{LN}(\text{TokenMixing}(\mathbf{X}_{n-1}) + \mathbf{X}_{n-1})$$

$$\mathbf{X}_{n} = \text{LN}(\text{PFFN}(\mathbf{S}_{n-1}) + \mathbf{S}_{n-1})$$

其中 $\mathbf{X}_{n} \in \mathbb{R}^{T \times D}$ 是第 $n$ 层 RankMixer Block 的输出，$D$ 是模型隐藏维度。

![[RankMixer_arch_v3.pdf|800]]

> 图1：RankMixer Block 架构。每个 Block 由 Multi-head Token Mixing 和 SMoE-based Per-token FFN 两个模块组成。Token Mixing 将每个 token 的 embedding 分为 $H$ 个 head，然后跨 token 重新组合这些 head，实现不同特征之间的信息交互。

#### 各模块详细说明

**模块1：Feature Tokenization（语义分组 Tokenizer）**
- **功能**：将异构特征 embedding 转化为维度对齐的 feature token
- **输入**：用户特征、候选特征、序列特征、交叉特征等原始 embedding，维度各不相同
- **输出**：$T$ 个维度为 $D$ 的 feature token $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$
- **处理流程**：
  1. 基于领域知识将特征按语义相似性分组（如用户画像特征一组、item 内容特征一组）
  2. 组内特征 embedding 拼接成一个向量 $e_{\text{input}} = [e_1; e_2; \dots; e_N]$
  3. 按固定维度 $d$ 切分，再通过 Proj 函数映射到统一维度 $D$
- **关键公式**：
  $$\mathbf{x}_i = \text{Proj}(e_{\text{input}}[d \cdot (i-1) : d \cdot i]), \quad i = 1, \dots, T$$
- **设计动机**：如果每个特征一个 token（如 AutoInt），则 token 数量过多（数百个），每个 token 的参数和计算量被碎片化，GPU 核心利用率极低。反之，如果只有一个 token，则退化为简单 DNN，无法区分不同特征空间。语义分组在两者之间取得平衡。

**模块2：Multi-head Token Mixing（多头 Token 混合）**
- **功能**：实现跨 token 的全局信息交换，是 Self-Attention 的替代方案
- **输入**：$T$ 个 token $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T$，每个维度为 $D$
- **输出**：$H$ 个混合后的 token $\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_H$
- **处理流程**：
  1. 将每个 token 均分为 $H$ 个 head：$[\mathbf{x}_t^{(1)} \| \mathbf{x}_t^{(2)} \| \dots \| \mathbf{x}_t^{(H)}] = \text{SplitHead}(\mathbf{x}_t)$
  2. 将不同 token 的相同 head 拼接形成新的混合 token：$\mathbf{s}^{h} = \text{Concat}(\mathbf{x}_1^{h}, \mathbf{x}_2^{h}, \dots, \mathbf{x}_T^{h})$
  3. 设 $H = T$ 以保持 token 数量不变，支持残差连接
- **关键技术**：这是一个**完全无参数**的操作——不需要学习任何权重矩阵，只是对 embedding 进行 reshape 和 transpose。这使得它的 FLOPs 为零（仅有内存移动），但能实现不同特征空间之间的信息融合。
- **优于 Self-Attention 的原因**：推荐系统的特征空间天然异构，用户 ID 和物品 ID 分属包含数亿元素的不同 ID 空间，计算它们之间的内积相似度既困难又低效。Token Mixing 避免了这个问题，同时消除了 Attention 权重矩阵的 $O(T^2)$ 内存和计算开销。

**模块3：Per-token FFN（逐 token 前馈网络）**
- **功能**：为每个 token 使用独立参数进行特征子空间内的深度建模
- **输入**：Token Mixing 后的 token $\mathbf{s}_t \in \mathbb{R}^D$
- **输出**：精炼后的 token $\mathbf{v}_t \in \mathbb{R}^D$
- **关键公式**：
  $$\mathbf{v}_t = f_{\text{pffn}}^{t,2}(\text{Gelu}(f_{\text{pffn}}^{t,1}(\mathbf{s}_t)))$$
  其中 $f_{\text{pffn}}^{t,i}(\mathbf{x}) = \mathbf{x}\mathbf{W}_{\text{pffn}}^{t,i} + \mathbf{b}_{\text{pffn}}^{t,i}$，$\mathbf{W}_{\text{pffn}}^{t,1} \in \mathbb{R}^{D \times kD}$，$\mathbf{W}_{\text{pffn}}^{t,2} \in \mathbb{R}^{kD \times D}$，$k$ 是 FFN 隐藏维度的倍数。
- **关键技术**：与 Transformer 的 shared FFN 不同，Per-token FFN 为每个 token 分配独立的权重矩阵。这使得参数量乘以 $T$ 倍但 FLOPs 不变（因为每个 token 的计算量相同，只是并行执行不同的矩阵乘法）。
- **与 MMoE 的区别**：MMoE 中所有 expert 看到的是相同的输入，而 Per-token FFN 中每个 FFN 处理的是不同的 token 输入。RankMixer 同时分割了输入和参数，有利于学习不同特征子空间的多样性。

**模块4：Sparse MoE 扩展**
- **功能**：将 Dense Per-token FFN 扩展为 Sparse MoE，进一步提升参数效率
- **输入**：Token Mixing 后的 token $\mathbf{s}_i$
- **输出**：经过稀疏专家加权的表示 $\mathbf{v}_i$
- **关键公式**：
  $$G_{i,j} = \text{ReLU}(h(\mathbf{s}_i)), \quad \mathbf{v}_i = \sum_{j=1}^{N_e} G_{i,j} \cdot e_{i,j}(\mathbf{s}_i)$$
  其中 $N_e$ 是每个 token 的专家数量，$h(\cdot)$ 是路由函数。稀疏性通过 $\ell_1$ 正则控制：
  $$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{reg}}, \quad \mathcal{L}_{\text{reg}} = \sum_{i=1}^{N_t}\sum_{j=1}^{N_e} G_{i,j}$$
- **两个关键创新**：
  1. **ReLU Routing**：用 ReLU 门控替代传统 Top-k + Softmax。ReLU 允许每个 token 动态激活不同数量的专家——高信息量 token 可以激活更多专家，低信息量 token 只需少量专家，更好地适配推荐数据的分布。
  2. **DTSI（Dense-Training Sparse-Inference）**：训练时使用两个路由器 $h_{\text{train}}$ 和 $h_{\text{infer}}$，$\mathcal{L}_{\text{reg}}$ 仅施加在 $h_{\text{infer}}$ 上。训练时所有专家都被充分更新（解决欠训练），推理时只激活稀疏子集（降低成本）。

### Scaling Up 方向

RankMixer 的参数量和计算量可以沿四个正交轴扩展：Token 数 $T$、模型宽度 $D$、层数 $L$ 和专家数 $E$。对于 Dense 版本：

$$\text{\#Param} \approx 2kLTD^2, \quad \text{FLOPs} \approx 4kLTD^2$$

实验发现与 LLM Scaling Law 类似的结论：模型质量主要与总参数量相关，不同扩展方向（$L$, $D$, $T$）在相同参数量下性能基本一致。从计算效率角度，增大宽度 $D$ 能产生更大的矩阵乘法形状，从而获得更高的 MFU。最终 100M 和 1B 的配置分别设为 $(D=768, T=16, L=2)$ 和 $(D=1536, T=32, L=2)$。

## 三、实验结果

### 3.1 数据集

实验在抖音推荐系统的生产数据上进行：

| 数据集 | 规模 | 特征数 | 用户/视频规模 | 训练周期 |
|--------|------|--------|--------------|---------|
| 抖音推荐在线日志 | 万亿级/天 | 300+ 特征 | 数十亿用户ID + 数亿视频ID | 2 周 |

特征类型包括：数值特征（各种统计量）、ID 特征（用户/视频ID等）、交叉特征和序列特征，全部转化为 embedding。

### 3.2 实验设置

训练在数百块 GPU 上进行，采用混合分布式训练框架：sparse 部分异步更新，dense 部分同步更新。Dense 部分使用 RMSProp 优化器（lr=0.01），sparse 部分使用 Adagrad 优化器，所有模型保持相同的超参数。

#### 3.2.1 基线方法

对比了以下 SOTA 基线：DLRM-MLP（vanilla MLP 基线）、DCNv2（显式特征交叉 SOTA）、RDCN（DCN 改进）、MoE（多专家扩展）、AutoInt（Attention 特征交叉）、HiFormer（异构注意力 + 低秩近似）、DHEN（混合特征交叉模块堆叠）、Wukong（FM + LCB 的 Scaling Law 研究）。

#### 3.3.2 评估指标

离线指标使用 Finish AUC/UAUC 和 Skip AUC/UAUC（视频完播/跳过预测），其中 AUC 提升 0.0001 即为显著。效率指标包括 Dense-Param、FLOPs/Batch 和 MFU。

### 3.3 实验结果与分析

#### 主实验（~100M 参数量对比）

| 方法 | Finish AUC | Finish UAUC | Skip AUC | Skip UAUC | Params | FLOPs/Batch |
|------|-----------|-------------|----------|-----------|--------|-------------|
| DLRM-MLP (base) | 0.8554 | 0.8270 | 0.8124 | 0.7294 | 8.7M | 52G |
| DLRM-MLP-100M | +0.15% | -- | +0.15% | -- | 95M | 185G |
| DCNv2 | +0.13% | +0.13% | +0.15% | +0.26% | 22M | 170G |
| RDCN | +0.09% | +0.12% | +0.10% | +0.22% | 22.6M | 172G |
| MoE | +0.09% | +0.12% | +0.08% | +0.21% | 47.6M | 158G |
| AutoInt | +0.10% | +0.14% | +0.12% | +0.23% | 19.2M | 307G |
| DHEN | +0.18% | +0.26% | +0.36% | +0.52% | 22M | 158G |
| HiFormer | +0.48% | -- | -- | -- | 116M | 326G |
| Wukong | +0.29% | +0.29% | +0.49% | +0.65% | 122M | 442G |
| **RankMixer-100M** | **+0.64%** | **+0.72%** | **+0.86%** | **+1.33%** | 107M | 233G |
| **RankMixer-1B** | **+0.95%** | **+1.22%** | **+1.25%** | **+1.82%** | **1.1B** | **2.1T** |

#### 结果分析

RankMixer-100M 在所有指标上大幅领先所有 SOTA 模型。以 Finish AUC 为例，RankMixer-100M 的 +0.64% 相比 Wukong 的 +0.29% 高出一倍多，而 FLOPs 仅为 Wukong 的约一半（233G vs 442G）。这说明 RankMixer 在性能-效率权衡上远优于现有方法。

值得注意的是，简单扩大 DLRM-MLP 到 100M 参数只带来 +0.15% 的 AUC 提升，验证了单纯堆参数不行，必须设计适配推荐数据特性的架构。DCNv2、RDCN 等经典交叉结构虽然参数量不大（~22M），但 FLOPs 已经很高（170G+），反映了参数-计算耦合的设计缺陷。

### Scaling Law 对比

![[scaling_law.png|800]]

> 图2：不同模型的 Scaling Law 曲线——Finish AUC gain vs Params/FLOPs（对数坐标）。RankMixer 在参数和 FLOPs 两个维度上都展现出最陡峭的 Scaling 曲线。

从图中可以观察到：RankMixer 在参数维度和 FLOPs 维度上均具有最陡峭的 Scaling Law。Wukong 虽然参数曲线较陡，但其 FLOPs 增长更快，导致在 AUC vs FLOPs 曲线上与 RankMixer 的差距被进一步拉大。HiFormer 依赖 feature-level token 分割和 Attention，效率略逊。DHEN 的 Scaling 表现不理想，反映了其交叉结构的有限扩展性。

### 消融实验

#### 组件消融（基于 RankMixer-100M）

| 设置 | ΔAUC |
|------|------|
| 移除 Skip Connections | -0.07% |
| 移除 Multi-head Token Mixing | -0.50% |
| 移除 Layer Normalization | -0.05% |
| Per-token FFN → Shared FFN | -0.31% |

Multi-head Token Mixing 的移除导致最大的性能下降（-0.50%），因为没有它每个 FFN 只能看到部分特征而无法进行全局交互。Per-token FFN 替换为 Shared FFN 也带来显著损失（-0.31%），说明独立参数对不同特征子空间的建模至关重要。

#### Token Routing 策略对比

| 路由策略 | ΔAUC | ΔParams | ΔFLOPs |
|---------|------|---------|--------|
| All-Concat-MLP | -0.18% | 0.0% | 0.0% |
| All-Share | -0.25% | 0.0% | 0.0% |
| Self-Attention | -0.03% | +16% | +71.8% |

All-Share（所有 token 共享相同输入给每个 FFN，类似 MoE）性能下降最严重（-0.25%），证明了特征子空间分割和独立建模的重要性。Self-Attention 虽然性能接近 Token Mixing（仅差 0.03%），但 FLOPs 暴增 71.8%，性价比极低。

### Sparse MoE 扩展性

![[MoE.png|800]]

> 图3：RankMixer 不同变体在递减稀疏激活比（1, 1/2, 1/4, 1/8）下的 AUC 表现。Dense-Training + ReLU-Routed SMoE 几乎保留了 1B Dense 模型的全部精度。

DTSI + ReLU Routing 组合在激活比降至 1/8 时仍能保持接近 Dense 模型的 AUC，实现 > 8× 的参数容量扩展和 50% 的推理吞吐提升。相比之下，Vanilla SMoE 随稀疏度增加性能单调下降，验证了论文指出的专家不平衡和欠训练问题。

### 专家激活分布

![[expert_balance.pdf|800]]

> 图4：不同 token 的专家激活比例分布。DTSI + ReLU Routing 使得激活比例根据 token 信息含量动态变化，高信息 token 激活更多专家，适配推荐数据的异构分布。

### 在线部署成本

| 指标 | OnlineBase-16M | RankMixer-1B | 变化 |
|------|---------------|-------------|------|
| #Param | 15.8M | 1.1B | ↑ **70×** |
| FLOPs | 107G | 2106G | ↑ 20.7× |
| FLOPs/Param (G/M) | 6.8 | 1.9 | ↓ 3.6× |
| MFU | 4.47% | **44.57%** | ↑ 10× |
| Hardware FLOPs | fp32 | fp16 | ↑ 2× |
| Latency | 14.5ms | **14.3ms** | 持平 |

参数增加 70 倍但延迟反而微降的关键公式：

$$\text{Latency} = \frac{\#\text{Param} \times \text{FLOPs/Param ratio}}{\text{MFU} \times \text{Theoretical Hardware FLOPs}}$$

三个杠杆共同抵消了参数增长：FLOPs/Param 比率降低 3.6×（架构设计使参数增长 70× 但 FLOPs 仅增长 20×），MFU 提升 10×（从 memory-bound 转为 compute-bound），fp16 量化带来 2× 理论算力提升。

### 在线 A/B 测试

#### Feed 推荐场景（抖音 + 抖音极速版，长期 8 个月观测）

| 用户群 | Active Day | Duration | Like | Finish | Comment |
|--------|-----------|----------|------|--------|---------|
| **Overall（抖音）** | +0.2908% | +1.0836% | +2.3852% | +1.9874% | +0.7886% |
| Low-active（抖音） | +1.7412% | +3.6434% | +8.1641% | +4.5393% | +2.9368% |
| Middle-active（抖音） | +0.7081% | +1.5269% | +2.5823% | +2.5062% | +1.2266% |
| High-active（抖音） | +0.1445% | +0.6259% | +1.828% | +1.4939% | +0.4151% |
| **Overall（极速版）** | +0.1968% | +0.9869% | +1.1318% | +2.0744% | +1.1338% |

#### 广告场景

| 指标 | ΔAUC | ADVV |
|------|------|------|
| 提升 | +0.73% | +3.90% |

在线结果非常亮眼：整体 Active Day 提升 0.29%，Duration 提升 1.08%。低活用户群体获得了最大的提升（Active Day +1.74%，Duration +3.64%），证明了模型的强泛化能力。广告场景的 ADVV 提升 3.90% 也验证了 RankMixer 作为统一 backbone 的跨场景通用性。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中虽然没有明确列出未来工作，但从 Sparse MoE 实验结果可以推断，作者团队正在探索将 RankMixer 从当前 1B 参数进一步扩展到 10B 级别。DTSI + ReLU Routing 的组合在稀疏度为 1/8 时仍能保持精度，为更大规模的部署提供了可行路径。

### 4.2 基于分析的未来方向

1. **方向1：10B+ 规模的 Sparse MoE RankMixer**
   - 动机：当前 1B Dense 模型的收益在 8 个月后仍未饱和，暗示更大参数量有进一步收益空间
   - 可能的方法：利用 DTSI + ReLU Routing，在保持 1B 激活参数的前提下将总参数扩展到 10B
   - 预期成果：在不增加推理成本的前提下获得额外的性能提升
   - 挑战：10B 级别的专家数量将更多，路由策略的稳定性和专家的充分训练将更具挑战

2. **方向2：跨模态统一 Token 空间**
   - 动机：当前 Tokenization 基于人工语义分组，未来可以纳入多模态特征（图像、文本）
   - 可能的方法：将视觉特征和文本特征作为额外 token 类型接入 RankMixer
   - 预期成果：统一处理多模态信号，减少独立特征处理模块
   - 挑战：不同模态的特征维度和分布差异更大，Tokenization 策略需要重新设计

3. **方向3：自适应 Tokenization**
   - 动机：当前 token 分组依赖领域知识，可能不是最优的
   - 可能的方法：学习式或数据驱动的特征分组策略
   - 预期成果：自动发现最优的特征组合方式
   - 挑战：搜索空间巨大，需要高效的 NAS 或优化方法

### 4.3 改进建议

1. **改进1：Token Mixing 引入轻量级可学习参数**
   - 当前问题：Multi-head Token Mixing 完全无参数，信息交换模式固定
   - 改进方案：加入少量可学习的 mixing 权重，在保持低 FLOPs 的同时增强表达力
   - 预期效果：进一步缩小与 Self-Attention 的 0.03% 差距，但保持效率优势

2. **改进2：动态 Token 数量**
   - 当前问题：Token 数量 $T$ 和 head 数 $H$ 是固定的
   - 改进方案：根据样本复杂度动态调整 token 数量
   - 预期效果：简单样本减少计算量，复杂样本获得更精细的建模

## 五、 我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - 一篇极具工程价值和学术深度的工业级论文，成功解决了推荐系统 Scaling Law 的核心瓶颈

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | Token Mixing + Per-token FFN 的组合巧妙地解决了异构特征交叉和 GPU 利用率的双重问题，MFU 从 4.5% 到 45% 的跃升具有范式意义 |
| 技术质量 | 9/10 | 从架构设计到工程优化形成完整闭环，Latency 分解公式清晰地展示了如何用 3 个杠杆抵消 70× 参数增长 |
| 实验充分性 | 9/10 | 万亿级生产数据集、8 个月长期 A/B 测试、覆盖推荐+广告两大场景、完整的消融实验和 Scaling Law 曲线 |
| 写作质量 | 8/10 | 整体结构清晰，核心思想表达到位，部分公式符号可以更统一 |
| 实用性 | 10/10 | 已在抖音全量部署，Active Day +0.29%、Duration +1.08% 都是极强的线上收益 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点
- Multi-head Token Mixing 的零参数设计：通过 reshape/transpose 实现跨 token 信息交换，FLOPs 为零
- Per-token FFN 的参数-计算解耦：参数量 ×T 但 FLOPs 不变
- DTSI + ReLU Routing 的专家训练策略：解决了 Sparse MoE 的欠训练和不平衡问题
- Latency 分解公式：清晰展示了如何在 70× 参数增长下保持延迟不变

#### 5.2.2 需要深入理解的部分
- Token Mixing 本质上是一种固定的排列变换，其表达力上限在理论上如何分析
- Per-token FFN 中不同 token 的 FFN 参数差异有多大？是否存在某些 token 的 FFN 可以共享
- ReLU Routing 的 $\lambda$ 调节如何在不同稀疏度下保持稳定

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans]] - 同期的统一 Transformer 精排架构（ByteDance/NTU，WWW 2026），采用混合参数化策略
- [[Wukong|Wukong]] - Meta 的 Scaling Law 研究，使用 FMB + LCB 堆叠，RankMixer 在 FLOPs 维度上大幅优于 Wukong
- [[HSTU|HSTU]] - Meta 的 Generative Recommender，侧重序列建模的 Scaling

### 6.2 背景相关
- [[DCN_V2|DCNv2]] - 经典的显式特征交叉模型，作为 RankMixer 的基线之一
- [[AutoInt|AutoInt]] - 基于 Self-Attention 的特征交叉，RankMixer 的 Token Mixing 对其进行了替代
- [[DeepFM|DeepFM]] - FM + DNN 的经典组合，代表了 CPU 时代的设计范式

### 6.3 后续工作
- 10B Sparse MoE RankMixer - 论文暗示的下一步扩展方向
- 跨模态 RankMixer - 将视觉/文本多模态特征接入统一 Token 空间

## 外部资源
- [arXiv HTML 版本（实验性）](https://arxiv.org/html/2507.15551v3)
- [知乎解读：一文搞懂 RankMixer](https://zhuanlan.zhihu.com/p/1975860443877748808)
- [知乎解读：字节 Scaling Law 方案 RankMixer](https://zhuanlan.zhihu.com/p/1934435946553664683)
- [知乎解读：推荐精排模型高 ROI 的 Scaling Law 结构](https://zhuanlan.zhihu.com/p/1986029801761436835)

> [!tip] 关键启示
> 推荐系统的 Scaling Law 不是简单地增加参数，而是需要从根本上重新设计架构以对齐现代 GPU 硬件——RankMixer 通过无参数 Token Mixing + 参数隔离 Per-token FFN 实现了参数量与计算量的解耦，用 3 个杠杆（FLOPs/Param 比率 ↓ 3.6×、MFU ↑ 10×、fp16 量化 ↑ 2×）成功抵消 70× 参数增长，在不增加推理成本的前提下实现了精排模型的百倍 Scaling。

> [!warning] 注意事项
> - Token Mixing 的信息交换模式是固定的（基于 reshape/transpose），当 token 语义分组不合理时可能限制性能
> - 论文的所有实验均在抖音推荐数据上完成，其他推荐场景（如电商、搜索）的泛化性有待验证
> - Sparse MoE 变体的训练稳定性在更大规模（10B+）下可能面临新的挑战

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是推荐系统 Scaling Law 领域的里程碑工作——首次在工业级推荐系统中实现 1B 参数全量部署且不增加推理成本，8 个月长期 A/B 测试验证了持续收益。与 OneTrans 一起代表了 2025 年推荐精排模型向 Transformer 统一架构演进的最前沿。
