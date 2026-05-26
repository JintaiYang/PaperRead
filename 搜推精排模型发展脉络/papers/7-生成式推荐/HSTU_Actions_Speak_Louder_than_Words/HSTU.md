---
paper_id: "[arXiv:2402.17152](https://arxiv.org/abs/2402.17152)"
title: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
authors: "Jiaqi Zhai, Lucy Liao, Xing Liu, Yanbin Gu, Boming Yin, Cathy Wu, et al."
institution: "Meta"
pushlication: "ICML 2024 (arXiv 2024-02-26)"
tags:
  - 精排论文
  - HSTU
  - 生成式推荐
  - Generative-Recommenders
  - Scaling-Law
  - 序列建模
  - 万亿参数
  - Pointwise-Attention
  - M-FALCON
  - 统一特征空间
quality_score: "9.8/10"
link:
  - "[PDF](https://arxiv.org/pdf/2402.17152)"
  - "[arXiv](https://arxiv.org/abs/2402.17152)"
  - "[GitHub](https://github.com/meta-recsys/generative-recommenders)"
date: "2024-02-26"
---

## 一、研究背景与动机

### 1.1 领域现状

2024 年，NLP 和 CV 领域已经充分验证了 Transformer + Scaling Law 的范式威力——GPT-3/LLaMa-2 等模型证明了"统一序列建模 + 持续增加计算量"可以带来可预测的性能提升。然而推荐系统领域仍然停留在 Deep Learning Recommendation Models（DLRMs）的范式中：依赖数千到数万个人工设计的异构特征（稀疏类别特征、稠密数值特征、交叉特征等），通过各种专门的模块（FM、DCN、MoE 等）进行特征交互和表示变换。

尽管 DLRMs 训练在海量数据上（Meta 每天处理数百亿用户行为），但它们无法随计算量增加而持续提升性能——这一现象在业界已被广泛观察到，却一直缺乏根本性的解决方案。

![[intro_pfsdays_dl_models_v3.png]]

> 图1：深度学习模型训练所用的总计算量随时间的变化。DLRM 的计算量在过去几年增长缓慢，而本文提出的 GRs 已经达到了 GPT-3/LLaMa-2 级别的计算规模（以年化 PetaFLOPs-days 计）。

### 1.2 现有方法的局限性

**DLRMs 的不可扩展性**：传统 DLRM 的核心瓶颈在于其异构特征范式。数千个特征由不同的工程管道生产和维护，每增加一个新特征都需要额外的工程投入。更关键的是，这种范式天然不具备 Scaling Law——增加模型参数并不能自动带来性能提升，因为模型的表达能力受限于人工设计的特征空间。

**序列化推荐的三大挑战**：将推荐问题转化为序列建模面临三个独特挑战：(1) 推荐系统的"词汇表"包含数十亿 Item 且持续动态变化（非平稳），远超 NLP 中 ~100K 的静态词表；(2) 最大互联网平台每天产生的用户行为 token 比 GPT-3 训练 1-2 个月用到的 token 还多几个数量级；(3) 用户序列长度可达 $10^5$，且需要 target-aware 的交互建模。

**标准 Transformer 的不适配**：直接将 Transformer 应用于推荐场景存在严重问题——softmax attention 在非平稳流式数据上表现不佳（无法捕获用户兴趣强度），训练不稳定（频繁 loss explosion），且 $O(N^2)$ 的计算复杂度在超长序列下不可接受。

### 1.3 本文解决方案概述

本文提出 **Generative Recommenders (GRs)**——一种全新的推荐范式，将排序和检索任务统一重新定义为序列转导任务（Sequential Transduction Tasks），并在生成式框架下训练。核心贡献包括：

1. **统一特征空间**：将 DLRMs 中数千个异构特征压缩为单一的时间序列表示，使推荐问题可以被纯粹的序列模型处理
2. **HSTU 架构**：专为推荐数据设计的高性能编码器，比 FlashAttention2-based Transformer 快 5.3x-15.2x
3. **M-FALCON 推理算法**：通过微批次化和 KV 缓存实现 285x 更复杂模型的部署，同时推理吞吐量反而提升 1.5x-3x
4. **首次验证推荐系统的 Scaling Law**：模型质量随训练计算量呈幂律增长，跨越三个数量级

## 二、解决方案

### 2.1 核心思想

GRs 的核心洞察是：**用户行为本身就是最有价值的"特征"**（Actions Speak Louder than Words）。传统 DLRMs 花费大量工程精力将用户行为加工成各种统计特征（如 CTR、加权衰减计数器等），但这个过程不可避免地丢失了信息。如果我们有一个足够强大的序列模型，直接建模原始行为序列就能隐式地捕获所有这些统计信息——而且随着序列长度和模型容量的增加，捕获的信息会越来越完整。

这一思想的数学基础是：当序列长度趋向无穷时，GRs 的特征空间可以逼近完整的 DLRM 特征空间。

### 2.2 整体架构

![[design_dlrms_vs_grs_features_training_v2.png]]

> 图2：DLRMs vs GRs 的特征和训练流程对比。左侧是传统 DLRM 的异构特征 + impression-level 训练；右侧是 GRs 的统一序列化特征 + 生成式训练。$\Phi_i$ 表示用户交互的第 $i$ 个 Item，$\Psi_k(t_j)$ 表示在时间 $t_j$ 产生的第 $k$ 个训练样本。

#### 模块1：统一的序列化特征空间

GRs 将 DLRMs 中的异构特征统一编码为单一时间序列：

**稀疏类别特征的序列化**：选择最长的时间序列（通常是用户交互的 Item 序列）作为主时间序列。其他缓慢变化的类别特征（如人口统计、关注的创作者等）通过"保留每个连续段的最早条目"进行压缩，然后合并到主时间序列中。由于这些特征变化很慢，合并后不会显著增加序列长度。

**稠密数值特征的隐式捕获**：传统 DLRMs 中的数值特征（如加权衰减计数器、比率等）本质上是对类别特征的聚合统计。在 GRs 中，这些类别特征已经被序列化编码，因此一个足够表达力的序列模型配合 target-aware 建模就能隐式捕获这些数值特征——无需显式计算。

#### 模块2：排序和检索的序列转导定义

给定按时间排序的 $n$ 个 token $x_0, x_1, \ldots, x_{n-1}$（$x_i \in \mathbb{X}$）及其观察时间 $t_0, t_1, \ldots, t_{n-1}$，序列转导任务将输入序列映射到输出 token $y_0, y_1, \ldots, y_{n-1}$（$y_i \in \mathbb{X} \cup \{\varnothing\}$），其中 $y_i = \varnothing$ 表示该位置无定义。

**检索任务**：学习分布 $p(\Phi_{i+1}|u_i)$，其中 $u_i$ 是用户在 token $i$ 处的表示。输入为 $(\Phi_0, a_0), (\Phi_1, a_1), \ldots$，输出为下一个正向交互的 Item。

**排序任务**：通过交错排列 Item 和 Action——$\Phi_0, a_0, \Phi_1, a_1, \ldots, \Phi_{n_c-1}, a_{n_c-1}$——使得排序可以被建模为 $p(a_{i+1} | \Phi_0, a_0, \ldots, \Phi_{i+1})$。这种交错设计使 target item $\Phi_{i+1}$ 能够在编码器内部与历史特征进行早期交互（target-aware cross-attention），而非像标准自回归设置那样在 softmax 之后才交互。

#### 模块3：生成式训练

传统 DLRMs 采用 impression-level 训练：每个 (user, item, label) 三元组作为一个独立样本。这导致编码器对同一用户的历史序列被重复计算 $n_i$ 次（$n_i$ 为用户交互数），总计算量为 $O(N^3d + N^2d^2)$。

GRs 采用生成式训练：在一次前向传播中同时预测序列中所有位置的输出，将编码器成本分摊到多个 target 上。通过以 $s_u(n_i) = 1/n_i$ 的速率采样用户，总计算量降至 $O(N^2d + Nd^2)$——减少了一个 $O(N)$ 因子。

#### 模块4：HSTU 编码器

![[design_dlrms_vs_grs_model_v2.png]]

> 图3：DLRMs vs GRs 的模型组件对比。左侧是完整的 DLRM 设置（包含 Feature Extraction、Feature Interaction、Transformation 三个阶段的多种异构模块）；右侧是简化的 HSTU（单一模块化 block 通过残差连接堆叠）。

HSTU 由相同的层通过残差连接堆叠而成，每层包含三个子层：

**Pointwise Projection（逐点投影）**：

$$U(X), V(X), Q(X), K(X) = \text{Split}(\phi_1(f_1(X)))$$

其中 $f_1(X) = W_1 X + b_1$ 是单层线性变换，$\phi_1 = \text{SiLU}$。将 Q、K、V 和门控权重 U 的计算融合为单个算子。

**Spatial Aggregation（空间聚合）**：

$$A(X)V(X) = \phi_2\left(Q(X)K(X)^T + \text{rab}^{p,t}\right)V(X)$$

这里 $\phi_2 = \text{SiLU}$ 是**逐点激活**（而非 softmax），$\text{rab}^{p,t}$ 是融合了位置和时间信息的相对注意力偏置。

**Pointwise Transformation（逐点变换）**：

$$Y(X) = f_2\left(\text{Norm}\left(A(X)V(X)\right) \odot U(X)\right)$$

其中 $\odot$ 是逐元素乘法，$\text{Norm}$ 是 Layer Norm。$\text{Norm}(A(X)V(X)) \odot U(X)$ 可以理解为 SwiGLU 的变体，同时实现了特征交互（类似 FM 的点积）和条件计算（类似 MoE 的门控）。

**HSTU 统一了 DLRMs 的三个阶段**：
- Feature Extraction → attention pooling（$A(X)V(X)$ 实现 target-aware pooling）
- Feature Interaction → 逐元素门控（$\text{Norm}(AV) \odot U$ 实现显式特征交互）
- Transformation → 门控操作（$U(X)$ 的 SiLU 激活实现条件计算/路由）

#### 模块5：Pointwise Aggregated Attention

HSTU 采用逐点聚合注意力（而非 softmax 注意力），动机有二：

1. **捕获兴趣强度**：在推荐中，与 target 相关的历史数据点数量本身就是用户偏好强度的强信号。softmax 归一化会抹去这一信息——无论用户看过 1 个还是 100 个美食视频，归一化后的权重和都是 1。逐点激活保留了绝对强度信息。

2. **适应非平稳词汇表**：softmax 对噪声具有鲁棒性，但在流式设置中面对动态变化的词汇表时表现不佳。论文通过 Dirichlet Process 合成数据验证了这一点——逐点注意力比 softmax 注意力高出 44.7%（HR@50: .3170 vs .2025）。

Layer Norm 在逐点 pooling 之后是必需的，用于稳定训练。

#### 模块6：Stochastic Length (SL)

推荐系统中用户历史序列的长度分布高度偏斜，且用户行为具有时间重复性——用户兴趣在不同时间尺度上反复出现。SL 利用这一特性人为增加稀疏性：

$$\begin{cases} (x_i)_{i=0}^{n_{c,j}} & \text{if } n_{c,j} \leq N_c^{\alpha/2} \\ (x_{i_k})_{k=0}^{N_c^{\alpha/2}} & \text{if } n_{c,j} > N_c^{\alpha/2}, \text{ w.p. } 1 - N_c^\alpha / n_{c,j}^2 \\ (x_i)_{i=0}^{n_{c,j}} & \text{if } n_{c,j} > N_c^{\alpha/2}, \text{ w.p. } N_c^\alpha / n_{c,j}^2 \end{cases}$$

这将 attention 相关复杂度从 $O(N^2d)$ 降至 $O(N^\alpha d)$，其中 $\alpha \in (1, 2]$。在 $\alpha=1.6$ 时，长度 4096 的序列大部分时间被压缩为长度 776（移除 80%+ 的 token），而模型质量几乎不受影响（NE 退化 < 0.002）。

![[exp_stochastic_length.png]]

> 图4：Stochastic Length 对模型指标的影响。左：$n=4096$；右：$n=8192$。即使在高稀疏率下，主要任务的 NE 退化也不超过 0.2%。

#### 模块7：M-FALCON 推理算法

![[m-falcon_inference.png]]

> 图5：M-FALCON 推理算法示意。通过修改 attention mask 和 rab 偏置，在单次前向传播中并行处理 $b_m$ 个候选 Item，将 cross-attention 成本从 $O(b_m n^2 d)$ 降至 $O((n+b_m)^2 d) \approx O(n^2 d)$。

M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）解决了推荐系统推理时需要处理数万候选 Item 的挑战：

1. **批内并行**：修改 attention mask 使 $b_m$ 个候选共享相同的 attention 计算
2. **跨批 KV 缓存**：将 $m$ 个候选分为 $\lceil m/b_m \rceil$ 个微批次，跨批次复用编码器级 KV 缓存
3. **$U(X)$ 缓存**：HSTU 的设计允许部分 dense 计算（$U(X)$）被缓存复用

最终效果：GR 模型比 DLRM 复杂 285x，但推理吞吐量反而提升 1.5x-3x。

#### 模块8：激活内存优化

推荐系统使用大 batch size（对训练吞吐和模型质量都至关重要），因此激活内存（而非参数内存）是主要的 scaling 瓶颈。HSTU 通过以下设计将每层激活内存从 Transformer 的 $33d$ 降至 $14d$：

- 将 attention 外的线性层从 6 个减少到 2 个（通过逐元素门控替代 MLP）
- 将 $\phi_1(f_1(\cdot))$ 融合为单个算子
- 将 Layer Norm + Dropout + Output MLP 融合为单个算子
- 消除了 $d_{ff} = 4d$ 的 FFN 层

这使得 HSTU 可以构建比 Transformer 深 2x 以上的网络，而不增加内存使用。

### 2.3 关键设计决策总结

| 设计选择 | HSTU/GRs | 标准 Transformer/DLRMs | 优势 |
|----------|----------|------------------------|------|
| 特征空间 | 统一时间序列 | 数千异构特征 | 消除特征工程，支持 Scaling |
| 训练方式 | 生成式（一次前向多 target） | Impression-level | 计算量减少 $O(N)$ 倍 |
| Attention | Pointwise SiLU | Softmax | 捕获强度信息，适应非平稳数据 |
| FFN | 无（融合为门控） | $d_{ff}=4d$ | 激活内存减半 |
| 稀疏性 | Raggified + Stochastic Length | Dense padding | 训练加速 5-15x |
| 推理 | M-FALCON 微批次 | 逐候选计算 | 285x 复杂度，1.5-3x 吞吐 |

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| Dirichlet Process 合成数据 | 动态词汇表 | 验证 pointwise attention |
| MovieLens-1M | 100万交互 | 公开基准（传统序列推荐设置） |
| MovieLens-20M | 2000万交互 | 公开基准 |
| Amazon Books | -- | 公开基准 |
| Meta 内部数据 | 100B+ 样本 | 工业级流式设置 |
| Meta 在线平台 | 数十亿 DAU | 在线 A/B 测试 |

### 3.2 实验设置

#### 3.2.1 基线方法

- SASRec（2023 最佳配置）：标准 Transformer 序列推荐
- Transformer（FlashAttention2）：工业级 Transformer 实现
- Transformer++（LLaMa-style）：RoPE + SwiGLU + RMSNorm
- DLRM（Meta 生产系统）：数百人多年迭代的成熟系统
- DLRM + DIN + DCN：增强版 DLRM
- DLRM（消融特征）：使用与 GR 相同的特征子集

#### 3.2.2 评估指标

- **HR@K（Hit Rate）**：检索任务的主要离线指标
- **NDCG@K**：公开数据集上的排序质量
- **NE（Normalized Entropy）**：排序任务的主要离线指标，0.001 的降低通常对应 0.5% 的在线指标提升
- **Log Perplexity**：检索任务的损失指标
- **在线 E-Task / C-Task**：主要参与事件和主要消费事件的在线指标

### 3.3 实验结果与分析

#### 公开数据集结果（传统序列推荐设置）

| 数据集 | 方法 | HR@10 | HR@50 | HR@200 | NDCG@10 | NDCG@200 |
|--------|------|-------|-------|--------|---------|----------|
| ML-1M | SASRec (2023) | .2853 | .5474 | .7528 | .1603 | .2498 |
| ML-1M | HSTU | .3097 (+8.6%) | .5754 (+5.1%) | .7716 (+2.5%) | .1720 (+7.3%) | .2606 (+4.3%) |
| ML-1M | **HSTU-large** | **.3294 (+15.5%)** | **.5935 (+8.4%)** | **.7839 (+4.1%)** | **.1893 (+18.1%)** | **.2771 (+10.9%)** |
| ML-20M | SASRec (2023) | .2906 | .5499 | .7655 | .1621 | .2521 |
| ML-20M | HSTU | .3252 (+11.9%) | .5885 (+7.0%) | .7943 (+3.8%) | .1878 (+15.9%) | .2774 (+10.0%) |
| ML-20M | **HSTU-large** | **.3567 (+22.8%)** | **.6149 (+11.8%)** | **.8076 (+5.5%)** | **.2106 (+30.0%)** | **.2971 (+17.9%)** |
| Books | SASRec (2023) | .0292 | .0729 | .1400 | .0156 | .0350 |
| Books | HSTU | .0404 (+38.4%) | .0943 (+29.5%) | .1710 (+22.1%) | .0219 (+40.6%) | .0450 (+28.6%) |
| Books | **HSTU-large** | **.0469 (+60.6%)** | **.1066 (+46.2%)** | **.1876 (+33.9%)** | **.0257 (+65.8%)** | **.0508 (+45.1%)** |

关键观察：(1) 相同配置下 HSTU 已显著优于 SASRec，证明了架构设计的有效性；(2) HSTU-large（4x 层数 + 2x heads）进一步大幅提升，最高达 +65.8% NDCG@10（Books），证明了 HSTU 的可扩展性。

#### 工业级流式设置结果

| 架构 | Retrieval (log pplx.) | Ranking E-Task (NE) | Ranking C-Task (NE) |
|------|----------------------|---------------------|---------------------|
| Transformers | 4.069 | NaN (训练崩溃) | NaN |
| HSTU (-rab, Softmax) | 4.024 | .5067 | .7931 |
| HSTU (-rab) | 4.021 | .4980 | .7860 |
| Transformer++ | 4.015 | .4945 | .7822 |
| HSTU (original rab) | 4.029 | .4941 | .7817 |
| **HSTU** | **3.978** | **.4937** | **.7805** |

关键发现：标准 Transformer 在排序任务中频繁 loss explosion 无法训练；HSTU 在所有设置中均为最优，且训练稳定性远超 softmax-based 方法。

#### GRs vs DLRMs 端到端对比

**检索任务**：

| 方法 | HR@100 | HR@500 | 在线 E-Task | 在线 C-Task |
|------|--------|--------|-------------|-------------|
| DLRM | 29.0% | 55.5% | +0% | +0% |
| DLRM (消融特征) | 28.3% | 54.3% | -- | -- |
| GR (content-based) | 11.6% | 18.8% | -- | -- |
| GR (interactions only) | 35.6% | 61.7% | -- | -- |
| **GR (new source)** | **36.9%** | **62.4%** | **+6.2%** | **+5.0%** |
| **GR (replace source)** | -- | -- | **+5.1%** | **+1.9%** |

**排序任务**：

| 方法 | E-Task NE | C-Task NE | 在线 E-Task | 在线 C-Task |
|------|-----------|-----------|-------------|-------------|
| DLRM | .4982 | .7842 | +0% | +0% |
| DLRM (DIN+DCN) | .5053 | .7899 | -- | -- |
| DLRM (消融特征) | .5053 | .7925 | -- | -- |
| GR (interactions only) | .4851 | .7903 | -- | -- |
| **GR** | **.4845** | **.7645** | **+12.4%** | **+4.4%** |

核心结论：GR 在排序任务中带来了 **+12.4%** 的在线主要参与指标提升——这在 Meta 数十亿用户的规模上是极其显著的。GR 仅使用原始交互特征就超越了使用数千个人工特征的 DLRM，验证了"Actions Speak Louder than Words"的核心论点。

#### 编码器效率

![[exp_encoder_efficiency_fwd+bwd_at_sparsity_v4.png]]

> 图6：编码器级效率对比——HSTU vs FlashAttention2-based Transformer。上：训练 NE；下：训练加速比。HSTU 在 8192 序列长度下实现了高达 15.2x 的训练加速。

HSTU 在 H100 GPU 上的效率优势：
- 训练加速：最高 **15.2x**（8192 序列长度，高稀疏率）
- 推理加速：最高 **5.6x**
- 激活内存：减少 50%+，支持 2x 更深的网络

#### 推理吞吐量

![[exp_efficiency_throughput_scaling_v3.png]]

> 图7：推理吞吐量对比。尽管 GR 模型复杂度是 DLRM 的 285x，通过 M-FALCON 算法，在评估 1024/16384 个候选时分别实现了 1.50x/2.99x 的吞吐量提升。

#### Scaling Law 验证

![[exp_scaling_retrieval_pflops_days_v5_100.png]]

> 图8：检索任务的 Scaling Law（HR@100）。GRs 的性能随计算量呈幂律增长，跨越三个数量级；DLRMs 在一定计算量后饱和。

![[exp_scaling_ranking_pflops_days_v5.png]]

> 图9：排序任务的 Scaling Law（NE）。GRs 展现出清晰的幂律 Scaling 行为，而 DLRM 的各种扩展方式（Transformer、DHEN、DCN）均在一定规模后饱和。

Scaling Law 的关键发现：

1. **GRs 展现幂律 Scaling**：HR@100、HR@500（检索）和 NE（排序）均随训练计算量呈幂律增长，跨越三个数量级
2. **DLRMs 饱和**：无论使用 Transformer、DHEN 还是 DCN 来扩展 DLRM，性能都在约 200B 参数后饱和
3. **GRs 扩展到 1.5 万亿参数**：最大模型使用 8192 序列长度、1024 embedding 维度、24 层 HSTU
4. **计算量达到 GPT-3/LLaMa-2 级别**：年化计算量接近这些大语言模型的总训练计算量
5. **序列长度的关键作用**：与语言建模不同，推荐中序列长度对 Scaling 的贡献更大，需要与其他参数协同扩展

### 消融实验

| 消融变体 | 效果 |
|----------|------|
| 移除 Pointwise Attention（改用 Softmax） | 排序训练不稳定/崩溃，检索 loss 显著恶化 |
| 移除 $\text{rab}^{p,t}$（相对注意力偏置） | 检索 log pplx +0.043，排序 NE +0.004 |
| 移除时间信息（仅保留位置） | 排序 NE +0.002 |
| GR 仅用 content 特征 | HR@100 从 36.9% 降至 11.6%（协同过滤信号至关重要） |
| GR 仅用 interaction（无 contextual 特征） | 排序 C-Task NE 从 .7645 恶化至 .7903 |
| DLRM 消融特征（使用 GR 相同特征） | 性能显著下降，证明 GR 架构能隐式捕获手工特征 |

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中指出：GRs 的特征空间统一化为推荐、搜索和广告领域的**首个基础模型**铺平了道路——统一的特征空间使跨域迁移成为可能。全序列化的设置也使推荐可以在端到端的生成式框架中建模，更好地将用户长期目标归因到短期决策。

### 4.2 基于分析的未来方向

1. **方向1：推荐基础模型（Foundation Model for Recommendations）**
   - 动机：GRs 的统一特征空间使跨域/跨场景的预训练成为可能
   - 可能的方法：在多个推荐场景（信息流、搜索、广告、电商）上联合预训练 HSTU，然后 fine-tune 到具体场景
   - 预期成果：减少每个场景的独立训练成本，提升冷启动性能
   - 挑战：不同场景的 action 语义差异大，如何统一建模

2. **方向2：多模态 GRs**
   - 动机：当前 GRs 主要使用 ID-based 特征，content 特征（文本、图像）的利用有限
   - 可能的方法：将多模态 embedding 作为额外的 token 类型融入序列
   - 预期成果：更好的冷启动能力和跨域迁移
   - 挑战：多模态 token 的计算成本高，如何在不显著增加延迟的情况下融入

3. **方向3：长期用户建模**
   - 动机：当前最长序列为 8192，但用户的完整历史可能有 $10^5$ 级别
   - 可能的方法：层次化编码（先压缩远期历史，再与近期历史联合建模）
   - 预期成果：更好的长期兴趣捕获和兴趣演化建模
   - 挑战：超长序列的训练效率和推理延迟

### 4.3 改进建议

1. **改进1：显式 MoE 集成**
   - 当前问题：HSTU 的门控机制隐式实现了条件计算，但不如显式 MoE 灵活
   - 改进方案：在 HSTU 层中引入 Sparse MoE，不同用户群体路由到不同专家
   - 预期效果：在不增加推理成本的情况下增加模型容量

2. **改进2：与特征交互方法的结合**
   - 当前问题：GRs 完全放弃了显式特征交互，但 FAT/Wukong 等工作证明了 field-aware 交互的价值
   - 改进方案：在 HSTU 的 attention 中引入 field-aware 结构（如 FAT 的 Field-Decomposed Attention）
   - 预期效果：在低计算量区间更快达到好的性能（论文中 GRs 在低计算量区间不如 DLRMs）

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.8/10** - HSTU/GRs 是推荐系统领域近年来最重要的工作，没有之一。它不仅提出了一个全新的范式（从 DLRMs 到 GRs），还在 Meta 数十亿用户的规模上完整验证了这一范式的有效性，并首次证明了推荐系统的 Scaling Law。这是推荐系统的"GPT 时刻"。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 10/10 | 范式级创新——将推荐从"特征工程+浅层模型"转变为"统一序列+生成式建模"，HSTU 的 pointwise attention + 门控融合设计优雅且有效 |
| 技术质量 | 10/10 | 从理论动机到架构设计到系统优化（SL、M-FALCON、激活内存）形成完整技术栈，每个组件都有严谨的实验验证 |
| 实验充分性 | 9.5/10 | 合成数据+公开数据+100B 工业数据+在线 A/B 测试+Scaling Law 验证，极其全面；唯一不足是部分消融实验的细节在附录中 |
| 写作质量 | 9/10 | 论文结构清晰，动机阐述充分，但由于内容极其丰富（正文+68页附录），部分细节需要反复阅读才能完全理解 |
| 实用性 | 10/10 | 已在 Meta 多个产品线部署并取得 12.4% 的在线提升，开源了代码，对工业界有直接指导价值 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Pointwise Aggregated Attention 的设计动机——softmax 在非平稳流式数据上的根本缺陷
- 生成式训练将计算量减少 $O(N)$ 倍的数学推导——这是 GRs 能 Scale 的关键前提
- M-FALCON 的"285x 复杂度 + 1.5-3x 吞吐"——证明了推理效率不是 GRs 部署的瓶颈
- Stochastic Length 的"80% token 移除 + <0.2% 质量损失"——用户行为的时间冗余性是推荐数据的独特优势
- 激活内存从 $33d$ 降至 $14d$——使 2x 更深的网络成为可能

#### 5.2.2 需要深入理解的部分

- 为什么 GRs 在低计算量区间不如 DLRMs？因为手工特征在数据量有限时提供了有效的归纳偏置，而 GRs 需要足够的计算量才能从原始序列中学到等价信息
- Pointwise attention 的训练稳定性为什么优于 softmax？Layer Norm 的位置和 SiLU 激活的梯度特性是关键
- 1.5 万亿参数中稀疏参数（embedding）和 dense 参数的比例是多少？论文提到 10B 词汇表 + 512d embedding 就需要 60TB，说明绝大部分参数是稀疏的

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wukong_Towards_a_Scaling_Law_for_Large_Scale_Recommendation|Wukong]] - Meta 的堆叠 FM Scaling Law，侧重特征交互维度的 Scaling，与 HSTU 侧重序列建模的路线互补
- [[FAT_From_Scaling_to_Structured_Expressivity|FAT]] - 阿里的 Field-Aware Transformer，在特征交互维度建立了理论 Scaling Law
- [[RankMixer_Scaling_Up_Ranking_Models_in_Industrial_Recommenders|RankMixer]] - 字节跳动的硬件感知 Scaling，在 FLOPs 效率上做了深入优化

### 6.2 背景相关
- [[DCN_V2_Improved_Deep_and_Cross_Network|DCN V2]] - 显式特征交叉的代表，HSTU 通过门控机制隐式实现了类似功能
- SASRec - 标准 Transformer 序列推荐的先驱，HSTU 的直接对比基线
- DIN - Target-aware attention 的开创者，GRs 的 interleaving 设计继承了这一思想

### 6.3 后续工作
- [[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans]] - 字节跳动的统一 Transformer 精排，受 HSTU 启发将特征交互和序列建模统一
- [[TokenMixer_Large_Scaling_Up_Large_Ranking_Models_in_Industrial_Recommenders|TokenMixer-Large]] - 字节跳动的大规模排序模型 Scaling
- 推荐基础模型的后续探索

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2402.17152)
- [GitHub 开源代码](https://github.com/meta-recsys/generative-recommenders)
- [ICML 2024 Slides](https://icml.cc/media/icml-2024/Slides/32684.pdf)
- [知乎解读：生成式推荐 HSTU 和 Scaling 方法总结](https://zhuanlan.zhihu.com/p/1959680193020028618)

> [!tip] 关键启示
> 推荐系统的 Scaling 瓶颈不在于数据量（推荐数据远超 NLP），而在于范式——异构特征工程范式天然不支持 Scaling。HSTU/GRs 通过将一切统一为序列，让推荐系统第一次具备了像 LLM 一样"越大越好"的能力。核心公式：**统一序列化 + 生成式训练 + 高效编码器 = 推荐系统的 Scaling Law**。

> [!warning] 注意事项
> - GRs 在低计算量区间不如 DLRMs——如果你的场景计算资源有限，传统特征工程仍然有价值
> - 1.5 万亿参数中绝大部分是稀疏 embedding 参数，dense 参数规模远小于 GPT-3
> - 论文的在线实验在 Meta 的特定产品线上进行，其他场景（如电商、搜索广告）的效果需要独立验证
> - Pointwise attention 需要配合 Layer Norm 使用，否则训练不稳定

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是推荐系统领域的里程碑论文，标志着从"特征工程时代"向"生成式建模时代"的范式转变。无论你是做推荐系统的研究者还是工程师，这篇论文都是必读的——它不仅改变了我们对推荐系统的认知，还提供了完整的技术方案（架构+训练+推理+Scaling），且已开源代码。如果只能读一篇 2024 年的推荐系统论文，就是这篇。
