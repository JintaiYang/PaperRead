---
paper_id: "[arXiv:2510.26104](https://arxiv.org/abs/2510.26104)"
title: "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
authors: "Zhaoqi Zhang, Haolei Pei, Jun Guo, et al."
institution: "ByteDance / Nanyang Technological University"
pushlication: "WWW 2026 2025-10-30"
tags:
  - 精排论文
  - OneTrans
  - 统一架构
  - Transformer
  - Scaling-Law
  - 特征交叉
  - 序列建模
  - KV-Cache
  - 推荐系统
quality_score: "9.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2510.26104)"
  - "[HTML](https://arxiv.org/html/2510.26104v3)"
date: "2025-10-30"
---

## 一、研究背景与动机

### 1.1 领域现状

工业推荐系统的精排阶段通常采用 DLRM（Deep Learning Recommendation Model）范式，核心由两个独立模块组成：**特征交叉模块**（Feature Interaction）负责学习用户画像、商品属性、上下文等非序列特征之间的高阶交互；**序列建模模块**（Sequence Modeling）负责从用户历史行为序列中提取时序兴趣表征。

近年来，这两个方向分别取得了显著进展。特征交叉方面，[[Wukong|Wukong]]（Meta, 2024）通过堆叠 FM-style 交叉层并建立 Scaling Law，[[RankMixer|RankMixer]]（快手, 2025）通过 token-mixing + sparse MoE 实现硬件友好的规模化。序列建模方面，LONGER（字节, 2025）将 Causal Transformer 应用于超长用户行为序列，验证了深度和宽度的单调增益。

### 1.2 现有方法的局限性

传统方法采用 **encode-then-interaction** 流水线：先将用户行为序列压缩为固定长度向量，然后与非序列特征拼接，再送入特征交叉模块。这种"先编码、后交互"的分离设计存在两个根本性问题：

- **双向信息流受阻**：序列编码阶段无法感知静态特征（如当前候选商品的属性），特征交叉阶段也只能看到压缩后的序列表征，而非原始行为细节。InterFormer（2024）尝试通过 summary-based 双向交叉架构来缓解，但仍保持两个独立模块，增加了架构复杂度
- **无法统一优化和扩展**：两个模块分别用不同的计算图和参数化策略，无法像 LLM 那样复用成熟的工程优化（FlashAttention、KV Cache、混合精度），也难以在一个统一框架内扩展 depth 和 width

### 1.3 本文解决方案概述

OneTrans 提出了一个统一的 Transformer 骨干网络，将序列建模和特征交叉合并到单一计算图中。其核心思路是：用一个 **Unified Tokenizer** 把序列特征和非序列特征都转换成 token 序列，然后用一个 **Mixed Parameterization Transformer** 联合处理，通过 causal attention 实现四类交互（序列内、跨序列、特征间、序列-特征间），同时利用 pyramid 策略和 KV Cache 保证效率。

## 二、解决方案

### 2.1 核心思想

OneTrans 的核心洞察很简单：既然 LLM 已经证明"把所有东西 token 化、用一个 Transformer 处理"是最具扩展性的范式，那推荐系统的精排模型为什么不能照做？

关键区别在于：LLM 的 token 来自同质的文本，而推荐系统的 token 来自异质的数据源——有些是时序行为序列（点击、加购、下单），有些是静态属性（用户画像、商品类目、价格）。OneTrans 通过 **混合参数化**（Mixed Parameterization）来处理这种异质性：同质的序列 token 共享一组参数，异质的非序列 token 各自拥有独立参数。

### 2.2 整体架构

![[onetrans_all.pdf|800]]

> 图1：OneTrans 系统架构。(a) 整体流程：序列特征（S，蓝色）和非序列特征（NS，橙色）分别 token 化后拼接为统一 token 序列，送入金字塔堆叠的 OneTrans Block 逐层压缩。(b) OneTrans Block：带 RMSNorm 的 causal pre-norm Transformer Block，包含 Mixed Causal Attention 和 Mixed FFN。(c) "Mixed"=混合参数化：S-token 共享一组 QKV/FFN 权重，每个 NS-token 拥有独立的 QKV/FFN。

整体流程为：

$$\hat{y}_{u,i} = f\left(i \,\big|\, \mathcal{NS}, \mathcal{S}; \Theta\right)$$

其中 $\mathcal{NS}$ 是非序列特征集合，$\mathcal{S}$ 是用户历史行为序列集合，$\Theta$ 是可训练参数。

#### 模块1：Unified Tokenizer

**功能**：将所有原始特征统一转化为维度为 $d$ 的 token 向量

**非序列特征 token 化**提供两种方案：

- **Group-wise Tokenizer**（对齐 RankMixer）：将特征手动分组为 $\{g_1, \dots, g_{L_{NS}}\}$，每组通过独立 MLP 映射

$$\text{NS-tokens} = \big[\text{MLP}_1(\text{concat}(g_1)), \dots, \text{MLP}_{L_{NS}}(\text{concat}(g_{L_{NS}}))\big]$$

- **Auto-Split Tokenizer**（默认推荐）：所有特征拼接后通过单个 MLP 投影，再 split 成 $L_{NS}$ 个 token

$$\text{NS-tokens} = \text{split}\Big(\text{MLP}(\text{concat}(\mathcal{NS})),\, L_{NS}\Big)$$

消融实验表明 Auto-Split 优于 Group-wise，说明让模型自动学习如何划分 token 比人工分组更有效，同时也减少了 kernel launch 开销。

**序列特征 token 化**：多行为序列 $\mathcal{S} = \{S_1, \dots, S_n\}$ 中每个事件通过 MLP 对齐到维度 $d$，再按两种策略合并：

- **Timestamp-aware**（优先推荐）：按时间戳交织所有行为事件
- **Timestamp-agnostic**：按用户意图强度排列（下单→加购→点击），序列间插入可学习的 `[SEP]` token

最终的统一 token 序列为：

$$\mathbf{X}^{(0)} = \big[\text{S-tokens};\, \text{NS-tokens}\big] \in \mathbb{R}^{(L_S + L_{NS}) \times d}$$

#### 模块2：OneTrans Block（Mixed Parameterization Transformer）

**功能**：联合执行序列建模和特征交叉

每个 OneTrans Block 是一个 pre-norm causal Transformer，包含两个子层：

$$\mathbf{Z}^{(n)} = \text{MixedMHA}\!\left(\text{Norm}\big(\mathbf{X}^{(n-1)}\big)\right) + \mathbf{X}^{(n-1)}$$

$$\mathbf{X}^{(n)} = \text{MixedFFN}\!\left(\text{Norm}\big(\mathbf{Z}^{(n)}\big)\right) + \mathbf{Z}^{(n)}$$

**Mixed Causal Attention** 的参数化策略：

$$\mathbf{W}^{\Psi}_i = \begin{cases} \mathbf{W}^{\Psi}_S, & i \le L_S \quad (\text{S-token 共享}) \\ \mathbf{W}^{\Psi}_{NS,i}, & i > L_S \quad (\text{NS-token 独立}) \end{cases}$$

其中 $\Psi \in \{Q, K, V\}$。这种设计的直觉是：S-token 来自同质的行为序列，语义相近可以共享参数；NS-token 来自完全不同的特征源（用户ID、商品类目、价格等），需要独立的投影来保持各自的语义区分度。

**Causal Mask** 产生四类交互：

- S-side：每个 S-token 只关注它之前的 S 位置（时序因果性）
- NS-side：每个 NS-token 关注全部 S 历史（target-attention 聚合序列证据）+ 之前的 NS-token（特征间交互）
- 跨序列：不同行为序列之间的信息传递
- 序列-特征：静态特征影响序列表征的双向信息流

**Mixed FFN** 遵循同样的参数化：

$$\text{MixedFFN}(\mathbf{x}_i) = \mathbf{W}^2_i \,\phi(\mathbf{W}^1_i \mathbf{x}_i)$$

#### 模块3：Pyramid Stack

**功能**：逐层压缩序列 token，节省计算开销

由于 causal masking 将信息向后面位置聚集，OneTrans 利用这种结构采用**金字塔调度**：每层只保留最近的一部分 S-token 发出 query，key/value 仍覆盖完整序列。

具体来说，OneTrans$_S$ 从 1190 个 S-query token 线性缩减到 12（匹配 NS-token 数量），OneTrans$_L$ 从 1500 缩减到 16。

两个好处：

- **渐进蒸馏**：长行为历史被逐层浓缩到少量 query 中，信息集中到 NS-token
- **计算效率**：attention 开销从 $O(L^2d)$ 降为 $O(LL'd)$，FFN 随 $L'$ 线性缩放

#### 模块4：Cross-Request KV Caching

同一请求中多个候选商品共享相同的 S-token。OneTrans 将推理分为两阶段：

- **Stage I**（每请求一次）：处理所有 S-token，缓存 key/value
- **Stage II**（每候选一次）：计算 NS-token，cross-attention 到缓存的 S-side KV

进一步，跨请求复用缓存：每次新请求只计算增量行为的 KV，复杂度从 $O(L)$ 降为 $O(\Delta L)$。

## 三、实验结果

### 3.1 数据集

| 指标 | 数值 |
|------|------|
| 曝光总量 | 29.1B |
| 独立用户数 | 27.9M |
| 独立商品数 | 10.2M |
| 日均曝光量 | 118.2M ± 14.3M |
| 日活用户 | 2.3M ± 0.3M |

数据来自字节跳动大规模工业排序场景的生产日志，按时间顺序切分，所有特征在曝光时刻快照以防止时间泄露。

### 3.2 实验设置

**模型配置**：

- **OneTrans$_S$**：6 层，$d=256$，$H=4$，约 91M 参数
- **OneTrans$_L$**：8 层，$d=384$，约 330M 参数

**优化器**：双优化器策略——稀疏 embedding 用 Adagrad，稠密参数用 RMSProp（lr=0.005）。训练在 16 张 H100 GPU 上用 data-parallel all-reduce，per-GPU batch size 2048。

**评估方式**：Next-batch evaluation（先记录预测，再训练），每天计算 AUC 和 UAUC，跨天宏平均。

#### 3.2.1 基线方法

在 encode-then-interaction 范式下，逐步增强两个模块：

- **特征交叉**：DCNv2 → Wukong → HiFormer → RankMixer
- **序列建模**：DIN → StackDIN → LONGER → Transformer

#### 3.3.2 评估指标

- **AUC**：全局曝光级别的 ROC-AUC
- **UAUC**：曝光加权的用户级 AUC
- **Params**：模型参数量（不含稀疏 embedding）
- **TFLOPs**：batch size 2048 下的训练计算量

### 3.3 实验结果与分析

![[compare_all.pdf|800]]

> 图2：架构对比。(a) 传统 encode-then-interaction 流水线。(b) OneTrans 在单一 Transformer 栈中联合建模序列和非序列特征。

| 类型 | 模型 | CTR AUC | CTR UAUC | CVR AUC | CVR UAUC | Params(M) | TFLOPs |
|------|------|---------|----------|---------|----------|-----------|--------|
| 基线 | DCNv2 + DIN | 0.79623 | 0.71927 | 0.90361 | 0.71955 | 10 | 0.06 |
| 特征交叉 | Wukong + DIN | +0.08% | +0.11% | +0.14% | +0.11% | 28 | 0.54 |
| 特征交叉 | HiFormer + DIN | +0.11% | +0.18% | +0.23% | -0.20% | 108 | 1.35 |
| 特征交叉 | RankMixer + DIN | +0.27% | +0.36% | +0.43% | +0.19% | 107 | 1.31 |
| 序列建模 | RankMixer + StackDIN | +0.40% | +0.37% | +0.63% | -1.28% | 108 | 1.43 |
| 序列建模 | RankMixer + LONGER | +0.49% | +0.59% | +0.47% | +0.44% | 109 | 1.87 |
| 序列建模 | RankMixer + Transformer | +0.57% | +0.90% | +0.52% | +0.75% | 109 | 2.51 |
| **统一框架** | **OneTrans$_S$** | **+1.13%** | **+1.77%** | **+0.90%** | **+1.66%** | **91** | **2.64** |
| **统一框架** | **OneTrans$_L$** | **+1.53%** | **+2.79%** | **+1.14%** | **+3.23%** | **330** | **8.62** |

> 注：在该系统中，+0.1% 以上被认为有意义，+0.3% 通常对应线上 A/B 的显著效果

#### 结果分析

OneTrans$_S$ 在与 RankMixer+Transformer 相近的 TFLOPs（2.64T vs 2.51T）下，CTR UAUC 从 +0.90% 提升到 +1.77%（几乎翻倍），证明统一建模的信息效率优势。OneTrans$_L$ 进一步将 CTR UAUC 提升到 +2.79%，CVR UAUC 达到 +3.23%，表现出随规模扩大的可预测性能增益。

### 消融实验

#### 实验设计

以 OneTrans$_S$ 为基准，分别消融输入设计和 Block 设计的关键选择：

| 类型 | 变体 | CTR AUC | CTR UAUC | CVR AUC | CVR UAUC |
|------|------|---------|----------|---------|----------|
| Input | Group-wise Tokenizer | -0.10% | -0.30% | -0.12% | -0.10% |
| Input | Timestamp-agnostic Fusion | -0.09% | -0.22% | -0.20% | -0.21% |
| Input | Agnostic w/o SEP Tokens | -0.13% | -0.32% | -0.29% | -0.33% |
| Block | Shared parameters | -0.15% | -0.29% | -0.14% | -0.29% |
| Block | Full attention | +0.00% | +0.01% | -0.03% | +0.06% |
| Block | w/o pyramid stack | -0.05% | +0.06% | -0.04% | -0.42% |

#### 消融结果和分析

- **Auto-Split > Group-wise**：自动学习 token 划分优于人工特征分组（CTR UAUC 差 0.30%），说明模型能学到比人类直觉更好的特征聚合方式
- **Timestamp-aware > Agnostic**：有时间戳时应优先按时间排序（差 0.22%），但无时间戳时 `[SEP]` token 帮助区分序列边界很重要（去掉后额外下降 0.10%）
- **Token-specific > Shared**：NS-token 用独立参数比全共享好 0.29% UAUC，验证了异质特征需要独立建模的假设
- **Causal ≈ Full attention**：性能几乎无差异，但 causal 可以用 KV Cache 优化推理
- **Pyramid 在效率上收益巨大**：去掉 pyramid，TFLOPs 从 2.64T 飙升到 8.08T（+206%），但性能几乎不变甚至略降，说明 OneTrans 有效地将信息浓缩到尾部

### 系统效率

| 优化 | 训练时间 | 训练内存 | 推理延迟(p99) | 推理内存 |
|------|----------|----------|---------------|----------|
| Pyramid stack | -28.7% | -42.6% | -8.4% | -6.9% |
| Cross-Request KV Cache | -30.2% | -58.4% | -29.6% | -52.9% |
| FlashAttention | -50.1% | -58.9% | -12.3% | -11.6% |
| Mixed Precision + Recomp | -32.9% | -49.0% | -69.1% | -30.0% |

最终 OneTrans$_L$（330M 参数）的推理 p99 延迟为 13.2ms，甚至**低于** DCNv2+DIN（10M 参数）的 13.6ms，MFU 从 13.4 提升到 30.8。

### Scaling Law 验证

![[tradeoff_scaling_singlecol.pdf|600]]

> 图3(a)：FLOPs vs ΔUAUC 的权衡。增加序列长度（length）带来最大收益，增加深度（depth）优于增加宽度（width）

![[scaling_law_singlecol_bw.pdf|600]]

> 图3(b)：Scaling Law——ΔUAUC vs FLOPs（对数尺度）。OneTrans 和 RankMixer 都呈 log-linear 趋势，但 OneTrans 斜率更陡

三个维度扩展的发现：**length > depth > width**。增加输入序列长度带来最大的 UAUC 提升；depth 扩展优于 width（更深的栈学到更高阶的交互和更丰富的抽象），但 width 更利于并行。OneTrans 的 scaling 斜率明显比 RankMixer 更陡，说明统一架构具有更好的参数效率和计算效率。

### 在线 A/B 实验

对照组为 RankMixer+Transformer（约 100M 参数，无序列 KV Cache），实验组为 OneTrans$_L$：

| 场景 | click/u | order/u | gmv/u | 延迟(p99) |
|------|---------|---------|-------|-----------|
| Feeds | +7.737%** | +4.351%* | +5.685%* | -3.91% |
| Mall | +5.143%** | +2.577%** | +3.670%* | -3.26% |

> *: p<0.05, **: p<0.01

此外，用户活跃天数增加 +0.75%，冷启动商品订单提升 +13.59%。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在 Scaling Law 部分提到，OneTrans$_L$ 可以在严格的线上 p99 延迟约束下部署，但进一步大幅扩展当前受限于线上效率，需要"system-model co-optimization"，这是留给未来工作的方向。

### 4.2 基于分析的未来方向

1. **方向1：MoE 化的 Mixed Parameterization**
   - 动机：当前 NS-token 数量固定时，token-specific 参数与 NS-token 数量线性相关，扩展受限
   - 可能的方法：引入 Sparse MoE 替代 token-specific FFN，每个 NS-token 通过路由选择专家
   - 预期成果：在参数更大时保持推理效率
   - 挑战：MoE 的 load balancing 和通信开销

2. **方向2：多域/跨场景统一**
   - 动机：论文已在 Feeds 和 Mall 验证，但不同业务场景的特征空间和行为模式差异大
   - 可能的方法：共享 S-side 参数，域特定的 NS-side token-specific 参数
   - 预期成果：单一模型服务多个场景，降低维护成本
   - 挑战：负迁移（negative transfer）和域间冲突

3. **方向3：与生成式推荐（GR）的融合**
   - 动机：论文在 Related Work 中提到 HSTU 等 GR 方法与 OneTrans 互补
   - 可能的方法：在 OneTrans 统一骨干上加入 generative head，直接生成推荐序列
   - 预期成果：兼具 DLRM 丰富特征利用 + GR 的端到端生成能力
   - 挑战：生成式解码的延迟控制

### 4.3 改进建议

1. **改进1：动态 Pyramid 调度**
   - 当前问题：pyramid 的 token 缩减策略是固定的线性调度
   - 改进方案：基于注意力权重或 token importance score 动态决定每层保留哪些 token
   - 预期效果：在相同 FLOPs 下保留更多信息量大的 token

2. **改进2：多粒度序列融合**
   - 当前问题：timestamp-aware 和 timestamp-agnostic 二选一
   - 改进方案：多粒度融合——长期行为用 agnostic（按意图），近期行为用 aware（按时间）
   - 预期效果：同时捕获长期偏好和近期兴趣

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - 这是一篇将"统一 Transformer 骨干"理念真正落地到工业推荐精排的里程碑论文，方法设计精巧、工程细节充实、实验说服力强。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次将序列建模和特征交叉完全统一到单一 Transformer，Mixed Parameterization 设计优雅 |
| 技术质量 | 9/10 | 方法论清晰严谨，每个设计选择都有消融验证，工程优化细节完整 |
| 实验充分性 | 9/10 | 29.1B 曝光的工业数据集，7+ 基线对比，6 项消融，效率分析，Scaling Law，在线 A/B |
| 写作质量 | 8/10 | 结构清晰，图表规范，但部分公式符号定义可以更早引入 |
| 实用性 | 10/10 | 已在字节跳动生产部署，GMV +5.68%，冷启动 +13.59%，延迟反而降低 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Mixed Parameterization 的思路非常通用——凡是存在"同质+异质" token 混合的场景都可以借鉴
- Pyramid Stack 的渐进蒸馏思想巧妙利用了 causal mask 的信息聚集特性
- Cross-Request KV Cache 从 $O(L)$ 到 $O(\Delta L)$ 的增量计算，对长序列场景价值巨大

#### 5.2.2 需要深入理解的部分

- Auto-Split Tokenizer 为什么比 Group-wise 好？模型是如何自动学习特征分组的？
- 金字塔调度的线性缩减策略是否最优？作者提到"heuristic"，有优化空间
- 330M 参数模型推理 p99 只有 13.2ms 的具体实现细节（half-precision、KV Cache 复用的工程栈）

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[RankMixer|RankMixer]] - OneTrans 的特征交叉基线，token-mixing + MoE 扩展方案
- [[Wukong|Wukong]] - Meta 提出的推荐 Scaling Law 先驱，FM-style 堆叠
- [[HSTU|HSTU]] - Meta 的生成式推荐万亿参数模型，互补的 GR 路线

### 6.2 背景相关
- [[DIN|DIN]] - 序列建模起点，target attention 机制
- [[DCN_V2|DCN V2]] - 特征交叉基线，Cross Network 改进版
- [[SIM|SIM]] - 长序列建模，candidate-specific 检索 + 精排

### 6.3 后续工作
- 统一骨干 + MoE 的扩展
- 多域跨场景的统一精排

## 外部资源

- [知乎解读：字节 OneTrans 用"一个 Transformer"统一精排](https://zhuanlan.zhihu.com/p/1969004009126880443)
- [Datawhale 教程：OneTrans 统一序列与特征交叉](https://datawhalechina.github.io/fun-rec/chapter_6_scaling/5.one_trans.html)
- [arXiv 论文页面](https://arxiv.org/abs/2510.26104)

> [!tip] 关键启示
> 推荐系统的序列建模和特征交叉不需要是两个独立模块——统一到单一 Transformer 骨干后，不仅能释放双向信息交换，还能直接复用 LLM 的全套系统优化（KV Cache、FlashAttention、混合精度），实现更高效的 Scaling。

> [!warning] 注意事项
> - 论文实验基于字节跳动电商场景，其他场景（如短视频、新闻推荐）的迁移效果未验证
> - Causal mask 限制了 NS-tokens 之间的交互多样性（只能看到前面的 NS-tokens）
> - 论文未开源代码，复现需要较强的工程能力

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是推荐系统精排架构的范式转变之作，首次成功将序列建模和特征交叉统一到单一 Transformer，并在工业级部署中验证了 Scaling Law，对搜推精排的未来发展具有重要指导意义。
