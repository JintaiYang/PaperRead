---
paper_id: "[arXiv:2006.05639](https://arxiv.org/abs/2006.05639)"
title: "Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction"
authors: "Pi Qi, Guorui Zhou, Yujing Zhang, et al."
institution: "阿里巴巴（阿里妈妈）"
pushlication: "CIKM 2020 2020-07-20"
tags:
  - 精排论文
  - SIM
  - 长序列建模
  - 检索范式
  - User Behavior Tree
  - 两阶段级联
  - CTR预估
  - 序列建模
quality_score: "9.2/10"
link:
  - "[PDF](https://arxiv.org/pdf/2006.05639)"
date: "2020-07-20"
---

## 一、研究背景与动机

### 1.1 领域现状

2020 年前后，阿里巴巴展示广告系统中用户行为序列建模已经经历了 DIN（target-aware attention）→ DIEN（GRU 序列演化）→ MIMN（memory network 压缩）的发展路线。核心矛盾在于：工业系统积累的用户行为数据越来越长（阿里妈妈的用户平均行为序列长度已达到数万甚至 54000+），但主流模型能处理的序列长度极其有限——DIEN 的 GRU 结构限制序列长度不超过 1000，MIMN 虽然通过 memory network 压缩打破了这一限制，但其固定容量的记忆单元不可避免地造成信息损失。

![[click_distribution_longterm.png|600]]

> 图4：DIEN 与 SIM 的点击样本分布对比。按"距离最近一次同类行为的天数"（$d_{category}$）分层统计，SIM 在远期行为（>14天）上的点击占比显著高于 DIEN，最高达 +32.1%。这说明长期历史行为中蕴含大量被短序列模型遗漏的有效信号。

### 1.2 现有方法的局限性

论文从**系统**和**算法**两个层面分析了现有方案的瓶颈：

**算法瓶颈**：DIEN 的 GRU 序列建模时间复杂度 $O(T)$，当 $T$ 增长到数万时，训练和推理的延迟都会线性增长到不可接受。MIMN 用固定大小的 memory 来压缩整条行为序列，memory 容量上限决定了信息保留的天花板——当序列长度从 1000 增加到 10000+ 时，大量长尾兴趣信息在压缩过程中丢失。

**系统瓶颈**：在阿里妈妈的 Real-Time Prediction（RTP）系统中，模型推理流程由 Prediction Server 和 Computation Node 两部分组成。Prediction Server 负责实时拼接特征和推理，Computation Node 负责离线计算用户行为特征。

![[RTP_system_original.png|600]]

> 图2：原始 RTP 系统架构。用户行为特征由 Computation Node 离线计算并存储，Prediction Server 在推理时拼接。当行为序列扩展到万级别时，行为数据的存储（每个用户的行为列表）和实时特征拼接的延迟都将成为不可承受的瓶颈。

### 1.3 本文解决方案概述

SIM 提出了一种全新的思路：**不压缩、不截断，而是搜索**。核心洞察是，面对万级甚至更长的行为序列，没有必要对全部行为做精细建模——只需根据候选 item 从用户的完整行为历史中"搜索"出最相关的 Top-K 条行为子序列，然后对这个短子序列做精细建模即可。这种"先粗选、再精排"的两阶段级联架构（GSU → ESU），类似于推荐系统中"召回 → 精排"的经典范式，是一种将检索思想引入特征工程的创新。

## 二、解决方案

### 2.1 核心思想

SIM 的核心设计哲学可以用一句话概括：**用 O(1) 的检索替代 O(T) 的遍历**。用户在阿里巴巴平台上的完整行为历史可能包含数万条记录，但面对某一个候选广告时，真正有参考价值的历史行为通常只有少量（论文验证 K=50 是最佳平衡点）。SIM 的两阶段设计正是基于这一洞察：第一阶段（GSU）快速从全量行为中检索出候选相关子集，第二阶段（ESU）对该子集做精细的 attention 建模。

### 2.2 整体架构

![[SIM_architecture.png|800]]

> 图1：SIM 的整体架构。左侧为 GSU（General Search Unit）的两种实现：Hard-search（基于品类的非参数检索）和 Soft-search（基于 embedding 的参数化检索 + ALSH 加速）。右侧为 ESU（Exact Search Unit），使用 multi-head target attention 对 GSU 输出的 Top-K 子序列做精细建模，并引入时间间隔编码。

整体流程分为两个阶段：

- **第一阶段 GSU（General Search Unit）**：从用户的完整行为序列 $B = [b_1, b_2, \ldots, b_T]$（$T$ 可达数万）中，基于与候选 item $a$ 的相关性，检索出 Top-K 条最相关的行为子序列 $B^* = [b_1^*, b_2^*, \ldots, b_K^*]$
- **第二阶段 ESU（Exact Search Unit）**：对 $B^*$ 使用 multi-head target attention 计算精确的用户兴趣表示，并融合时间间隔信息

#### 各模块详细说明

**模块1：GSU Hard-search（非参数检索）**

- **功能**：基于品类匹配的规则检索，无需学习参数
- **输入**：候选 item $a$ 的品类 $C_a$，用户完整行为序列 $B$
- **输出**：与候选 item 同品类的行为子序列 $B^*$
- **关键技术**：User Behavior Tree（UBT），一种分布式的 Key-Key-Value 数据结构

相关性评分公式：

$$r_i = \text{Sign}(C_a = C_i)$$

其中 $C_i$ 是用户第 $i$ 条行为所属的品类。当 $C_i$ 与候选 item 的品类 $C_a$ 相同时 $r_i = 1$，否则 $r_i = 0$。这本质上是一种"精确匹配"过滤——在所有行为中保留与候选 item 同品类的子集。

UBT 是一棵存储在分布式系统中的树形索引结构，总占用 22TB 存储。第一层 Key 是 user_id，第二层 Key 是 category_id，Value 是该用户在该品类下的行为列表。这种设计使得检索时间复杂度从 $O(T)$ 降为 $O(1)$——给定 user 和 category，一次查表即可获得所有相关行为。

**模块2：GSU Soft-search（参数化检索）**

- **功能**：基于 embedding 相似度的参数化检索
- **输入**：候选 item 的 embedding $\mathbf{e}_a$，用户行为中每条行为的 embedding $\mathbf{e}_i$
- **输出**：Top-K 最相关的行为子序列 $B^*$
- **关键技术**：独立的 embedding 空间 + ALSH（Asymmetric Locality-Sensitive Hashing）加速

相关性评分公式：

$$r_i = \text{Softmax}\left(\text{MLP}(\mathbf{e}_a \odot \mathbf{e}_i)\right)$$

其中 $\mathbf{e}_a$ 和 $\mathbf{e}_i$ 分别是候选 item 和第 $i$ 条行为在 GSU 独立 embedding 空间中的向量表示，$\odot$ 表示逐元素乘。

Soft-search 的一个重要设计是：**GSU 和 ESU 使用独立的 embedding 参数**。论文指出，长期行为和短期行为的分布不同，如果共享 embedding 参数，长期行为的 embedding 训练会受到短期行为分布的误导。因此 soft-search 模型独立训练一个辅助 CTR 预测任务，用该模型的 embedding 做 soft-search 检索。

用户行为表示 $\mathbf{U}_r$ 通过加权求和得到：

$$\mathbf{U}_r = \sum_{i=1}^{T} r_i \mathbf{e}_i$$

$\mathbf{U}_r$ 与候选 item $\mathbf{e}_a$ 拼接后送入 MLP 预测点击概率，作为 GSU 的辅助训练信号。

在线推理时，为了高效地从万级行为中选出 Top-K，论文采用 ALSH（Asymmetric LSH）将最大内积搜索（MIPS）转化为近似最近邻搜索（ANN），在亚线性时间内完成检索。具体做法是将用户行为序列的 embedding 离线建索引存储在 UBT 中，在线只需要一次 ALSH 查询即可获得 Top-K。

**模块3：ESU（Exact Search Unit）**

- **功能**：对 GSU 筛选出的 Top-K 子序列做精细建模
- **输入**：GSU 输出的子序列 $B^* = [b_1^*, \ldots, b_K^*]$、候选 item $a$、时间间隔信息
- **输出**：用户对当前候选 item 的精确兴趣表示向量
- **关键技术**：Multi-head target attention + 时间间隔编码

ESU 的 attention 计算延续了 DIN 的 target-aware 范式，但做了两个增强：

（1）**Multi-head Attention**：使用多头注意力替代 DIN 的单头注意力，能在不同子空间中捕捉不同维度的兴趣相关性。

（2）**时间间隔编码**：将行为发生时刻与当前时刻的时间差 $\Delta t_i$ 离散化为 embedding，拼接到行为的 embedding 中作为额外特征。这让模型能区分"3天前浏览过同品类"和"3个月前浏览过同品类"的行为——直觉上，越近期的同品类行为对当前点击预测的参考价值越高。

### 2.3 联合训练策略

SIM 的训练目标是端到端联合优化 GSU 和 ESU，总损失函数为：

$$L = \alpha \cdot L_{GSU} + \beta \cdot L_{ESU}$$

其中 $L_{GSU}$ 和 $L_{ESU}$ 分别是 GSU 辅助 CTR 任务和 ESU 主 CTR 任务的交叉熵损失。注意对于 hard-search 的 GSU，由于不涉及可学习参数，$L_{GSU} = 0$，总损失退化为 $L = \beta \cdot L_{ESU}$。

对于 soft-search 的 GSU，其 embedding 参数通过辅助 CTR 任务的梯度信号优化。论文的关键设计是 **GSU 和 ESU 共享 embedding 参数**（soft-search 版本中，指的是 soft-search 的 embedding 训练是与 ESU 联合进行的——但实际 embedding 空间是独立的，联合训练的含义是两个损失函数共同反向传播更新各自的 embedding），通过联合训练让两个阶段的目标对齐。

### 2.4 系统部署架构

![[SIM_RTP_system_with_UBT.png|600]]

> 图3：集成 SIM 的 RTP 系统架构。与原始系统的关键区别在于：增加了 User Behavior Tree（UBT）和 Search Index 模块。用户的完整行为历史离线构建成 UBT 索引，推理时只需从 UBT 中检索 Top-K 子序列送入模型，避免了万级行为特征的实时拼接开销。

部署的核心工程优化：

**离线阶段**：实时用户行为事件（Realtime user behavior event）持续更新 UBT，UBT 的索引构建在离线 Computation Node 上完成。对于 soft-search，每条行为的 embedding 也在离线阶段计算好并存储在 UBT 中。

**在线阶段**：Prediction Server 收到请求后，根据候选 item 的品类（hard-search）或 embedding（soft-search + ALSH）从 UBT 中取出 Top-K 行为子序列，拼接后送入 ESU 模型推理。由于只拼接 K 条（K=50）而非数万条行为特征，特征拼接延迟大幅降低。

## 三、实验分析

### 3.1 实验设置

论文使用三个规模递增的数据集：

| 数据集 | 用户量 | 物品量 | 品类数 | 样本量 | 最大序列长度 |
|--------|--------|--------|--------|--------|-------------|
| Amazon Books | 75,053 | 358,367 | 1,583 | 150,016 | 100 |
| Taobao | 7,956,431 | 34,106,612 | 5,597 | 7,956,431 | 500 |
| Industrial | 0.29 billion | 0.6 billion | 100,000 | 12.2 billion | 54,000 |

Amazon Books 将每个用户最近 10 条行为作为短期特征，其余 90 条作为长期行为。Taobao 将最近 50 条作为短期、其余 450 条作为长期。Industrial 数据集中最大序列长度达到 54000，这是当时公开文献中最大规模的行为序列建模实验。

### 3.2 公开数据集结果

| 模型 | Amazon AUC | Taobao AUC |
|------|-----------|-----------|
| DIN (short) | 0.7340 | 0.9336 |
| DIEN (short) | 0.7410 | 0.9371 |
| DIN (long+short) | 0.7440 | 0.9395 |
| DIEN (long+short) | 0.7387 | 0.9460 |
| Avg-Pooling Long | 0.7414 | 0.9398 |
| MIMN | 0.7467 | 0.9471 |
| SIM (hard) | 0.7475 | 0.9483 |
| SIM (soft) | **0.7510** | 0.9486 |
| SIM (soft) + Timeinfo | 0.7500 | **0.9501** |

关键观察：

（1）**长期行为确实有价值**：DIN/DIEN 加入长期行为后 AUC 均有提升（Amazon: 0.7340→0.7440，Taobao: 0.9371→0.9460），证实了长期历史行为中包含有效信号。

（2）**SIM 优于 MIMN**：在 Taobao 数据集上，SIM(soft)+Timeinfo 达到 0.9501，超过 MIMN 的 0.9471。SIM 通过"检索+精确建模"范式比 MIMN 的"压缩记忆"范式更有效地利用了长期行为。

（3）**时间编码在 Taobao 上显著有效**：SIM(soft) 从 0.9486 提升到 0.9501（+0.0015），但在 Amazon 上反而略降。这可能是因为 Taobao 的行为序列更长（500 vs 100），时间跨度更大，时间信息的区分度更高。

### 3.3 工业数据集结果

| 模型 | AUC |
|------|-----|
| DIEN (1000) | 0.6452 |
| MIMN (1000) | 0.6541 |
| SIM (hard) | 0.6604 |
| SIM (soft) | 0.6625 |
| SIM (hard) + Timeinfo | 0.6624 |

在 12.2 billion 样本、序列长度最长达 54000 的超大规模工业数据集上：

（1）**SIM(soft) 大幅领先 MIMN**：AUC 从 0.6541 提升到 0.6625，在 AUC 已经很高的工业级数据集上提升了 0.0084，这是非常显著的效果。注意 DIEN 和 MIMN 的序列长度被截断在 1000，而 SIM 可以使用完整的万级序列。

（2）**Hard-search 非常有竞争力**：SIM(hard) AUC 0.6604 已经超过 MIMN 的 0.6541，而 hard-search 是零参数的纯规则检索。这说明品类匹配在工业场景中是非常强的先验——同品类行为确实是预测点击概率的关键线索。

（3）**时间编码叠加 hard-search 效果更突出**：SIM(hard)+Timeinfo 达到 0.6624，几乎追平 SIM(soft) 的 0.6625，说明时间信息能有效弥补 hard-search 相比 soft-search 在相关性建模上的不足。

### 3.4 消融实验

**两阶段 vs 单阶段**：论文对比了两种配置：(a) 直接对全量序列做 attention（single-stage），(b) GSU 先筛选再 ESU 精确建模（two-stage）。结果表明两阶段方案始终优于单阶段，证明 GSU 的检索过程本身就是一种有效的噪声过滤。

**Hard-search 与 Soft-search 的重叠度**：实验发现 hard-search（品类匹配）选出的 Top-K 行为中，有 **75%** 与 soft-search（embedding 相似度）选出的重叠。这个高重叠率解释了为什么 hard-search 的效果能接近 soft-search——品类匹配已经覆盖了大部分有效行为。

**K 值选择**：论文测试了不同的 K 值（Top-K 中的 K），结果表明 **K=50** 是最佳平衡点。K 太小会遗漏相关行为，K 太大则引入噪声且增加 ESU 的计算开销。

### 3.5 在线 A/B 测试

SIM 在阿里巴巴展示广告系统上进行了 A/B 测试（baseline 为 MIMN）：

- **CTR 提升 +7.1%**
- **RPM（千次展示收入）提升 +4.4%**

这是非常显著的在线效果。CTR +7.1% 意味着在 MIMN 的基础上，SIM 让用户的点击意愿显著增强；RPM +4.4% 意味着广告主的投放 ROI 也因此受益。

### 3.6 系统性能

![[system_performance_qps_latency.png|600]]

> 图5：不同模型在不同 QPS 下的延迟对比。DIEN 的最大吞吐量只有 200 QPS，且延迟超过 30ms。MIMN 在 500 QPS 下延迟约 13.29ms。SIM 在 500 QPS 下延迟约 18.03ms，略高于 MIMN 但支持的序列长度从 1000 扩展到了 10000+。

虽然 SIM 的绝对延迟（18.03ms@500QPS）略高于 MIMN（13.29ms@500QPS），但考虑到 SIM 处理的行为序列长度是 MIMN 的 10 倍以上（10000+ vs 1000），这个延迟增加是可以接受的。而 DIEN 在行为长度仅 1000 时就只能支撑 200 QPS，根本无法扩展到万级序列。

## 四、个人评价与思考

### 4.1 核心贡献

SIM 最大的贡献在于**将"检索"思想引入用户行为建模**，开创了 "search-based" 这一全新的长序列建模范式。在此之前，行为序列建模的主流思路是"如何更好地编码全量序列"（attention、GRU、memory network），SIM 则跳出了这个框架——不是去编码全量序列，而是先"搜"出相关子集再编码。这种 "retrieve then attend" 的思想后来被 ETA、SDIM 等后续工作沿用和发展，成为了长序列建模的主流范式。

### 4.2 设计优势

**工程友好性**：SIM 的架构设计非常贴合工业部署需求。UBT 的 Key-Key-Value 结构使得行为检索可以在线上以 O(1) 完成；离线建索引 + 在线查索引的模式完美契合 RTP 系统的"存算分离"架构。Hard-search 更是零模型参数的纯工程方案，部署成本极低。

**渐进式升级路径**：Hard-search → Soft-search → + Timeinfo 是一条清晰的效果提升路径，团队可以根据系统资源选择不同配置：资源紧张时用 hard-search（0.6604），资源充裕时用 soft-search + timeinfo（0.6625+）。

### 4.3 局限性

**Hard-search 的品类粒度问题**：品类是一个人工定义的粗粒度标签体系。用户浏览了一件"红色雪纺连衣裙"，这条行为在品类层面是"女装/连衣裙"，但用户真正的兴趣可能是"红色"、"雪纺材质"或"某个特定风格"。品类匹配会将同品类但语义不相关的行为也纳入 Top-K，同时遗漏跨品类但语义相关的行为（如"红色高跟鞋"对"红色连衣裙"的兴趣关联）。这一局限性催生了后来 ETA（基于 SimHash 的语义检索）等工作。

**Soft-search 的训练-推理不一致**：训练时 soft-search 使用精确的内积相似度排序，但推理时使用 ALSH 近似检索，这引入了排序一致性的偏差。

**GSU 与 ESU 的目标对齐**：虽然论文设计了联合训练策略，但 GSU 的辅助 CTR 损失和 ESU 的主 CTR 损失之间并不完全等价——GSU 优化的是"能否从全量行为中找到有用的子集"，ESU 优化的是"给定子集能否精确预测点击"。两者的最优解不一定一致。

### 4.4 在模型发展脉络中的位置

SIM 是阿里巴巴 CTR 预估序列建模四部曲（DIN → DIEN → MIMN → SIM）的最后一篇，完成了从"单行为 attention"→"序列演化"→"记忆压缩"→"检索范式"的发展。从技术路线看，SIM 标志着序列建模从"编码器"范式向"检索器+编码器"的两阶段范式转变。这种转变在后续工作中被广泛延续：ETA 用 SimHash 替代 hard-search 实现更快的端到端语义检索，SDIM 进一步将 hash-based 检索嵌入到模型内部，QIN 引入了 query-aware 的兴趣网络。SIM 作为这一范式的开山之作，其影响深远。

### 4.5 总评

SIM 是一篇工程性极强的论文，它成功地将一个看似不可能的目标（在线推理中处理万级行为序列）变成了现实。论文的写作风格也很工业化——比起理论推导，更侧重系统设计和 A/B 测试结果。这正是阿里妈妈系列论文的一贯风格：从真实的工业痛点出发，用 scalable 的方案解决问题，用在线指标验证效果。如果说 DIN 教会了我们"看什么"（attention），DIEN 教会了我们"怎么看变化"（evolution），那 SIM 教会了我们"在海量信息中如何快速找到该看的东西"（search）——这在信息爆炸的时代，可能是更本质的能力。
