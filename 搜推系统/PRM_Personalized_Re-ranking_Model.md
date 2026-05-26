---
paper_id: "[arXiv:1904.06813](https://arxiv.org/abs/1904.06813)"
title: "Personalized Re-ranking for Recommendation"
authors: "Changhua Pei, Yi Zhang, Yongfeng Zhang, Fei Sun, Xiao Lin, Hanxiao Sun, Jian Wu, Peng Jiang, Junfeng Ge, Wenwu Ou"
institution: "Alibaba Group, Rutgers University, Kwai Inc."
publication: "RecSys 2019, 2019-09-16"
tags:
  - Re-ranking
  - Transformer
  - 个性化推荐
  - Self-Attention
  - 列表级建模
  - 电商推荐
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/1904.06813)"
  - "[Dataset](https://github.com/rank2rec/rerank)"
date: "2026-05-25"
---

## 一、研究背景与动机

### 1.1 领域现状

推荐系统中的排序是一个多阶段过程：召回 → 粗排 → 精排 → 重排。传统的 Learning to Rank（LTR）方法（如 Pointwise/Pairwise/Listwise）为每个 item 独立打分，未考虑 item 之间在展示列表中的相互影响（mutual influence）。2019 年前后，业界开始关注重排序阶段——即在精排输出的有序列表基础上，通过建模 item 间的交互关系来进一步优化展示顺序。

### 1.2 现有方法的局限性

论文指出当时的重排方法存在两个主要问题：

**RNN-based 方法的局限**：当时的 SOTA 重排方法（如 GlobalRerank 和 DLCM）采用 RNN 结构对列表进行编码。RNN 的序列编码方式导致距离较远的 item 之间信息衰减，难以有效建模列表中任意两个 item 之间的交互关系。此外，基于 decoder 的方法（如 Seq2Slate、GlobalRerank）需要逐个生成重排结果，推理时间复杂度为 $O(n) \times RT$，在在线服务的严格延迟要求下难以部署。

**缺乏个性化建模**：已有重排方法仅关注 item 之间的相互影响，而忽略了不同用户对 item 间交互的感知程度不同。例如，对价格敏感的用户，相似 item 之间的价格差异会更强烈地影响其点击决策；而无明显购买意图的用户，列表中 item 的多样性可能更为重要。已有方法缺乏用户特定的编码函数来刻画这种差异。

### 1.3 本文解决方案概述

本文提出 PRM（Personalized Re-ranking Model），核心贡献包括：(1) 使用 Transformer 的 Self-Attention 机制替代 RNN，可在 $O(1)$ 编码距离内建模任意 item 对之间的交互，且支持并行推理；(2) 引入预训练的个性化向量（Personalized Vector, PV）作为额外输入，使模型学到用户特定的编码函数；(3) 发布了电商重排数据集供后续研究使用。

## 二、解决方案

### 2.1 核心思想

PRM 的核心思想是：将精排输出的 item 列表作为输入，通过 Transformer Encoder 对列表中的 item 进行全局交互编码，同时注入用户个性化信息，最终为每个 item 生成考虑了上下文和用户偏好的重排分数。这本质上是将 NLP 中 Transformer 处理序列的能力迁移到推荐系统的列表重排问题上。

直觉上可以类比：精排为每个 item 独立"看材料写作文"，而 PRM 相当于让这些 item "坐在一起讨论"，每个 item 根据列表中其他 item 的情况以及面对的具体用户来调整自己的分数。

### 2.2 整体架构

PRM 由三部分组成：Input Layer（输入层）、Encoding Layer（编码层）和 Output Layer（输出层）。

![[PRM_Personalized_Re-ranking_Model/images/prm_model_architecture.png|800]]
> 图1：PRM 模型整体架构。(a) Transformer Encoder 单个 block 的内部结构（Multi-Head Attention + FFN + Add&Norm）；(b) PRM 主架构——输入层拼接特征向量与个性化向量，加上位置编码后送入 $N_x$ 个 Transformer block，最终通过 Softmax 输出重排分数；(c) 预训练模型生成个性化向量 $pv_i$。

#### 模块详细说明

**模块1：Input Layer（输入层）**

输入层的目标是为编码层准备综合的 item 表示。给定精排输出的初始有序列表 $\mathcal{S} = [i_1, i_2, ..., i_n]$，输入层包含三个组件：

**原始特征矩阵 $\mathbf{X}$**：$\mathbf{X} \in \mathbb{R}^{n \times d_{\text{feature}}}$，其中每一行 $\mathbf{x_i}$ 是 item $i$ 的原始特征向量（与精排使用的特征相同）。

**个性化向量 $\mathbf{PV}$**：$\mathbf{PV} \in \mathbb{R}^{n \times d_{\text{pv}}}$，由预训练模型产生，表征用户与每个 item 的交互偏好。将 $\mathbf{X}$ 和 $\mathbf{PV}$ 拼接得到中间嵌入：

$$\mathbf{E}^{'} = \begin{bmatrix} \mathbf{x_{i_1}} \, ; \, \mathbf{pv_{i_1}} \\ \mathbf{x_{i_2}} \, ; \, \mathbf{pv_{i_2}} \\ \dots \\ \mathbf{x_{i_n}} \, ; \, \mathbf{pv_{i_n}} \end{bmatrix}$$

其中 $;$ 表示向量拼接操作。$\mathbf{pv_{i_k}}$ 是用户 $u$ 对 item $i_k$ 的个性化向量，编码了用户在该 item 上的潜在偏好。

**位置编码 $\mathbf{PE}$**：$\mathbf{PE} \in \mathbb{R}^{n \times (d_{\text{feature}} + d_{\text{pv}})}$，采用可学习的位置编码（论文发现略优于 Vaswani 等人使用的固定正弦位置编码）。加入位置编码后：

$$\mathbf{E}^{''} = \mathbf{E}^{'} + \mathbf{PE}$$

最后通过一层前馈网络将维度映射到编码层所需的隐维度 $d$：

$$\mathbf{E} = \mathbf{E}^{''}\mathbf{W}^E + b^E$$

其中 $\mathbf{W}^E \in \mathbb{R}^{(d_{\text{feature}}+d_{\text{pv}}) \times d}$ 为投影矩阵。

**模块2：Encoding Layer（编码层）**

编码层采用标准的 Transformer Encoder 结构，由 $N_x$ 个 block 堆叠而成。每个 block 包含一个 Multi-Head Attention 层和一个 FFN 层。

**Attention Layer**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$

其中 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 分别为查询、键、值矩阵。$\sqrt{d}$ 为缩放因子，防止内积过大导致 softmax 饱和。在本文中使用 Self-Attention，即 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$ 均从同一输入矩阵投影而来。

**Multi-Head Attention**：

$$\mathbf{S}^{'} = \text{MH}(\mathbf{E}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$
$$\text{head}_i = \text{Attention}(\mathbf{E}\mathbf{W}^Q, \mathbf{E}\mathbf{W}^K, \mathbf{E}\mathbf{W}^V)$$

其中 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d}$，$\mathbf{W}^O \in \mathbb{R}^{hd \times d_{\text{model}}}$。$h$ 为 head 数量。

Self-Attention 相比 RNN 的优势在于：列表中任意两个 item 之间的交互距离为 $O(1)$，不存在信息沿序列传播时的衰减问题。这对于重排任务尤为重要——用户在浏览列表时，头部 item 可能影响尾部 item 的点击决策，这种远程依赖 RNN 难以有效捕捉。

**FFN（Feed-Forward Network）**：为模型引入非线性，增强不同维度间的交互能力。

**堆叠多个 Block**：通过堆叠 $N_x$ 个 block，模型可以捕获更复杂的高阶 item 间交互信息。

**模块3：Output Layer（输出层）**

输出层为每个 item 生成重排分数。使用一层线性变换加 softmax：

$$Score(i) = P(y_i | \mathbf{X}, \mathbf{PV}; \hat{\theta}) = \text{softmax}(\mathbf{F}^{(N_x)}\mathbf{W}^F + \mathbf{b}^F), \quad i \in \mathcal{S}_r$$

其中 $\mathbf{F}^{(N_x)}$ 是经过 $N_x$ 个 Transformer block 编码后的输出。softmax 将分数归一化为概率分布，表示每个 item 被点击的概率。

训练使用负对数似然损失：

$$\mathcal{L} = -\sum_{r \in \mathcal{R}} \sum_{i \in \mathcal{S}_r} y_i \log(P(y_i | \mathbf{X}, \mathbf{PV}; \hat{\theta}))$$

其中 $y_i \in \{0, 1\}$ 为点击标签。

**模块4：Personalized Module（个性化模块）**

个性化向量 $\mathbf{PV}$ 的生成采用预训练方式，而非与 PRM 端到端联合训练。原因在于：重排任务的训练数据仅来自精排输出的列表，数据量有限且需要频繁更新以适配上游模型变化；而预训练模型可以利用平台全量的点击日志学习更泛化的用户偏好表示。

预训练模型的结构（如论文图1(c)所示）接收三部分输入：用户行为历史序列 $\mathcal{H}_u$、当前 item $i$ 的特征、用户 $u$ 的侧信息（性别、年龄、消费等级等）。模型输出为 item $i$ 对用户 $u$ 的点击概率预测，使用 pointwise cross-entropy 训练：

$$\mathcal{L} = \sum_{i \in \mathcal{D}} y_i \log(P(y_i|\mathcal{H}_u, u; \theta^{'})) + (1-y_i)\log(1 - P(y_i|\mathcal{H}_u, u; \theta^{'}))$$

其中 $\mathcal{D}$ 是平台展示给用户 $u$ 的 item 集合。受 YouTube DNN 推荐论文启发，取 sigmoid 层之前的隐向量作为个性化向量 $\mathbf{pv_i}$。

论文指出预训练模型的架构不与 PRM 强耦合，可以替换为 FM、FFM、DeepFM、DCN、FNN、PNN 等模型。

## 三、实验结果

### 3.1 数据集

| 数据集 | 用户数 | 文档/Item数 | 记录数 | 反馈类型 |
|--------|--------|-------------|--------|----------|
| Yahoo Letor v2.0 | -- | 709,877 | 29,921 | {0,1,2,3,4} |
| E-commerce Re-ranking | 743,720 | 7,246,323 | 14,350,968 | {0,1} |

**Yahoo Letor 数据集**处理方式：将评级（0-4）通过阈值 $T_b=1.5$ 转为二值标签；使用衰减因子 $\eta=0.2$，以 $1/pos(i)^{\eta}$ 模拟移动端推荐场景中 item 的曝光概率衰减。

**E-commerce Re-ranking 数据集**来自真实电商推荐系统，包含推荐列表、用户基本信息、点击标签和排序特征。该数据集的平均点击率低于 5%，远低于 Yahoo 数据集的约 30%，因此排序难度更大。

### 3.2 实验设置

#### 3.2.1 基线方法

**LTR 方法**（也可作为重排基线，即对初始列表再次排序）：
- SVMRank：基于 pairwise loss 的经典 LTR 方法
- LambdaMART：基于 listwise loss 的 SOTA LTR 方法
- DNN-based LTR：线上部署的 Wide&Deep 模型

**Re-ranking 方法**：
- DLCM：基于 GRU 编码列表上下文信息的重排模型，将全局向量与每个 item 特征结合打分

论文未选择 Seq2Slate 和 GlobalRerank 作为基线，原因是这两者使用 decoder 结构逐个生成结果，推理延迟不满足线上要求。

#### 3.2.2 评估指标

- **离线**：Precision@5、Precision@10、MAP@5、MAP@10、MAP@30
- **线上**：PV（展示量）、IPV（点击量）、CTR（点击率）、GMV（成交额）

#### 3.2.3 训练细节

- 隐维度 $d_{\text{model}}$：Yahoo 数据集为 1024，E-commerce 数据集为 64
- 学习率：与 Vaswani et al. (2017) 一致的 warmup schedule
- Dropout：$p=0.1$
- Batch size：Yahoo 为 256，E-commerce 为 512
- 初始列表长度：30
- 默认配置：$N_x=4$ blocks，$h=3$ heads

### 3.3 实验结果与分析

**Yahoo Letor 数据集离线结果**（初始列表由 SVMRank 生成）：

| 初始列表 | 重排方法 | Precision@5(%) | Precision@10(%) | MAP@5(%) | MAP@10(%) | MAP(%) |
|----------|----------|----------------|-----------------|----------|-----------|--------|
| SVMRank | SVMRank | 50.42 | 42.25 | 73.71 | 68.28 | 62.14 |
| SVMRank | LambdaMART | 51.35 | 43.08 | 74.94 | 69.54 | 63.38 |
| SVMRank | DLCM | 52.54 | 43.26 | 76.52 | 70.86 | 64.50 |
| SVMRank | **PRM-BASE** | **53.29** | **43.66** | **77.62** | **72.02** | **65.60** |
| LambdaMART | SVMRank | 50.41 | 42.34 | 73.82 | 68.27 | 62.13 |
| LambdaMART | LambdaMART | 52.04 | 43.00 | 75.77 | 70.49 | 64.04 |
| LambdaMART | DLCM | 52.54 | 43.16 | 77.81 | 71.88 | 65.24 |
| LambdaMART | **PRM-BASE** | **53.63** | **43.41** | **78.62** | **72.67** | **65.72** |

**E-commerce Re-ranking 数据集离线结果**：

| 初始列表 | 重排方法 | Precision@5 | Precision@10 | MAP@5(%) | MAP@10(%) | MAP(%) |
|----------|----------|-------------|--------------|----------|-----------|--------|
| DNN-LTR | DLCM | 12.21 | 9.73 | 29.32 | 30.28 | 28.19 |
| DNN-LTR | PRM-BASE | 12.71 | 9.99 | 29.80 | 30.83 | 28.85 |
| DNN-LTR | **PRM-Personalized-Pretrain** | **13.58** | **10.52** | **31.18** | **32.12** | **30.15** |

**线上 A/B 测试结果**（相对于无重排的 DNN-LTR 基线的提升）：

| 重排方法 | PV | IPV | CTR | GMV |
|----------|-----|------|------|------|
| DLCM | +0.77% | +1.75% | +0.97% | +0.13% |
| PRM-BASE | +1.27% | +2.44% | +1.16% | +0.36% |
| **PRM-Personalized-Pretrain** | **+3.01%** | **+5.69%** | **+2.6%** | **+6.65%** |

#### 结果分析

**PRM-BASE vs. DLCM**：在 Yahoo 数据集上，PRM-BASE 在 SVMRank 初始列表上相比 DLCM 取得 MAP +1.7%、Precision@5 +1.4% 的提升；在 E-commerce 数据集上 MAP +2.3%、Precision@5 +4.1%。E-commerce 数据集的提升幅度更大，论文分析这与数据集难度有关——Yahoo 数据集平均 CTR 约 30%，而 E-commerce 数据集低于 5%，排序任务越难，PRM 的优势越明显。

**PRM-Personalized-Pretrain vs. PRM-BASE**：引入预训练个性化向量后，离线 MAP 进一步提升 4.5%、Precision@5 提升 6.8%。线上 GMV 提升从 PRM-BASE 的 +0.36% 跃升至 +6.65%（绝对增加 6.29%），说明个性化编码函数能更精准地捕捉不同用户面对同一列表时的偏好差异。

**初始列表的影响**：无论初始列表由 SVMRank 还是 LambdaMART 生成，PRM 均能稳定提升性能，说明模型对上游排序方法的选择不敏感。

### 3.4 消融实验

在 Yahoo Letor 数据集上（初始列表由 SVMRank 生成）的消融结果：

| 配置 | P@5 | P@10 | MAP@5 | MAP@10 | MAP |
|------|------|------|-------|--------|-----|
| DLCM (baseline) | 52.54 | 43.26 | 76.52 | 70.86 | 64.50 |
| Default (b=4, h=3) | 53.29 | 43.66 | 77.62 | 72.02 | 65.60 |
| Remove PE | 52.55 | 43.56 | 76.11 | 70.74 | 64.73 |
| Remove RC | 53.24 | 43.63 | 77.52 | 71.92 | 65.52 |
| Remove Dropout | 53.17 | 43.42 | 77.41 | 71.80 | 65.17 |
| Block(b=1) | 53.12 | 43.59 | 77.58 | 71.91 | 65.49 |
| Block(b=2) | 53.19 | 43.58 | 77.51 | 71.86 | 65.49 |
| Block(b=6) | 53.22 | 43.63 | 77.64 | 72.02 | 65.61 |
| Block(b=8) | 52.85 | 43.32 | 77.43 | 71.65 | 65.14 |
| Multiheads(h=1) | 53.17 | 43.67 | 77.65 | 71.96 | 65.55 |
| Multiheads(h=2) | 53.29 | 43.60 | 77.68 | 72.00 | 65.57 |
| Multiheads(h=4) | 53.20 | 43.61 | 77.72 | 72.00 | 65.58 |

消融实验的关键发现：

**位置编码（PE）影响最大**：移除 PE 后 MAP 下降约 0.9pp，接近 DLCM 水平。这证实了初始列表的顺序信息对重排至关重要——没有位置编码，模型退化为对无序候选集的打分。但即使去掉 PE，PRM-BASE 仍与 DLCM 持平，说明 Transformer 的编码能力本身就优于 GRU。

**残差连接和 Dropout**：移除后性能下降较小（MAP 分别 -0.1% 和 -0.7%），说明在当前数据规模下梯度消失和过拟合问题不严重。

**Block 数量**：性能随 block 数从 1 到 4 递增，之后在 8 blocks 时下降（过拟合）。

**Head 数量**：不同 head 数（1-4）之间差异不大。论文分析认为重排列表中的 item 高度同质，将特征投影到更多子空间的收益有限。论文建议在实际部署中使用 1 个 head 以节省计算资源。

### 3.5 Attention 可视化分析

![[drr_img_cat_attention.png|800]]
> 图2：Category 维度的平均 Attention 权重热力图。相似品类间（如"男鞋"与"女鞋"、"电脑"与"手机"）attention 权重较大，说明 Self-Attention 能有效捕捉品类层面的 item 间相互影响。

![[drr_img_payclass_attention.png|800]]
> 图3：Price 维度的平均 Attention 权重热力图。价格越接近的 item 之间 attention 权重越大，说明模型学到了价格维度上的 item 间相互影响。

![[drr_img_pos_with_PE_attention.png|800]]
> 图4：带位置编码的 Attention 权重分布。位置1（排在最前面）的 item 对位置30的 item 仍有较大影响权重，说明 Self-Attention 不受编码距离限制。此外，头部位置的 item 普遍拥有更大的影响力（位置偏置效应）。

![[drr_img_pos_without_PE_attention.png|800]]
> 图5：去掉位置编码后的 Attention 权重分布。权重分布更加均匀，缺乏位置区分性，验证了位置编码在捕获初始排序信号中的重要性。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文提出两个方向：(1) 在标签空间进行优化——构建更丰富的 pairwise 或 listwise 排序关系（如基于停留时间的排序信号），挖掘标签空间中的更多信息；(2) 在重排模型中引入多样性目标——虽然实验表明 PRM 未损害列表多样性，但将多样性作为优化目标融入重排模型中值得探索。

### 4.2 基于分析的未来方向

1. **方向1：多目标重排**
   - 动机：当前 PRM 仅优化 CTR，但实际业务中需要兼顾多样性、新颖性、公平性等目标
   - 可能的方法：在损失函数中引入多样性正则项，或采用多任务学习框架
   - 预期成果：在保持点击率的同时提升用户体验指标
   - 挑战：多目标之间可能存在冲突，需要合理的权重分配机制

2. **方向2：在线学习与增量更新**
   - 动机：论文提到重排模型需要频繁更新以适配上游模型变化，但每次全量训练成本较高
   - 可能的方法：引入增量学习或在线学习机制，实时吸收新的用户反馈
   - 预期成果：模型能更快适应数据分布变化，提升新 item 的排序精度
   - 挑战：在线学习的稳定性和灾难性遗忘问题

3. **方向3：跨场景重排**
   - 动机：PRM 的个性化向量来自单一场景的预训练，未利用用户在其他场景的行为
   - 可能的方法：利用全域行为数据预训练个性化向量，或引入跨场景迁移学习
   - 预期成果：改善冷启动用户的重排效果
   - 挑战：不同场景的行为分布差异较大，需要对齐机制

### 4.3 改进建议

1. **改进1：更丰富的输入信号**
   - 当前问题：论文中的个性化向量仅编码用户-item 的偏好，未考虑实时上下文（时间、地理位置、session 内行为等）
   - 改进方案：将 session 内实时行为序列和上下文特征融入输入层
   - 预期效果：提升模型对用户即时意图的感知能力

2. **改进2：Softmax 损失的局限**
   - 当前问题：使用 softmax + 交叉熵本质上是 pointwise 的分类损失，与排序目标（如 NDCG）不一致
   - 改进方案：使用 ListNet/LambdaLoss 等 listwise 排序损失，或引入对比学习框架
   - 预期效果：更直接优化排序质量指标

## 五、我的综合评价

### 5.1 价值评分

**7.5/10** - 本文在重排领域有较好的先驱性价值，将 Transformer 引入重排并加入个性化模块的思路清晰且实用，在 2019 年的工业界背景下具有较好的创新性和实际落地价值。

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 将 Transformer 引入重排并提出个性化编码函数是有价值的组合创新，但各组件本身并非全新（Transformer + 预训练 embedding 的组合较为直接） |
| 技术质量 | 7/10 | 方法论清晰，数学形式化完整；但 softmax 输出层的设计缺乏对排序目标的直接对齐，预训练模块的架构细节偏简略 |
| 实验充分性 | 8/10 | 离线+线上双验证，消融实验覆盖面较广，attention 可视化分析有说服力；但缺少与 Seq2Slate 等方法的离线对比（仅以延迟为由排除） |
| 写作质量 | 7/10 | 结构清晰，动机阐述充分；但部分实验分析较为表面，E-commerce 数据集的消融结果被注释掉未展示 |
| 实用性 | 9/10 | 模型简洁，可并行推理，可直接部署为任意精排模型的后置模块，线上 AB 效果突出 |

### 5.2 重点关注

#### 值得关注的技术点
- Self-Attention 在重排中的应用方式：不做 decode，而是 one-step 并行输出所有 item 的重排分数
- 预训练个性化向量的设计：将用户偏好与重排模型解耦，既保证了灵活性又提供了有效的个性化信号
- 位置编码对重排的重要性：可学习的 PE 编码了初始排序信号，是模型的关键组件

#### 需要深入理解的部分
- 预训练模型如何设计才能产生对重排有帮助的 PV，而非仅仅复制精排的能力
- 在列表长度变化较大时，位置编码的泛化性问题
- 线上 GMV +6.65% 的增益中，有多少来自个性化向量本身带来的"更好打分"而非"更好的列表级建模"

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DLCM|Learning a Deep Listwise Context Model for Ranking Refinement]] - GRU-based 重排模型，本文的主要对比基线
- [[Seq2Slate|Seq2Slate: Re-ranking and Slate Optimization with RNNs]] - Pointer Network 生成重排列表
- [[GlobalRerank|Globally Optimized Mutual Influence Aware Ranking in E-Commerce Search]] - 阿里搜索重排，RNN+attention decoder

### 6.2 背景相关
- [[Attention_Is_All_You_Need|Attention Is All You Need]] - Transformer 架构原始论文
- [[YouTube_DNN|Deep Neural Networks for YouTube Recommendations]] - 预训练 embedding 的设计启发来源
- [[SASRec|Self-Attentive Sequential Recommendation]] - Self-Attention 在推荐序列建模中的应用

### 6.3 后续工作
- [[PEAR|PEAR: Personalized Re-ranking with Contextualized Transformer]] - 对 PRM 的改进工作
- [[SetRank|SetRank: Learning a Permutation-Invariant Ranking Model]] - 集合级别的重排模型

## 外部资源
- [arXiv 论文页面](https://arxiv.org/abs/1904.06813)
- [开源数据集](https://github.com/rank2rec/rerank)
- [知乎解读](https://zhuanlan.zhihu.com/p/101596475)

> [!tip] 关键启示
> 重排阶段是推荐系统中"投入产出比"较高的优化点：模型结构相对轻量（仅处理短列表），但通过显式建模 item 间交互和用户个性化，能带来可观的线上收益。PRM 验证了"Transformer + 个性化"在重排中的有效性，为后续的重排研究奠定了基础架构范式。

> [!warning] 注意事项
> - 论文使用 softmax 输出，本质上是将重排建模为列表内的多分类问题，这与 pointwise CTR 预估类似，并未真正利用 listwise 的排序信号
> - 预训练 PV 的具体模型架构（图1(c)）描述较为简略，实际复现时需要考虑如何选择合适的预训练模型
> - 线上实验中未报告统计显著性测试和置信区间

> [!success] 推荐指数
> ⭐⭐⭐⭐ 作为重排领域的经典工作，对理解 Transformer 在重排中的应用方式和个性化重排的建模思路有较好的参考价值，尤其适合做重排方向的同学作为入门阅读材料。
