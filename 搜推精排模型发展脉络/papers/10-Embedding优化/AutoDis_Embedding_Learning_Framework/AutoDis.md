---
paper_id: "[arXiv:2012.08986](https://arxiv.org/abs/2012.08986)"
title: "An Embedding Learning Framework for Numerical Features in CTR Prediction"
authors: "Huifeng Guo, Bo Chen, Ruiming Tang, Weinan Zhang, Zhenguo Li, Xiuqiang He"
institution: "Huawei Noah's Ark Lab, Shanghai Jiao Tong University"
publication: "KDD 2021, 2021-08-14"
tags:
  - Embedding
  - CTR预测
  - 数值特征
  - 自动离散化
  - Meta-Embedding
  - 端到端学习
  - 华为
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2012.08986)"
  - "[KDD 2021](https://dl.acm.org/doi/10.1145/3447548.3467077)"
date: "2026-07-10"
---

## 一、研究背景与动机

### 1.1 领域现状

CTR 预测中，特征 embedding 是将原始输入映射到低维稠密空间的核心步骤。对于 categorical 特征（如 user_id、item_category），标准做法是通过 embedding table 为每个特征值分配独立的 embedding 向量。然而对于 numerical 特征（如年龄、价格、CTR 统计值），如何高效地学习其 embedding 表示仍是一个开放问题。

![[AutoDis_Embedding_Learning_Framework/images/embedding_fi_architecture.png|800]]
> 图1：CTR 预测中 Embedding & Feature Interaction 的标准范式。所有特征先映射为等维 embedding 向量，再通过特征交互层（FM、DCN、DeepFM 等）进行组合建模。

### 1.2 数值特征的困境

论文系统分析了数值特征 embedding 学习的三大挑战性质：

**TPP（Two-Phase Problem，两阶段问题）**：数值特征的离散化和 embedding 学习应当是一个统一过程。但传统方法将二者分开——先用手工规则或无监督方法离散化，再学习 embedding。离散化阶段无法感知下游任务的目标，可能导致次优的分桶方案。

**SBD（Similar value But Dissimilar embedding，近值远嵌）**：当两个相近的数值被划分到不同的桶时（如年龄 17 和 18 分属不同桶），它们获得完全不同的 embedding，丢失了数值相近应当表示相似的归纳偏置。

**DBS（Dissimilar value But Same embedding，远值同嵌）**：同一个桶内所有数值共享相同的 embedding。例如收入 1000 元和 9000 元若处于同一桶中，模型无法区分二者，损失了信息粒度。

![[AutoDis_Embedding_Learning_Framework/images/limitations.png|600]]
> 图4：传统硬离散化方法的 SBD 和 DBS 问题示意。左图展示了两个相邻数值被桶边界分隔导致的 SBD 问题；右图展示了同桶内差异较大数值获得相同 embedding 的 DBS 问题。

### 1.3 已有方案分析

论文总结了已有的数值特征 embedding 方法及其局限：

**Field Embedding（FE）**：整个数值特征域共享一个 embedding 向量，数值 $x_j$ 的表示为 $x_j \cdot \mathbf{e}_j$（标量乘法）。优点是参数少、端到端训练；缺点是所有值共线性（方向相同，仅幅度不同），表示能力极弱——这是最严重的 DBS 问题。

**Hard Discretization（HD）**：等宽（EDD）/等频（EFD）分桶或基于决策树（如 GBDT+LR、XGBoost+NN）的分桶。解决了 FE 的共线性问题，但引入 TPP（两阶段不可联合优化），且存在 SBD 和 DBS。

**AutoDis（本文方案）**：同时满足三大理想性质——(1) 高模型容量（Model Capacity），每个数值可获得独特表示，不受 FE 的共线性约束；(2) 端到端训练（End-to-End Training），离散化与 embedding 学习在统一框架下联合优化，避免 TPP；(3) 唯一表示（Unique Representation），每个不同数值映射为唯一的 embedding 向量，同时消除 SBD 和 DBS。

## 二、解决方案

### 2.1 核心思想

AutoDis 的核心洞察是：与其用硬边界将数值划入某个桶，不如让每个数值以「软」方式关联到所有候选桶（Meta-Embeddings），然后通过可学习的聚合策略生成最终 embedding。这样做的好处是：(1) 离散化变成可微分操作，可以梯度回传到离散化参数；(2) 相近数值会获得相似的软分布从而得到相近表示（解决 SBD）；(3) 每个不同数值获得唯一的聚合权重从而得到唯一表示（解决 DBS）。

![[AutoDis_Embedding_Learning_Framework/images/autodis_overall.png|800]]
> 图2：AutoDis 整体框架。对于数值特征域 $j$ 的某个数值 $x_j$，首先通过 Automatic Discretization 模块生成归属于 $H_j$ 个 Meta-Embedding 的概率分布，然后通过 Aggregation 模块将 Meta-Embeddings 按此分布加权聚合为最终 embedding $\mathbf{e}_{x_j}$。

### 2.2 整体架构

![[AutoDis_Embedding_Learning_Framework/images/autodis_architecture.png|800]]
> 图3：AutoDis 的详细架构。展示了 Meta-Embeddings 矩阵、两层 MLP + Softmax(τ) 生成离散化概率分布、以及 Weighted-Average 聚合的完整流程。

AutoDis 由三个核心模块组成：

#### 模块1：Meta-Embeddings

对于数值特征域 $j$（如"年龄"、"价格"），AutoDis 定义一组 $H_j$ 个 Meta-Embedding 向量：

$$\mathbf{ME}_j = [\mathbf{me}_j^1, \mathbf{me}_j^2, ..., \mathbf{me}_j^{H_j}] \in \mathbb{R}^{H_j \times d}$$

这里的 Meta-Embeddings 不同于 categorical embedding table 中的 embedding。Categorical 特征中，每个类别独占一个 embedding 向量；而 Meta-Embeddings 是所有数值共享的「基底向量」，每个具体数值通过独特的加权组合来生成自己的表示。论文将其命名为 "Meta" 正是强调这种共享-组合的特性。

Meta-Embedding 的数量 $H_j$ 是超参数，决定了该数值特征域的表示粒度和参数量。实验表明 $H_j$ 在 20~40 之间表现稳定。

#### 模块2：Automatic Discretization（自动离散化）

给定特征域 $j$ 的某个数值 $x_j$，自动离散化模块计算 $x_j$ 归属于各个 Meta-Embedding 的概率分布：

$$\tilde{\mathbf{h}}_j = \text{Leaky\_ReLU}(\mathbf{W}_j^{(1)} \cdot x_j + \boldsymbol{\alpha}_j)$$

$$\hat{\mathbf{h}}_j = \text{Leaky\_ReLU}(\mathbf{W}_j^{(2)} \cdot \tilde{\mathbf{h}}_j + \boldsymbol{\beta}_j)$$

$$\bar{\mathbf{h}}_j = \text{Softmax}(\hat{\mathbf{h}}_j / \tau)$$

其中 $\mathbf{W}_j^{(1)} \in \mathbb{R}^{H_j \times 1}$，$\mathbf{W}_j^{(2)} \in \mathbb{R}^{H_j \times H_j}$，$\boldsymbol{\alpha}_j, \boldsymbol{\beta}_j \in \mathbb{R}^{H_j}$ 均是可学习参数。$\tau$ 是温度系数，控制分布的尖锐程度：

- $\tau \to 0$：分布退化为 one-hot，等效于硬离散化
- $\tau \to \infty$：分布趋于均匀，所有 Meta-Embedding 等权贡献
- 适中的 $\tau$（实验最优约 $\tau = 1$~$5$）：产生有意义的软分布

两层 MLP 的设计使得离散化函数可以学习非线性的分桶边界。相比单纯的 one-hot 硬分配，这种软分配保证了：(a) 梯度可以从聚合后的 embedding 回传到 MLP 参数和 Meta-Embeddings，实现端到端训练；(b) 相近数值产生相近的概率分布。

![[AutoDis_Embedding_Learning_Framework/images/softmax_adjacent.png|500]]
![[AutoDis_Embedding_Learning_Framework/images/softmax_distant.png|500]]
> 图5-6：（上图）相邻数值的 Softmax 输出分布高度相似，验证 AutoDis 解决了 SBD 问题；（下图）差异较大数值的分布明显不同，验证 AutoDis 解决了 DBS 问题。

#### 模块3：Aggregation（聚合）

最终 embedding 由 Meta-Embeddings 按离散化概率加权聚合得到。论文对比了三种聚合策略：

**Max-Pooling（MP）**：$\mathbf{e}_{x_j} = \mathbf{me}_j^{k^*}$, 其中 $k^* = \arg\max_k \bar{h}_j^{(k)}$。取概率最大的 Meta-Embedding 作为结果。问题是 argmax 不可微，且同一桶内所有数值得到相同表示（DBS 问题未解决）。

**Top-K-Sum（TKS）**：$\mathbf{e}_{x_j} = \sum_{k \in \text{TopK}} \mathbf{me}_j^k$。取概率最高的 K 个 Meta-Embedding 求和。引入超参 K，且仍存在部分 DBS 问题。

**Weighted-Average（WA）**：$\mathbf{e}_{x_j} = \sum_{k=1}^{H_j} \bar{h}_j^{(k)} \cdot \mathbf{me}_j^k$。以概率为权重对所有 Meta-Embedding 加权求和。这是最终选择的方案，因为：(1) 完全可微，梯度可回传到所有组件；(2) 不同数值必然有不同的权重分布，从而得到唯一表示。

$$\mathbf{e}_{x_j} = \sum_{k=1}^{H_j} \bar{h}_j^{(k)} \cdot \mathbf{me}_j^k$$

### 2.3 理论性质分析

论文通过严格分析证明 AutoDis（使用 WA 聚合）同时满足三大理想性质：

**Model Capacity**：每个数值 $x_j$ 的 embedding $\mathbf{e}_{x_j}$ 是 $H_j$ 个 $d$ 维向量的加权组合。当 $H_j$ 个 Meta-Embedding 线性无关时（通用情况下成立），可表示的空间维度为 $H_j \times d$（embedding 空间的整个子空间），远大于 FE 方法的 1 个自由度。

**End-to-End Training**：所有组件——Meta-Embeddings $\mathbf{ME}_j$、MLP 参数 $\{\mathbf{W}_j^{(1)}, \mathbf{W}_j^{(2)}, \boldsymbol{\alpha}_j, \boldsymbol{\beta}_j\}$、温度 $\tau$——都是可微分的，通过标准 CTR loss（如 binary cross-entropy）梯度更新。

**Unique Representation**：对于任意两个不同数值 $x_a \neq x_b$，由于 MLP 的非线性映射 + Softmax 归一化，$\bar{\mathbf{h}}_j(x_a) \neq \bar{\mathbf{h}}_j(x_b)$（除退化情况外），因此 $\mathbf{e}_{x_a} \neq \mathbf{e}_{x_b}$。

### 2.4 参数量分析

对于数值特征域 $j$，AutoDis 引入的额外参数为：

- Meta-Embeddings：$H_j \times d$
- 第一层 MLP：$H_j \times 1 + H_j$（权重 + 偏置）
- 第二层 MLP：$H_j \times H_j + H_j$（权重 + 偏置）
- 总计：$H_j \times d + H_j^2 + 3H_j$

以 $H_j = 20$, $d = 16$ 为例，每个数值特征域的额外参数约 $20 \times 16 + 400 + 60 = 780$ 个，极其轻量。相比之下，EDD/EFD 的 embedding table 同样需要 $H_j \times d = 320$ 个参数，AutoDis 仅多出 $H_j^2 + 3H_j = 460$ 个 MLP 参数即可获得端到端可训练的软离散化能力。

## 三、实验评估

### 3.1 实验设置

**数据集**：

| 数据集 | 样本数 | 特征数 | 正样本比 | 来源 |
|--------|--------|--------|----------|------|
| Criteo | 45M | 39（13 numerical + 26 categorical） | ~26% | 在线广告点击 |
| AutoML | 1M | 27 numerical + 1 label | 未公开 | KDD Cup 2019 |
| Huawei | 工业级 | 未公开 | 未公开 | 华为应用市场（在线AB） |

**Backbone 模型**：实验在 6 个主流 CTR 模型上验证 AutoDis 的通用性——FM、DeepFM、IPNN、PIN、FiBiNet、AutoInt+。

**Baseline 方法**：FE（Field Embedding）、EDD（等宽分桶）、EFD（等频分桶）、LD（Logarithm Discretization）、AutoDis。

**评估指标**：AUC、LogLoss（Criteo/AutoML），以及在线 CTR 和 eCPM（Huawei）。

### 3.2 主要结果

**Criteo 离线效果**：

| Backbone | FE (AUC) | EDD (AUC) | EFD (AUC) | LD (AUC) | AutoDis (AUC) | 提升 vs 最佳Baseline |
|----------|----------|-----------|-----------|----------|---------------|---------------------|
| FM | 0.7876 | 0.7923 | 0.7930 | 0.7923 | **0.7959** | +0.0029 |
| DeepFM | 0.8062 | 0.8065 | 0.8065 | 0.8066 | **0.8077** | +0.0011 |
| IPNN | 0.8079 | 0.8082 | 0.8086 | 0.8085 | **0.8098** | +0.0012 |
| PIN | 0.8079 | 0.8083 | 0.8080 | 0.8087 | **0.8099** | +0.0012 |
| FiBiNet | 0.8082 | 0.8085 | 0.8085 | 0.8089 | **0.8098** | +0.0009 |
| AutoInt+ | 0.8065 | 0.8069 | 0.8072 | 0.8071 | **0.8082** | +0.0010 |

**AutoML 离线效果**：AutoDis 同样在所有 backbone 上取得最优 AUC，对 FE baseline 的提升更显著（因为 AutoML 数据集全是数值特征，数值 embedding 质量的影响更大）。

**关键发现**：

1. AutoDis 在所有 6 个 backbone 模型上一致优于所有 baseline，证明其通用性。
2. 在 FM（纯线性交互）上提升最大（+0.0029 AUC），说明高质量的数值 embedding 可以部分弥补模型表达力不足。
3. EDD/EFD/LD 三种硬离散化方法之间差异较小，而 AutoDis 与它们的差距一致且显著。

### 3.3 消融实验

**聚合策略对比**：

![[AutoDis_Embedding_Learning_Framework/images/criteo_aggregation.png|450]]
![[AutoDis_Embedding_Learning_Framework/images/automl_aggregation.png|450]]
> 图7：三种聚合策略在 Criteo 和 AutoML 上的 AUC 对比。Weighted-Average 一致优于 Max-Pooling 和 Top-K-Sum。

Weighted-Average 在两个数据集上均表现最优。Max-Pooling 最差（因为退化为硬分配），Top-K-Sum 居中。这验证了「软加权 > 硬选择」的设计直觉。

**温度系数 $\tau$ 的影响**：

![[AutoDis_Embedding_Learning_Framework/images/criteo_temp.png|450]]
![[AutoDis_Embedding_Learning_Framework/images/automl_temp.png|450]]
> 图8：温度系数 τ 在 [0.01, 100] 范围内对 AUC 的影响。最优值在 1~5 之间，过小（硬离散化）和过大（均匀分布）都会降低性能。

- $\tau = 0.01$（极端尖锐）：性能接近 EDD，因为退化为硬分桶
- $\tau = 1$~$5$：最优区间，代表有意义的软分配
- $\tau = 100$（极端平坦）：性能下降，因为无法区分不同数值

**Meta-Embedding 数量 $H_j$ 的影响**：

![[AutoDis_Embedding_Learning_Framework/images/criteo_bins.png|450]]
![[AutoDis_Embedding_Learning_Framework/images/automl_bins.png|450]]
> 图9：Meta-Embedding 数量 $H_j$ 从 5 到 80 的 AUC 变化。20~40 为最优区间，继续增加带来边际收益递减甚至过拟合。

- $H_j$ 过小（如 5）：表示容量不足，无法充分刻画数值分布
- $H_j$ 在 20~40：最佳区间
- $H_j$ 过大（如 80）：过拟合风险增加，且计算代价上升

### 3.4 数值特征重要性分析

![[AutoDis_Embedding_Learning_Framework/images/numerical_feature_fix.png|700]]
![[AutoDis_Embedding_Learning_Framework/images/numerical_feature_random.png|700]]
> 图10-11：（上图）按原始顺序排列的 13 个数值特征的 AUC 增益对比；（下图）随机打乱顺序后的对比。每个数值特征的贡献程度不同，AutoDis 对所有特征均一致优于 baseline。

论文逐个分析了 Criteo 13 个数值特征各自的贡献。关键发现：不同数值特征从 AutoDis 中获得的提升幅度不同，这与特征本身的分布特性有关（如分布越复杂、越非线性的特征，AutoDis 的优势越大）。

### 3.5 Embedding 可视化

![[AutoDis_Embedding_Learning_Framework/images/embedding_autodis_tsne.png|450]]
![[AutoDis_Embedding_Learning_Framework/images/embedding_edd_tsne.png|450]]
> 图12：AutoDis vs EDD 的 embedding t-SNE 可视化。AutoDis（左图）产生了更平滑、更有结构的 embedding 空间——数值相近的点聚在一起，而 EDD（右图）由于硬边界导致明显的断裂和聚类不连续。

这组可视化直观验证了 AutoDis 解决 SBD 问题的能力：数值相近的样本在 embedding 空间中也相近，不存在硬分桶造成的突变。

### 3.6 在线 A/B 实验

在华为应用市场的真实流量上部署 AutoDis，对比生产环境中的硬离散化 baseline：

| 指标 | 提升 |
|------|------|
| CTR | **+2.1%** |
| eCPM | **+2.7%** |

在工业级在线系统中获得如此显著的提升，充分说明了 AutoDis 的实用价值。论文特别指出，该提升在统计上显著（p < 0.01），且在持续多天的实验期间保持稳定。

## 四、优势与创新点

1. **首个系统性框架**：论文首次明确定义了数值特征 embedding 的三大理想性质（Model Capacity、End-to-End、Unique Representation），并证明了已有方法均无法同时满足，为该问题提供了清晰的理论框架。

2. **优雅的软离散化设计**：通过 MLP + 温度控制 Softmax 的组合，将离散化从不可微的硬操作转化为可微的软操作，实现了端到端训练。设计简洁但数学性质完备。

3. **通用性与轻量级**：AutoDis 作为数值特征 embedding 模块可以无缝嵌入任何 CTR 模型（仅替换数值特征的 embedding 方式），额外参数量极小（每个特征域 < 1K 参数），推理开销可忽略。

4. **工业验证**：在华为应用市场的在线实验验证了其工业可部署性，CTR +2.1% 和 eCPM +2.7% 是非常有说服力的业务效果。

## 五、局限性与讨论

1. **超参数选择**：$H_j$（Meta-Embedding 数量）和 $\tau$（温度）需要为每个数值特征域调参。虽然论文表明对这些超参数相对鲁棒（最优区间较宽），但在特征数量极多的场景下仍需要一定的调参代价。论文未探索自动确定 $H_j$ 的方法。

2. **数值特征间的交互**：AutoDis 为每个数值特征域独立设计离散化，未考虑不同数值特征之间的关联。例如"年龄"和"收入"的联合分布可能比独立分布更有信息量，但 AutoDis 无法利用这种跨域信息。

3. **分布假设的隐含限制**：两层 MLP 的函数族虽然比线性分桶更强大，但对于极端复杂的多模态分布，是否存在更好的参数化方式（如 Mixture of Experts、更深的网络）未被探索。

4. **与其他数值编码方法的对比不足**：论文主要对比了离散化系列方法和 FE，但未与一些其他数值编码方案（如 Periodic Embedding、Piece-wise Linear Encoding）进行对比。

5. **大规模场景下的效率**：当数值特征域数量极多（如数百个连续特征）时，为每个域独立维护 MLP 参数是否会成为训练瓶颈，论文未讨论。

## 六、与相关工作的关系

| 方法                   | Model Capacity | End-to-End | Unique Repr. | 核心局限            |
| -------------------- | :------------: | :--------: | :----------: | --------------- |
| Field Embedding (FE) |       ✗        |     ✓      |      ✓       | 共线性，表示能力极弱      |
| EDD/EFD              |       ✓        |     ✗      |      ✗       | TPP + SBD + DBS |
| GBDT + LR/NN         |       ✓        |     ✗      |      ✗       | TPP（两阶段不可联合优化）  |
| AutoDis              |       ✓        |     ✓      |      ✓       | （完备）            |

论文还讨论了与 NAS/AutoML 方法的关系——部分工作（如 AutoFIS）自动搜索特征交互结构，但未涉及数值特征的 embedding 方式；AutoDis 可与这些方法正交组合。

## 七、个人思考

### 7.1 方法论洞察

AutoDis 的核心贡献不仅是一个具体的技术方案，更是对数值特征 embedding 问题的系统性建模。TPP/SBD/DBS 三大性质的提出使得该领域有了清晰的评价维度。从方法论上看，AutoDis 延续了 "hard→soft" 的经典思路（类似于 hard attention → soft attention、hard clustering → soft clustering），将不可微的离散操作通过温度控制的 Softmax 软化，是一种被反复验证有效的工程范式。

### 7.2 工程部署思考

对于搜推系统的工程落地：
- AutoDis 的推理开销仅是一个小型 MLP 前向 + 加权求和，latency 增量在微秒级别，完全可以在线部署
- Meta-Embeddings 可以预加载到 GPU 显存中，不存在稀疏查表的 cache miss 问题
- 训练时每个 sample 的每个数值特征都需要计算完整的 $H_j$ 维概率分布，相比查表有一定训练速度的代价

### 7.3 后续发展

AutoDis 启发了后续一系列工作：
- **DIEN+AutoDis**：将 AutoDis 应用于序列模型中的数值时间特征
- **特征交叉中的应用**：将 soft discretization 的思想扩展到特征交叉的离散化
- **其他数值编码工作**：如 Periodic Activation（ICLR 2022）、Feature Tokenizer（NeurIPS 2021）等，虽然方法不同但都在解决数值特征表示问题

### 7.4 适用性评估

AutoDis 特别适合以下场景：
- 数值特征重要且分布复杂（如电商的价格、竞价的 bid、用户行为统计量）
- 现有系统使用硬离散化且调参困难
- 追求端到端训练，不希望引入额外的预处理步骤
- 模型对数值特征的embedding质量敏感（如 CTR 预估、CVR 预估）

不太适合的场景：
- 数值特征本身已经很少或不重要
- 模型结构本身对数值特征有较好的处理（如 tree-based models）
- 对推理延迟极端敏感且特征数极多（AutoDis 增加的参数量与 $H \times d$ 成正比）

## 八、总结

### 核心贡献

AutoDis 提出了一种优雅且实用的数值特征 embedding 学习框架，通过 **Meta-Embeddings + Soft Discretization + Aggregation** 三步，解决了传统硬离散化方法的三大固有缺陷（TPP、SBD、DBS），同时满足了高模型容量、端到端训练和唯一表示三个理想性质。

### 关键 Takeaways

1. **数值特征值得精心对待**：在 CTR 模型中，数值特征的 embedding 方法对最终效果有显著影响，简单的归一化或等频分桶远非最优。
2. **Soft 优于 Hard**：用可学习的 soft discretization 替代固定的 hard discretization，既保留了离散化的表示能力，又避免了人工规则的局限。
3. **Meta-Embeddings 是关键创新**：通过共享 meta-embeddings 而非为每个分桶独立分配 embedding，大幅降低了参数量，同时 aggregation 机制保证了表示的唯一性。
4. **温度参数的双面性**：$\tau$ 控制了"软"与"硬"的平衡——太软失去判别力，太硬退化为硬离散化。实验建议 $\tau \in [1, 5]$ 区间。
5. **工业验证充分**：在华为广告系统的在线 A/B 测试中获得 CTR +2.1%、eCPM +2.7% 的显著提升，证明了方法的工业可行性。

### 评分理由

- **创新性 8/10**：Meta-Embeddings + Soft Discretization 的组合是原创性的，问题建模清晰
- **实验充分性 8/10**：双数据集 + 在线 A/B + 丰富消融实验
- **写作质量 7/10**：结构清晰，但部分符号定义可更精确
- **工业价值 9/10**：直接解决工业级 CTR 模型的实际痛点
- **综合 8.0/10**
