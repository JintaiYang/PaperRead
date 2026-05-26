---
paper_id: "[arXiv:1706.06978](https://arxiv.org/abs/1706.06978)"
title: "Deep Interest Network for Click-Through Rate Prediction"
authors: "Guorui Zhou, Xiaoqiang Zhu, Chengru Song, et al."
institution: "阿里巴巴（阿里妈妈）"
pushlication: "KDD 2018 2018-06-19"
tags:
  - 精排论文
  - DIN
  - Attention
  - 用户兴趣建模
  - CTR预估
  - 序列建模
  - 激活函数
  - 正则化
quality_score: "9.0/10"
link:
  - "[Github](https://github.com/zhougr1993/DeepInterestNetwork)"
  - "[PDF](https://arxiv.org/pdf/1706.06978)"
date: "2018-06-19"
---

## 一、研究背景与动机

### 1.1 领域现状

2018 年前后，CTR 预估模型普遍遵循 **Embedding & MLP** 范式：大规模稀疏特征通过 Embedding 层映射为低维稠密向量，经 Pooling 层压缩为固定长度后拼接送入 MLP。这一范式在 Wide & Deep、DeepFM、PNN 等模型中被广泛采用。阿里巴巴展示广告系统每天处理数十亿次请求，用户行为序列（浏览、点击、购买的商品列表）是最重要的特征来源之一。

![[DIN_fig1_page3.png|800]]

> 图1：阿里巴巴展示广告系统的运行流程。用户在淘宝浏览商品时，系统需要从海量候选广告中选出最相关的进行展示，CTR 预估模型是排序阶段的核心。

### 1.2 现有方法的局限性

传统 Embedding & MLP 范式中，用户行为序列通过 **sum pooling** 或 **average pooling** 被压缩为一个固定维度的向量表示。这种设计的本质问题在于：**不论当前候选广告是什么，用户的行为表示向量始终相同**。

对于一个在电商场景中同时浏览过服装、电子产品、图书的用户，当系统推荐一本书时，理想的用户表示应当更多地激活其与"图书"相关的历史行为，而非笼统地将所有行为等权求和。论文通过对阿里巴巴电商数据的分析，提炼出两个关键洞察：

- **用户兴趣多样性（Diversity）**：一个用户在一段时间内可能对多个完全不同的品类感兴趣（如同时关注泳装和鞋类），这些兴趣之间可能完全没有关联
- **局部激活特性（Local Activation）**：面对某一个候选广告时，用户的决策仅由其部分历史兴趣驱动

### 1.3 本文解决方案概述

DIN 设计了一个 **Local Activation Unit**（局部激活单元），本质上是一种 target-aware 的 attention 机制，根据候选广告对用户每一条历史行为赋予不同的激活权重，从而为不同广告生成不同的用户兴趣表示向量。此外，论文还提出了 **Mini-batch Aware Regularization** 和 **Dice 激活函数** 两项工程创新。

## 二、解决方案

### 2.1 核心思想

DIN 的核心洞察是：用户表示向量应该随候选 item 动态变化。通过引入 attention 机制，让同一维度空间承载随候选 item 变化的不同语义，而无需增大向量维度。这比朴素地增大向量维度更优雅——后者在有限训练样本上容易过拟合，且带来额外的存储和计算开销。

### 2.2 整体架构

![[DIN_fig2_page4.png|800]]

> 图2：DIN 网络架构。左侧为 Base Model（Embedding & MLP），右侧为 DIN 模型。关键区别在于 DIN 在用户行为序列的 Pooling 层引入了 Activation Unit，根据候选广告对每条行为赋予不同权重。

整体预测公式为：

$$\mathbf{v}_U(A) = f(\mathbf{v}_A, \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_H) = \sum_{j=1}^{H} a(\mathbf{e}_j, \mathbf{v}_A) \cdot \mathbf{e}_j = \sum_{j=1}^{H} w_j \cdot \mathbf{e}_j$$

其中 $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_H\}$ 是用户 $U$ 的 $H$ 条历史行为的 Embedding 向量，$\mathbf{v}_A$ 是候选广告 $A$ 的 Embedding 向量，$a(\cdot)$ 是 Activation Unit 网络。

#### 各模块详细说明

**模块1：Embedding 层**

- **功能**：将大规模稀疏 categorical 特征映射为低维稠密向量
- **输入**：用户画像特征（性别、年龄等）、用户行为特征（浏览商品 ID、店铺 ID、品类 ID 列表）、候选广告特征（广告 ID、店铺 ID、品类 ID）、上下文特征（时间等）
- **输出**：各特征的 Embedding 向量
- **关键技术**：论文没有使用人工组合特征，完全依赖 DNN 来捕获特征交叉

**模块2：Local Activation Unit（核心创新）**

- **功能**：根据候选广告对用户每条历史行为计算激活权重
- **输入**：行为 Embedding $\mathbf{e}_j$、广告 Embedding $\mathbf{v}_A$、以及两者的外积 $\mathbf{e}_j \otimes \mathbf{v}_A$
- **输出**：标量激活权重 $w_j$
- **关键设计**：**权重不做 softmax 归一化**，即不要求 $\sum_i w_i = 1$。这样做的目的是保留用户兴趣的强度信息——如果一个有 100 条浏览历史的用户中只有 3 条与候选广告相关，不归一化能保留"只有极少行为相关"这一重要信号

**模块3：Mini-batch Aware Regularization**

- **功能**：解决大规模稀疏特征的正则化计算效率问题
- **关键技术**：在每个 mini-batch 中，只对当前 batch 中实际出现的稀疏特征所对应的 Embedding 参数计算 L2 正则

$$L_2(\mathbf{W}) = \sum_{j=1}^{K} \sum_{m=1}^{B} \frac{\alpha_{mj}}{n_j} \|\mathbf{w}_j\|_2^2$$

其中 $n_j$ 是特征 $j$ 在所有样本中的出现次数。低频特征获得更强的正则化（$n_j$ 小），高频特征正则化相对较弱。

![[DIN_fig4_page7.png|800]]

> 图3：不同正则化方法在阿里巴巴数据集上的训练曲线对比。无正则化时模型在训练一个 epoch 后就出现严重过拟合，MBA Regularization 能有效缓解这一问题。

**模块4：Dice 激活函数**

- **功能**：数据自适应的激活函数，PReLU 的泛化版本
- **数学形式**：

$$f(s) = p(s) \cdot s + (1 - p(s)) \cdot \alpha s, \quad p(s) = \frac{1}{1 + e^{-\frac{s - E[s]}{\sqrt{Var[s] + \epsilon}}}}$$

训练阶段 $E[s]$ 和 $Var[s]$ 是当前 mini-batch 的均值和方差；推理阶段使用滑动平均值。Dice 本质上是用 Batch Normalization 的思想对激活函数的校正点进行自适应调整。

![[DIN_fig3_page5.png|800]]

> 图4：PReLU 和 Dice 的控制函数对比。PReLU 的校正点固定为 0，而 Dice 根据输入数据的统计量自适应调整，控制函数更平滑。

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征 | 数据类型 |
|--------|--------|------|----------|
| Amazon | 用户评论数据 | 商品 ID、品类 | 公开数据集 |
| MovieLens-20M | 2000万条评分 | 电影 ID、类型 | 公开数据集 |
| Alibaba | 20亿+ 样本 | 用户画像+行为+广告+上下文 | 工业生产数据 |

### 3.2 实验设置

#### 3.2.1 基线方法

- LR：逻辑回归
- Wide & Deep：Google 的双流模型
- PNN：Product-based Neural Network
- DeepFM：FM + Deep 的端到端模型
- Base Model：Embedding & MLP（DIN 的基础模型）

#### 3.3.2 评估指标

- **GAUC（用户加权 AUC）**：$GAUC = \frac{\sum_{i=1}^{n} \#impression_i \times AUC_i}{\sum_{i=1}^{n} \#impression_i}$，比全局 AUC 更能反映模型对每个用户的排序质量
- **RelaImpr**：$\text{RelaImpr} = \left(\frac{AUC(\text{model}) - 0.5}{AUC(\text{base}) - 0.5} - 1\right) \times 100\%$

### 3.3 实验结果与分析

| 方法 | Amazon AUC | MovieLens AUC | Alibaba GAUC | RelaImpr |
|------|-----------|---------------|--------------|----------|
| LR | 0.7194 | 0.7325 | -- | -- |
| Wide & Deep | 0.7283 | 0.7538 | -- | -- |
| PNN | 0.7312 | 0.7555 | -- | -- |
| DeepFM | 0.7324 | 0.7590 | -- | -- |
| Base Model | 0.7337 | 0.7602 | ~0.60 | baseline |
| **DIN** | **0.7348** | **0.7632** | -- | **+3.4%** |

#### 结果分析

DIN 在所有数据集上均优于对比方法。在阿里巴巴真实生产数据集上，DIN 相比 Base Model 实现了约 3.4% 的 RelaImpr 提升。虽然绝对提升看似不大，但在阿里巴巴的广告收入体量下，千分位级别的 AUC 提升即可带来数百万级的 eCPM 收益。

### 消融实验

#### 实验设计

论文通过消融实验验证了各个组件的贡献。

#### 消融结果和分析

- **MBA Regularization vs 无正则化/传统 L2/Dropout**：MBA 正则化在所有对比中均有明显优势
- **Dice vs ReLU/PReLU**：Dice 在多个数据集上取得了一致的提升
- **Attention 权重不归一化 vs 归一化**：不归一化效果更好，验证了保留兴趣强度信息的重要性

### 实验结果图

![[DIN_fig5_page9.png|800]]

> 图5：DIN 的自适应激活可视化。与候选广告高度相关的历史行为获得高激活权重（红色），不相关的行为权重接近零（蓝色）。

![[DIN_fig6_page9.png|800]]

> 图6：DIN 学到的商品 Embedding 可视化。不同形状代表不同品类，可以看到同品类商品在 Embedding 空间中聚集，验证了模型学到了有意义的语义表示。

### 在线 A/B 测试

DIN 在阿里巴巴展示广告系统进行了在线 A/B 测试：**CTR 提升 10.0%，RPM（Revenue Per Mille）提升 3.8%**。论文还介绍了服务大规模在线流量时的工程优化技巧，包括用户行为序列的截断策略和 Activation Unit 的高效实现。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中提到，未来的研究方向包括探索更复杂的序列建模方法来捕捉用户兴趣的时序演化，以及将 DIN 的思想扩展到更长的行为序列。

### 4.2 基于分析的未来方向

1. **方向1：用户兴趣的时序演化建模**
   - 动机：DIN 将行为序列视为无序集合，忽略了兴趣随时间的演化
   - 可能的方法：引入 GRU/LSTM 等序列模型捕捉兴趣演化（即后来的 DIEN）
   - 预期成果：更准确地建模用户当前兴趣状态
   - 挑战：序列模型的计算效率和长序列建模能力

2. **方向2：超长行为序列的高效建模**
   - 动机：DIN 的 attention 计算复杂度为 O(H)，当行为序列长度 H 达到数千时效率下降
   - 可能的方法：两阶段检索架构，先粗筛再精排（即后来的 SIM）
   - 预期成果：支持数万级行为序列的高效建模
   - 挑战：粗筛阶段的信息损失控制

### 4.3 改进建议

1. **改进1：引入序列位置信息**
   - 当前问题：DIN 的 attention 不考虑行为的时间顺序
   - 改进方案：加入位置编码或时间衰减因子
   - 预期效果：更好地区分近期兴趣和远期兴趣

2. **改进2：多粒度兴趣建模**
   - 当前问题：DIN 在单一粒度（商品 ID 级别）建模兴趣
   - 改进方案：同时在品类、店铺、品牌等多个粒度建模
   - 预期效果：更全面的用户兴趣表示

## 五、 我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - DIN 是推荐系统精排领域引入 attention 机制的开创性工作，其"用户表示应随候选 item 动态变化"的核心思想深刻影响了后续所有行为序列建模工作。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次在精排中引入 target-aware attention，"不归一化以保留兴趣强度"的设计尤为精妙 |
| 技术质量 | 8/10 | 模型结构简洁但配套的 MBA 正则化和 Dice 激活函数体现了扎实的工程功底 |
| 实验充分性 | 9/10 | 公开数据集 + 工业数据集 + 在线 A/B 测试 + 可视化分析，实验体系完备 |
| 写作质量 | 8/10 | 结构清晰，motivation 阐述充分，图表直观 |
| 实用性 | 10/10 | 在阿里巴巴展示广告系统上线服务主要流量，是工业界最具影响力的 CTR 模型之一 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Attention 权重不归一化的设计思想——保留兴趣强度信息，这与 NMT 中的标准 attention 不同
- MBA 正则化利用稀疏特征在 mini-batch 中的分布特性，将计算复杂度从 O(全量参数) 降至 O(batch 内活跃参数)
- Dice 激活函数通过 BN 思想自适应调整校正点，在后续众多 CTR 模型中被广泛采用

#### 5.2.2 需要深入理解的部分

- 为什么不归一化比归一化效果好？从信息论角度，归一化丢失了"相关行为占比"这一信号
- Activation Unit 中外积操作的必要性——引入显式交叉特征有助于更好地建模行为与广告的相关性
- GAUC vs 全局 AUC 的区别——GAUC 更能反映模型对每个用户的排序质量

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DIEN|DIEN]] - DIN 的直接后续，引入 GRU+AUGRU 捕捉用户兴趣的时序演化
- [[SIM|SIM]] - 两阶段检索架构，将 DIN 的 attention 思想扩展到超长行为序列

### 6.2 背景相关
- [[Wide_and_Deep|Wide & Deep]] - Embedding & MLP 范式的代表，DIN 的 Base Model 基础
- [[DeepFM|DeepFM]] - 同期的 CTR 模型，DIN 的对比基线之一

### 6.3 后续工作
- [[DIEN|DIEN]] - AAAI 2019，兴趣演化网络
- [[SIM|SIM]] - CIKM 2020，搜索式用户兴趣建模
- BST (DLP-KDD 2019) - 将 Transformer 引入行为序列建模

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1706.06978)
- [GitHub 官方代码](https://github.com/zhougr1993/DeepInterestNetwork)
- [知乎经典解读](https://zhuanlan.zhihu.com/p/51623339)

> [!tip] 关键启示
> 用户兴趣是多样的，面对不同候选 item 时只有部分历史行为被"激活"——DIN 通过 target-aware attention 实现了动态用户表示，打破了"用户向量不随候选变化"的限制，这一思想成为后续所有序列建模工作的起点。

> [!warning] 注意事项
> - DIN 将行为序列视为无序集合，忽略了时序信息，这在 DIEN 中被改进
> - Activation Unit 的计算复杂度为 O(H)，当行为序列很长时需要截断，这在 SIM 中被解决
> - 论文中的 GAUC 绝对值较低（~0.60），这是用户加权 AUC 的特点，不能与全局 AUC 直接比较

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！DIN 是序列建模在推荐精排中的奠基之作，其 target-aware attention 的思想几乎成为后续所有行为序列建模工作的起点。理解 DIN 是理解 DIEN、SIM、BST 等后续工作的必备基础。
