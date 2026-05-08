---
paper_id: "[arXiv:1606.07792](https://arxiv.org/abs/1606.07792)"
title: "Wide & Deep Learning for Recommender Systems"
authors: "Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, et al."
institution: "Google Inc."
pushlication: "DLRS 2016 (RecSys Workshop) 2016-06-24"
tags:
  - 精排论文
  - Wide-and-Deep
  - 记忆与泛化
  - CTR预估
  - 推荐系统
  - 深度学习
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/1606.07792)"
  - "[TensorFlow 官方实现](https://www.tensorflow.org/tutorials/wide_and_deep)"
date: "2016-06-24"
---

## 一、研究背景与动机

### 1.1 领域现状

2016 年，推荐系统正处于从传统机器学习向深度学习转型的关键节点。Google Play 应用商店面对超过 10 亿活跃用户和超过 100 万个应用的推荐任务，需要在极低延迟（约 10ms）下完成候选集排序。当时主流方案有两类：广义线性模型（如 LR + 人工交叉特征）擅长"记忆"（memorization）已观察到的特征组合，但泛化能力弱；基于 Embedding 的深度模型擅长"泛化"（generalization）到未见过的特征组合，但容易过度泛化（over-generalize），推荐出不相关的结果。

### 1.2 现有方法的局限性

线性模型通过大量人工交叉特征（cross-product transformation）实现"记忆"——例如 `AND(user_installed_app=netflix, impression_app=pandora)` 这样的规则，直接记住了"安装了 Netflix 的用户倾向于点击 Pandora"。然而，这种记忆是脆弱的：如果训练集中没有出现过某个交叉组合，模型就无法预测。

深度模型通过 Embedding 将稀疏特征映射到低维稠密空间，使得语义相似的特征（如 Netflix 和 Hulu）在 Embedding 空间中距离较近，从而能泛化到训练集未见的组合。但论文指出，当 user-item 交互矩阵非常稀疏且高秩时（典型的长尾推荐场景），Embedding 容易过度泛化，为用户推荐看起来语义相关但实际不相关的内容。

### 1.3 本文解决方案概述

Wide & Deep 模型将线性模型（Wide）和深度神经网络（Deep）联合训练，使 Wide 部分负责记忆，Deep 部分负责泛化，最终将两者的输出加权求和后通过 sigmoid 得到预测概率。这是深度学习时代 CTR 预估的开山之作，首次在工业级推荐系统中验证了"Wide+Deep"的联合范式。

## 二、解决方案

### 2.1 核心思想

Wide & Deep 的核心洞察是：记忆（memorization）和泛化（generalization）是推荐系统的两种互补能力，不应该用单一模型同时追求，而应该用两个专门的子网络分别优化，再联合训练。

Wide 部分是一个带交叉特征的线性模型，参数少但能精确记住重要的特征交叉模式；Deep 部分是一个 DNN，通过 Embedding 泛化到长尾和未见组合。联合训练（joint training）让两个部分互补——Wide 不需要过多特征（因为 Deep 已经做了泛化），Deep 也不需要过于复杂（因为 Wide 补充了记忆），最终模型比单独任一部分都更紧凑高效。

### 2.2 整体架构

![[PrexWideDeepModelStructure.png|800]]

> 图1：Wide & Deep 模型结构。左侧为 Wide 部分（线性模型 + 交叉特征），右侧为 Deep 部分（Embedding + 多层 DNN），两者输出在最后一层合并，经 sigmoid 输出预测概率。

整体预测公式为：

$$P(Y=1|\mathbf{x}) = \sigma\left(\mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T \mathbf{a}^{(l_f)} + b\right)$$

其中 $\phi(\mathbf{x})$ 是交叉特征变换，$\mathbf{a}^{(l_f)}$ 是 Deep 部分最后一层的激活输出。

#### 各模块详细说明

**模块1：Wide 部分**

- **功能**：通过交叉特征实现记忆功能
- **输入**：原始特征 $\mathbf{x}$ 和交叉特征 $\phi(\mathbf{x})$
- **输出**：线性得分 $\mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})]$
- **关键技术**：Cross-product transformation

$$\phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{c_{ki}}, \quad c_{ki} \in \{0, 1\}$$

其中 $c_{ki}$ 指示第 $i$ 个特征是否参与第 $k$ 个交叉组合。论文中 Wide 部分只使用了"用户已安装应用 × 当前曝光应用"这一组交叉特征，而非大量人工特征。

- **优化器**：FTRL（Follow-the-regularized-leader）with L1 正则，产生稀疏解

**模块2：Deep 部分**

- **功能**：通过 Embedding + DNN 实现泛化功能
- **输入**：类别特征的 Embedding 向量 + 连续特征，拼接后形成约 1200 维的输入
- **输出**：最后一层的隐藏激活 $\mathbf{a}^{(l_f)}$
- **处理流程**：
  1. 每个类别特征通过 Embedding 层映射为 32 维稠密向量
  2. 所有 Embedding 和连续特征拼接
  3. 通过 3 层全连接层（1024→512→256），激活函数为 ReLU
- **优化器**：Adagrad

**模块3：联合训练（Joint Training）**

- **功能**：同时优化 Wide 和 Deep 部分的参数
- 论文区分了 **joint training** 和 **ensemble**：ensemble 中各模型独立训练，最后只合并预测；joint training 中两个部分共享梯度信号，Wide 知道 Deep 已经做了泛化，所以只需少量交叉特征即可，反之亦然
- 具体做法：将 Wide 和 Deep 的输出 logit 相加，通过 sigmoid 后计算 logloss，反向传播同时更新两部分参数

![[RecommenderSystemOverview.png|800]]

> 图2：Google Play 推荐系统整体架构。检索系统（Retrieval）从百万级候选中召回约百级候选，然后 Wide & Deep 模型对每个候选打分排序。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 特征 | 数据类型 |
|--------|------|------|----------|
| Google Play | 约 10 亿用户 | 用户画像、应用属性、上下文、行为历史 | 工业生产数据 |

### 3.2 实验设置

实验在 Google Play 应用商店的真实流量上进行，采用 A/B 实验。训练数据基于用户曝光和安装行为，标签为"是否安装了曝光的应用"。

#### 3.2.1 基线方法

- Wide-only：LR + 交叉特征
- Deep-only：DNN + Embedding
- Wide & Deep（本文方法）

#### 3.3.2 评估指标

- **离线 AUC**：离线数据集上的 ROC-AUC
- **在线 Acquisition Rate**：线上 A/B 实验中应用安装率（核心业务指标）

### 3.3 实验结果与分析

| 方法 | 离线 AUC | 在线 Acquisition Rate |
|------|----------|-----------------------|
| Wide-only | -- | baseline |
| Deep-only | -- | +2.9% |
| **Wide & Deep** | -- | **+3.9%** |

> 注：论文未报告具体 AUC 数值，只报告了在线 A/B 的相对提升

#### 结果分析

Wide & Deep 相比 Deep-only 在线安装率高出 1%（绝对值），说明 Wide 部分的"记忆"能力确实补充了 Deep 的不足。相比 Wide-only，提升 3.9%，说明 Deep 的泛化能力也是不可或缺的。论文还强调，Wide & Deep 在 Google Play 上线后，服务延迟保持在 10ms 以内（单次 batch 评估约 500 个候选），满足工业级低延迟要求。

### 消融实验

#### 实验设计

论文主要通过对比 Wide-only、Deep-only、Wide & Deep 三种变体来进行消融，同时讨论了 joint training 与 ensemble 的区别。

#### 消融结果和分析

论文指出 joint training 优于 ensemble 的关键原因在于互补效应：joint training 让 Wide 只需要少量关键交叉特征（而非数千个人工特征），Deep 也不需要过深的网络，两者合在一起比各自独立训练时更紧凑。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中提到，未来的研究方向包括探索更多类型的 Wide 特征变换、改进 Deep 部分的架构，以及在更多推荐场景中验证 Wide & Deep 的有效性。

### 4.2 基于分析的未来方向

1. **方向1：自动化 Wide 部分的特征交叉**
   - 动机：Wide 部分仍需人工定义交叉特征，这是工程瓶颈
   - 可能的方法：用 Cross Network 自动学习特征交叉（即后来的 DCN）
   - 预期成果：消除人工特征工程
   - 挑战：自动交叉的可解释性不如人工定义

2. **方向2：统一的特征交叉网络**
   - 动机：Wide 和 Deep 的分离设计增加了系统复杂度
   - 可能的方法：设计一个统一网络同时具备记忆和泛化能力
   - 预期成果：更简洁的模型架构
   - 挑战：在单一网络中平衡记忆和泛化

### 4.3 改进建议

1. **改进1：替换 Wide 中的人工交叉特征**
   - 当前问题：Wide 部分依赖人工定义的 cross-product 特征
   - 改进方案：用 FM 或 attention 机制自动学习交叉（即 DeepFM 的思路）
   - 预期效果：完全端到端，无需人工特征

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - Wide & Deep 是深度学习推荐系统的里程碑之作，"记忆+泛化"的双流架构成为后续大量工作的基础范式，其影响力远超模型本身的技术创新。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | "记忆+泛化"的双流思想简洁深刻，joint training 的思路也很重要 |
| 技术质量 | 7/10 | 方法简单直接，但理论分析较薄，Wide 部分仍依赖人工交叉特征 |
| 实验充分性 | 6/10 | 基于真实工业系统的 A/B 实验有说服力，但缺乏离线指标对比和消融深度 |
| 写作质量 | 9/10 | Google 论文一贯的简洁风格，3 页正文把核心思想讲得非常清楚 |
| 实用性 | 10/10 | 已在 Google Play 部署，TensorFlow 提供官方实现，引发了整个行业的范式转变 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- Joint training vs Ensemble 的区分非常重要——前者让两个子网络互相知道对方的存在，从而各自更"紧凑"
- Wide 部分只用了一组交叉特征（installed_app × impression_app），说明 joint training 下 Wide 不需要海量特征
- 10ms 延迟约束下服务 500 个候选的工程实现，体现了工业化部署的考量

#### 5.2.2 需要深入理解的部分

- 论文对"过度泛化"（over-generalize）的描述较为定性，缺乏量化分析
- Wide 部分为什么选择 FTRL+L1 而非 SGD？——L1 正则产生稀疏解，让 Wide 只保留最重要的交叉特征

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[GBDT_LR|GBDT+LR]] - 前序工作，GBDT 作为自动特征工程的思路启发了 Wide 部分的设计
- [[DeepFM|DeepFM]] - 用 FM 替代 Wide 中的人工交叉特征，实现完全端到端

### 6.2 背景相关
- Rendle, S. "Factorization Machines" - FM 模型，二阶自动特征交叉的经典方法

### 6.3 后续工作
- [[DCN|DCN]] - 用 Cross Network 自动化 Wide 部分的交叉特征学习
- [[xDeepFM|xDeepFM]] - 在 vector-wise 层面进行显式高阶交叉
- [[DCN_V2|DCN V2]] - Cross Network 的改进版本

## 外部资源

- [Google AI Blog：Wide & Deep Learning](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
- [TensorFlow 官方教程](https://www.tensorflow.org/tutorials/wide_and_deep)
- [arXiv 论文页面](https://arxiv.org/abs/1606.07792)

> [!tip] 关键启示
> 记忆（Memorization）和泛化（Generalization）是推荐系统的两种互补能力——Wide & Deep 通过双流架构 + 联合训练实现了两者的有机统一，这一思想深刻影响了后续所有特征交叉网络的设计。

> [!warning] 注意事项
> - Wide 部分仍需人工定义交叉特征，这在后续 DeepFM 和 DCN 中被自动化机制取代
> - 论文实验主要在 Google Play 验证，其他场景的效果需要实际验证
> - 3 页的篇幅限制了技术细节和实验深度的展示

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！这是深度学习推荐系统的奠基之作，"Wide+Deep"的双流范式成为后续 DeepFM、DCN、xDeepFM 等一系列工作的起点。理解 Wide & Deep 是理解整个精排模型演进的必备基础。
