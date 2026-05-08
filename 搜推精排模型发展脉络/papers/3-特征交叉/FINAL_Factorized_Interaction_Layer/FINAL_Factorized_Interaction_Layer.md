---
paper_id: "[arXiv:2304.00902](https://arxiv.org/abs/2304.00902)"
title: "FINAL: Factorized Interaction Layer for CTR Prediction"
authors: "Jieming Zhu, Qinglin Jia, Guohao Cai, et al."
institution: "Huawei Noah's Ark Lab"
pushlication: "SIGIR 2023 2023-04-03"
tags:
  - 精排论文
  - FINAL
  - 因子化交互
  - 特征交叉
  - CTR预估
  - Bilinear
  - 门控机制
quality_score: "8.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2304.00902)"
  - "[Github](https://github.com/reczoo/FuxiCTR)"
date: "2023-04-03"
---

## 一、研究背景与动机

### 1.1 领域现状

经过多年发展，CTR 预估中的特征交叉方法已极为丰富。然而，不同方法（FM、DCN、DeepFM、FiBiNET 等）的设计看似各不相同，缺乏统一的理论框架来理解和比较它们。这使得从业者在选择和改进模型时缺乏系统性指导。

### 1.2 现有方法的局限性

论文观察到，现有的特征交叉方法都可以视为某种形式的 bilinear interaction 的特殊情况，但它们在两个维度上做了不同的设计选择：交互矩阵的参数化方式（共享/独立/因子化）和残差连接结构。缺乏统一视角导致两个问题：一是很多设计选择缺乏理论支撑，二是可能错过了更优的组合方式。

### 1.3 本文解决方案概述

FINAL 从统一视角重新审视特征交叉，提出了一个因子化的交互层，支持 feature-level 和 bit-level 两种粒度的交互，并通过门控残差连接稳定训练。论文系统化地推导了现有方法（FM、FiBiNET、DCN V2 等）都是 FINAL 的特殊实例，并通过消融实验验证了各设计选择的贡献。

## 二、解决方案

### 2.1 核心思想

FINAL 的核心贡献是提出了一个统一的特征交互框架，将交互表达为 $\mathbf{e}_i \odot \mathbf{W}_{ij} \mathbf{e}_j$（bilinear with Hadamard），其中矩阵 $\mathbf{W}_{ij}$ 的参数化方式决定了模型的复杂度和表达能力。通过因子化（$\mathbf{W}_{ij} = \mathbf{U}_i \mathbf{V}_j^T$），在特征对数量 $O(m^2)$ 的场景下将参数从 $O(m^2 d^2)$ 降到 $O(md^2)$。

### 2.2 整体架构

![[overview.pdf|800]]

> 图1：FINAL 整体架构。支持 Feature-level（Block1）和 Bit-level（Block2）两种交互粒度，通过门控残差连接输出。

FINAL 包含两个交互 Block：

**Block 1：Feature-level Interaction**

$$\mathbf{o}_i = \sum_{j \neq i} \alpha_{ij} \cdot (\mathbf{e}_i \odot \mathbf{W}_{ij} \mathbf{e}_j)$$

其中 $\alpha_{ij}$ 是注意力权重，$\mathbf{W}_{ij}$ 通过因子化参数化。这对应 vector-wise 的特征交叉。

**Block 2：Bit-level Interaction**

类似 DCN V2 的 Cross Layer，在拼接后的特征向量上做全局交叉：

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

**门控残差连接**

$$\mathbf{h} = \alpha \cdot \mathbf{h}_{interaction} + (1 - \alpha) \cdot \mathbf{h}_{input}$$

其中 $\alpha$ 是可学习的门控参数，控制交互信息和原始信息的混合比例。论文发现这个门控机制对训练稳定性至关重要。

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 数据类型 |
|--------|--------|----------|
| Criteo | 4500万 | 广告点击 |
| Avazu | 4000万 | 移动广告 |
| MovieLens-1M | 约100万 | 电影评分 |
| Frappe | 约29万 | 应用推荐 |

### 3.2 实验设置

#### 3.2.1 基线方法

- FM、DeepFM、DCN V2、xDeepFM、AutoInt、FiBiNET、AFN+

#### 3.3.2 评估指标

- **AUC**、**Logloss**

### 3.3 实验结果与分析

![[avazu.pdf|600]]

> 图2：在 Avazu 数据集上的实验结果对比。

![[frappe.pdf|600]]

> 图3：在 Frappe 数据集上的实验结果对比。

FINAL 在 4 个数据集上取得了与最佳基线持平或更优的结果。特别值得注意的是，FINAL 的消融实验表明，门控残差连接（gated residual）是最重要的设计选择，贡献了约 50% 的性能提升。

### 消融实验

#### 消融结果和分析

- **去掉门控**：AUC 显著下降，说明门控残差是 FINAL 最关键的组件
- **Feature-level vs Bit-level**：两者互补，同时使用效果最佳
- **因子化 vs 全矩阵**：因子化在大多数数据集上接近全矩阵性能，但参数量大幅降低

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议在更大规模的工业数据集上验证 FINAL，以及探索因子化交互与其他网络结构的组合。

### 4.2 基于分析的未来方向

1. **方向1：自适应粒度选择**
   - 动机：不同特征对可能适合不同粒度的交互（有些适合 feature-level，有些适合 bit-level）
   - 可能的方法：用 routing 机制自动为每个特征对选择最佳交互粒度

### 4.3 改进建议

1. **改进1：动态门控**
   - 当前问题：门控 $\alpha$ 是全局参数
   - 改进方案：输入自适应的门控（类似 GRU 的 gate）

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.0/10** - FINAL 的最大价值在于提供了统一的理论框架来理解各种特征交叉方法，门控残差连接的发现也有重要的实践意义。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 统一框架的视角有学术价值，门控残差的发现实用 |
| 技术质量 | 8/10 | 理论推导严谨，消融设计科学 |
| 实验充分性 | 8/10 | 四个数据集，充分消融 |
| 写作质量 | 8/10 | 结构清晰，统一视角的展示到位 |
| 实用性 | 7/10 | 门控残差可直接应用于现有模型，但 FINAL 整体架构的工业落地信息不足 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- 统一视角：FM、DCN V2、FiBiNET 等都是 FINAL 的特殊情况
- 门控残差连接是最关键的设计选择（贡献约50%提升）
- 因子化参数化有效降低了参数量

#### 5.2.2 需要深入理解的部分

- 门控残差为什么如此重要？是因为特征交叉容易引入噪声，需要残差来"保底"？
- 因子化的秩设置对不同数据集的敏感度

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DCN_V2|DCN V2]] - FINAL Bit-level Block 的基础
- [[FiBiNET|FiBiNET]] - FINAL Feature-level Block 中 Bilinear 交互的前序

### 6.2 背景相关
- [[DeepFM|DeepFM]] - FM-based 交叉的代表
- [[AutoInt|AutoInt]] - Attention-based 交叉

### 6.3 后续工作
- [[FinalMLP|FinalMLP]] - 挑战显式交叉的必要性
- [[DCN_V3|DCN V3]] - 新一代 Cross Network

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2304.00902)
- [FuxiCTR 开源框架](https://github.com/reczoo/FuxiCTR)

> [!tip] 关键启示
> 特征交叉领域看似方法繁多，实则都可统一为 bilinear interaction 的变体——理解这一统一视角，比追逐单个模型的设计细节更有价值。而门控残差连接可能是比交叉方式本身更重要的设计选择。

> [!warning] 注意事项
> - FINAL 的统一框架虽然优雅，但可能忽略了一些方法的独特设计思想
> - 门控残差的 $\alpha$ 初始化策略对收敛有显著影响
> - 缺乏工业级大规模验证

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。FINAL 的统一分析框架对理解特征交叉领域的全貌非常有帮助，门控残差连接的实践发现也值得在自己的模型中尝试。
