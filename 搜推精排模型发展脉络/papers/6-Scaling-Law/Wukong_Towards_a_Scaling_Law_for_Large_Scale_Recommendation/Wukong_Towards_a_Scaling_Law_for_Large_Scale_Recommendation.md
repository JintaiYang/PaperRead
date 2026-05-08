---
paper_id: "[arXiv:2403.02545](https://arxiv.org/abs/2403.02545)"
title: "Wukong: Towards a Scaling Law for Large-Scale Recommendation"
authors: "Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, Guna Lakshminarayanan, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, Wenlin Chen"
institution: "Meta"
pushlication: "ICML 2024 (arXiv 2024-03-04)"
tags:
  - 精排论文
  - Wukong
  - Scaling-Law
  - 大规模推荐
  - 因子分解机
  - 参数扩展
  - Dense参数
  - 堆叠FM
quality_score: "9.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2403.02545)"
  - "[arXiv](https://arxiv.org/abs/2403.02545)"
date: "2024-03-04"
---

## 一、研究背景与动机

### 1.1 领域现状

2024 年，NLP 和 CV 领域的 Scaling Law（模型性能随参数量/数据量/计算量的增加而持续可预测地提升）已成为驱动进步的核心范式。然而，推荐系统一直未能建立起类似的规律——已有的推荐模型在增大 Dense 参数到一定规模后，性能往往达到饱和甚至恶化，无法像 LLM 那样"越大越好"。

### 1.2 现有方法的局限性

**稀疏参数扩展的局限**：传统推荐模型的参数扩展主要集中在稀疏参数（Sparse Scaling）——通过增大 Embedding Table 的大小来容纳更多特征和更高维度的 Embedding。这种扩展方式增加的是查表参数而非计算参数，带来的性能增益有限且容易饱和。

**Dense 参数扩展的瓶颈**：在 Dense 参数方向上，现有模型（如 DLRM、DCN V2、DeepFM 等）的 Dense 部分通常是 MLP。增大 MLP 的宽度和深度在一定范围内有效，但很快就会遇到性能饱和。原因在于 MLP 本身的特征交互能力有限——它学习的是隐式的、无结构的非线性映射，当参数量超过数据复杂度所需时，额外参数变成了冗余。

![[Wukong_fig1_page1.png|800]]

> 图1：Wukong 的 Scaling Law 展示。在 Meta 内部大规模数据集上，Wukong 的性能随计算量增加持续平稳提升，而其他模型出现饱和。

### 1.3 本文解决方案概述

Wukong 的核心洞察是：要建立推荐系统的 Scaling Law，需要一种**可扩展的显式特征交互机制**。论文提出基于**堆叠因子分解机**（Stacked Factorization Machines）的网络架构，以及配套的**协同扩展策略**（Synergistic Upscaling Strategy），使推荐模型首次展现出在两个数量级的模型复杂度范围内持续有效的 Scaling Law。

## 二、解决方案

### 2.1 核心思想

因子分解机（FM）天然具备捕获显式二阶交互的能力。如果将 FM 进行堆叠，第一层捕获二阶交互，第二层在一阶和二阶交互的基础上捕获最高四阶交互，以此类推，$l$ 层可以捕获最高 $2^l$ 阶交互。这种指数级增长的交互阶数为模型扩展提供了天然的"深度扩展轴"——增加层数→捕获更高阶交互→覆盖更复杂的交互模式→持续提升预测精度。

相比之下，传统 MLP 的参数增加只是增强了已有交互模式的拟合精度，而非引入新的交互模式，因此容易饱和。DCN 系列的 Cross Network 虽然引入了显式交互，但其交互阶数与层数线性增长（而非指数级），扩展效率不如 Wukong。

### 2.2 整体架构

![[Wukong_fig2_page3.png|800]]

> 图2：Wukong 的整体架构。主体是堆叠了 $l$ 个交互层（Interaction Layer），每层包含并行的 FMB（Factorization Machine Block）和 LCB（Linear Compress Block），通过残差连接和 Layer Normalization 串联。

#### 各模块详细说明

**模块1：Embedding 层**

Wukong 的 Embedding 层生成统一维度 $d$ 的 Embedding 向量。对于重要的类别特征，分配多个 Embedding（增加表达能力）；对于不太重要的类别特征，分配较小的底层维度，然后将多个小维度 Embedding 拼接并通过 MLP 转换为 $d$ 维。对于 Dense 特征，通过 MLP 映射为 $d$ 维 Embedding。所有 Embedding 拼接后得到初始表示矩阵 $X_0 \in \mathbb{R}^{n \times d}$，其中 $n$ 是 Embedding 的数量。

**模块2：Factorization Machine Block（FMB）**

FMB 是交互层的核心，包含一个 FM 模块和一个 MLP。FM 模块计算输入 Embedding 之间的显式成对交互，输出一个 2D 交互矩阵（每个元素表示一对 Embedding 之间的交互强度）。交互矩阵被展平后经过 MLP，输出被 reshape 为 $n_F$ 个 $d$ 维的 Embedding：

$$\text{FMB}(X_i) = \text{reshape}(\text{MLP}(\text{LN}(\text{flatten}(\text{FM}(X_i)))))$$

FMB 的关键作用是**交互阶数倍增**：如果输入包含最高 $k$ 阶交互，经过 FMB 后输出包含最高 $2k$ 阶交互。

**模块3：Linear Compress Block（LCB）**

LCB 通过线性变换 $W_L \in \mathbb{R}^{n_L \times n_i}$ 对输入 Embedding 进行线性重组，输出 $n_L$ 个 Embedding。LCB 不增加交互阶数，仅传递线性信息。其关键作用是确保第 $i$ 层的交互阶数精确为 $1$ 到 $2^i$——FMB 负责交互阶数的倍增，LCB 保持原始阶数不变，两者拼接后确保了所有阶数的覆盖。

**模块4：层间连接**

两个分支的输出拼接后加上残差连接和 Layer Normalization：

$$X_{i+1} = \text{LN}(\text{concat}(\text{FMB}_i(X_i), \text{LCB}_i(X_i)) + X_i)$$

**模块5：FM 的效率优化**

标准 FM 的交互矩阵计算复杂度为 $O(n^2 d)$。论文引入一个可学习的降维矩阵 $Y \in \mathbb{R}^{n \times k}$（$k \ll n$），将交互计算转化为 $XX^T Y$，先计算 $X^T Y$（复杂度 $O(nkd)$）再左乘 $X$，总复杂度从 $O(n^2 d)$ 降至 $O(nkd)$。

### 2.3 协同扩展策略（Synergistic Upscaling）

Wukong 的扩展策略涉及四个可调超参数：

- **层数 $l$**：控制最高交互阶数（$2^l$），是最重要的扩展维度，优先增大
- **$n_F$ 和 $n_L$**：分别控制 FMB 和 LCB 输出的 Embedding 数量，增大它们可以让模型在每一层捕获更多样化的交互模式
- **压缩参数 $k$**：控制 FM 的低秩近似精度
- **MLP 宽度和深度**：控制非线性变换的能力

扩展的核心原则是**先深后宽**：首先增加层数 $l$ 以提升交互阶数，然后增大 $n_F$、$n_L$、$k$ 等宽度参数以增强每层的表达能力。这种协同扩展策略确保了新增参数不会形成冗余，每增加一单位的计算量都能带来可预测的性能增益。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 特征数 | 用途 |
|--------|------|--------|------|
| Criteo | 4500万 | 39 | 公开基准 |
| Avazu | 4000万 | 22 | 公开基准 |
| KDD Cup 2012 | 1.5亿 | 11 | 公开基准 |
| Terabyte | 40亿 | 39 | 大规模公开 |
| MovieLens-1M/25M | 100万/2500万 | - | 推荐基准 |
| Meta 内部数据集 | ~1460亿 | 720 | Scaling Law 验证 |

### 3.2 实验设置

#### 3.2.1 基线方法

- DLRM（Meta 标准推荐模型）
- DCN V2（显式特征交叉 SOTA）
- FinalMLP（双流 MLP 架构）
- DLRM 的多种扩展变体（增大 MLP 宽度/深度）

#### 3.2.2 评估指标

- **NE（Normalized Entropy）**：归一化交叉熵，越低越好
- **AUC**：公开数据集上的标准指标

### 3.3 实验结果与分析

#### 公开数据集结果

在 Criteo、Avazu、KDD Cup 2012、Terabyte、MovieLens-1M、MovieLens-25M 六个公开数据集上，Wukong 在低复杂度（即不做特别的参数扩展）下即优于 DLRM、DCN V2、FinalMLP 等 SOTA 模型，证明了其基础架构设计的有效性。

![[Wukong_fig3_page7.png|800]]

> 图3：公开数据集上的性能对比。Wukong 在多个数据集上均优于现有 SOTA 方法。

#### Scaling Law 验证

![[Wukong_fig4_page8.png|800]]

> 图4：Scaling Law 曲线。随着计算量从 1 GFLOP/example 增加到 100 GFLOP/example，Wukong 的 NE 持续下降，表现出清晰的 Scaling Law 趋势。其他模型在一定规模后出现饱和。

核心结果：随着计算量从 1 GFLOP/example 增加到 100 GFLOP/example（约两个数量级），Wukong 的 NE 持续下降——**计算量每增加四倍，性能提升约 0.1%**，且未出现饱和迹象。而其他模型在计算量增加到一定程度后出现性能饱和或波动。

Wukong 的 Dense 参数从 0.74B 扩展到 17B，性能始终保持单调提升。这是推荐系统领域首次在如此大的参数范围和计算范围内展示出 Scaling Law。

![[Wukong_fig5_page8.png|800]]

> 图5：不同扩展策略的对比。协同扩展策略（先深后宽）优于单一维度的扩展。

### 消融实验

![[Wukong_fig6_page12.png|800]]

> 图6：架构消融实验。移除 FMB 后性能显著下降，证实了 FM 交互的核心价值；移除 LCB 也导致性能下降，说明保持低阶交互信息同样重要。

消融实验验证了各组件的贡献：移除 FMB 后性能显著下降，证实了 FM 交互的核心价值；移除 LCB 也导致性能下降，说明保持低阶交互信息对整体效果同样重要；残差连接和 Layer Normalization 对训练稳定性贡献显著，尤其在深层网络中。

![[Wukong_fig7_page12.png|800]]

> 图7：不同模型在 Scaling 过程中的性能变化趋势对比。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文建议探索更高效的 FM 计算方法、与序列建模的结合、以及在更多推荐场景中验证 Scaling Law 的普适性。

### 4.2 基于分析的未来方向

1. **方向1：硬件感知的 Scaling 优化**
   - 动机：Wukong 的 FLOPs 增长速度快于参数增长，在 GPU 上的 MFU 可能不高
   - 可能的方法：结合 RankMixer 的思路，优化 FM 计算的 GPU 利用率
   - 预期成果：在相同延迟约束下部署更大的模型
   - 挑战：FM 的交互矩阵计算天然是 memory-bound 的

2. **方向2：与序列建模的统一**
   - 动机：Wukong 主要处理非序列特征的交互，未涉及用户行为序列
   - 可能的方法：将序列 token 纳入 FM 的交互计算（类似 OneTrans 的统一思路）
   - 预期成果：单一架构同时处理特征交叉和序列建模
   - 挑战：序列 token 数量大，FM 的二次复杂度可能成为瓶颈

3. **方向3：Sparse MoE 扩展**
   - 动机：17B Dense 参数的推理成本极高
   - 可能的方法：将 FMB 中的 MLP 替换为 Sparse MoE
   - 预期成果：在保持模型容量的同时降低推理成本

### 4.3 改进建议

1. **改进1：自适应层数**
   - 当前问题：所有样本使用相同深度的网络
   - 改进方案：引入 early-exit 机制，简单样本在浅层即可输出
   - 预期效果：降低平均推理成本

2. **改进2：交互矩阵的稀疏化**
   - 当前问题：FM 计算所有特征对的交互，但很多交互可能无意义
   - 改进方案：学习稀疏的交互掩码，只计算有意义的特征对
   - 预期效果：减少计算量同时可能提升泛化

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - Wukong 首次在推荐系统领域建立了可验证的 Scaling Law，这是一个范式级别的突破。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | 首次在推荐系统领域建立 Scaling Law，堆叠 FM 的架构设计既简洁又深刻，"交互阶数指数级增长"作为扩展轴的洞察非常精彩 |
| 技术质量 | 9/10 | 从 FM 到堆叠 FM，从低秩近似到协同扩展策略，技术链条完整且逻辑严密 |
| 实验充分性 | 8/10 | 六个公开数据集 + 超大规模内部数据集的 Scaling 实验非常充分，但缺少在线 A/B 测试结果 |
| 写作质量 | 8/10 | 论文结构清晰，对 Scaling Law 动机的阐述充分，部分细节可以更详尽 |
| 实用性 | 9/10 | 为推荐系统的"大模型化"方向提供了重要的理论基础和实践指引 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- 堆叠 FM 的交互阶数指数级增长——为推荐模型提供了天然的"深度扩展轴"
- 协同扩展策略的"先深后宽"原则——确保新增参数不形成冗余
- FM 的低秩近似——将复杂度从 $O(n^2 d)$ 降至 $O(nkd)$
- FMB + LCB 的互补设计——确保所有阶数的交互都被覆盖

#### 5.2.2 需要深入理解的部分

- 为什么 MLP 不能 Scale 而 FM 能？根本原因是 MLP 增加参数只增强拟合精度，FM 增加层数引入新的交互模式
- 协同扩展策略中各超参数的最优比例如何确定？
- 17B 参数模型的实际推理延迟和部署成本是多少？论文未详细讨论

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[RankMixer_Scaling_Up_Ranking_Models_in_Industrial_Recommenders|RankMixer]] - 字节跳动的硬件感知 Scaling Law 方案，在 FLOPs 维度上优于 Wukong
- [[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans]] - 统一 Transformer 精排架构，将序列建模和特征交叉合并
- [[HSTU_Actions_Speak_Louder_than_Words|HSTU]] - Meta 的生成式推荐 Scaling，侧重序列建模

### 6.2 背景相关
- [[DCN_V2_Improved_Deep_and_Cross_Network|DCN V2]] - 显式特征交叉的代表，Cross Network 交互阶数线性增长
- [[DeepFM_A_Factorization_Machine_based_Neural_Network|DeepFM]] - FM + DNN 的经典组合
- [[FinalMLP_An_Enhanced_Two_Stream_MLP_Model|FinalMLP]] - 双流 MLP 架构

### 6.3 后续工作
- RankMixer（ByteDance, 2025）- 硬件感知的推荐 Scaling Law
- OneTrans（ByteDance/NTU, 2025）- 统一 Transformer 精排
- 推荐系统大模型化的更多探索

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2403.02545)
- [ICML 2024 论文列表](https://icml.cc/virtual/2024/poster/33453)

> [!tip] 关键启示
> 推荐系统的 Scaling Law 不是不存在，而是需要匹配数据特性的架构设计。推荐数据的核心信息蕴含在特征之间的高阶交互中，堆叠 FM 通过指数级增长的交互阶数为模型扩展提供了天然的"深度扩展轴"——这与 LLM 中"更深的 Transformer 能学习更复杂的语言模式"异曲同工。

> [!warning] 注意事项
> - Wukong 的 FLOPs 增长速度快于参数增长，在 AUC vs FLOPs 维度上优势被削弱（RankMixer 论文指出了这一点）
> - 17B Dense 参数的推理成本极高，论文未讨论实际部署方案
> - 内部数据集无法复现，公开数据集规模不足以验证大规模 Scaling

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！Wukong 是推荐系统 Scaling Law 方向的开创性工作，首次证明了推荐模型可以像 LLM 一样"越大越好"。虽然后续的 RankMixer 和 OneTrans 在工程效率上有所超越，但 Wukong 的理论贡献和方向性指引不可替代。
