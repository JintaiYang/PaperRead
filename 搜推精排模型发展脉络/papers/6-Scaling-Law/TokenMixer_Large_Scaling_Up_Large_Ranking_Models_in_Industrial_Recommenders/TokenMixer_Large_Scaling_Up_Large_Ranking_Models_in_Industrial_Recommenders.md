---
title: "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders"
authors: "Yuanhao Zhu, Jieming Zhu, Guohao Cai, Jianrong Zhang, Zhenhua Dong, Ruiming Tang"
venue: "arXiv 2602.06563"
year: 2025
tags:
  - ranking-model
  - scaling-law
  - MoE
  - TokenMixer
  - industrial-recommender
  - ByteDance
url: "https://arxiv.org/abs/2602.06563"
status: "已读"
rating: 5
date_read: 2026-05-08
category: "6-Scaling-Law"
---

# TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders

## 论文基本信息

| 属性 | 内容 |
|------|------|
| 标题 | TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders |
| 作者 | Yuanhao Zhu, Jieming Zhu, Guohao Cai, Jianrong Zhang, Zhenhua Dong, Ruiming Tang (ByteDance) |
| 机构 | 字节跳动 |
| 发表 | arXiv:2602.06563, 2025 |
| 链接 | [arXiv](https://arxiv.org/abs/2602.06563) |

## 一句话总结

TokenMixer-Large 是 RankMixer/TokenMixer 的升级版本，通过 Mixing & Reverting 操作修复残差路径语义对齐问题、引入 Sparse-Pertoken MoE 实现"稀疏训练+稀疏推理"、设计 Token Parallel 分布式策略，成功将推荐精排模型扩展到 15B 参数规模，在抖音电商/广告/直播三大场景取得显著线上收益。

## 研究动机与问题

### 背景

RankMixer（TokenMixer）在推荐精排领域展示了优秀的 Scaling Law 特性，但在进一步扩展到更大规模时面临三个核心问题：

1. **残差路径语义不对齐**：RankMixer 的 mixing 操作改变了 token 数量和语义，导致残差连接中 $F(x') + x$ 的 $x'$ 和 $x$ 语义不一致
2. **原始 token 信息丢失**：如果 40 个 token 被 mix 成 16 个新 token，原始 40 个 token 的语义信息无法完整传递到后续层
3. **训练效率与推理效率的平衡**：RankMixer 采用"dense train, sparse infer"策略，而更大规模模型需要"sparse train, sparse infer"来降低训练成本

### 核心挑战

- 如何在保持 TokenMixer 高效 mixing 能力的同时修复残差路径问题？
- 如何设计适合 Pertoken 架构的 MoE 方案？
- 如何在数百 GPU 上高效训练/推理 7B-15B 规模的推荐模型？

## 方法详解

### 整体架构

![[images/tokenmixer-large-new3.pdf]]

TokenMixer-Large 的核心架构由以下模块组成：

1. **Tokenization**：将异构特征转化为统一的 token 表示
2. **Mixing & Reverting**：无参数的 token 混合与还原操作
3. **Pertoken SwiGLU / Sparse-Pertoken MoE**：每个 token 独立的前馈网络
4. **Inter-layer Residual + Auxiliary Loss**：深层模型训练稳定性保障
5. **Token Parallel**：面向 Pertoken 专家的模型并行策略

### 2.1 Mixing & Reverting 操作

这是 TokenMixer-Large 相对于 RankMixer 最核心的改进。

**问题分析**：RankMixer 中 mixing 操作将 $T$ 个 token 混合为 $T'$ 个新 token（$T' < T$），然后直接在混合后的 token 上做残差连接。这导致三个问题：

| 属性 | Group Transformer | RankMixer | TokenMixer-Large |
|------|:-:|:-:|:-:|
| Standard Residual (SR) | ✓ | ✓ | ✓ |
| Original Token Residual (OTR) | ✓ | ✗ | ✓ |
| Token Semantic Alignment (TSA) | ✓ | ✗ | ✓ |

- **SR (Standard Residual)**：block 之间是否存在标准残差连接
- **OTR (Original Token Residual)**：原始 token 的语义信息是否能传播到后续层
- **TSA (Token Semantic Alignment in Residual)**：残差操作 $F(x') + x$ 中 token 语义是否一致

**解决方案**：TokenMixer-Large 引入 **Reverting** 操作——在 mixing 之后、残差连接之前，将混合后的 token 还原回原始 token 维度，确保残差路径上的语义对齐。

具体流程：
```
Input tokens (T个) → Split → Concat (Mixing, 得到 T'个新token) 
→ Pertoken SwiGLU → Split → Concat (Reverting, 还原为 T个token) 
→ Residual Add (与原始 T个token 对齐)
```

**关键发现**：不同的 split-concat 策略（垂直分割、对角分割、随机分割）对性能没有影响，**只要每个新混合的 token 包含所有原始 token 的信息**。仅混合一半原始 token 信息会导致 -0.08% AUC 下降。

### 2.2 Pertoken SwiGLU

将 RankMixer 中的 Pertoken FFN 升级为 Pertoken SwiGLU：

$$\text{SwiGLU}(x) = \text{FC}_{\text{down}}(\text{FC}_{\text{up}}(x) \odot \text{Swish}(\text{FC}_{\text{gate}}(x)))$$

每个 token 位置拥有独立的 $\text{FC}_{\text{up}}$、$\text{FC}_{\text{gate}}$、$\text{FC}_{\text{down}}$ 参数矩阵。

**消融实验**：
- Pertoken SwiGLU → SwiGLU（共享参数）：-0.21% AUC
- Pertoken SwiGLU → Pertoken FFN：-0.10% AUC

### 2.3 Inter-layer Residual & Auxiliary Loss

![[images/interRes.pdf]]

为解决深层模型训练不稳定问题，引入两个机制：

**Inter-layer Residual（层间残差）**：每隔若干层添加一个跨层残差连接，将浅层的输出直接加到深层的输入上，缓解梯度消失。

**Auxiliary Loss（辅助损失）**：在中间层添加辅助预测头，提供额外的梯度信号，帮助深层网络收敛。

消融结果：移除 Inter-Residual & AuxLoss 导致 -0.04% AUC 下降。

### 2.4 Sparse-Pertoken MoE

![[images/fisrt_enlarge_then_sparse.pdf]]

这是 TokenMixer-Large 实现高效扩展的核心技术。

#### 设计哲学："First Enlarge, Then Sparse"

与传统 MoE 直接增加专家数量不同，TokenMixer-Large 采用：
1. 先设计一个性能最优的 dense 模型
2. 将 SwiGLU 中的大 FC 矩阵拆分为多个细粒度的小矩阵（专家）
3. 通过稀疏激活实现效率提升

这种方法类似 DeepSeek-MoE 的细粒度专家设计，但关键区别是**不增加激活参数数量（topk）**。

#### Pertoken 特性

Sparse-Pertoken MoE 的核心创新：每个 token 位置的专家是**不共享**的。这等价于给标准 MoE 一个 routing prior——每个 token 能激活的专家集合是预先确定的，避免了 router 在训练初期难以学习的问题。

对比实验：Sparse-Pertoken MoE → Sparse MoE（标准）：-0.10% AUC

#### Shared Expert

引入一个始终激活的共享专家，捕获所有 token 的通用模式：

$$y = \alpha \cdot \sum_{i=1}^{K-1} g(x)_i \cdot \text{Expert}_i(x) + \text{SharedExpert}(x)$$

移除 Shared Expert：-0.02% AUC

#### Gate Value Scaling

在 router 输出上乘以缩放因子 $\alpha$，放大被选中专家的梯度更新：

$$y = \alpha \cdot \sum_{i=1}^{K-1} g(x)_i \cdot \text{Expert}_i(x) + \text{SharedExpert}(x)$$

**关键发现**：最优 $\alpha$ 与稀疏率成**反比**关系：

| 稀疏率 | 最优 $\alpha$ | AUC 变化 |
|--------|:---:|:---:|
| 1:2 | 2 | -0.00% (vs dense) |
| 1:4 | 4 | -0.03% (vs dense) |

**原理解释**："First Enlarge, Then Sparse" 将大 kernel 拆分为多个小 kernel，每个专家被激活的概率降低。Gate Value Scaling 通过放大激活时的梯度来补偿这一点。

移除 Gate Value Scaling：-0.03% AUC

#### Down-Matrix Small Init

![[images/smallInit.pdf]]

将 SwiGLU 中最后一层 $\text{FC}_{\text{down}}$ 的初始化 scale 设为 0.01（$\text{FC}_{\text{up}}/\text{FC}_{\text{gate}}$ 保持 1.0）：

$$W \sim \mathcal{N}\left(0, \frac{2 \cdot \text{scale}}{n_{\text{in}} + n_{\text{out}}}\right)$$

| 版本 | Init Value [up, gate, down] | ΔAUC |
|------|:---:|:---:|
| Base | [1, 1, 1] | -- |
| SmallInit-001 | [1, 1, 0.01] | **+0.03%** |
| SmallInit-01 | [1, 1, 0.1] | +0.02% |
| SmallInit-001-All | [0.01, 0.01, 0.01] | -0.10% |
| SmallInit-001-Reverse | [0.01, 0.01, 1] | -0.01% |

**原理**：使早期训练阶段 $F(x) + x ≈ x$（近似恒等映射），同时约束 SwiGLU 中间层的激活幅度，增强深层模型训练稳定性。灵感来自 ReZero。

移除 Down-Matrix Small Init：-0.03% AUC

#### 稀疏率选择

当前线上部署选择 **1:2 稀疏率**（ROI 最高）：
- 1:2 稀疏：离线/在线效果与 dense 模型几乎无损
- 1:4 稀疏：有轻微下降
- 更高稀疏率（1:8+）：负载均衡恶化，仍在探索中

### 2.5 Token Parallel

![[images/operators_workflow.pdf]]

针对 Pertoken 专家设计的模型并行策略。

**核心思想**：由于每个 token 位置有独立的专家参数，可以将不同 token 的计算分配到不同的 GPU 上，实现 token 维度的并行。

**与传统并行的区别**：
- Data Parallel：切分 batch 维度
- Tensor Parallel：切分 hidden 维度
- Expert Parallel（传统 MoE）：切分 expert 维度
- **Token Parallel**：切分 token 维度，每个 GPU 负责一部分 token 的完整计算

这种策略特别适合 Pertoken 架构，因为不同 token 的参数本身就是独立的，无需 All-to-All 通信。

### 2.6 Pure Model Design

随着 TokenMixer-Large 参数规模扩大，各种小型碎片化算子（DCN、LHUC 等）的增益逐渐被主干网络吸收：

| 参数规模 | DCN 增益 |
|---------|:---:|
| 150M | +0.09% |
| 500M | +0.04% |
| 700M | +0.00% |

最终 TokenMixer-Large **仅包含**无参数的 mixing/reverting 操作和大量 GroupedGemm 操作，MFU 高达 **60%**。

### 2.7 Normalization

- 使用 RMSNorm 替代 LayerNorm（去除所有 bias），端到端吞吐量提升 **8.4%**
- 采用 Pre-Norm（稳定训练，Post-Norm 虽效果更好但容易梯度爆炸）

### 2.8 FP8 量化

TokenMixer-Large 支持 FP8 训练/推理，进一步提升计算效率。

## 实验结果

### 离线实验设置

- **数据集**：抖音电商主 feed 真实训练数据，500+ 特征，数亿用户，采样后约 4 亿条/天，训练周期 2 年
- **广告场景**：采样后 3 亿条/天
- **直播场景**：采样后 170 亿条/天
- **硬件**：电商 64 GPU，广告/直播 256 GPU
- **优化器**：Adagrad（dense lr=0.01, sparse lr=0.05）

### 主实验结果（电商场景，~500M 参数对比）

| 模型 | CTCVR ΔAUC | Params | FLOPs/Batch |
|------|:---:|:---:|:---:|
| DLRM-MLP-500M | baseline | 499M | 125.1T |
| HiFormer | +0.44% | 570M | 28.8T |
| DCNv2 | +0.49% | 502M | 125.8T |
| DHEN | +0.63% | 415M | 103.4T |
| AutoInt | +0.75% | 549M | 138.6T |
| Wukong | +0.76% | 513M | 4.6T |
| Group Transformer | +0.81% | 550M | 4.5T |
| FAT | +0.82% | 551M | 4.59T |
| RankMixer | +0.84% | 567M | 4.6T |
| **TokenMixer-Large 500M** | **+0.94%** | 501M | 4.2T |
| **TokenMixer-Large 4B** | **+1.14%** | 4.6B | 29.8T |
| **TokenMixer-Large 7B** | **+1.20%** | 7.6B | 49.0T |
| **TokenMixer-Large 4B SP-MoE** | **+1.14%** | 2.3B in 4.6B | 15.1T |

**关键发现**：
- 同等 500M 参数下，TokenMixer-Large 比 RankMixer 高 +0.10% AUC，且 FLOPs 更低（4.2T vs 4.6T）
- 4B SP-MoE（稀疏）与 4B Dense 效果持平，但 FLOPs 减半（15.1T vs 29.8T）
- 7B 模型达到 +1.20% AUC，展示了持续的 scaling 收益

### Scaling Law

![[images/bestlines.png]]

TokenMixer-Large 在参数量/FLOPs 两个维度上都展示了比 RankMixer 更陡峭的 scaling 曲线。

**跨场景 Scaling Law 验证**：

![[images/15B_new.png]]
![[images/7B_new.png]]
![[images/4B_new.png]]

- Feed Ads：成功扩展到 **15B**（离线）/ **7B**（线上）
- E-Commerce：成功扩展到 **7B**（离线）/ **4B**（线上）
- Live Streaming：成功扩展到 **4B**（离线）/ **2B**（线上）

**模型收敛与数据量关系**（直播场景）：

| 参数规模 | 收敛所需天数 | ΔUAUC |
|---------|:---:|:---:|
| 30M → 90M | 14天 | +0.94% |
| 90M → 500M | 30天 | +0.62% |
| 500M → 2.3B (30d) | 30天 | +0.41% |
| 500M → 2.3B (60d) | 60天 | +0.70% |

**关键发现**：
- 超过 1B 参数后，需要**均衡扩展**宽度 $D$、深度 $L$、缩放因子 $N$，单一维度扩展会遇到瓶颈
- 更大模型需要更多数据才能充分收敛

### 消融实验

#### TokenMixer-Large Block 消融（4B 模型）

| 设置 | ΔAUC |
|------|:---:|
| w/o Global Token | -0.02% |
| w/o Mixing & Reverting | **-0.27%** |
| w/o Residual | -0.15% |
| w/o Internal Residual & AuxLoss | -0.04% |
| Pertoken SwiGLU → SwiGLU | **-0.21%** |
| Pertoken SwiGLU → Pertoken FFN | -0.10% |

**结论**：Mixing & Reverting 和 Pertoken SwiGLU 是最关键的两个组件。

#### Sparse-Pertoken MoE 消融

| 设置 | ΔAUC | ΔParams | ΔFLOPs |
|------|:---:|:---:|:---:|
| w/o Shared Expert | -0.02% | 0% | 0% |
| w/o Gate Value Scaling | -0.03% | 0% | 0% |
| w/o Down-Matrix Small Init | -0.03% | 0% | 0% |
| Sparse-Pertoken MoE → Sparse MoE | **-0.10%** | 0% | 0% |

**结论**：所有 MoE 改进都是"免费"的（不增加参数和计算量），Pertoken 特性贡献最大。

### 在线实验结果

| 场景 | ΔAUC/ΔUAUC | 核心业务指标 |
|------|:---:|:---:|
| 抖音电商 | +0.51% AUC | **订单 +1.66%，人均预览支付 GMV +2.98%** |
| Feed 广告 | +0.35% AUC | **ADSS +2.0%** |
| 直播 | +0.7% UAUC | **收入 +1.4%** |

线上 baseline 分别为 RankMixer-1B（广告）、RankMixer-150M（电商）、RankMixer-500M（直播），对应扩展到 TokenMixer-Large 7B、4B、2B。

### RankMixer 在 Feed 推荐的长期 AB 实验

RankMixer-1B 在抖音主 feed 推荐的长期实验结果（8 个月观察，收益仍未收敛）：

| 用户群 | Active Day | Duration | Like | Finish |
|--------|:---:|:---:|:---:|:---:|
| Overall | +0.29% | +1.08% | +2.39% | +1.99% |
| Low-active | **+1.74%** | +3.64% | +8.16% | +4.54% |
| Middle-active | +0.71% | +1.53% | +2.58% | +2.51% |
| High-active | +0.14% | +0.63% | +1.83% | +1.49% |

**关键发现**：低活跃用户获益最大（Active Day +1.74%），体现了模型强大的泛化能力。

## 负载均衡分析

![[images/loadBalance1-2-2.png]]
![[images/loadBalance1-8-2.png]]

- 1:2 稀疏率下负载均衡良好
- 1:8 稀疏率下负载均衡有所恶化
- 尝试了 Switch Transformer 的负载均衡方法和 Z-loss，有一定效果
- 当前线上 1:2 版本无负载均衡问题

## 与 RankMixer 的关系

TokenMixer-Large 是 RankMixer（TokenMixer）的直接升级版本：

| 维度 | RankMixer | TokenMixer-Large |
|------|-----------|-----------------|
| Token 混合 | Mixing only | **Mixing & Reverting** |
| 残差语义对齐 | ✗ | ✓ |
| 原始 token 保留 | ✗ | ✓ |
| FFN 类型 | Pertoken FFN | **Pertoken SwiGLU** |
| 稀疏策略 | Dense train, Sparse infer | **Sparse train, Sparse infer** |
| MoE 类型 | N/A | **Sparse-Pertoken MoE** |
| 并行策略 | Data Parallel | **Token Parallel** |
| 最大规模 | 1B | **15B** |
| 碎片算子 | 保留 DCN 等 | **Pure Model（移除所有碎片算子）** |
| MFU | - | **60%** |

## 核心创新点

1. **Mixing & Reverting**：通过"混合-计算-还原"三步操作，在保持 token mixing 高效性的同时修复了残差路径的语义对齐问题
2. **Sparse-Pertoken MoE**："First Enlarge, Then Sparse" 策略 + Pertoken routing prior，实现稀疏训练+稀疏推理
3. **Gate Value Scaling**：发现缩放因子与稀疏率的反比关系，零成本提升 MoE 性能
4. **Down-Matrix Small Init**：仅初始化 $\text{FC}_{\text{down}}$ 为小值，显著改善深层模型收敛
5. **Token Parallel**：面向 Pertoken 专家的新型模型并行策略
6. **Pure Model Design**：随着模型规模扩大，移除所有碎片化算子，MFU 达 60%

## 个人思考与评价

### 优点

1. **工程与理论结合紧密**：每个设计决策都有清晰的理论分析和充分的消融实验支撑
2. **规模验证充分**：在三个不同业务场景（电商/广告/直播）都验证了 scaling law，且线上部署规模达到 15B
3. **"First Enlarge, Then Sparse" 思路新颖**：不同于传统 MoE 直接增加专家，而是先设计最优 dense 模型再稀疏化
4. **Pertoken MoE 的 routing prior 设计巧妙**：避免了 router 学习困难的问题，且是零成本改进
5. **Pure Model 理念前瞻**：随着主干网络能力增强，碎片化算子的增益被吸收，简化架构反而提升效率

### 局限性

1. **高稀疏率（>1:4）仍有挑战**：负载均衡恶化，性能有损失
2. **数据需求巨大**：2.3B 模型需要 60 天数据才能充分收敛
3. **Token Parallel 的通信开销**：论文未详细讨论跨 GPU 通信的具体实现和开销
4. **Mixing 策略的理论解释不够深入**：为什么"包含所有原始 token 信息"是充分条件？

### 对推荐系统领域的启示

1. **推荐模型的 Scaling Law 已被充分验证**：从 30M 到 15B，持续有收益
2. **"Pure Model" 是大模型时代的趋势**：小规模时需要各种 trick，大规模时简洁架构反而更优
3. **Pertoken 设计是推荐模型的独特优势**：利用特征异构性，为每个 token 分配独立参数
4. **稀疏化是推荐大模型落地的关键**：1:2 稀疏可以无损，大幅降低推理成本

## 相关论文

- [[RankMixer_Exploring_the_Scaling_Laws_of_Ranking_Models_in_Recommendation]] - 前作，TokenMixer 的原始版本
- [[FAT_Scaling_Laws_of_Feature_Interactions_in_Recommendation]] - 同期工作，基于正交基组合的 Group Transformer 升级
- DeepSeek-MoE - 细粒度专家设计的灵感来源
- ReZero - Small Init 的理论基础
- Switch Transformer - MoE 负载均衡参考

## 引用

```bibtex
@article{zhu2025tokenmixerlarge,
  title={TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders},
  author={Zhu, Yuanhao and Zhu, Jieming and Cai, Guohao and Zhang, Jianrong and Dong, Zhenhua and Tang, Ruiming},
  journal={arXiv preprint arXiv:2602.06563},
  year={2025}
}
```
