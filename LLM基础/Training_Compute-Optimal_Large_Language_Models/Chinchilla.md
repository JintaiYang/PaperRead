---
paper_id: "[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)"
title: "Training Compute-Optimal Large Language Models"
authors: "Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre"
institution: "DeepMind"
publication: "arXiv preprint 2022-03-29"
tags:
  - Scaling-Laws
  - 大语言模型
  - Compute-Optimal-Training
  - Chinchilla
  - 训练效率
quality_score: "8.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2203.15556)"
date: "2022-03-29"
---

## 一、研究背景与动机

### 1.1 领域现状

2020-2022年间，大语言模型（LLM）经历了参数规模的快速膨胀。GPT-3（175B）、Jurassic-1（178B）、Gopher（280B）、Megatron-Turing NLG（530B）等模型相继问世，最大的稠密模型已超过 5000 亿参数。这些大规模自回归 Transformer 在零样本、少样本和微调等多种评估协议下展现了出色的性能。

然而，一个被广泛忽视的问题是：这些模型几乎都是在大约 300B tokens 上训练的。这源于 Kaplan et al.（2020）提出的 Scaling Law 的一个推论——当计算预算增加 10 倍时，模型参数应增加约 5.5 倍，而训练 tokens 只需增加 1.8 倍。这一结论导致了业界"优先增大模型、保持数据量不变"的惯性做法。

### 1.2 现有方法的局限性

本文指出 Kaplan et al.（2020）的分析存在两个关键方法论缺陷：

第一，Kaplan 等人对不同规模的模型使用了固定的训练 tokens 和固定的学习率调度策略。这意味着对于训练步数较少的模型，学习率调度（cosine schedule 到 130B tokens）会导致中间 loss 被高估，从而低估了"在较少数据上训练模型"的有效性。这最终偏向了"增大模型规模优于增加数据量"的结论。

第二，Kaplan 等人使用的模型多数小于 1 亿参数，而本文发现在大模型区间存在轻微的曲率（curvature），使得从极小模型外推的结论与大模型的实际表现存在偏差。

### 1.3 本文解决方案概述

本文重新审视了"给定固定计算预算，如何在模型大小和训练 tokens 数量之间做最优权衡"这一问题。通过训练超过 400 个模型（参数量从 70M 到 16B，训练 tokens 从 5B 到 500B），作者提出了三种不同的方法来估计最优分配策略，并得出了一致的结论：模型参数和训练 tokens 应以近似相等的比例增长——即计算预算每翻倍，模型大小和训练 tokens 都应翻倍。

基于这一发现，作者训练了 Chinchilla（70B 参数，1.4T tokens），使用与 Gopher（280B）相同的计算预算，在多数下游任务上取得了更优的表现。

![[combined_predictions_v9.png|800]]
> 图1：三种方法的预测叠加图。本文三种方法预测了最优模型大小和训练 tokens 数量，与 Kaplan et al.（2020）的预测形成对比。三种方法均预测当前大模型应当更小、训练更长时间。右侧展示了 Chinchilla 相比 Gopher 和其他大模型的性能优势。

## 二、解决方案

### 2.1 核心思想

本文的核心洞察可以概括为一句话：**当前的大语言模型在给定计算预算下是"欠训练"的**——它们的参数量过大而训练数据过少。最优的计算资源分配策略应当让模型大小和训练数据量以近似相等的速率增长。

形式化地，给定计算预算 $C$，目标是找到最优的参数量 $N$ 和训练 tokens 数 $D$：

$$N_{opt}(C), D_{opt}(C) = \arg\min_{N, D \text{ s.t. } \text{FLOPs}(N, D) = C} L(N, D)$$

其中 $L(N, D)$ 是训练损失（作为模型参数量和训练 tokens 的函数），计算量约束为 $\text{FLOPs}(N,D) \approx 6ND$。

### 2.2 整体架构

论文提出三种互补的方法来估计 $N_{opt}(C)$ 和 $D_{opt}(C)$，并验证它们给出一致的结论。

#### 方法 1：固定模型大小，变化训练 tokens

对一组固定规模的模型（70M 到 10B 参数），分别以 4 种不同的训练 token 数量（通过 cosine schedule 长度控制）进行训练。从这些训练曲线中提取"给定 FLOPs 下的最低 loss"组成的包络线。然后在 1500 个对数均匀分布的 FLOP 值上，找到使 loss 最低的模型规模和对应的训练 tokens。最后用幂律拟合 $N_{opt} \propto C^a$ 和 $D_{opt} \propto C^b$。

结果：$a = 0.50$，$b = 0.50$。

![[scaling_11.png|800]]
> 图2：训练曲线包络线方法。左图展示不同规模模型的训练曲线；中图为给定计算预算下的最优模型大小；右图为最优训练 tokens 数。绿色标记为 Gopher 计算预算对应的预测值。

#### 方法 2：IsoFLOP 曲线

固定 9 个不同的训练 FLOP 预算（从 $6\times10^{18}$ 到 $3\times10^{21}$），对每个预算训练不同大小的模型（最大到 16B），观察最终训练 loss。对每条 IsoFLOP 曲线拟合抛物线，找到 loss 最低点对应的模型大小。

结果：$a = 0.49$，$b = 0.51$。

![[isoflop_7.png|800]]
> 图3：IsoFLOP 曲线。对不同规模的模型，固定总 FLOPs 后绘制 loss 与参数量的关系。左图展示了每个 FLOP 预算下存在一个明确的"loss 谷底"（最优模型大小）；中、右图为最优参数和 tokens 的幂律拟合。

#### 方法 3：参数化损失函数拟合

将所有实验的最终 loss 建模为参数量和 tokens 的参数化函数。基于经典的风险分解，提出如下函数形式：

$$\hat{L}(N,D) \triangleq E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中：
- $E$：数据分布上的理想生成过程的损失（对应自然文本的熵），即"不可约误差"
- $\frac{A}{N^\alpha}$：函数逼近项，反映 $N$ 参数的 transformer 相比理想模型的差距
- $\frac{B}{D^\beta}$：优化不充分项，反映有限训练步数（仅在 $D$ 个 token 上训练一个 epoch）带来的次优性

**参数估计方法**：通过最小化 Huber 损失进行拟合：

$$\min_{A, B, E, \alpha, \beta} \sum_{\text{Run } i} \text{Huber}_\delta \Big(\log \hat{L}(N_i, D_i) - \log L_i\Big)$$

使用 L-BFGS 算法在初始化网格上搜索局部最优。Huber 损失（$\delta=10^{-3}$）对异常值具有鲁棒性。

**最优前沿的闭式解**：在约束 $\text{FLOPs}(N,D) \approx 6ND$ 下最小化 $\hat{L}$，得到幂律形式的解：

$$N_{opt}(C) = G \left(\frac{C}{6}\right)^a, \quad D_{opt}(C) = G^{-1} \left(\frac{C}{6}\right)^b$$

其中 $G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}$，$a = \frac{\beta}{\alpha+\beta}$，$b = \frac{\alpha}{\alpha+\beta}$。

拟合结果：$E=1.69$，$A=406.4$，$B=410.7$，$\alpha=0.34$，$\beta=0.28$，对应 $a=0.46$，$b=0.54$。

![[approach_3_v2.png|800]]
> 图4：参数化拟合方法。左图为 $\hat{L}(N,D)$ 的等高线图，蓝色线为高效计算前沿（在对数空间中为一条直线）；右图为不同 IsoFLOP 切片上拟合值与实际值的对比。

### 2.3 三种方法的对比总结

三种方法虽然采用不同的拟合方法和模型子集，但给出了高度一致的预测：

| 方法 | $a$（$N_{opt} \propto C^a$） | $b$（$D_{opt} \propto C^b$） |
|------|------|------|
| 方法1：训练曲线包络线 | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 方法2：IsoFLOP 曲线 | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 方法3：参数化损失拟合 | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

本文的三种方法均预测 $a \approx b \approx 0.5$，即参数量和训练 tokens 应当等比例增长；而 Kaplan et al. 预测应将大部分增量计算分配给增大模型（$a=0.73$），这与本文结论形成鲜明对比。

### 2.4 Chinchilla 模型

基于以上分析，Gopher（280B 参数，300B tokens）的计算预算对应的最优模型应当在 40-70B 参数之间。作者训练了 Chinchilla：70B 参数，1.4T tokens，与 Gopher 使用相同的计算预算。

Chinchilla 与 Gopher 的主要差异：

| 配置 | Gopher 280B | Chinchilla 70B |
|------|------|------|
| 层数 | 80 | 80 |
| 注意力头数 | 128 | 64 |
| Key/Value 维度 | 128 | 128 |
| d_model | 16,384 | 8,192 |
| 最大学习率 | $4\times10^{-5}$ | $1\times10^{-4}$ |
| Batch Size | 3M → 6M | 1.5M → 3M |
| 优化器 | Adam | AdamW |
| 训练 tokens | 300B | 1.4T |

由于 Chinchilla 比 Gopher 小 4 倍，其推理和微调的计算成本也相应降低，这为下游应用带来了实际的效率优势。

## 三、实验结果

### 3.1 数据集

Chinchilla 在 MassiveText 数据集上训练，采样分布如下：

| 子集 | 磁盘大小 | 文档数 | 采样比例 | 1.4T tokens 中的 epoch 数 |
|------|------|------|------|------|
| MassiveWeb | 1.9 TB | 604M | 45% | 1.24 |
| Books | 2.1 TB | 4M | 30% | 0.75 |
| C4 | 0.75 TB | 361M | 10% | 0.77 |
| News | 2.7 TB | 1.1B | 10% | 0.21 |
| GitHub | 3.1 TB | 142M | 4% | 0.13 |
| Wikipedia | 0.001 TB | 6M | 1% | 3.40 |

值得注意的是，MassiveWeb 和 Wikipedia 子集训练时超过了一个 epoch。

### 3.2 实验设置

#### 3.2.1 基线方法

- Gopher（280B）：DeepMind 之前的旗舰模型
- GPT-3（175B）：OpenAI 的大语言模型
- Jurassic-1（178B）：AI21 Labs 的模型
- Megatron-Turing NLG（530B，MT-NLG）：NVIDIA/Microsoft 的超大模型

#### 3.2.2 评估指标

评估覆盖 6 大类共 150+ 个任务：

| 类别 | 任务数 | 示例 |
|------|------|------|
| 语言建模 | 20 | WikiText-103, The Pile 各子集 |
| 阅读理解 | 3 | RACE-m, RACE-h, LAMBADA |
| 问答 | 3 | Natural Questions, TriviaQA, TruthfulQA |
| 常识推理 | 5 | HellaSwag, Winogrande, PIQA, SIQA, BoolQ |
| MMLU | 57 | 高中化学、天文学、临床知识等 |
| BIG-bench | 62 | 因果判断、时序推理等 |

#### 3.2.3 训练细节

- 硬件：TPUv3/TPUv4
- 框架：JAX + Haiku
- 优化器：AdamW（$\beta_1=0.9$, $\beta_2=0.95$）
- 精度：前向/反向用 bfloat16，优化器状态保存 float32 权重副本
- Tokenizer：SentencePiece（32K 词表，不使用 NFKC 归一化）
- Batch size 在训练中期翻倍（1.5M → 3M tokens）

### 3.3 实验结果与分析

#### 语言建模

Chinchilla 在 The Pile 的所有子集上均优于 Gopher，如下图所示：

![[chinchilla_pile_3.png|800]]
> 图5：The Pile 评估。展示 Chinchilla 相对 Gopher 在各子集上的 bits-per-byte 改善（越低越好）。Chinchilla 在所有子集上均优于 Gopher。

在 Wikitext-103 上，Chinchilla 取得了 7.16 的困惑度（Gopher 为 7.75）。

#### MMLU

| 模型/基准 | 准确率 |
|------|------|
| 随机 | 25.0% |
| 平均人类评估者 | 34.5% |
| GPT-3 5-shot | 43.9% |
| Gopher 5-shot | 60.0% |
| **Chinchilla 5-shot** | **67.6%** |
| 平均人类专家 | 89.8% |

Chinchilla 在 MMLU 上相比 Gopher 提升了 7.6 个百分点，超越了竞赛预测者对 2023 年 6 月 SOTA 的预估（63.4%）。在 57 个子任务中，Chinchilla 在 51 个上优于 Gopher，2 个持平，仅 4 个稍逊。

![[mmlu_0.png|800]]
> 图6：MMLU 各子任务对比。Chinchilla 在绝大多数任务上优于 Gopher。

#### BIG-bench

Chinchilla 在 BIG-bench 上的平均准确率为 65.1%，相比 Gopher 的 54.4% 提升了 10.7 个百分点。62 个任务中仅 4 个表现略逊于 Gopher。

![[bigbench_2.png|800]]
> 图7：BIG-bench 各任务对比。Chinchilla 在绝大多数 BIG-bench 任务上优于 Gopher。

#### 阅读理解

| 任务 | Chinchilla | Gopher | GPT-3 | MT-NLG |
|------|------|------|------|------|
| LAMBADA Zero-Shot | **77.4** | 74.5 | 76.2 | 76.6 |
| RACE-m Few-Shot | **86.8** | 75.1 | 58.1 | -- |
| RACE-h Few-Shot | **82.3** | 71.6 | 46.8 | 47.9 |

Chinchilla 在 RACE-h 和 RACE-m 上分别比 Gopher 提升超过 10 个百分点。

#### 常识推理

| 任务 | Chinchilla | Gopher | GPT-3 | MT-NLG |
|------|------|------|------|------|
| HellaSwag | **80.8%** | 79.2% | 78.9% | 80.2% |
| PIQA | 81.8% | 81.8% | 81.0% | **82.0%** |
| Winogrande | **74.9%** | 70.1% | 70.2% | 73.0% |
| SIQA | **51.3%** | 50.6% | -- | -- |
| BoolQ | **83.7%** | 79.3% | 60.5% | 78.2% |

Chinchilla 在 5 个常识任务上全面超越 Gopher 和 GPT-3，在 4/5 个任务上优于 530B 参数的 MT-NLG。

#### 闭卷问答

| 数据集 | 方式 | Chinchilla | Gopher | GPT-3 |
|------|------|------|------|------|
| Natural Questions | 0-shot | 16.6% | 10.1% | 14.6% |
| Natural Questions | 5-shot | 31.5% | 24.5% | -- |
| Natural Questions | 64-shot | 35.5% | 28.2% | 29.9% |
| TriviaQA (unfiltered) | 0-shot | 67.0% | 52.8% | 64.3% |
| TriviaQA (unfiltered) | 64-shot | 72.3% | 61.3% | 71.2% |

Chinchilla 在 Natural Questions 上达到了新的闭卷 SOTA。

#### TruthfulQA

Chinchilla 在 TruthfulQA 上达到 43.6%（0-shot）和 66.7%（10-shot），相比 Gopher 的 29.5%（0-shot）和 43.7%（10-shot）有大幅提升。作者认为这表明更好地建模预训练数据本身就可以在真实性基准上带来较大的改善。

### 3.4 消融实验

#### Cosine Schedule 长度

![[cosine_v2.png|800]]
> 图8：Cosine 调度长度的影响。当 cosine cycle 设置为目标训练步数的 1x、1.1x、1.25x、1.5x、2x、5x 时的效果。超过 25% 会导致明显的性能下降。

一个关键发现是：cosine cycle 长度应当大致匹配训练步数。当 cosine schedule 超过目标步数的 25% 以上时，性能明显下降。

#### Adam vs AdamW

![[Adam_AdamW.png|800]]
> 图9：Adam vs AdamW。对 417M 和 1.4B 模型，AdamW 训练的模型在训练后期（约80%进程后）开始超越 Adam 模型，最终达到更低的 loss。

![[ablate_v1.png|800]]
> 图10：Chinchilla 与 Gopher 训练配置对比消融。使用 680M 参数模型，Chinchilla 的训练配置（AdamW + 高精度权重副本，橙色线）明显优于 Gopher 的配置（绿色线）。

#### 不同数据集的一致性

![[gh_c4_2.png|800]]
> 图11：C4 和 GitHub 数据集的 IsoFLOP 曲线。在 C4 上 $a=0.50, b=0.50$；在 GitHub 上 $a=0.53, b=0.47$，与 MassiveText 上的结论一致。

#### FLOP-loss 前沿的曲率

![[curvature_v3.png|800]]
> 图12：训练曲线包络线的曲率。分别用前1/3、中间1/3、后1/3的前沿点进行拟合，显示在高计算预算区间存在负曲率——这暗示对于非常大的计算预算，最优模型可能比本文预测的还要更小。

### 3.5 最优计算预算预测

| 参数量 | 所需 FLOPs | 相对 Gopher | 所需 Tokens |
|------|------|------|------|
| 400M | 1.92e+19 | 1/29,968 | 8.0B |
| 1B | 1.21e+20 | 1/4,761 | 20.2B |
| 10B | 1.23e+22 | 1/46 | 205.1B |
| 67B | 5.76e+23 | 1 | 1.5T |
| 175B | 3.85e+24 | 6.7 | 3.7T |
| 280B | 9.90e+24 | 17.2 | 5.9T |
| 520B | 3.43e+25 | 59.5 | 11.0T |
| 1T | 1.27e+26 | 221.3 | 21.2T |
| 10T | 1.30e+28 | 22515.9 | 216.2T |

这一表格表明：若要计算最优地训练一个 175B 参数的模型，需要 3.7T tokens（而非 GPT-3 使用的 300B）；一个万亿参数的模型则需要超过 21T tokens。

![[tokens_vs_params4.png|800]]
> 图13：给定训练 FLOP 预算下的最优 tokens 和参数量。三种方法的预测总体一致。

## 四、未来工作建议

### 4.1 作者建议的未来工作

作者在论文中指出了几个值得探索的方向：

1. 在多个中间规模上验证 Chinchilla 与 Kaplan 预测的差异（目前仅有 Chinchilla/Gopher 一组大规模对比）
2. 研究高计算预算下 FLOP-loss 前沿的曲率效应，探索是否应进一步缩小模型
3. 探索多 epoch 训练场景下的 scaling 规律
4. 将方法论推广到其他模态（视觉、多模态等）
5. 关注高质量大规模数据集的构建——本文的结论意味着数据规模将成为 scaling 的瓶颈

### 4.2 基于分析的未来方向

1. **数据质量与规模的平衡**
   - 动机：本文预测训练万亿 token 级别的模型需要数十万亿 token 的数据，但目前公开可用的高质量文本数据远不足此规模
   - 可能的方法：数据增强、合成数据、多语言数据利用、retrieval-augmented 方法
   - 挑战：在扩大数据规模的同时维持质量；重复数据和 train-test 泄漏的风险

2. **在线/增量训练的 scaling law**
   - 动机：本文所有分析均假设单 epoch 训练，实际场景中模型可能需要持续更新
   - 可能的方法：研究多 epoch 场景下的最优模型大小-数据量权衡，以及灾难性遗忘对 scaling law 的影响

3. **MoE 模型的 compute-optimal 训练**
   - 动机：稠密模型的 FLOPs ≈ 6ND 关系在 MoE 中不直接适用
   - 可能的方法：将分析拓展到 MoE 的"有效参数量"和"实际计算量"

### 4.3 改进建议

1. **损失函数形式的改进**
   - 当前问题：三项分解 $E + A/N^\alpha + B/D^\beta$ 假设 $N$ 和 $D$ 的效应独立可分，未建模两者的交互
   - 改进方案：引入交叉项或更灵活的参数化形式
   - 预期效果：可能在高/低计算预算区间都给出更准确的预测

2. **考虑推理成本的联合优化**
   - 当前问题：本文仅优化训练 loss，未显式考虑推理效率
   - 改进方案：将推理 FLOPs（与 $N$ 成正比）纳入目标函数，在"训练性能"和"推理效率"之间做帕累托优化

## 五、我的综合评价

### 5.1 价值评分

**8.5/10** - 这是一篇对 LLM 训练实践产生了深远影响的工作，其核心结论（模型大小和数据量应当等比例增长）改变了后续模型的训练策略（如 LLaMA 系列），具有很高的实用价值。

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 核心方法（拟合 scaling law）并非全新，但通过修正 Kaplan 的方法论缺陷得出了不同结论，改变了整个领域的训练范式 |
| 技术质量 | 8/10 | 三种方法相互验证的设计合理且严谨；但受限于大规模实验成本，仅有一组 Chinchilla/Gopher 对比验证 |
| 实验充分性 | 8/10 | 训练了超过 400 个模型覆盖广泛参数范围，三种分析方法交叉验证；但大规模验证仅 Chinchilla vs Gopher 一组 |
| 写作质量 | 9/10 | 结构清晰，三种方法层层递进，图表丰富直观，论述逻辑严密 |
| 实用性 | 9/10 | 直接指导 LLM 训练资源分配决策，后续 LLaMA/Mistral 等模型明确参考了该结论 |

### 5.2 重点关注

#### 值得关注的技术点

三种独立方法（训练曲线包络、IsoFLOP profiles、参数化损失函数拟合）得出一致结论的实验设计思路值得学习。特别是方法3中对损失函数的理论分解（贝叶斯风险 + 函数近似误差 + 随机近似误差），为 scaling law 提供了理论根基。此外，cosine schedule 长度需要与训练步数匹配这一发现，对实际训练具有直接指导意义。

#### 需要深入理解的部分

参数化损失函数 $\hat{L}(N,D) = E + A/N^\alpha + B/D^\beta$ 的拟合过程，特别是使用 Huber loss 和 L-BFGS 优化的细节。FLOP-loss frontier 的负曲率现象（大模型可能需要比预测更小的规模）也值得关注，这暗示 scaling law 可能存在更复杂的非幂律行为。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关

- [[Scaling Laws for Neural Language Models|Kaplan et al. 2020 - Scaling Laws]] - 本文修正和扩展的直接前序工作
- [[LLaMA - Open and Efficient Foundation Language Models|LLaMA]] - 采纳 Chinchilla 结论训练的开源模型

### 6.2 背景相关

- [[Language Models are Few-Shot Learners|GPT-3]] - 300B tokens 训练范式的代表
- [[Scaling Language Models - Methods, Analysis & Insights from Training Gopher|Gopher]] - Chinchilla 的直接对比模型
- [[Attention Is All You Need|Transformer 原文]] - 模型架构基础

### 6.3 后续工作

- [[Llama 2 - Open Foundation and Fine-Tuned Chat Models|LLaMA 2]] - 后续进一步验证 compute-optimal 策略
- [[Scaling Data-Constrained Language Models|数据受限 Scaling]] - 探讨数据不足时的 scaling 策略

## 外部资源

- [arXiv PDF](https://arxiv.org/pdf/2203.15556)
- [Yannic Kilcher 视频解读](https://www.youtube.com/watch?v=Pf3RkSXlJBk)
- [DeepMind Blog](https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training)

> [!tip] 关键启示
> 在给定计算预算下，模型参数量和训练数据量应当等比例增长。当前多数大模型存在 "过大而训练不足" 的问题——用同样的算力训练一个更小但数据更充分的模型，往往能获得更好的性能。

> [!warning] 注意事项
> - 本文结论基于单 epoch 训练场景，多 epoch 训练的 scaling 行为可能不同
> - FLOP-loss frontier 在大计算预算时存在负曲率，外推到更大规模时需谨慎
> - 预测所需的训练数据量（万亿 token 级别）对数据质量提出了很高要求

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 大模型训练必读论文。核心结论简洁有力且已被后续工作广泛验证，对理解 LLM scaling 和指导训练资源分配具有重要参考价值。
