---
paper_id: "[arXiv:2505.18654](https://arxiv.org/abs/2505.18654)"
title: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
authors: "Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Menglei Zhou, Lei Yu, Chuan Liu, Wei Lin"
institution: "美团 (Meituan)"
publication: "CIKM 2025 (34th ACM International Conference on Information and Knowledge Management)"
tags:
  - 精排论文
  - MTGR
  - HSTU
  - 生成式推荐
  - Scaling-Law
  - DLRM融合
  - 混合式架构
  - 工业落地
  - 外卖推荐
  - TorchRec
  - Group-LayerNorm
  - Dynamic-Masking
  - 用户粒度压缩
quality_score: "9.2/10"
link:
  - "[PDF](https://arxiv.org/pdf/2505.18654)"
  - "[DOI](https://doi.org/10.1145/3746252.3761565)"
date: "2025-05-24"
venue: "CIKM 2025, Seoul, Republic of Korea"
keywords: "Scaling Law, Generative Recommendation"
---

## 一、研究背景与动机

### 1.1 领域现状

Scaling Law 已在 NLP、CV、信息检索等领域得到广泛验证。在推荐系统中，近期工作（Meta HSTU/GR、快手 OneRec）采用生成式推荐实现了可扩展性，但其生成式方法要求放弃传统推荐模型精心构建的交叉特征。美团外卖推荐场景经过近十年面向交易目标的迭代，已形成成熟的 DLRM 体系，积累了大量有价值的交叉特征（如用户对候选商家的历史点击率、曝光次数等）。

### 1.2 现有方法的局限性

**DLRM 的 Scaling 困境**：传统 DLRM 的 Scaling 存在两条路径，但都有根本性局限。Scaling Cross Module（特征交互 MLP）的推理成本与候选数量线性增长，无法承受；Scaling User Module（用户表征模块）虽然推理友好（用户表征可在候选间共享），但无法直接增强用户-物品的特征交互能力。

**生成式推荐（GRM）的代价**：GRM 通过 next token prediction 建模完整用户行为序列，具有优秀的可扩展性（训练样本从候选数压缩到用户数，推理成本与候选数解耦）。但 GRM 在生成时不感知具体待预测物品，因此无法使用任何交叉特征。论文通过实验直接证明：去除交叉特征后性能严重下降，且 Scaling 无法弥补这一损失。

![[traditional-recsys.png|800]]

> 图1：传统 DLRM 的数据组织与工作流程。每个候选独立处理，用户特征被重复编码 K 次。

### 1.3 核心问题

如何构建一个既能利用交叉特征保证效果、又具备 GRM 可扩展性的排序模型？

### 1.4 本文解决方案概述

MTGR（Meituan Generative Recommendation）采用混合式架构：基于 HSTU 架构进行统一序列编码，同时保留与 DLRM 完全一致的特征体系（包括交叉特征）。通过用户粒度数据压缩实现训练加速，通过 Group-Layer Normalization 处理异构 Token 的语义对齐，通过动态掩码策略避免信息泄露。最终实现 65 倍 FLOPs 提升下训练成本持平、推理成本降 12%、外卖首页订单量 +1.22% 的工业化落地。

## 二、解决方案

### 2.1 核心思想

MTGR 的设计哲学是"两全其美"——既享受 HSTU 的 Scaling 红利，又保留交叉特征的信息增益。具体而言，MTGR 将 GRM 的用户粒度聚合（训练效率）和 Transformer 架构（可扩展性）与 DLRM 的完整特征体系（包括交叉特征）相结合，使用判别式损失（而非生成式 next token prediction）进行学习。

### 2.2 整体架构

![[GR-workflow.png|800]]

> 图2：MTGR 的数据组织与整体架构。(a) 用户粒度聚合后的数据流：多个候选的特征与同一用户特征聚合为一个样本，Token 化后输入 HSTU 编码器；(b) HSTU 自注意力模块的详细结构：Group LayerNorm → Q/K/V/U 投影 → 带掩码的注意力 → 点积门控 → 残差连接；(c) 动态掩码策略示例。

#### 模块1：数据组织——用户粒度聚合

**传统 DLRM 的数据表示**：对于用户和 K 个候选，第 i 个样本表示为 $\mathbb{D}_i = [\mathbf{U}, \overrightarrow{\mathbf{S}}, \overrightarrow{\mathbf{R}}, \mathbf{C}_i, \mathbf{I}_i]$，其中 $\mathbf{U}$ 为用户画像特征（年龄、性别等标量特征），$\overrightarrow{\mathbf{S}}$ 为历史行为序列（每个 item 包含 ID、标签、平均 CTR 等），$\overrightarrow{\mathbf{R}}$ 为实时行为序列（最近几小时内的交互），$\mathbf{C}_i$ 为交叉特征（用户对候选的历史 CTR、曝光次数等），$\mathbf{I}_i$ 为候选物品特征。

**MTGR 的用户粒度聚合**：将同一用户的 K 个候选聚合为一个样本：
$$\mathbb{D} = [\mathbf{U}, \overrightarrow{\mathbf{S}}, \overrightarrow{\mathbf{R}}, [\mathbf{C}, \mathbf{I}]_1, ..., [\mathbf{C}, \mathbf{I}]_K]$$

这一聚合带来两个关键优势：训练样本数从"所有候选数"压缩到"所有用户数"，大幅减少训练冗余；推理时对所有候选只需一次前向计算，推理成本与候选数量解耦（亚线性增长）。

#### 模块2：输入 Token 化

将聚合后的特征统一转换为 Token 序列：

- **用户画像 Token**：每个标量特征自然转换为一个 Token，$\mathbf{Feat}_{\mathbf{U}} \in \mathbb{R}^{N_{\mathbf{U}} \times d_{model}}$
- **行为序列 Token**：每个历史交互 item 作为一个 Token，其多个特征 embedding 拼接后通过 MLP 映射到统一维度 $d_{model}$
- **候选 Token**：每个候选将 ItemID、side info、交叉特征、时空 Context 的 embedding 拼接后通过另一个 MLP 映射为一个 Token

最终形成统一的 Token 序列：
$$\mathbf{Feat}_{\mathbb{D}} = \text{Concat}([\mathbf{Feat}_{\mathbf{U}}, \mathbf{Feat}_{\overrightarrow{\mathbf{S}}}, \mathbf{Feat}_{\overrightarrow{\mathbf{R}}}, \mathbf{Feat}_{\mathbf{I}}]) \in \mathbb{R}^{(N_{\mathbf{U}} + N_{\overrightarrow{\mathbf{S}}} + N_{\overrightarrow{\mathbf{R}}} + N_{\mathbf{I}}) \times d_{model}}$$

#### 模块3：统一 HSTU 编码器

采用多层自注意力的 Encoder-only 架构。每层的计算流程：

1. **Group Layer Normalization**：对不同类型的 Token（用户画像、行为序列、候选）使用不同的 LayerNorm 参数，实现异构语义空间的对齐：$\tilde{X} = \text{GroupLN}(\mathbf{X})$
2. **四路投影**：$\mathbf{K}, \mathbf{Q}, \mathbf{V}, \mathbf{U} = \text{MLP}_{\mathbf{K}/\mathbf{Q}/\mathbf{V}/\mathbf{U}}(\tilde{\mathbf{X}})$
3. **带掩码的注意力计算**（使用 SiLU 激活，除以总序列长度作为归一化因子）：
$$\tilde{\mathbf{V}} = \frac{\text{silu}(\mathbf{K}^T \mathbf{Q})}{(N_{\mathbf{U}} + N_{\overrightarrow{\mathbf{S}}} + N_{\overrightarrow{\mathbf{R}}} + N_{\mathbf{I}})} \mathbf{M} \mathbf{V}$$
4. **点积门控 + 残差**：$\mathbf{X} = \text{MLP}(\text{GroupLN}(\tilde{\mathbf{V}} \odot \mathbf{U})) + \mathbf{X}$

注意：与标准 Transformer 不同，MTGR（同 HSTU）不使用 FFN 层，而是通过 $\mathbf{U}$ 的点积门控实现非线性变换。

#### 模块4：Group Layer Normalization (GLN)

**动机**：不同于 LLM 中 Token 语义相对统一，MTGR 的输入包含多种异构 Token（用户画像是标量特征、行为序列是 item 特征、候选包含交叉特征），它们的分布差异很大。

**设计**：在每层自注意力的输入和输出各加一个 Group LayerNorm。同一"组"（如所有用户画像 Token）共享一组 LayerNorm 参数（$\gamma$, $\beta$），不同组使用不同参数。这确保了不同语义空间的 Token 在进入注意力计算前具有相似的分布。

**消融实验证明**：去除 GLN 后 CTCVR GAUC 下降 0.0018，相当于从 MTGR-small 到 MTGR-medium 的增益幅度。

#### 模块5：动态混合掩码 (Dynamic Masking)

**动机**：HSTU 原始使用因果掩码（causal mask），但在 MTGR 中效果不佳。更重要的是，训练时用户聚合将不同时间的候选合并到一个样本中，而实时行为序列 $\overrightarrow{\mathbf{R}}$ 记录的是最近交互，其时间可能与聚合窗口重叠——简单的因果掩码会导致信息泄露（如晚上的交互被下午的候选看到）。

**三条掩码规则**：

1. **静态序列（$\mathbf{U}$, $\overrightarrow{\mathbf{S}}$）全可见**：这些信息来自聚合窗口之前，不存在因果性问题，使用 Full Attention
2. **动态序列（$\overrightarrow{\mathbf{R}}$）因果可见**：每个实时行为 Token 只对时间上晚于它的 Token 可见（Auto-Regressive Mask）
3. **候选 Token 仅自身可见**：每个候选只能看到自己（Diagonal Mask），避免候选间信息泄露

这种设计在发挥 HSTU 作为 Encoder 的学习能力的同时，严格保证了因果性。

#### 模块6：训练系统优化

基于 TorchRec 构建的高性能分布式训练框架，相比原始 TorchRec 提升 1.6x-2.4x 吞吐量：

- **动态哈希表**：替代 TorchRec 的固定大小 embedding 表，支持实时新增/删除稀疏 ID。采用解耦架构——key 存储（轻量映射）与 value 存储（embedding 向量 + 元数据）分离，支持动态扩容和高效 key 扫描
- **Embedding Lookup 优化**：All-to-all 通信前后进行 ID 去重，减少跨设备重复传输
- **动态负载均衡**：根据实际序列长度动态调整每张 GPU 的 batch size，保证 total_tokens 基本相同；梯度聚合按 batch size 加权
- **三流水线并行**：copy stream（CPU→GPU 数据传输）、dispatch stream（embedding lookup + 通信）、compute stream（前向/反向）重叠执行
- **Fused HSTU Kernel**：基于 Cutlass 实现的融合注意力算子，借鉴 FlashAttention 思想，单算子性能相比 Triton 版本提升 2-3 倍
- **BF16 混合精度训练**

最终效果：65 倍计算复杂度的模型训练成本与 DLRM 持平。

#### 模块7：推理系统优化

基于 TensorRT + Triton Inference Server 构建：

- **用户粒度聚合推理**：一次请求中所有候选共享用户序列计算，推理成本与候选数量亚线性增长
- **特征 H2D 优化**：Host-to-Device 传输耗时从 7.5ms 降至 12μs
- **CUDA Graph**：吞吐提升 13%
- **FP16 计算**：吞吐提升 50%

最终效果：MTGR 在线推理资源比 DLRM 节省 12%。

### 2.3 与现有方法的关键区别

| 维度 | DLRM | GRM (HSTU/OneRec) | MTGR |
|------|------|-------------------|------|
| 特征体系 | 完整（含交叉特征） | 仅 ID + side info | 完整（含交叉特征） |
| 序列建模 | Target Attention（受限长度） | 全序列 Next Token Prediction | 全序列 Self-Attention |
| 训练粒度 | 样本级（每个候选一行） | 用户级 | 用户级 |
| 推理成本 | 线性于候选数 | 亚线性于候选数 | 亚线性于候选数 |
| 损失函数 | 判别式 | 生成式 | 判别式 |
| 可扩展性 | 受限 | 优秀 | 优秀 |

## 三、实验结果

### 3.1 数据集

| 数据集 | 用户数 | 物品数 | 曝光数 | 点击数 | 购买数 |
|--------|--------|--------|--------|--------|--------|
| Train (10天) | 2.1亿 | 430万 | 237.4亿 | 10.8亿 | 1.8亿 |
| Test | 302万 | 314万 | 7686万 | 455万 | 77万 |

在线实验使用超过 6 个月的数据训练，对比经过 2 年以上持续学习的 DLRM 基线。

### 3.2 实验设置

#### 模型配置

| 模型 | 配置 | 学习率 | GFLOPs/样本 |
|------|------|--------|-------------|
| UserTower-SIM (DLRM最强基线) | - | 8×10⁻⁴ | 0.86 |
| MTGR-small | 3层, d=512, 2头 | 3×10⁻⁴ | 5.47 |
| MTGR-medium | 5层, d=768, 3头 | 3×10⁻⁴ | 18.59 |
| MTGR-large | 15层, d=768, 3头 | 1×10⁻⁴ | 55.76 |

训练设置：Adam 优化器，DLRM 使用 8×A100（batch size 2400/GPU），MTGR 使用 16×A100（batch size 96/GPU）。序列最大长度：$\overrightarrow{\mathbf{S}}$ 为 1000，$\overrightarrow{\mathbf{R}}$ 为 100。

#### 基线方法

DLRM 系列：DNN-SIM、MoE-SIM（4 experts）、MultiEmbed-SIM、Wukong-SIM、UserTower-SIM、UserTower-E2E。其中 UserTower 使用 learnable queries + qFormer + MoE(16 experts)，计算复杂度为 MoE 的 3 倍，是 DLRM 中最强基线。

### 3.3 离线实验结果

| 模型 | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|------|---------|----------|-----------|------------|
| DNN-SIM | 0.7432 | 0.6679 | 0.8737 | 0.6504 |
| MoE-SIM | 0.7484 | 0.6698 | 0.8750 | 0.6519 |
| MultiEmbed-SIM | 0.7501 | 0.6715 | 0.8766 | 0.6525 |
| Wukong-SIM | 0.7568 | 0.6759 | 0.8800 | 0.6530 |
| UserTower-SIM | 0.7593 | 0.6792 | 0.8815 | 0.6550 |
| UserTower-E2E | 0.7576 | 0.6787 | 0.8818 | 0.6548 |
| **MTGR-small** | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| **MTGR-medium** | 0.7645 | 0.6843 | 0.8849 | 0.6625 |
| **MTGR-large** | **0.7661** | **0.6865** | **0.8862** | **0.6646** |
| 相对提升% | 0.90% | 1.07% | 0.50% | 1.47% |

关键发现：即使最小的 MTGR-small（5.47 GFLOPs）也超越了最强 DLRM 基线 UserTower-SIM（0.86 GFLOPs）。三种尺寸展现出平滑的 Scaling 趋势。

### 3.4 消融实验

| 模型 | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|------|---------|----------|-----------|------------|
| MTGR-small | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| w/o 交叉特征 | 0.7495 | 0.6689 | 0.8736 | 0.6514 |
| w/o GLN | 0.7606 | 0.6809 | 0.8826 | 0.6585 |
| w/o Dynamic Mask | 0.7620 | 0.6810 | 0.8828 | 0.6587 |

关键发现：

- **交叉特征至关重要**：去除后 CTCVR GAUC 下降 0.0089，甚至抹平了 MTGR-large 相对 DLRM 的全部增益
- **GLN 和 Dynamic Mask 贡献显著**：各自的去除导致的性能下降幅度，与从 small 到 medium 的 Scaling 增益相当

### 3.5 Scaling Law 验证

![[scaling.png|800]]

> 图3：MTGR 的 Scaling Law 验证。(a) 层数 Scaling；(b) d_model Scaling；(c) 序列长度 Scaling；(d) 计算复杂度与性能的幂律关系。

论文验证了三个维度的 Scaling：层数、模型维度、输入序列长度。图3(d) 展示了性能与计算复杂度之间的幂律关系（Power-Law），横轴为相对 UserTower-SIM 的计算复杂度对数倍数，纵轴为 CTCVR GAUC 增益。离线最大实验了 22 层、d=1024、137.87 GFLOPs（约 160 倍 DLRM）的超大模型。

### 3.6 在线 A/B 测试

| 模型 | CTR GAUC diff | CTCVR GAUC diff | PV_CTR | UV_CTCVR |
|------|---------------|-----------------|--------|----------|
| MTGR-small | +0.0036 | +0.0154 | +1.04% | +0.04% |
| MTGR-medium | +0.0071 | +0.0182 | +2.29% | +0.62% |
| MTGR-large | +0.0153 | +0.0288 | +1.90% | +1.02% |

MTGR-large 在 2% 流量 A/B 测试中：离线 CTCVR GAUC +2.88pp（超过过去一年所有优化的累计增益），UV_CTCVR +1.02%。2025 年 4 月底在外卖首页、频道页、小程序等核心场景全量上线。

### 3.7 系统效率

- 训练成本：与 DLRM 持平（65 倍 FLOPs，但用户粒度压缩 + 框架优化抵消了计算增长）
- 推理成本：比 DLRM 节省 12%（用户序列计算在所有候选间共享）
- 训练规模：100+ GPU 良好扩展性

## 四、未来工作与改进方向

### 4.1 作者提出的未来方向

论文结尾明确提出将探索 MTGR 的多场景建模扩展，类似大语言模型，建立具有广泛知识的推荐基础模型（Recommendation Foundation Model）。

### 4.2 基于分析的潜在方向

**方向1：更大规模模型的在线部署**
- 动机：离线实验显示 160 倍 DLRM（22层, d=1024）的模型有更高性能，但受推理限制未上线
- 可能的方法：Sparse MoE 减少激活参数、更激进的 KV Cache 压缩、模型蒸馏
- 预期成果：在不增加推理成本的前提下进一步提升模型容量

**方向2：端到端特征学习替代手工交叉特征**
- 动机：当前仍依赖人工设计的交叉特征，特征工程成本高
- 可能的方法：随着模型容量增大，逐步用模型学习到的交互替代手工交叉特征
- 预期成果：减少特征工程成本同时保持或提升效果

**方向3：多场景统一建模**
- 动机：当前 MTGR 仅在外卖场景验证
- 可能的方法：跨场景共享用户表征，场景特定的 adapter 或 prompt
- 预期成果：建立推荐基础模型，实现知识迁移

**方向4：自适应 Group LayerNorm**
- 动机：当前 Token 分组是预定义的（用户画像/序列/候选）
- 可能的方法：让模型自动学习 Token 的分组方式，或使用更细粒度的分组
- 预期成果：更灵活的语义空间对齐

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.2/10** — MTGR 成功解决了"如何在真实工业场景中落地生成式推荐"这一高难度工程问题，证明了生成式推荐可以兼容传统特征工程的信息优势，并在美团核心业务全量上线取得显著收益。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7.5/10 | 核心架构基于 HSTU，主要创新在混合式架构设计思想、GLN、动态掩码和系统优化。"保留交叉特征 + 用户粒度聚合"的组合虽非全新概念，但工程实现和验证极为扎实 |
| 技术深度 | 8.5/10 | 从数据组织、Token 化、掩码设计到训推引擎的全链路优化，技术链条完整。Fused HSTU Kernel、动态哈希表、三流水线并行等系统优化有深度 |
| 实验充分性 | 9.0/10 | 多尺寸 Scaling 验证、充分的消融实验、三维度 Scaling Law 曲线、在线 A/B 测试（2% 流量，百万级曝光），实验设计严谨 |
| 写作质量 | 8.5/10 | 从业务背景到技术方案到系统优化层层递进，公式推导清晰，图表信息量大。CIKM 论文篇幅限制下信息密度很高 |
| 实用性 | 9.5/10 | 在美团核心外卖业务全量上线，UV_CTCVR +1.02% 是非常实在的业务收益。训练成本持平、推理成本降 12% 的工程约束满足极为出色 |
| 可复现性 | 6.5/10 | 依赖美团内部数据和基础设施，公开数据集缺乏交叉特征，外部复现困难。但技术方案描述足够详细，思路可迁移 |

### 5.2 重点关注

#### 5.2.1 核心技术贡献

1. **混合式架构设计哲学**：不必在 DLRM 和 GRM 之间二选一，可以"两全其美"。将交叉特征编码到候选 Token 中，用判别式损失学习，同时享受用户粒度聚合的效率优势
2. **Group Layer Normalization**：解决异构 Token 的语义空间对齐问题，简单有效，消融实验证明贡献显著
3. **动态混合掩码**：在 Encoder 能力（Full Attention）和因果性（避免信息泄露）之间取得精确平衡，三条规则设计清晰
4. **用户粒度数据压缩**：将训练样本从候选级压缩到用户级，配合变长序列处理（JaggedTensor）完全消除 padding
5. **推理成本与候选数解耦**：用户序列计算在所有候选间共享，推理资源反而比 DLRM 节省 12%

#### 5.2.2 值得深入理解的技术细节

- **交叉特征的具体贡献量化**：去除交叉特征后 CTCVR GAUC 下降 0.0089，这个下降幅度甚至超过了 MTGR-large 相对 DLRM 的全部增益（0.0096），说明交叉特征的信息量极大
- **SiLU 激活 + 长度归一化的注意力**：不同于标准 softmax attention，MTGR 使用 $\text{silu}(\mathbf{K}^T\mathbf{Q}) / L$ 作为注意力权重，这是 HSTU 的设计，避免了 softmax 的计算开销
- **点积门控（$\tilde{\mathbf{V}} \odot \mathbf{U}$）替代 FFN**：通过额外的 $\mathbf{U}$ 投影实现非线性变换，减少参数量和计算量
- **动态 batch size 的梯度一致性**：按 batch size 加权梯度聚合，保证与固定 batch size 的计算逻辑一致
- **65 倍 FLOPs 如何做到训练成本持平**：用户粒度压缩（样本数大幅减少）+ 框架优化（1.6-2.4x 吞吐提升）+ 变长序列处理（消除 padding 浪费）

#### 5.2.3 局限性分析

- 仅在美团外卖场景验证，其他场景（短视频、电商、社交）的迁移效果未知
- 依赖大量人工设计的交叉特征，特征工程成本仍然存在
- 160 倍 DLRM 的超大模型受推理限制未能上线，说明 Scaling 仍有工程瓶颈
- 公开数据集缺乏交叉特征，外部研究者难以复现和对比

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[HSTU_Actions_Speak_Louder_than_Words|HSTU (Actions Speak Louder than Words)]] — MTGR 的基础架构来源，Meta 提出的万亿参数生成式推荐框架
- [[Wukong_Towards_a_Scaling_Law_for_Large_Scale_Recommendation|Wukong]] — 特征交互 Scaling Law，Scaling Cross Module 的代表
- [[FAT_From_Scaling_to_Structured_Expressivity|FAT]] — 特征交互 Scaling 的另一代表工作
- [[TokenMixer_Large_Scaling_Up_Large_Ranking_Models_in_Industrial_Recommenders|TokenMixer-Large]] — 工业推荐模型 Scaling 的相关工作
- [[Scaling_Law_of_Large_Sequential_Recommendation_Models|Scaling Law of Large Sequential Recommendation Models]] — 序列推荐模型的 Scaling Law

### 6.2 背景相关
- OneRec (快手) — 使用语义编码 + DPO 的统一生成式推荐模型
- SRP4CTR (美团) — 序列推荐预训练增强 CTR 预测，同一团队的前序工作
- DCN V2 — DLRM 中的特征交叉模块
- FlashAttention — MTGR Fused Kernel 的灵感来源
- TorchRec — MTGR 训练框架的基础

### 6.3 后续展望
- 多场景统一推荐基础模型
- 更大规模模型的在线部署（Sparse MoE / 蒸馏）
- 端到端特征学习替代手工交叉特征

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2505.18654)
- [ACM Digital Library](https://doi.org/10.1145/3746252.3761565)

> [!tip] 关键启示
> 生成式推荐不必完全抛弃传统 DLRM 的特征工程积累——混合式架构可以"两全其美"，既享受 HSTU 的 Scaling 红利，又保留交叉特征的信息增益。MTGR 证明了在低点击率高复购率的外卖场景中，交叉特征的价值不可替代（去除后性能下降甚至超过 Scaling 带来的全部增益）。

> [!warning] 注意事项
> - MTGR 的核心创新偏工程导向（混合架构 + 系统优化），理论贡献相对有限
> - 仅在美团外卖场景验证，其他场景的迁移效果未知
> - 160 倍 DLRM 的超大模型受推理限制未能上线，说明 Scaling 仍有工程瓶颈
> - 公开数据集缺乏交叉特征，外部复现困难

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！MTGR 是生成式推荐工业落地的标杆案例，为"如何在已有成熟 DLRM 体系的基础上引入 GR"提供了完整的技术方案和系统优化经验。对于正在探索 GR 落地的推荐团队，MTGR 的混合式架构思路极具参考价值。论文信息密度极高，每个技术点都有清晰的动机-方案-验证链条。
