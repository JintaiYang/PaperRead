---
title: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
authors: [Meituan Search & Recommendation Team]
year: 2025
venue: arXiv
arxiv_id: "2505.18654"
tags: [generative-recommendation, scaling-law, HSTU, industrial-system, ranking-model]
status: completed
created: 2025-05-15
---

# MTGR: 美团工业级生成式推荐排序框架

## 一句话总结

MTGR 结合了 DLRM 的交叉特征优势与 GRM 的扩展性，基于 HSTU 架构通过用户级样本聚合实现训推加速，提出 Group-Layer Normalization 和 Dynamic Masking 两项关键技术，在美团外卖主流量达到 65x DLRM FLOPs 并取得近两年最大业务收益。

---

## 1. 研究背景与动机

### 1.1 推荐系统中的 Scaling 困境

Scaling law 已在 NLP、CV 领域被广泛验证，但在推荐排序模型上存在独特困境：

**传统 DLRM 的两大扩展瓶颈**：

1. **用户行为序列爆炸增长**：传统 DLRM 无法高效处理完整行为序列，只能依赖序列检索（SIM）或设计低复杂度模块，限制了模型学习能力
2. **推理成本线性增长**：DLRM 的 cross module 对每个候选项独立推理，成本与候选数线性增长，推理延迟不可承受

**GRM（生成式推荐）的扩展优势与致命缺陷**：

GRM（如 HSTU）具有优秀的扩展性源于两个因素：(1) 直接建模完整用户行为链，将同用户多样本压缩为一条，减少计算冗余；(2) Transformer + FlashAttention 满足工业系统的训推要求。

但 GRM 严重依赖 next-token prediction 建模完整行为序列，**必须移除候选-用户之间的交叉特征**（cross features）。实验表明，移除交叉特征会严重损害模型性能，且这种退化**无法通过 scaling up 弥补**。

### 1.2 MTGR 的核心问题

> 如何构建一个既能利用交叉特征保证效果，又具备 GRM 扩展性的排序模型？

![[images/traditional_recsys_workflow.png]]
*图1：传统 DLRM 的数据组织与工作流程。每个候选项独立推理，交叉特征不可或缺。*

---

## 2. MTGR 方法详解

### 2.1 数据组织：从 Sample-Level 到 User-Level 聚合

**传统 DLRM 的数据组织**：对用户和 K 个候选，第 i 个样本为：
$$\mathbb{D}_i = [\mathbf{u}, \overrightarrow{\mathbf{s}}, \mathbf{rt}, \mathbf{x}_i, \mathbf{c}_i]$$

其中：
- $\mathbf{u} = [u^1, ..., u^{N_u}]$：用户画像特征（年龄、性别等标量）
- $\overrightarrow{\mathbf{s}} = [s^1, ..., s^{N_s}]$：历史交互序列，每个元素是一个 item 的特征集合
- $\mathbf{rt}$：实时交互序列（最近几小时内的行为）
- $\mathbf{x}_i$：用户-候选交叉特征（如用户对该候选的历史 CTR、曝光次数）
- $\mathbf{c}_i$：候选项特征（ID、标签、品牌等）

**MTGR 的关键创新——用户级聚合**：将同一用户的所有候选项聚合为一个样本：
$$\mathbb{D} = [\mathbf{u}, \overrightarrow{\mathbf{s}}, \mathbf{rt}, [\mathbf{x}, \mathbf{c}]_1, ..., [\mathbf{x}, \mathbf{c}]_K]$$

核心区别在于：**交叉特征 $\mathbf{x}$ 被重新组织为候选项特征的一部分**，与 item 特征拼接。这使得 MTGR 可以同时享有：
- 用户特征只计算一次（共享 user representation）→ 推理成本与候选数解耦
- 保留完整的交叉特征 → 不损失模型效果

**训练与推理效率提升**：
- 训练：样本数从"所有候选数"降为"所有用户数"，大幅减少数据量
- 推理：一次前向传播为同一用户的所有候选评分，推理成本亚线性增长

### 2.2 Token 化统一输入

为适配 Transformer 架构，MTGR 将所有特征统一转化为固定维度 $d_{uni}$ 的 token：

- **用户画像 $\mathbf{u}$**：每个标量特征→一个 token，$\mathbf{F}_u \in \mathbb{R}^{N_u \times d_{uni}}$
- **行为序列 $\overrightarrow{\mathbf{s}}$**：每个历史 item→一个 token（Embed+Concat+MLP），$\mathbf{F}_{\overrightarrow{s}} \in \mathbb{R}^{N_{\overrightarrow{s}} \times d_{uni}}$
- **实时序列 $\mathbf{rt}$**：同理，按时间顺序排列
- **候选项 $[\mathbf{x}, \mathbf{c}]_i$**：交叉特征与 item 特征拼接→一个 token

最终形成统一的 token 序列：
$$\mathbf{F}_\mathbb{D} = \text{Concat}[\mathbf{F}_u, \mathbf{F}_{\overrightarrow{s}}, \mathbf{F}_{rt}, \mathbf{F}_{item}] \in \mathbb{R}^{(N_u + N_{\overrightarrow{s}} + N_{rt} + N_{item}) \times d_{uni}}$$

### 2.3 HSTU Encoder 架构

![[images/GR_workflow_architecture.png]]
*图2：MTGR 的数据组织与架构。(a) 整体工作流：用户级聚合后 token 化输入 self-attention；(b) Self-attention 模块细节：Group-LN → Q/K/V/U 投影 → Masked Attention → V·U 点积 → Group-LN + Residual；(c) Dynamic Masking 策略示例。*

MTGR 采用 HSTU（Hierarchical Sequential Transduction Units）的 encoder-only 架构，每层包含：

**Step 1 - Group Layer Normalization（GLN）**：
$$\tilde{\mathbf{X}} = \text{GroupLN}(\mathbf{X})$$

不同于标准 LayerNorm 对整个序列归一化，GLN 将同一语义域（如用户画像、行为序列、候选项）的 token 分为一组独立归一化。原因：不同域的特征分布差异大（如稠密的用户特征 vs 稀疏的 item ID），统一归一化会互相干扰。GLN 使不同语义空间的 token 在进入 attention 前对齐到相似分布。

**Step 2 - 四向投影**：
$$\mathbf{K}, \mathbf{Q}, \mathbf{V}, \mathbf{U} = \text{MLP}_{K/Q/V/U}(\tilde{\mathbf{X}})$$

注意这里有额外的 $\mathbf{U}$ 投影（HSTU 的特有设计，用于替代 FFN）。

**Step 3 - Self-Attention with Dynamic Masking**：
$$\tilde{\mathbf{V}} = \frac{\text{silu}(\mathbf{K}^T \mathbf{Q})}{(N_u + N_{\overrightarrow{s}} + N_{rt} + N_{item})} \odot \mathbf{M} \cdot \mathbf{V}$$

使用 SiLU 替代 softmax（HSTU 的设计），除以总序列长度作为平均因子，然后施加定制化 mask $\mathbf{M}$。

**Step 4 - 点积融合与残差**：
$$\mathbf{X} = \text{MLP}(\text{GroupLN}(\tilde{\mathbf{V}} \odot \mathbf{U})) + \mathbf{X}$$

$\mathbf{U}$ 与 updated $\mathbf{V}$ 的逐元素点积（类似 gated mechanism），再经 GroupLN + MLP + 残差连接。MTGR 不使用 FFN（同 HSTU 设计）。

### 2.4 Dynamic Masking 策略

这是 MTGR 最精巧的设计之一，解决了用户级聚合带来的信息泄露问题。

**问题本质**：训练时同一用户不同时段的候选被聚合到一个样本中，而实时序列 $\mathbf{rt}$ 记录的是用户最近的交互——可能在时间上与聚合窗口重叠。例如：晚上的交互不应暴露给下午的候选项。

**三条 Masking 规则**：

| Token 类型                                          | 可见性规则                                | 理由               |
| ------------------------------------------------- | ------------------------------------ | ---------------- |
| 静态序列（$\mathbf{u}$, $\overrightarrow{\mathbf{s}}$） | 对所有 token 可见（full attention）         | 信息来自聚合窗口之前，无因果错误 |
| 动态序列（$\mathbf{rt}$）                               | 仅对时间上晚于自己的 token 可见（auto-regressive） | 保持时间因果性          |
| 候选项 token                                         | 仅对自身可见（diagonal mask）                | 候选间不应互相泄露信息      |

**关键洞察**：候选项只能看到自身 + 所有用户侧信息，但看不到其他候选。这保证了每个候选的评分独立性，同时充分利用用户上下文。

### 2.5 Group-Layer Normalization（GLN）详解

**动机**：MTGR 的 token 序列由异构信息组成——用户画像是稠密连续特征，行为序列是稀疏 ID 嵌入，交叉特征是统计类特征。它们的分布特性迥异：

- 用户画像 token：激活值集中、方差小
- 行为序列 token：稀疏 ID 经嵌入后方差大
- 交叉特征 token：统计特征数值范围不一

标准 LayerNorm 统一归一化会导致某些语义空间的信息被压缩或放大。GLN 将同一语义域的 token 视为一组，独立计算均值和方差，实现语义空间对齐。

---

## 3. 训练框架优化

MTGR 基于 PyTorch 生态重建训练框架（替代 TensorFlow），在 TorchRec 基础上做了大量优化：

### 3.1 动态哈希表（Dynamic Hash Table）

**问题**：TorchRec 使用固定大小的 Embedding 表，不适合工业流式训练——新用户/新商品无法实时分配 embedding，静态表需预留过多空间造成内存浪费。

**方案**：解耦式哈希表架构，将 Key 存储与 Value 存储分离：
- Key 存储：轻量映射系统（key → pointer to embedding vector）
- Value 存储：embedding 向量 + 元数据（计数器、时间戳用于淘汰策略）

**优势**：(1) 动态扩容只需复制 key 存储而非整个 embedding 表；(2) key 紧凑排列提升扫描效率。

### 3.2 Embedding 去重与表合并

跨设备 embedding 交换采用 All-to-all 通信。为减少重复 ID 传输，实现两步去重：通信前后均保证 ID 唯一性。同时支持自动表合并，减少 lookup 算子数量。

### 3.3 动态负载均衡（Dynamic BS）

**问题**：用户行为序列呈长尾分布（少数用户有极长序列），固定 batch size 导致 GPU 间计算负载严重不均。

**方案**：动态 Batch Size——根据输入数据的实际序列长度调整每个 GPU 的 local BS，确保计算负载均衡。梯度聚合按 BS 加权，保持计算逻辑一致性。

> 对比方案 Sequence Packing 需要复杂的 mask 调整防止不同序列在 attention 中互相干扰，实现成本高。动态 BS 更简洁高效。

### 3.4 流水线与混合精度

三流水线并行：copy（CPU→GPU）、dispatch（embedding lookup + 通信）、compute（前向/反向）。采用 BF16 混合精度训练，并基于 CUTLASS 设计定制 attention kernel 加速。

**最终效果**：相比原版 TorchRec，训练吞吐提升 1.6x–2.4x，支持 100+ GPU 良好扩展。

---

## 4. 实验结果

### 4.1 实验设置

**数据集**：美团外卖工业级数据，包含丰富的交叉特征和长用户行为序列：

| 数据集 | 用户数 | 商品数 | 曝光数 | 点击数 | 购买数 |
|---|---|---|---|---|---|
| Train (10天) | 2.1亿 | 430万 | 237.4亿 | 10.8亿 | 1.8亿 |
| Test | 302万 | 314万 | 7686万 | 455万 | 77万 |

**模型配置**：

| 模型 | 配置 | 学习率 | GFLOPs/sample |
|---|---|---|---|
| UserTower-SIM (DLRM最强) | - | 8×10⁻⁴ | 0.86 |
| MTGR-small | 3层, d=512, 2头 | 3×10⁻⁴ | 5.47 |
| MTGR-medium | 5层, d=768, 3头 | 3×10⁻⁴ | 18.59 |
| MTGR-large | 15层, d=768, 3头 | 1×10⁻⁴ | 55.76 |

**关键参数**：行为序列最大长度 $N_{\overrightarrow{s}}=1000$，实时序列 $N_{rt}=100$。Embedding 维度策略：优先扩大低基数特征的维度，极稀疏特征维度保持不变（避免稀疏参数过度膨胀）。

### 4.2 离线性能对比

| 模型 | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|---|---|---|---|---|
| DNN-SIM | 0.7432 | 0.6679 | 0.8737 | 0.6504 |
| MoE-SIM | 0.7484 | 0.6698 | 0.8750 | 0.6519 |
| MultiEmbed-SIM | 0.7501 | 0.6715 | 0.8766 | 0.6525 |
| Wukong-SIM | 0.7568 | 0.6759 | 0.8800 | 0.6530 |
| UserTower-SIM | 0.7593 | 0.6792 | 0.8815 | 0.6550 |
| UserTower-E2E | 0.7576 | 0.6787 | 0.8818 | 0.6548 |
| **MTGR-small** | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| **MTGR-medium** | 0.7645 | 0.6843 | 0.8849 | 0.6625 |
| **MTGR-large** | **0.7661** | **0.6865** | **0.8862** | **0.6646** |
| Impr.% | 0.90% | 1.07% | 0.50% | 1.47% |

**关键发现**：
- MTGR-small（5.47 GFLOPs）已超越最强 DLRM（UserTower-SIM, 0.86 GFLOPs）
- 三个版本展现平滑的扩展性，性能随计算量稳定提升
- UserTower-E2E 相比 UserTower-SIM 反而略有下降——说明在 DLRM 框架下模型复杂度不足以建模完整序列（欠拟合）

### 4.3 消融实验

| 模型 | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|---|---|---|---|---|
| MTGR-small (full) | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| w/o cross features | 0.7495 | 0.6689 | 0.8736 | 0.6514 |
| w/o GLN | 0.7606 | 0.6809 | 0.8826 | 0.6585 |
| w/o dynamic mask | 0.7620 | 0.6810 | 0.8828 | 0.6587 |

**核心结论**：
1. **交叉特征至关重要**：移除后性能暴跌（CTCVR GAUC -0.0089），直接抹平了 MTGR-large 相对 DLRM 的全部增益。这证实了论文的核心论点——GRM 不使用交叉特征是其最大缺陷
2. **GLN 和 Dynamic Masking 各贡献显著**：去除任一组件的退化幅度约等于从 small 到 medium 的提升量

### 4.4 扩展性验证

![[images/scaling_law.png]]
*图3：MTGR 在不同超参数（HSTU 层数、模型维度 $d_{model}$、序列长度）下的扩展性，以及性能与计算量的 power-law 关系。*

实验基于 MTGR-small，分别增加 HSTU 层数、$d_{model}$ 和序列长度，均展现良好的扩展性。性能（CTCVR GAUC 增益）与计算复杂度呈 **power-law 关系**。

### 4.5 在线实验

在美团外卖平台 2% 流量 AB 测试，对比连续学习 2 年的 DLRM 基线（UserTower-SIM）：

| 模型 | CTR GAUC diff | CTCVR GAUC diff | PV_CTR | UV_CTCVR |
|---|---|---|---|---|
| MTGR-small | +0.0036 | +0.0154 | +1.04% | +0.04% |
| MTGR-medium | +0.0071 | +0.0182 | +2.29% | +0.62% |
| MTGR-large | +0.0153 | +0.0288 | +1.90% | +1.02% |

**核心结论**：
- MTGR 仅用 6 个月数据训练，就超越了 DLRM 基线 2 年的持续优化积累
- MTGR-large 的 CTCVR GAUC 提升超过了过去一年所有优化的累计增量
- 线上 UV_CTCVR +1.02%（对于美团外卖体量而言，转化订单量 +1.22%，极为显著）
- 训练成本与 DLRM 持平，推理成本反而降低 12%（亚线性扩展）

---

## 5. 技术贡献总结

| 贡献点 | 解决的问题 | 技术手段 |
|---|---|---|
| DLRM+GRM 融合 | GRM 无法使用交叉特征 | 将 cross features 编入候选 token，用判别式 loss |
| 用户级样本聚合 | 训推成本高 | 同用户候选聚合为一个样本，推理成本亚线性 |
| Group-Layer Norm | 异构 token 分布不一致 | 按语义域分组归一化 |
| Dynamic Masking | 用户聚合后的信息泄露 | 静态全可见/动态因果/候选对角 |
| TorchRec 训练框架 | 原框架效率不足 | 动态哈希表、去重、动态 BS、流水线 |

---

## 6. 个人思考与评价

### 6.1 创新亮点

**方法论层面**：MTGR 的核心贡献是**重新定义了推荐系统中 scaling 的范式**——既不是单纯 scale user module，也不是 scale cross module，而是通过数据重组织让整个系统天然具备扩展性。这个思路比"设计更复杂的网络结构"更本质。

**工程层面**：Dynamic Masking 的设计非常精巧，将"用户级聚合导致的因果错误"这一实际工程问题转化为灵活的 attention mask 策略，既保证了正确性，又不增加额外计算开销。

### 6.2 局限与思考

1. **序列长度受限**：$N_{\overrightarrow{s}}=1000$, $N_{rt}=100$ 在外卖场景可能够用，但迁移到行为更密集的场景（如短视频）可能需要进一步的序列压缩方案
2. **交叉特征依赖**：实验证明移除交叉特征损失巨大，但交叉特征本身需要大量人工特征工程——未来是否可以让模型自动学习交叉
3. **单场景训练**：论文结尾提到未来会探索多场景建模（类似 LLM 的 foundation model），这是很值得期待的方向
4. **与 HSTU 的关系**：本质上 MTGR = HSTU + 交叉特征 + Dynamic Masking + GLN。其中 HSTU 已证明扩展性，MTGR 的核心贡献更多在"如何把交叉特征塞回去"

### 6.3 与其他工作的对比

| 维度 | HSTU (Meta) | OneRec | MTGR |
|---|---|---|---|
| 架构 | Transformer encoder | Transformer + DPO | HSTU encoder |
| 交叉特征 | ❌ 不支持 | ❌ 不支持 | ✅ 完整保留 |
| 扩展性 | ✅ 万亿参数 | ✅ 验证 | ✅ power-law |
| 推理效率 | 亚线性 | - | 亚线性，成本降 12% |
| 训练框架 | TF | - | PyTorch/TorchRec |
| 落地场景 | Instagram Reels | - | 美团外卖（全量） |

---

## 7. 相关引用

- HSTU: [[zhai2024actions]] - Actions Speak Louder than Words (Meta, 2024)
- OneRec: [[deng2025onerec]] - 语义编码 + DPO + 统一生成模型
- SIM: [[pi2020search]] - Search-based Interest Model
- TorchRec: [[ivchenko2022torchrec]] - PyTorch 推荐系统训练框架
- Wukong: [[zhang2024wukong]] - 可堆叠的 Wukong Layer 扩展
- MultiEmbed: [[guo2023embedding]] - 多 embedding 策略解决 embedding collapse

---

## 附录：关键公式速查

**Target Attention（传统 DLRM）**：
$$\mathbf{F}_{\overrightarrow{s}} = \text{Attention}(\mathbf{E}_{item}, \mathbf{E}_{\overrightarrow{s}}, \mathbf{E}_{\overrightarrow{s}})$$

**MTGR Self-Attention**：
$$\tilde{\mathbf{V}} = \frac{\text{silu}(\mathbf{K}^T\mathbf{Q})}{L_{total}} \odot \mathbf{M} \cdot \mathbf{V}$$

**残差输出**：
$$\mathbf{X}^{l+1} = \text{MLP}(\text{GroupLN}(\tilde{\mathbf{V}} \odot \mathbf{U})) + \mathbf{X}^l$$
