---
paper_id: "[arXiv:2312.10743](https://arxiv.org/abs/2312.10743)"
title: "A Unified Framework for Multi-Domain CTR Prediction via Large Language Models"
authors: "Zichuan Fu, Xiangyang Li, Chuhan Wu, Yichao Wang, Kuicai Dong, Xiangyu Zhao, Mengchen Zhao, Huifeng Guo, Ruiming Tang"
institution: "Huawei Noah's Ark Lab, City University of Hong Kong, Renmin University of China, University of Science and Technology of China"
publication: "[ACM TOIS] [2023-12-17]"
tags:
  - 多域CTR预测
  - 大语言模型
  - 推荐系统
  - 零样本预测
  - 领域特定网络
  - Prompt-based-Modeling
  - Multi-Domain-Recommendation
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2312.10743)"
date: "2026-05-13"
---

## 一、研究背景与动机

### 1.1 领域现状

多域CTR（Multi-Domain CTR, MDCTR）预测是现代推荐系统面临的核心挑战之一。随着商业平台不断扩展服务范围（电商、打车、外卖、视频等），推荐系统需要在多个业务场景中提供精准的点击率预估。传统做法有两种极端：为每个域单独训练模型（忽略跨域共性），或简单混合所有域数据训练统一模型（忽略域间差异）。现有的多域CTR方法如 MMOE、PLE、STAR 等通过共享专家网络或星形拓扑结构来平衡共性与特性建模，但仍面临诸多挑战。

### 1.2 现有方法的局限性

论文指出现有MDCTR系统主要存在三个根本性问题：

**域数据稀疏性（Domain Seesaw Phenomenon）**：实际场景中许多域缺乏充足训练数据，模型容易被数据丰富的域主导，导致长尾域性能严重下降。MMOE 和 PLE 虽然用多专家和门控机制缓解了跷跷板现象，但在极度数据稀疏的域仍表现不佳。

**可扩展性有限（Limited Scalability）**：传统方法引入新域时需要重新训练整个模型或从头构建新模型。STAR 虽然用拓扑结构改善了扩展性，但其逐元素乘法要求所有域特定网络与骨干网络结构一致，参数量随域数增加呈倍增。MMOE 和 PLE 的组件紧耦合，移除某个过时域的专家网络也很困难。

**弱泛化能力（Weak Generalization）**：传统方法将特征转化为离散ID后通过 embedding 映射，这一过程丢失了关键的语义信息。面对冷启动场景（新域零样本预测），模型很难快速泛化。

### 1.3 本文解决方案概述

论文提出 **Uni-CTR**，一个基于大语言模型的统一多域CTR预测框架。核心思路是用自然语言作为通用信息载体，使不同域的知识可以统一编码和利用。Uni-CTR 包含三大组件：LLM骨干网络（捕获跨域共性）、域特定网络DSN（通过梯形网络提取域特性）、通用网络（使能零样本预测），并通过 Masked Loss 策略确保各DSN解耦训练。

## 二、解决方案

### 2.1 核心思想

Uni-CTR 的核心洞察是：**LLM 预训练积累的世界知识天然是跨域的**。通过将用户-物品特征转化为自然语言 prompt，LLM 骨干可以理解"电脑"和"键盘"的语义关联，而非仅仅视其为两个无关ID。同时，不同 transformer 层的表示捕获了从浅层词汇到深层语义的多级别信息，通过梯形网络（Ladder Network）从不同层提取表示，可以更精细地建模域特性。

关键设计理念包括：用 LLM 建模跨域共性；用轻量级 DSN 捕获各域特性；通过 Masked Loss 确保各 DSN 独立训练互不干扰，使得新域可以"即插即用"地加入而不影响已有域。

### 2.2 整体架构

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/Uni-CTR.png|800]]
> 图1：Uni-CTR 整体架构。输入 prompt 经 LLM 编码后，域特定网络（DSN）通过 Ladder 接收多层表示学习域特性，General Network 直接使用最后一层表示学习跨域共性，Masked Loss 确保各 DSN 解耦。

Uni-CTR 由三大模块组成：**LLM Backbone**（语义编码）、**Domain-Specific Networks**（域特性建模）、**General Network**（跨域共性与零样本预测）。

#### 模块1：Prompt-based Semantic Modeling

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/Prompt.png|800]]
> 图2：Prompt 模板设计，将域、用户、商品特征整合为自然语言序列。

**功能**：将非文本和文本特征整合为结构化的自然语言序列
**输入**：域上下文 $d$、用户信息 $u$（包含用户ID和点击历史）、商品信息 $p$（ID、标题、品牌、价格）
**输出**：文本序列 $x_{\text{text}}$

Prompt 模板格式为：
$$x_{\text{text}} = \text{[Domain Name]: The user ID is user\_[ID], who clicked product `[Title1]' and product `[Title2]' recently. The ID of the current product is product\_[ID], the title is [Name], the brand is [Brand], the price is [Price].}$$

这种设计保留了完整的语义信息，而非传统方法中的 one-hot 编码导致的语义丢失。

#### 模块2：LLM Backbone

**功能**：编码输入序列的丰富语义上下文信息
**处理流程**：prompt 文本经 tokenizer 分词 → embedding 层映射为向量序列 $\boldsymbol{h}_0$ → 通过 $L$ 层 transformer 逐层编码得到 $\boldsymbol{h}_1, \ldots, \boldsymbol{h}_L$
**输出**：所有层的表示集合 $\boldsymbol{H} = \{\boldsymbol{h}_0, \boldsymbol{h}_1, \ldots, \boldsymbol{h}_L\}$

论文使用 Sheared-LLaMA（1.3B参数，24层 transformer）作为骨干，并用 LoRA（rank=8, alpha=32）加速训练。

#### 模块3：Domain-Specific Network (DSN)

每个域对应一个独立的 DSN，包含三个子模块：

**Ladder Network**：以频率超参 $\phi$ 为间隔，从 LLM 不同层接收中间表示。第 $f$ 个 ladder 的输出为：
$$\boldsymbol{lad}_f = \begin{cases} Ladder_1(\boldsymbol{h}_\phi) & \text{if } f=1 \\ Ladder_f(\boldsymbol{h}_{f\cdot\phi} + \boldsymbol{lad}_{f-1}) & \text{if } f \in \{2,\ldots,F\} \end{cases}$$

每个 Ladder 可以是 MLP、Attention Network 或 Transformer Block。论文使用4个 ladder 层（每个为小型 transformer encoder block）。

**Gate Network**：自适应平衡域特性表示（ladder输出）和跨域共性表示（LLM最后层）。先拼接 $\boldsymbol{O} = \text{concat}(\boldsymbol{h}_L, \boldsymbol{lad}_F)$，再通过 attention pooling 计算动态权重：
$$\boldsymbol{score} = \text{tanh}(\boldsymbol{W}_k \boldsymbol{O})\boldsymbol{W}_q, \quad \boldsymbol{A} = \text{softmax}(\boldsymbol{score}), \quad \boldsymbol{R} = \boldsymbol{A}^T \boldsymbol{O}$$

**Tower Network**：3层MLP（512→256→128），对池化表示 $\boldsymbol{R}^{d_m}$ 做最终预测。

#### 模块4：General Network

**功能**：学习所有域的共性特征，使能零样本预测
**结构**：仅包含一个 Tower Network（MLP），直接使用 LLM 最后一层隐状态 $\boldsymbol{h}_L$
$$\hat{y}^G = \text{MLP}(\boldsymbol{h}_L; \boldsymbol{W}_\sigma^G, \boldsymbol{b}_\sigma^G)$$

对于已知域，使用对应 DSN 的预测；对于未知域，使用 General Network 的预测。

#### 模块5：Masked Loss Strategy

**核心创新**：通过 mask 机制确保各 DSN 参数解耦更新。给定域 $d_m$ 的样本，mask 向量为：
$$\boldsymbol{mask}^{d_m} = [I(d_1=d_m), I(d_2=d_m), \ldots, I(d_M=d_m)]$$

梯度传播特性：
- **LLM骨干**：接收来自当前域 DSN 和 General Network 两部分梯度
- **DSN参数**：只有当前域 $d_m$ 的 DSN 参数被更新，其他 DSN 梯度为0
- **General Network**：接收所有域样本的梯度，学习跨域共性

这确保了各 DSN 可独立训练、插拔、移除，系统具备真正的可扩展性。

## 三、实验结果

### 3.1 数据集

**公开数据集**：Amazon Review Data (2018)，选取5个类别作为不同域：

| 域 | 用户数 | 商品数 | 样本数 |
|---|---|---|---|
| Fashion | 749,233 | 186,189 | 883,636 |
| Digital Music | 127,174 | 66,010 | 1,584,082 |
| Musical Instruments | 903,060 | 112,132 | 1,512,530 |
| Gift Cards | 128,873 | 1,547 | 147,194 |
| All Beauty | 319,335 | 32,486 | 371,345 |

**工业数据集**：来自大规模工业推荐系统的一个月用户行为数据，按业务需求分为 Domain 0 和 Domain 1。

### 3.2 实验设置

#### 3.2.1 基线方法

单域模型8个：PNN、DCN、DeepFM、xDeepFM、DIEN、AutoInt、FiBiNET、IntTower。多域模型6个：Shared Bottom、MMOE、PLE、STAR、SAR-Net、DFFM。

#### 3.2.2 评估指标

- **AUC**：ROC曲线下面积，衡量模型区分正负样本的能力
- **RelaImpr**：相对改进，$(AUC_{model}-0.5)/(AUC_{base}-0.5) - 1$

#### 3.2.3 训练细节

8卡 Tesla V100，batch size=128，dropout=0.3，L2正则化，AdamW优化器，CyclicLR学习率调度（范围 $[1\times10^{-6}, 8\times10^{-5}]$），LoRA rank=8, alpha=32。

### 3.3 实验结果与分析

**主实验（RQ1）**：

| 方法 | Fashion AUC | Musical Instruments AUC | Gift Cards AUC |
|------|-------------|------------------------|----------------|
| xDeepFM (最佳单域) | 0.7031 | 0.6893 | 0.6121 |
| PLE (最佳多域) | 0.6842 | 0.6813 | 0.6375 |
| **Uni-CTR** | **0.7523** | **0.7569** | **0.7246** |

Uni-CTR 在三个域分别取得 **24.22%、35.71%、63.35%** 的相对改进。特别值得注意的是，在数据最稀疏的 Gift Cards 域（仅147K样本），改进幅度最大（63.35%），证明 LLM 预训练知识有效补偿了数据稀疏问题。

**工业数据集结果**：

| 方法 | Domain 0 AUC | Domain 1 AUC |
|------|-------------|-------------|
| STAR (最佳多域) | 0.7000 | 0.6638 |
| **Uni-CTR** | **0.7387** | **0.6881** |

相对改进超过 10.26%，验证了方法在真实工业场景的有效性。

### 3.4 零样本预测（RQ2）

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/zeroshot.png|800]]
> 图3：零样本预测对比。在未见域 All Beauty 上，Uni-CTR 的 General Network 比最佳多域基线高出超过6个百分点。

用3个域训练、在第4个域（All Beauty）做零样本预测：单域模型AUC约0.51（接近随机），多域模型有一定提升，Uni-CTR 的 General Network 大幅超越所有基线，证明其真正的零样本泛化能力。

### 3.5 规模化定律（RQ3）

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/scale.png|800]]
> 图4：不同规模LLM骨干的性能对比（TinyBERT 14M → BERT 110M → DeBERTa-V3-Large 340M → Sheared-LLaMA 1.3B）。

随着LLM规模增大，性能稳步提升，验证了 scaling law 在 Uni-CTR 中同样适用。值得注意的是，仅用110M参数的BERT作骨干，Uni-CTR 已超越所有传统多域模型。

### 3.6 可扩展性验证（RQ4）

冻结已训练好的3域 Uni-CTR，仅添加并训练新 DSN 适配 Digital Music 域：AUC=0.6140，比 STAR 扩展方案高9.83%，比完全重训的单域模型高19.12%。证明 LLM 骨干已有效学习跨域共性，新域仅需训练轻量 DSN。

### 3.7 消融实验（RQ6）

| 配置 | Fashion | Musical Instruments | Gift Cards |
|------|---------|--------------------|----|
| Uni-CTR (full) | 0.7391 | 0.7395 | 0.7073 |
| w/o ladder | 0.7084 | 0.6975 | 0.6723 |
| w/o LLM | 0.6954 | 0.6923 | 0.6100 |
| MMOE (340M) | 0.7038 | 0.7005 | 0.6712 |
| STAR (340M) | 0.7107 | 0.7016 | 0.6775 |

去掉 Ladder Network 后性能明显下降，说明多层语义信息的提取至关重要。去掉 LLM（用同层数DNN替代+ID输入）后性能最大幅下降，尤其在稀疏域 Gift Cards（从0.7073降至0.6100）。即使将MMOE/STAR参数量扩至340M（与DeBERTa相同），仍远不如 Uni-CTR。

Prompt 消融实验表明：Full Prompt > Only Feature ID + Name > Only Feature ID，语义信息越丰富性能越好。

### 3.8 可视化分析（RQ5）

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/dsn-untrained.png|600]]
> 图5a：未训练DSN的t-SNE可视化，三域表示完全混杂。

![[搜推精排模型发展脉络/papers/8-LLM赋能CTR/Uni-CTR/assets/dsn-trained.png|600]]
> 图5b：训练后DSN的t-SNE可视化，三域表示清晰分离。

LLM不同层的可视化显示：低层表示各域混合（捕获共性），高层表示逐渐分离（捕获粗粒度特性），DSN进一步将各域表示清晰区分。

### 3.9 推理加速

工业部署方案：导出为ONNX格式 → TensorRT FP16量化 → Tesla V100上batch_size=32, seq_len=256时总延迟80ms，单样本约2ms，AUC损失<0.01。满足工业推荐系统rank阶段延迟要求。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文结论中提到将继续研究多域CTR预测中输入模态（modalities）的增强，暗示可能探索多模态特征（图片、视频）的融合。

### 4.2 基于分析的未来方向

1. **方向1：多模态特征融合**
   - 动机：当前仅用文本prompt，商品图片、用户行为序列中的视觉信息被忽略
   - 可能的方法：在 LLM 骨干前加入视觉编码器（如CLIP），将图片特征与文本prompt拼接
   - 预期成果：在图片信息丰富的域（如Fashion）获得额外提升
   - 挑战：推理延迟大幅增加，需要更极致的加速方案

2. **方向2：更轻量的 LLM 骨干**
   - 动机：1.3B参数对在线推理仍有挑战，单样本2ms已接近上限
   - 可能的方法：知识蒸馏到更小模型、使用专为推荐设计的预训练模型、Speculative decoding
   - 预期成果：保持80%+性能的同时延迟降至0.5ms
   - 挑战：如何在压缩中保留世界知识

3. **方向3：动态域适应**
   - 动机：现实中域的边界是模糊的，新域可能与某些旧域有重叠
   - 可能的方法：用 meta-learning 让 DSN 能快速适应，或设计域间知识迁移机制
   - 预期成果：新域冷启动阶段用更少样本达到较好性能
   - 挑战：如何平衡域间迁移与域特定建模

### 4.3 改进建议

1. **改进1：Prompt 工程优化**
   - 当前问题：prompt模板是人工设计的固定格式，可能不是最优
   - 改进方案：用 prompt tuning 或 soft prompt 学习最优输入表示
   - 预期效果：进一步提升LLM对推荐特征的理解精度

2. **改进2：更精细的 Ladder 连接策略**
   - 当前问题：固定频率 $\phi$ 从所有层均匀采样，可能不是最优
   - 改进方案：用 NAS 或可学习的选择机制自动确定最佳连接层
   - 预期效果：用更少ladder获得相同或更好性能

## 五、我的综合评价

### 5.1 价值评分

**7.5/10** - 将LLM引入多域CTR预测的开创性工作，架构设计合理，实验充分，但受限于特定数据集选择和公开数据集规模。

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 首次将LLM作为多域CTR的共享骨干，Masked Loss保证DSN解耦是巧妙设计，但LLM+推荐的大方向并非首创 |
| 技术质量 | 8/10 | 数学推导严谨（梯度解耦证明完整），架构各组件设计合理，Ladder+Gate+Tower层次清晰 |
| 实验充分性 | 7/10 | 6个RQ覆盖性能、零样本、规模律、扩展性、可视化、消融，但仅用Amazon数据集（公开）和未具名工业数据集，缺少Ali-CCP等常用CTR基准的对比 |
| 写作质量 | 8/10 | 组织清晰，符号一致，图表丰富，数学推导易读，水平较高 |
| 实用性 | 7/10 | 工业场景已验证可行（2ms/样本），但1.3B参数对中小公司仍有门槛；在语义特征缺失的场景（如Ali-CCP匿名ID）不适用 |

### 5.2 重点关注

#### 值得关注的技术点
- Masked Loss 策略的数学证明：确保各域DSN参数独立更新，这是实现"可插拔"的关键
- Ladder Network 从LLM多层提取表示的思路：类似于feature pyramid在CV中的作用，低层捕获通用模式，高层捕获域特定模式
- General Network 的零样本能力：在全新域上 AUC 比传统多域模型高6个百分点

#### 需要深入理解的部分
- Gate Network 的注意力池化如何平衡 LLM 最后层表示与 Ladder 输出的权重分配
- LoRA 微调 1.3B LLM 时的学习率调度策略（CyclicLR 范围 $[10^{-6}, 8\times10^{-5}]$）
- 推理加速的 ONNX+TensorRT 方案中 AUC 损失 <0.01 的具体 trade-off

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[STAR|STAR: One Model to Serve All]] - 星形拓扑多域CTR基线，参数共享乘法设计
- [[PLE|PLE: Progressive Layered Extraction]] - 渐进式分层提取，共享与域特定专家分离
- [[MMOE|MMOE: Multi-gate Mixture-of-Experts]] - 多门混合专家网络，多域/多任务经典方法

### 6.2 背景相关
- [[Sheared-LLaMA|Sheared-LLaMA]] - 论文使用的1.3B LLM骨干，从LLaMA裁剪而来
- [[LoRA|LoRA: Low-Rank Adaptation]] - 用于高效微调LLM骨干的关键技术
- [[LST|Ladder Side-Tuning]] - Ladder Network的灵感来源，侧网络提取骨干中间表示

### 6.3 后续工作
- [[CTRL|CTRL: Connect Tabular and Language Model for CTR]] - 同时期工作，LLM辅助CTR但未做多域
- [[FLIP|FLIP: Towards Fine-grained Alignment between ID and Language]] - ID与语言对齐用于推荐

## 外部资源

论文未提供开源代码仓库链接，但在实验部分提到"settings can be seen in our open source code"，可能后续会开源。

> [!tip] 关键启示
> LLM 的世界知识是解决多域CTR中数据稀疏和冷启动的天然武器。通过自然语言作为跨域"通用语言"，不同业务场景的特征可以在统一语义空间中交互。Masked Loss 保证的 DSN 解耦是工业落地的关键——新业务上线只需训练一个轻量 DSN，不影响任何已有服务。

> [!warning] 注意事项
> - 方法依赖文本语义特征，对纯ID匿名数据集（如Ali-CCP）不适用
> - 1.3B LLM骨干的推理成本较高（2ms/样本），需要ONNX+TensorRT加速
> - 论文工业数据集细节未公开，难以完全复现
> - Prompt模板为人工设计，迁移到不同业务需重新设计

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐搜推从业者阅读。架构思路优雅，对"LLM如何融入工业推荐系统"给出了清晰答案。特别适合有多域CTR需求且具备LLM serving能力的团队参考。
