---
paper_id: "[arXiv:2311.11351](https://arxiv.org/abs/2311.11351)"
title: "Scaling Law of Large Sequential Recommendation Models"
authors: "Gaowei Zhang, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen"
institution: "Renmin University of China (Gaoling School of AI) & WeChat, Tencent & UC San Diego"
pushlication: "RecSys 2024 (arXiv 2023-11-19)"
tags:
  - 精排论文
  - Scaling-Law
  - 序列推荐
  - Sequential-Recommendation
  - Transformer
  - 大规模推荐模型
  - ID-based
  - 冷启动
  - 长尾推荐
  - 鲁棒性
quality_score: "7.5/10"
link:
  - "[PDF](https://arxiv.org/pdf/2311.11351)"
  - "[arXiv](https://arxiv.org/abs/2311.11351)"
date: "2023-11-19"
---

## 一、研究背景与动机

### 1.1 领域现状

Scaling Law 在 NLP 和 CV 领域已被广泛验证——模型性能与模型规模/数据规模之间存在幂律关系，这为大规模模型的设计和优化提供了重要指导。GPT-4、LLaMA 等模型已经证明了数十亿甚至万亿参数的 Transformer 模型可以取得卓越性能。然而，在推荐系统领域，尤其是序列推荐（Sequential Recommendation）中，对模型 Scaling 行为的理解仍然非常有限。

序列推荐的主流方法（如 SASRec、BERT4Rec）通常只使用 2 层 Transformer，模型参数量极小（~100K 级别）。尽管 NLP 中的序列建模与推荐中的行为序列建模在本质上高度相关（都是对 token 序列的建模），但推荐领域一直缺乏系统性的 Scaling Law 研究。

### 1.2 现有方法的局限性

已有的推荐系统 Scaling 研究存在明显不足：

- Ardalani et al. (Meta, 2022)：研究了 CTR 模型的 Scaling，但使用 MLP 架构且依赖丰富的用户/物品特征，非序列建模
- Guo et al. (2023)：研究了 embedding 参数的 Scaling 效率，发现当前 CTR 模型的参数 Scaling 已经"out of steam"
- Chitlangia et al. (2023)：研究了广告序列模型的 Scaling，但依赖多种特征输入
- Shin et al. (2023)：研究了基于文本的预训练推荐模型 Scaling，但使用 Encoder 架构且依赖文本特征

**核心空白**：没有工作系统研究过纯 ID-based 序列推荐模型的 Scaling Law。这是推荐系统研究社区最主流的数据格式，且在实际场景中额外特征往往不可用。

推荐数据的独特挑战使 Scaling 研究更加困难：(1) 数据极度稀疏——MovieLens-20M 仅有 18.5M 交互，而同等规模的语言模型通常需要 10B+ token；(2) 数据噪声大——用户行为包含大量随机点击；(3) 训练不稳定——直接放大 SASRec 会导致性能退化。

### 1.3 本文解决方案概述

本文首次系统研究了纯 ID-based 序列推荐模型的 Scaling Law。采用 Decoder-only Transformer 架构（与 SASRec 一致），通过两项训练策略创新（Layer-wise Adaptive Dropout + Switching Optimizer）成功将模型扩展到 0.8B 参数。核心发现包括：(1) Scaling Law 在序列推荐中成立，即使在高度数据受限的场景下；(2) 可以用小模型性能预测大模型性能（Predictable Scaling）；(3) 大模型在五个挑战性推荐任务上展现出显著优势。

## 二、解决方案

### 2.1 核心思想

本文的核心思想是：**将序列推荐视为与语言建模类似的序列转导任务，验证 Scaling Law 是否同样适用**。与语言模型不同的是，推荐模型的"词汇表"是 Item ID 集合（不做 tokenization），且面临严重的数据受限问题。作者通过精心设计的训练策略解决了大规模训练的不稳定性，从而首次在推荐领域建立了从 98.3K 到 0.8B 参数的完整 Scaling 曲线。

### 2.2 整体架构

![[illustration.png]]

> 图1：模型架构和不同规模版本的示意图。采用标准的 Decoder-only Transformer（与 SASRec 相同），通过增加层数和宽度实现 Scaling。

#### 模块1：模型架构

采用标准的 Decoder-only Transformer 作为骨干网络：

- **输入层**：Item Embedding $\mathbf{E} \in \mathbb{R}^{n \times d}$（$n$ 为物品总数，$d$ 为隐层维度）+ 可学习位置编码 $\mathbf{P} \in \mathbb{R}^{s \times d}$（$s=50$ 为最大序列长度）
- **编码层**：堆叠多层 Transformer Decoder Block，每层包含 Multi-Head Self-Attention + Position-wise FFN（GeLU 激活）
- **注意力掩码**：单向因果掩码，每个 token 只能 attend 到过去的 token 和自身
- **实现**：基于 HuggingFace Transformers + RecBole 框架

模型规模配置：

| 非嵌入参数 | 层数 | $d_{model}$ | 注意力头数 | 训练轮数 |
|-----------|------|-------------|-----------|---------|
| 98.3K | 2 | 64 | 2 | 30 |
| 786K | 4 | 128 | 4 | 30 |
| 1.57M | 8 | 128 | 4 | 30 |
| 9.44M | 12 | 256 | 8 | 27 |
| 75.5M | 24 | 512 | 8 | 12 |
| 829M | 48 | 1200 | 24 | 12 |

#### 模块2：Layer-wise Adaptive Dropout

直接放大推荐模型时，训练极不稳定——容易在欠拟合和过拟合之间摇摆。本文提出分层自适应 Dropout：

- **低层**设置较大的 Dropout 率：低层直接处理原始数据信息，过程相对简单，需要防止过拟合
- **高层**设置较小的 Dropout 率：高层将低层的语义信息加工为更抽象的表示，需要防止信息丢失和欠拟合

消融实验表明：对于小模型（2层），有无 Layer-wise Adaptive Dropout 影响不大（CE Loss: 5.6013 vs 5.6249）；但对于大模型（24层），移除后性能显著下降（CE Loss: 4.7504 vs 4.7182），说明固定 Dropout 策略不足以支撑大规模模型的稳定训练。

#### 模块3：Switching Optimizer Strategy

训练过程中发现 Adam 优化器在初始阶段表现良好，但最终收敛 loss 高于 SGD。因此采用两阶段优化策略：

1. **第一阶段**：使用 Adam 优化器训练至收敛
2. **切换点**：Adam 收敛点即为切换点（无需显式学习）
3. **第二阶段**：切换为 SGD 优化器继续训练至最终收敛

实验发现不同切换点对最终性能影响很小，因此直接使用 Adam 收敛点作为切换点即可。

#### 模块4：训练数据与评估

**数据集**：

| 数据集 | 用户数 | 物品数 | 交互数 |
|--------|--------|--------|--------|
| MovieLens-20M | 138,493 | 26,427 | 18,476,840 |
| Amazon (mix) | 367,710 | 240,320 | 21,787,957 |

Amazon 数据集混合了 29 个域的交互记录，按时间戳排序形成统一序列。仅保留 Item ID，不使用任何辅助信息（文本、类别等）。

**评估方式**：
- Scaling Law 评估：使用交叉熵损失（Cross-Entropy Loss）
- 推荐任务评估：HR@N、NDCG@N（N∈{5,10,50}）、Coverage

### 2.3 Scaling Law 分析

#### 模型规模 Scaling

![[model_scaling.png]]

> 图2：MovieLens 数据集上模型规模的 Scaling 曲线。蓝点为用于拟合的小模型，红虚线为拟合的幂律曲线，红点为大模型的实际性能（与预测高度吻合）。

Scaling Law 的数学形式：

$$L(N) = E_N + (N_0 / N)^{\alpha_N}$$

其中 $N$ 为非嵌入参数量，$E_N$ 为不可约损失（估计数据分布的熵），$\alpha_N$ 为数据集相关的 Scaling 指数。

**MovieLens-20M 拟合结果**：$E_N = 4.9$，$N_0 = 6.8 \times 10^5$，$\alpha_N = 0.121$

关键发现：
1. **Scaling Law 成立**：测试损失与非嵌入参数量呈幂律关系
2. **$\alpha_N$ 大于语言模型**：推荐中 $\alpha_N = 0.121$ vs NLP 中 $\alpha_N = 0.07$，说明推荐模型从 Scaling 中获益更快
3. **Predictable Scaling**：用 4 个小模型（98.3K~9.4M）拟合的曲线可以准确预测 75.5M 和 0.8B 模型的性能
4. **数据稀疏性影响 $\alpha_N$**：数据越稀疏，$\alpha_N$ 越小，Scaling 曲线越平坦

#### 数据规模 Scaling

![[data_scaling.png]]

> 图3：不同数据规模下的 Scaling 曲线。即使在高度数据受限的场景（1.8M 交互），Scaling Law 仍然成立。

关键发现：
1. **数据受限下 Scaling Law 仍成立**：即使将数据减少到 10%（1.8M），幂律关系依然存在
2. **大模型更加数据高效**：75.5M 模型在 9.2M 数据上的 loss 等于 98.3K 模型在 18.5M 数据上的 loss——大模型用一半数据就能达到小模型的全量数据性能

#### 数据重复的影响

![[ml_loss.png]]

> 图4：数据重复（多 epoch 训练）对测试损失的影响。前 2-5 个 epoch 有明显收益，之后收益递减。

推荐数据量远小于语言模型（18.5M vs 10B+），因此需要多 epoch 训练。实验发现：
- 单 epoch 无法收敛
- 2-5 epoch 有显著收益
- 6-12 epoch 收益快速递减
- 13-30 epoch 收益归零，大模型开始出现过拟合风险

#### 模型形状的影响

![[aspect.png]]

> 图5：不同 Aspect Ratio（$d_{model}/n_{layer}$）对性能的影响。即使形状变化范围很大，loss 增加也很小。

关键发现：模型性能对形状（深 vs 宽）的依赖很弱，且随模型规模增大，这种依赖进一步减弱。这意味着在推荐系统中，不需要过度搜索最优的深度/宽度配比。

## 三、实验结果

### 3.1 数据集

| 数据集 | 用户数 | 物品数 | 交互数 | 特点 |
|--------|--------|--------|--------|------|
| MovieLens-20M | 138,493 | 26,427 | 18.5M | 电影评分，单域 |
| Amazon (mix) | 367,710 | 240,320 | 21.8M | 29 域混合，跨域 |

### 3.2 实验设置

#### 3.2.1 基线方法

- SASRec (2层, 64d)：标准 Transformer 序列推荐
- GRU4Rec (2层, 64d)：GRU-based 序列推荐
- FMLP (2层, 64d)：MLP-based 序列推荐
- Caser (2层, 64d)：CNN-based 序列推荐
- LSRM_wide (2层, 1200d)：仅增加宽度的大模型
- LSRM_deep (48层, 1200d)：增加深度的大模型（0.83B 参数）

#### 3.2.2 评估指标

- **Cross-Entropy Loss**：Scaling Law 的主要度量，每个 item 视为一个类别，计算下一个 item 的预测概率
- **HR@N (Hit Ratio)**：推荐列表中命中目标 item 的比例
- **NDCG@N**：考虑排序位置的推荐准确度
- **Coverage@N**：推荐多样性度量

### 3.3 实验结果与分析

#### 整体推荐性能

| 方法 | 层数 | $d_{model}$ | 非嵌入参数 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@50 | NDCG@50 |
|------|------|-------------|-----------|------|--------|-------|---------|-------|---------|
| SASRec | 2 | 64 | 98.3K | 0.1463 | 0.0930 | 0.2337 | 0.1212 | 0.5166 | 0.1834 |
| GRU4Rec | 2 | 64 | 180.3K | 0.1559 | 0.1007 | 0.2466 | 0.1281 | 0.5267 | 0.1965 |
| FMLP | 2 | 64 | 167.6K | 0.1498 | 0.0978 | 0.2427 | 0.1255 | 0.5214 | 0.1918 |
| Caser | 2 | 64 | 703.7K | 0.1407 | 0.0877 | 0.2219 | 0.1140 | 0.5036 | 0.1793 |
| LSRM_wide | 2 | 1200 | 34.56M | 0.0595 | 0.0384 | 0.0976 | 0.0506 | 0.2639 | 0.0865 |
| **LSRM_deep** | **48** | **1200** | **0.83B** | **0.2794** | **0.1827** | **0.3786** | **0.2319** | **0.6264** | **0.2794** |

核心观察：
1. **LSRM_deep 显著优于所有基线**：HR@5 和 NDCG@5 几乎是 SASRec 的 2 倍（0.2794 vs 0.1463，0.1827 vs 0.0930）
2. **仅增加宽度会导致性能崩溃**：LSRM_wide（2层1200d）性能远低于 SASRec（2层64d），验证了 embedding collapse 现象
3. **Scaling 模型规模的收益远大于改变模型结构**：LSRM_deep 的提升幅度远超 GRU4Rec/FMLP/Caser 等结构变化带来的提升

#### 长尾物品推荐

![[ml_pop.png]]

> 图6：MovieLens 上不同流行度分组的模型性能。G1 为最热门组，G4 为最冷门组。大模型在冷门物品上的优势更加显著。

![[amazon_pop.png]]

> 图7：Amazon 上不同流行度分组的模型性能。趋势与 MovieLens 一致。

关键发现：在最冷门的 G4 组中，48 层大模型的性能是 2 层小模型的 **3 倍**（而在最热门的 G1 组中仅为 2 倍）。大模型对长尾物品的推荐能力提升更为显著，原因在于大模型具有更强的 few-shot 泛化能力和记忆能力。

#### 冷启动用户推荐

![[ml_length.png]]

> 图8：MovieLens 上不同输入长度（模拟冷启动程度）的模型性能。

![[amazon_length.png]]

> 图9：Amazon 上不同输入长度的模型性能。

关键发现：当用户历史交互极短（5 条）时，大模型的优势最为明显。大模型能为冷启动用户提供更多样化的推荐结果（Coverage@10: 大模型 0.2046 vs 小模型 0.1362），从而更可能命中用户兴趣。

#### 多域迁移推荐

![[cross_num.png]]

> 图10：不同规模模型在 Mix-domain 和 Diff-domain 设置下的绝对性能。

![[cross_percent.png]]

> 图11：不同规模模型相对于 2 层基线的性能提升百分比。Diff-domain 的提升远大于 Mix-domain。

关键发现：在完全跨域（Diff-domain）设置中，大模型的性能提升百分比远超混合域（Mix-domain）设置，说明大模型在跨域知识迁移方面具有显著优势。

#### 鲁棒性挑战

![[robustness.png]]

> 图12：面对噪声输入（删除/替换序列中的 item）时，不同规模模型的性能退化百分比。大模型退化更小。

关键发现：面对输入序列的噪声扰动（删除、替换），大模型的性能退化幅度显著小于小模型。大模型能够捕获更长程的依赖关系，因此对局部扰动不敏感。

#### 用户轨迹预测

![[ml_trajectory.png]]

> 图13：MovieLens 上不同预测长度的轨迹预测性能退化。大模型在长程预测中保持稳定。

![[amazon_trajectory.png]]

> 图14：Amazon 上不同预测长度的轨迹预测性能退化。

关键发现：随着预测长度从 1 增加到 10，小模型性能急剧下降（累积误差放大），而大模型保持了显著的稳定性。这表明大模型具有更强的长期用户兴趣建模能力。

### 消融实验

| 训练策略 | LSRM_2层 (CE Loss) | LSRM_24层 (CE Loss) |
|----------|--------------------|--------------------|
| Both (LAD + SO) | 5.6249 | **4.7182** |
| w/o LAD | 5.6127 | 4.7296 |
| w/o SO | 5.6281 | 4.7230 |
| None | **5.6013** | 4.7504 |

观察：对于小模型，训练策略影响不大（甚至 None 略好）；但对于大模型，两种策略都有明显贡献，且组合使用效果最佳。这验证了训练策略对大规模模型的必要性。

## 四、未来工作建议

### 4.1 作者建议的未来工作

作者在结论中指出两个关键方向：(1) 扩展可用的用户交互行为数据量，缓解推荐系统中的数据稀缺问题；(2) 开发更高效、更稳定的优化方法来训练大规模推荐模型。此外，更高效的部署方法（如量化）对于支持大规模序列模型在实际场景中的应用也至关重要。

### 4.2 基于分析的未来方向

1. **方向1：突破数据瓶颈**
   - 动机：本文最大的限制是数据量（18.5M 交互 vs 语言模型的 10B+ token），Scaling 曲线在 0.8B 参数后可能因数据不足而饱和
   - 可能的方法：数据增强（序列扰动、对比学习）、跨平台数据融合、合成数据生成
   - 预期成果：在更大数据量下验证 Scaling Law 是否持续有效
   - 挑战：推荐数据的隐私限制、跨平台 ID 对齐

2. **方向2：融合特征信息的 Scaling**
   - 动机：本文仅使用纯 ID 序列，但实际系统中有丰富的 side information
   - 可能的方法：将文本/图像特征作为额外 token 融入序列，或作为 embedding 初始化
   - 预期成果：在保持 Scaling 特性的同时利用多模态信息
   - 挑战：如何避免特征信息干扰 Scaling Law 的纯粹性

3. **方向3：工业级在线验证**
   - 动机：本文仅在公开数据集上进行离线实验，缺乏在线 A/B 测试验证
   - 可能的方法：在工业级推荐系统中部署不同规模的模型，验证 Scaling 收益是否转化为在线指标提升
   - 预期成果：建立从离线 Scaling Law 到在线收益的映射关系
   - 挑战：大模型的推理延迟和部署成本

### 4.3 改进建议

1. **改进1：序列长度的 Scaling**
   - 当前问题：固定序列长度为 50，未探索序列长度作为 Scaling 维度的影响
   - 改进方案：将序列长度从 50 扩展到 1000+，研究长度-性能的 Scaling 关系
   - 预期效果：HSTU 论文已证明序列长度对推荐 Scaling 的关键作用

2. **改进2：更先进的架构设计**
   - 当前问题：直接使用标准 Transformer，未针对推荐数据特点做架构优化
   - 改进方案：引入 HSTU 的 Pointwise Attention、Stochastic Length 等推荐专用设计
   - 预期效果：在相同参数量下获得更好的性能，或在相同性能下使用更少参数

3. **改进3：计算效率优化**
   - 当前问题：0.8B 模型需要 8 张 A100 训练，部署成本高
   - 改进方案：模型量化（INT8/INT4）、知识蒸馏、稀疏化
   - 预期效果：在保持性能的同时大幅降低推理成本

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - 这是推荐系统 Scaling Law 研究的重要先驱工作，首次在纯 ID-based 序列推荐中系统验证了 Scaling Law 的存在性。虽然实验规模和架构创新相对有限（相比后续的 Wukong、HSTU），但其开创性意义和对五个挑战性任务的深入分析具有重要参考价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 首次在纯 ID-based 序列推荐中验证 Scaling Law，但架构本身无创新（直接使用 SASRec），训练策略（LAD + SO）也相对简单 |
| 技术质量 | 7/10 | 实验设计合理，Scaling 曲线拟合严谨，但模型规模（0.8B）和数据规模（18.5M）相对有限，未达到工业级验证 |
| 实验充分性 | 8/10 | 五个挑战性任务的设计很有洞察力（长尾、冷启动、跨域、鲁棒性、轨迹预测），消融实验完整，但缺乏在线实验 |
| 写作质量 | 7/10 | 结构清晰，但部分内容重复，且论文中保留了大量注释掉的文字（说明是较早期的版本） |
| 实用性 | 7/10 | 验证了 Scaling Law 的存在性，但未提供工业级部署方案，训练策略的通用性有待验证 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- **$\alpha_N = 0.121$ vs NLP 的 0.07**：推荐模型从 Scaling 中获益的速度比语言模型更快，这为推荐系统的大模型化提供了理论支撑
- **Predictable Scaling 在推荐中可行**：用 <100x 的小模型可以准确预测大模型性能，这对工业界的资源规划有重要意义
- **仅增加宽度会导致 Embedding Collapse**：LSRM_wide 的惨败说明推荐模型的 Scaling 必须增加深度
- **大模型对长尾/冷启动的 3x 提升**：这是推荐系统最痛的问题，Scaling 提供了一条新的解决路径

#### 5.2.2 需要深入理解的部分

- 为什么推荐中 $\alpha_N$ 大于 NLP？可能是因为推荐数据的信息密度更低（稀疏 ID 序列 vs 富语义文本），大模型能更有效地从稀疏信号中提取模式
- 数据重复训练的收益递减规律：2-5 epoch 有效，之后无效——这对工业界的训练资源分配有直接指导意义
- 模型形状的弱依赖性：与 NLP 一致，说明推荐模型也不需要过度调参，可以直接按标准比例放大

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[Wukong_Towards_a_Scaling_Law_for_Large_Scale_Recommendation|Wukong]] - Meta 的堆叠 FM Scaling Law，侧重特征交互维度的 Scaling，与本文侧重序列建模的路线互补
- [[HSTU_Actions_Speak_Louder_than_Words|HSTU]] - Meta 的生成式推荐 Scaling，在工业级规模（1.5T 参数）验证了推荐 Scaling Law，是本文思想的工业级实现
- [[FAT_From_Scaling_to_Structured_Expressivity|FAT]] - 阿里的 Field-Aware Transformer Scaling Law，从理论角度建立了特征交互的 Scaling 理论

### 6.2 背景相关
- SASRec (Kang et al., 2018) - 本文的基础架构，2 层 Transformer 序列推荐的开创者
- Kaplan et al. (2020) "Scaling Laws for Neural Language Models" - NLP Scaling Law 的奠基工作，本文的直接灵感来源
- Ardalani et al. (Meta, 2022) - CTR 模型 Scaling 的先驱研究，发现 MLP-based 模型 Scaling 已饱和

### 6.3 后续工作
- [[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans]] - 字节跳动的统一 Transformer 精排，在更大规模上验证了序列+特征交互的联合 Scaling
- [[TokenMixer_Large_Scaling_Up_Large_Ranking_Models_in_Industrial_Recommenders|TokenMixer-Large]] - 字节跳动的 15B 参数排序模型，工业级 Scaling 的代表
- Climber (arXiv:2502.09888) - 高效 Scaling Law 框架，解决了大规模推荐模型的资源效率问题

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2311.11351)
- [RecSys 2024 论文页面](https://dl.acm.org/doi/10.1145/3640457.3688129)
- [知乎解读：推荐模型 Scaling Up](https://zhuanlan.zhihu.com/p/17891194938)

> [!tip] 关键启示
> 推荐系统的 Scaling Law 确实存在，且 Scaling 指数 $\alpha_N = 0.121$ 大于 NLP 的 0.07——这意味着推荐模型从 Scaling 中获益的潜力比语言模型更大。但推荐数据的稀缺性是最大瓶颈：18.5M 交互远不足以支撑 0.8B 模型的充分训练。核心公式：$L(N) = E_N + (N_0/N)^{\alpha_N}$。

> [!warning] 注意事项
> - 本文仅在公开数据集上验证，缺乏工业级在线实验——后续 HSTU、Wukong 等工作在 Meta 数十亿用户规模上验证了类似结论
> - 序列长度固定为 50，这在实际系统中过短——HSTU 证明了序列长度是推荐 Scaling 的关键维度
> - 仅增加宽度会导致 Embedding Collapse——推荐模型的 Scaling 必须以增加深度为主
> - 训练策略（LAD + SO）的通用性未在其他数据集/架构上验证

> [!success] 推荐指数
> ⭐⭐⭐⭐ 推荐阅读。作为推荐系统 Scaling Law 的先驱工作，本文系统验证了 Scaling Law 在纯 ID-based 序列推荐中的存在性，并通过五个挑战性任务展示了大模型的潜力。虽然实验规模和架构创新不如后续的 HSTU/Wukong，但其开创性意义和清晰的实验设计使其成为理解推荐 Scaling 的重要入门论文。建议与 HSTU、Wukong 对比阅读，形成完整的推荐 Scaling Law 知识体系。
