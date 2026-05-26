---
title: "MTmixAtt：美团推荐排序模型的Scaling Up技术应用"
authors:
  - 田园
  - 齐宪阳
  - 蒯支睿
  - 胡朝煜
  - 刘畅
  - 王磊
  - 孙权
  - 谷体林
  - 张维民
  - 于航
  - 于磊
  - 张帅
  - 王旭东
  - 吴晓琳
  - 姜春阳
organization: 美团 搜索和推荐平台部
date: 2025-10-14
source: https://km.sankuai.com/collabpage/2729162274
type: internal-tech-blog
tags:
  - recommendation-system
  - scaling-law
  - ranking-model
  - GPU-optimization
  - feature-interaction
  - MoE
  - attention-mechanism
  - 美团
status: read
---

## 一句话总结

MTmixAtt（MulTi-mix Attention）是美团针对推荐排序模型提出的可堆叠、GPU友好的特征交叉架构，通过 Token Mixing + Per-Token FFN（MoE）的统一结构替代传统手工交叉模块，在首页推荐精排场景验证了推荐系统的 Scaling Law，模型参数从 3.5M 扩展至 0.4B 持续带来在线业务收益。

## 背景与动机

### 推荐系统 Scaling 的困境

NLP 领域通过 Scaling Law（模型性能随算力与规模扩展显著提升）实现了 LLM 的技术跃迁，但推荐系统在规模化扩展时面临独特瓶颈：

**三大核心挑战：**

1. **异构特征建模困难** — 推荐系统需处理用户 ID、商品属性、时间、上下文等多类型特征，稀疏性与稠密性交织，语义关联复杂。Transformer 原生设计针对等长、同质 token 序列，直接拼接特征输入会导致复杂关系建模困难。

2. **GPU 算力利用率极低** — 传统推荐模型深受 CPU 时代影响，依赖手工交叉特征模块（如 DCN、DeepFM），核心操作在现代 GPU 上多为内存密集型（Memory-Bound）而非计算密集型（Compute-Bound），导致 MFU（浮点运算利用率）极低（传统模型约 6%，远低于 LLM 的 40-50%）。

3. **规模化扩展收益不确定** — 推荐系统的业务目标（CTR 预估）是否具备类似 NLP 的"问题复杂度-模型规模"对应关系仍需验证，且 CPU 时代模型的计算成本与参数量线性相关，使得简单扩展参数的 ROI 有限。

### 现有方案的局限

此前推荐系统的 Scaling Law 探索主要集中在两个维度：行为序列长度和候选集规模。这两个方向已逼近增长天花板，且未触及模型结构本身的可扩展性问题。

## 核心方法：MTmixAtt 架构

### 设计理念

MTmixAtt 的全称是 **MulTi-mix Attention**，核心设计目标是：用统一的、可堆叠的 Mixer Block 替代传统推荐模型中各种复杂的子网络建模和特征交叉模块（MLP、DCN、DeepFM 等），使模型具备类似 LLM 的规模扩展特性。

### 整体架构

```
输入特征 → Tokenizer → [Mixer Block × N] → 输出预测
```

模型整体由三部分组成：

1. **Tokenizer**：将异构特征转化为统一维度的 token 序列
2. **Mixer Block 堆叠**：核心特征交叉模块，可通过堆叠 N 层扩展参数
3. **预测头**：输出 CTR/CTCVR 等预估值

### Tokenizer 设计

如何将推荐系统的异构特征（稀疏 ID + 稠密特征 + 序列特征）转化为高质量的 token 序列，是决定模型表达能力的关键。文章探索了三种方案：

1. **Group-wise Tokenizer**：按语义将特征分组，同组特征 Concat 后过独立 MLP 生成 token。保留了人工先验的语义分组信息。
   $$tokens = [MLP_1(concat(g_1)), ..., MLP_{len}(concat(g_{len}))]$$

2. **Auto-Split Tokenizer**：所有特征连接后通过单一 MLP，然后自动分割为 token。完全自动化但可能丢失语义结构。

3. **混合方案**：结合人工先验和自动学习的优势。

最终选择了 Group-wise Tokenizer，因为它在保留特征异构性的同时提供了足够的多样性。

### Mixer Block 结构

每个 Mixer Block 由两大模块构成：

**Token Mixing 模块：**
- 负责将不同 token 重新切割、充分混合
- 实现跨 token 的信息交互
- 类比 NLP 中的 Self-Attention，但针对推荐场景做了适配

**Per-Token FFN 模块（Multi Mix / MoE）：**
- 对混合后的 token 做高效特征交叉
- 采用 MoE（Mixture of Experts）机制实现稀疏计算
- 每个 token 独立经过专家网络，实现差异化建模

### 关键设计创新

1. **多头融合门控（Multi-head Gated Fusion）**：在多个注意力头的输出之间引入门控机制，动态调节不同头的贡献权重

2. **层间残差连接**：确保深层堆叠时梯度流动稳定，解决深层结构的训练退化问题

3. **归一化策略**：探索了适合推荐模型的归一化方案，保证训练稳定性

4. **混合残差设计**：同时保留原始特征、浅层交互特征与深层交互特征，在特征保留与梯度流动上提供更大灵活性

## 工程优化

### 训练性能优化

1. **同步 Batch Norm**：解决大规模训练时的显存问题
2. **半精度训练**：Multi Mix 模块主要由大矩阵乘操作构成，适合使用 FP16 进行性能优化
3. **样本格式与存储优化**：样本存储成本降低约 95%，训练时间大幅缩短

### 推理性能优化

1. **基于 Triton 的算子融合**：在 Multi Mix 层引入 MoE 后，通过 Triton 编写融合算子，减少 kernel launch 开销
2. **手写融合算子**：针对 MTmixAtt 结构设计专用融合算子
3. **非计算密集型算子优化**：优化访存密集型操作
4. **Sparse Lib 优化**：稀疏参数库的性能优化

**性能数据：** 0.2B 未优化模型中 Multi Mix 层占端到端显存开销约 60%、时间开销约 55%。优化后在模型参数扩大 10 倍以上的情况下，耗时与基线几乎持平。

### MFU 提升

| 版本 | 参数量 | MFU |
|------|--------|-----|
| 传统 Base 模型 | 8M | 6% |
| ScalingUp 一期 | 27M | 11%（+83%） |
| ScalingUp 三期（0.2B）| 200M | ~37%（Multi Mix 模块） |

## Scaling Law 验证

### 离线实验

参数量从 3.5M 增长到 1B 的实验中，离线 AUC 和 GAUC 指标随着参数量增加持续提升，验证了推荐系统下 MTmixAtt 结构符合 Scaling Law。

| 阶段 | 参数量 | 离线 AUC 变化 |
|------|--------|--------------|
| Base | 8M | baseline |
| 二期 | ~18M | Ctr-G-AUC +181bp |
| 三期 | 200M (0.2B) | 持续提升 |
| 四期 | 400M (0.4B) | 进一步提升 |

### 在线业务收益

**二期上线效果（2025年8月）：**
- 推荐大盘：支付PV +1.28%（显著），实付GTV +0.95%（显著）
- Feed 低卡：支付PV +2.65%（显著），实付GTV +1.98%（显著）
- 新颖性：7日新颖item曝光占比 +0.49%（显著）

**三期上线效果（2025年10月，0.2B）：**
- Feed 低卡：支付PV +1.28%（显著），点击PV +1.02%（显著）
- 推荐整体：支付PV +0.72%（显著），点击PV +0.73%（显著）
- 性能：API AVG +1.28ms，工程侧优化叠加后耗时预计持平

**四期上线效果（2026年1月，0.4B → MTLightAttention）：**
- Feed 低卡：实付GTV +1.02%（显著）
- 推荐整体：实付GTV +0.75%

## 后续演进

### MTLightAttention（0.4B）

四期将 MTmixAtt 进一步演进为 **MTLightAttention**（Multi-Token Light Attention），引入自注意力机制替代原有的 mix attention，原因是：
- 自注意力机制在 NLP/CV 中已有成熟的开源优化技术（如 FlashAttention）
- 能够复用大模型生态的工程优化成果
- 直接使用原始自注意力效果不佳，需要针对推荐场景做定制化改造

### 行业扩展

MTmixAtt 架构已在美团内部多个场景得到验证和应用：
- 首页推荐精排（主战场）
- 酒店频道搜索排序
- 综合搜索排序
- 频道前置页推荐

酒店场景一期实验（27M 参数）即取得频道订单 +1.06%（显著）、实付 +1.39%（显著）的收益。

## 核心洞察与思考

1. **统一可堆叠结构是推荐模型 Scaling 的关键前提**：只有设计出单一的、可无限堆叠的基础 Block，才能像 LLM 一样通过简单地增加层数来扩展模型。MTmixAtt 的成功说明了"追求推荐领域单一且具备 scale 特性的特征交叉结构"这条路径的可行性。

2. **工程与算法的协同设计（Co-design）不可或缺**：仅有好的模型结构不够，必须配合 GPU 友好的计算模式（Compute-Bound > Memory-Bound）、算子融合、稀疏计算等工程优化，才能在有限资源下实现真正的 Scaling。

3. **推荐系统的 Token 化需要特殊处理**：不同于 NLP 中的同质 token，推荐特征是异构的（ID、数值、序列混合），好的 Tokenizer 设计是模型效果的前提。

4. **Scaling 收益存在边际递减但远未饱和**：从 3.5M 到 0.4B，每次参数翻倍都带来在线收益，说明推荐排序任务的复杂度足以支撑更大规模的模型。

5. **MoE 是推荐模型 Scaling 的重要工具**：通过稀疏激活，在不显著增加推理成本的前提下扩展模型容量。

## 与相关工作的对比

| 方案 | 机构 | 核心思路 | 规模 |
|------|------|----------|------|
| MTmixAtt | 美团 | Token Mixing + Per-Token MoE，判别式 | 0.4B |
| MetaGR/HSTU | Meta | 生成式推荐，Causal Transformer 建模行为序列 | 1.5T |
| RankMixer | 字节跳动 | TokenMixer 架构，mixing & FFN | ~1B |
| TokenMixer-Large | 字节跳动 | RankMixer 升级版，Mixing & Reverting + Sparse MoE | 15B(离线)/7B(在线) |

MTmixAtt 与 RankMixer 在结构理念上高度相似（都是 Token Mixing + Channel Mixing），但 MTmixAtt 是美团的独立演进版本，重点强调了工程协同优化和实际业务落地。

## 关联笔记

- [[Training_Compute-Optimal_Large_Language_Models]] — Chinchilla Scaling Law，LLM 领域的 Scaling 理论基础
- RankMixer 论文 — 字节跳动提出的 TokenMixer 架构
- MetaGR/HSTU — Meta 的生成式推荐 Scaling 方案
