---
paper_id: "[arXiv:2604.15650](https://arxiv.org/abs/2604.15650)"
title: "Sample Is Feature: Beyond Item-Level, Toward Sample-Level Tokens for Unified Large Recommender Models"
authors: "Shuli Wang, Junwei Yin, Changhao Li, Senjie Kou, Chi Wang, Yinqiu Huang, Yinhua Zhu, Haitao Wang, Xingxing Wang"
institution: "Meituan"
publication: "arXiv 2026-04-17"
tags:
  - SIF
  - 推荐系统
  - 搜推系统
  - Sample-Level-Token
  - HGAQ
  - Sample-Tokenizer
  - SIF-Mixer
  - Transformer
  - 向量化
  - 美团
quality_score: "[8.5]/10"
link:
  - "[arXiv](https://arxiv.org/abs/2604.15650)"
  - "[PDF](https://arxiv.org/pdf/2604.15650)"
date: "2026-05-14"
---

# Sample Is Feature: Beyond Item-Level, Toward Sample-Level Tokens for Unified Large Recommender Models

## 一、研究背景与动机

### 1.1 领域现状

工业推荐系统 Scaling 的两条并行范式：

**范式一：Sample Information Scaling（样本信息 Scaling）**

通过更深、更长的行为序列来丰富每个训练样本的信息含量，包含两个子方向：
- **Sequence Lengthening（序列延长）**：将行为历史从数十条扩展到数千条（SIM、ETA、TWIN、LONGER）
- **Sequence Widening（序列拓宽）**：在每条历史 token 中加入额外上下文特征（DSAN、CAIN、HSTU）

**范式二：Model Capacity Scaling（模型容量 Scaling）**

通过统一的 Transformer 主干将序列建模与特征交互融合（HyFormer、OneTrans、MixFormer）。

### 1.2 现有方法的结构性局限

尽管两条范式各自取得进展，但都存在**两个根本性局限**：

**局限一：样本信息不完整**

现有 Sample Information Scaling 方法仅将历史交互的**子集**编码进序列 token：
- 受存储和 Serving 成本限制，只能保留 bare item embedding，或仅加入精选特征子集
- 训练日志中存储的完整样本上下文（用户画像、上下文信号、行为结果）未被充分利用
- **结构性缺陷**：无法建模 **sample-level、时变特征**（如实时商品热度、竞争曝光度、时间需求变化）

**局限二：特征异构性**

Model Capacity Scaling 方法中，历史 token（item-level）与当前请求的丰富多字段 token 在**信息密度**上存在根本差异，放在同一注意力空间中造成表征不对称，限制了模型表达能力。

### 1.3 本文解决方案概述

提出 **SIF（Sample Is Feature）**：将每个历史 Raw Sample 直接编码为序列 token，在最大化保留样本信息的同时，一并解决序列特征与非序列特征之间的异构性问题。

核心洞察：训练时每个历史交互在训练日志中已有完整请求记录（用户画像、商品特征、上下文信号、预计算交叉特征），瓶颈不在数据——瓶颈在于**表征方式**。

---

## 二、解决方案

### 2.1 核心思想

SIF 的核心思想是将每个历史交互的 **Raw Sample**（完整的特征快照）量化为一个 **Token Sample**，以 token 均质序列替代传统 item embedding 序列。

### 2.2 整体架构

![[SIF_p1.png|800]]
> 图1：SIF 整体架构。（a）Sample Tokenizer 通过 HGAQ 将 Raw Sample $\mathcal{S}$ 压缩为 Token Sample $\mathcal{Q}$。（b）Sample Splicing 用 Token Sample 替换 item embedding，生成 token 均质的序列。（c）Sample Serialization 将历史 Token Sample 按时间顺序排列，附上时间位置编码。（d）SIF-Mixer 通过 codebook 查找将每个 Token Sample $\mathcal{Q}$ 嵌入为 $\mathbf{H}^0$，然后堆叠 $N$ 个 SIF Block（Token-level Mixer + Sample-level Mixer + Token-level FFN），最后接预测头。

SIF 由两个核心组件构成：

| 组件 | 功能 | 关键机制 |
|------|------|----------|
| **Sample Tokenizer** | 将 Raw Sample 量化为 Token Sample | HGAQ（分层组自适应量化） |
| **SIF-Mixer** | 对均质样本表征进行深度特征交互 | Token-level + Sample-level Mixing |

### 2.3 Sample Tokenizer：分层组自适应量化（HGAQ）

#### 2.3.1 Group-Wise Decomposition（分组分解）

将 Raw Sample $\mathcal{S}$ 按语义划分为 $G=4$ 个组：

$$
\mathcal{S} = [\underbrace{\mathbf{f}^{\text{user}}}_{G_1} \mid \underbrace{\mathbf{f}^{\text{item}}}_{G_2} \mid \underbrace{\mathbf{f}^{\text{ctx}}}_{G_3} \mid \underbrace{\mathbf{f}^{\text{cross}}}_{G_4}]
$$

| 组别 | 特征类型 | 示例字段 |
|------|----------|----------|
| $G_1$ | 用户特征 | user_id, age, gender, city_tier, avg_ctr_7d |
| $G_2$ | 商品特征 | item_id, category, price_bucket, popularity_bucket |
| $G_3$ | 上下文特征 | hour_of_day, day_of_week, device_type, promotion_flag |
| $G_4$ | 交叉特征 | user-category affinity, item-context co-occurrence, ALS CF scores |

#### 2.3.2 Adaptive Intra-Group Sub-tokenization（自适应组内子 token 化）

每组内的特征数差异很大，直接压缩到单一 token 会限制 Token-level Mixer 的解耦能力。

**子 token 数自适应公式**：

$$
K_g = \left\lceil \frac{|\mathcal{F}_g|}{B} \right\rceil
$$

其中 $B$ 是 **sub-token granularity**（每个子 token 对应的特征字段数，默认 $B=32$）。每组特征均匀划分为 $K_g$ 个非重叠子集，每个子集通过组+槽特定线性层投影到固定维度 $d_0$（默认 $d_0=16$）：

$$
\tilde{\mathbf{f}}^{(g,k)} = W^{(g,k)}_{\mathrm{proj}}\, \mathbf{f}^{(g,k)} \in \mathbb{R}^{d_0}
$$

#### 2.3.3 Residual Vector Quantization（残差量化）

对每个子 token $(g,k)$，应用 $M$ 层（默认 $M=3$）残差向量量化（RVQ）：

$$
q^{(g,k,m)} = \arg\min_v \|\mathbf{r}^{(g,k,m)} - \mathbf{c}^{(g,k,m)}_{v}\|_2, \quad \mathbf{r}^{(g,k,m)} = \mathbf{r}^{(g,k,m-1)} - \mathbf{c}^{(g,k,m-1)}_{q^{(g,k,m-1)}}
$$

初始残差 $\mathbf{r}^{(g,k,1)} = \tilde{\mathbf{f}}^{(g,k)}$。

**存储压缩比**：
- 原始快照：$|\mathcal{F}_{\text{non-seq}}| \times d_e \times 32 = 600 \times 8 \times 32 = 153{,}600$ bits
- HGAQ 存储：$T \times M \times \log_2 V = 27 \times 3 \times 8 = 648$ bits
- **压缩比 ≈ 237×**

#### 2.3.4 Label-Supervised Codebook Training（标签监督的 Codebook 训练）

Sample Tokenizer 与排序目标联合训练，通过辅助 pCTR loss 确保 codebook 按预测相关性组织，而非仅按重建误差：

$$
\hat{y}^{\text{token}} = \sigma\bigl(\text{MLP}(\hat{\mathbf{s}})\bigr), \quad \hat{\mathbf{s}} = \Bigl[\sum_m \mathbf{c}^{(g,k,m)}_{q^{(g,k,m)}}\Bigr]_{g,k}
$$

$$
\mathcal{L}_{\text{token}} = \mathcal{L}_{\text{BCE}}(\hat{y}^{\text{token}}, y)
$$

### 2.4 Sample Splicing and Serialization（样本拼接与序列化）

标准行为序列仅记录 item ID。SIF 将每个条目升级为完整特征快照并压缩为 Token Sample：

$$
\underbrace{\{i_1,\;\ldots,\;i_L\}}_{\text{item sequence (standard)}}
\;\xrightarrow{\;\mathcal{T}\;}
\underbrace{\{\mathcal{Q}_1,\;\ldots,\;Q_L\}}_{\text{sample sequence (SIF)}}
$$

### 2.5 SIF-Mixer

#### 2.5.1 输入布局

输入为 $(L+1) \times T \times d_0$ 的初始隐藏状态 $\mathbf{H}^0$：
- **Seq Token Samples**（$l=1,\ldots,L$）：通过 codebook 查找嵌入 + recency 编码
- **Target Token Sample**（$l=0$）：当前请求特征在线投影到 codebook 空间

#### 2.5.2 SIF Block

每个 Block 包含三个子操作：

**(i) Token-level Mixer（行注意力)**

在每个样本的 $T$ 个子 token 上运行自注意力，建模单个样本内的特征交互：

$$
\tilde{\mathbf{H}}^n_l = \mathbf{H}^{n-1}_l + \text{MHA}\bigl(\text{LN}(\mathbf{H}^{n-1}_l)\bigr), \quad l = 0,\ldots,L
$$

**(ii) Sample-level Mixer（列注意力)**

在每个子 token 位置的所有 $L+1$ 个样本上运行自注意力，建模样本间的时间交互关系：

$$
\bar{\mathbf{H}}^n_{*,p} = \tilde{\mathbf{H}}^n_{*,p} + \text{MHA}\bigl(\text{LN}(\tilde{\mathbf{H}}^n_{*,p})\bigr), \quad p = 1,\ldots,T
$$

**(iii) Token-level FFN**

对每个位置执行逐位非线性变换：

$$
\mathbf{H}^n_{l,p} = \bar{\mathbf{H}}^n_{l,p} + \text{FFN}\bigl(\text{LN}(\bar{\mathbf{H}}^n_{l,p})\bigr)
$$

#### 2.5.3 预测头与训练目标

**预测头**：目标样本表征 = $T$ 个子 token 输出的平均池化：

$$
\mathbf{h} = \frac{1}{T}\sum_{p=1}^{T} \mathbf{H}^N_{0,p}
$$

$$
\hat{y} = \sigma\!\left(\mathbf{w}_2^\top \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + b_2\right)
$$

**训练目标**：

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \beta\,\mathcal{L}_{\text{VQ}} + \gamma\,\mathcal{L}_{\text{align}}
$$

其中 $\beta = 1.0$（VQ commitment loss），$\gamma = 0.25$（对齐损失，使 Target Token 投影与 codebook 空间对齐）。

**复杂度**：$O(L^2 \cdot T \cdot d_0)$，与标准序列注意力复杂度相同（因为 $T \ll L$）。

### 2.6 关键创新点总结

1. **从 Item-Level 到 Sample-Level 的范式跃迁**：每个历史 token 从 bare item embedding 升级为完整 Raw Sample，弥补了样本信息 Scaling 和模型容量 Scaling 各自的结构性缺陷
2. **HGAQ 分层量化**：Group-wise 分解 + 自适应子 token 化 + 标签监督的 RVQ，首次对完整 Raw Sample 进行高效量化存储（237×压缩比）
3. **Token 均质序列**：统一 Token Sample 表示消除历史 token 与当前请求 token 之间的表征不对称
4. **SIF-Mixer 分解注意力**：Row/Column 分解的注意力设计，同时建模样本内特征交互和样本间时序交互，复杂度与标准序列注意力持平

---

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 序列长度 | 特征字段数 |
|--------|------|----------|------------|
| **Industrial Dataset**（美团本地生活） | 10亿+条曝光记录，5000万+用户，500万+商品 | $L=1000$ | $\sim$600+（含4个语义组） |

### 3.2 基线方法

| 类型 | 模型 |
|------|------|
| Feature Interaction 变体 | DCNv2、Wukong、RankMixer |
| Sequence Modeling 变体 | DIN、SIM、LONGER |
| 统一架构 | HyFormer、OneTrans |

### 3.3 主要结果

| 模型 | CTR GAUC | CTR AUC 提升 | CVR GAUC | CVR AUC 提升 | 参数量 | TFLOPs |
|------|----------|-------------|----------|-------------|--------|--------|
| DCNv2+DIN（基线） | 0.7614 | — | 0.7891 | — | 48M | 0.31 |
| Wukong+LONGER | — | +0.72% | — | +0.63% | 62M | 0.42 |
| RankMixer+LONGER | — | +0.79% | — | +0.68% | 53M | 0.40 |
| HyFormer | — | +1.01% | — | +0.88% | 120M | 0.87 |
| OneTrans | — | +0.96% | — | +0.83% | 115M | 0.82 |
| **SIF（本文）** | **0.7803** | **+1.89%** | **—** | **+1.61%** | **128M** | **0.93** |

SIF 在所有指标上均显著优于所有基线（$p < 0.01$），相比最强基线 HyFormer：
- CTR GAUC 提升 **+0.88%**（相对 +1.14%）
- 离线 AUC 绝对提升 **+0.0071**（CTR），在线 A/B 实测 CTR 提升 **+2.03%**

### 3.4 消融实验

#### 3.4.1 Sample Tokenizer 消融

| Token 表征方式 | ΔCTR-GAUC | ΔCVR-GAUC | 存储压缩比 |
|----------------|----------|----------|----------|
| **SIF (HGAQ token)** | — | — | ≈237× |
| Item ID only | −1.00% | −0.86% | ≈2400×（信息丢弃，非真正压缩） |
| Item ID + key features | −0.60% | −0.51% | ≈185× |
| Raw sample emb (d=512, dense) | −0.27% | −0.23% | ≈9× |

**关键发现**：HGAQ 在存储压缩比和效果上均优于所有变体。Dense embedding 虽保留所有特征但效果更差，原因：
1. 连续表征比离散 codebook 更难优化
2. RVQ 隐式聚类提供正则化
3. 所有历史位置共享 codebook → 时序对齐

#### 3.4.2 SIF-Mixer 架构消融

| 注意力策略 | ΔCTR-GAUC | ΔCVR-GAUC | 复杂度 |
|-----------|----------|----------|--------|
| **SIF-Mixer (factored row+col)** | — | — | $O(L^2 T + LT^2)$ |
| Flat attention | −0.24%±0.01% | −0.20%±0.01% | $O((LT)^2)$ |
| Pooled-then-attend | −0.81%±0.02% | −0.68%±0.02% | $O(L^2)$ |

**关键发现**：因子分解的 SIF-Mixer 效果最好，将 $T$ 个子 token 池化会丢失样本内结构，直接扁平注意力复杂度在 $L=1000$ 时不可接受。

### 3.5 敏感性分析：Sub-token Granularity $B$

![[k_sweep_gauc.png|600]]
> 图2：CTR GAUC vs. sub-token granularity $B$。$B=32$（$T=20$）达到最优 GAUC (0.7758)，SIF 在所有 $B$ 值下均优于 HyFormer（虚线，GAUC=0.7691）。

### 3.6 Scaling 分析

#### 模型深度 Scaling

![[k_sweep_single.png|600]]
> 图3：CTR GAUC vs. TFLOPs（模型深度 $N \in \{1,2,3,4,6\}$）。SIF 在整个深度范围内均实现更优的 GAUC-FLOPs 权衡，$N=4$ 为最优。

#### 序列长度 Scaling

| 序列长度 $L$ | OneTrans | HyFormer | SIF | Δ(SIF−HyFormer) |
|-------------|----------|----------|-----|----------------|
| 100 | 0.7674 | 0.7680 | **0.7693** | +0.0013 |
| 200 | 0.7689 | 0.7695 | **0.7745** | +0.0050 |
| 500 | 0.7706 | 0.7712 | **0.7782** | +0.0070 |
| 1000 | 0.7710 | 0.7715 | **0.7803** | **+0.0088** |
| 2000 | 0.7713 | 0.7718 | **0.7820** | **+0.0102** |

**关键发现**：SIF 的优势随序列长度增加而**单调扩大**（$L=100$ 时 +0.0013 → $L=2000$ 时 +0.0102），体现了 sample-level token 的结构性优势。HyFormer 和 OneTrans 在 $L \geq 500$ 后快速饱和。

![[scaling_curve.png|600]]
> 图4：CTR GAUC vs. 序列长度 $L$。SIF（红线）的 Scaling 曲线最陡，随序列增长持续拉开与基线的差距。

### 3.7 在线 A/B 实验

在美团外卖平台 **5% 流量 Holdout，持续7天**：

| 指标 | 提升幅度 |
|------|----------|
| **CTR** | **+2.03%** |
| **CVR（转化率）** | **+1.21%** |
| **GMV/session** | **+1.35%** |

**按用户序列长度分层**：

| 用户序列长度 | ΔCTR | ΔCVR | ΔGMV/session |
|------------|------|------|-------------|
| $L < 10$（冷启动用户） | +0.53% | +0.31% | +0.37% |
| $10 \le L < 100$ | +1.18% | +0.71% | +0.84% |
| $100 \le L < 500$ | +2.07% | +1.24% | +1.38% |
| **$L \ge 500$（重度用户）** | **+3.12%** | **+1.87%** | **+2.06%** |

**洞察**：收益随历史序列长度单调增长，与离线 Scaling 分析一致。冷启动用户也有改善（+0.53% CTR），得益于 Target Token 投影到 codebook 空间后的语义对齐效果。

### 3.8 部署效率

| 指标 | HyFormer | SIF |
|------|---------|-----|
| 训练耗时（小时/epoch） | 4.2h | 4.5h (+7%) |
| 推理 P99 延迟 | 18ms | 19ms (+6%) |
| 参数量 | 120M | 128M (+6.7%) |

SIF 的额外开销完全可接受，满足工业级 SLA（≤25ms）。

---

## 四、与相关论文对比

### 4.1 直接对比

| 论文 | 关系 | 差异 | 改进 |
|------|------|------|------|
| **HSTU** (Meta, 2024) | 最相关基线 | 在 item token 中编码丰富的边信息，但仍是 selected 特征子集 | SIF 压缩**完整** Raw Sample，存储开销相同但信息更完整 |
| **HyFormer** (2024) | 主要对比基线 | 统一架构但历史 token 停留在 item-level | SIF 提升 token 粒度，+0.88% GAUC |
| **TIGER** (2023) | 相关工作 | 对 item ID 做生成式检索的 RVQ | SIF 对 Raw Sample 做判别式排序的 RVQ |
| **IAT** (2026) | 相关工作 | 两阶段将历史交互压缩为统一实例嵌入 | SIF 使用 HGAQ + SIF-Mixer，分解得更细 |

### 4.2 技术路线定位

本文属于**搜推系统统一大模型路线**，主要关注**序列 token 表征增强**子方向。

技术演进路径：
1. **Item Embedding**（DIN/DIEN）→ 引入注意力
2. **序列拓宽**（DSAN/HSTU）→ 丰富单 token 上下文
3. **统一架构**（HyFormer/MixFormer）→ 序列+特征交互统一
4. **→ SIF（本文）**：从 item-level token 跃迁到 sample-level token

---

## 五、未来工作建议

### 5.1 作者建议的方向

论文未明确列出，但隐含方向包括：
- 在更多推荐场景（电商、短视频）验证泛化性
- 探索 Codebook collapse 的更鲁棒预防策略

### 5.2 基于分析的未来方向

1. **跨域迁移**：IAT 已验证了跨域迁移潜力，SIF 的 sample-level token 是否同样具备跨域迁移能力值得探索
2. **与其他 Scaling 技术结合**：SIF 与 Sequence Lengthening（如 LONGER）结合，在更长序列上是否仍有 Scaling 优势
3. **实时特征更新**：当前 Cross Feature（user-category affinity 等）每日更新，探索更细粒度（如小时级）更新对效果的影响

---

## 六、我的综合评价

### 6.1 价值评分

**[8.5]/10**

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 8/10 | 从 item-level 到 sample-level 的范式跃迁；HGAQ 量化的工程化应用；SIF-Mixer 的分解注意力设计 |
| 技术质量 | 9/10 | 方法论严谨；消融实验全面；Scaling 分析系统；工业部署验证充分 |
| 实验充分性 | 9/10 | 离线+在线 A/B；多维度消融；敏感性分析；真实工业数据集 |
| 写作质量 | 8/10 | 动机清晰；图表规范；与相关工作对比充分 |
| 实用性 | 9/10 | 已在美团外卖平台生产部署；延迟可控；收益显著（+2.03% CTR） |

### 6.2 突出亮点

1. **范式创新**：提出 sample-level token 代替 item-level token，弥补了样本信息 Scaling 的结构性缺陷，同时解决了特征异构性问题
2. **237× 存储压缩**：HGAQ 在保留完整样本信息的同时实现高压缩比，且效果优于 dense embedding，证明了"可学习性 > 信息量"的洞察
3. **Scaling 优势可持续**：在线收益随序列长度单调增长，在超长序列（$L=2000$）时优势最大（+1.02% GAUC），工程价值极高

### 6.3 重点关注

#### 值得关注的技术点

- **HGAQ 自适应子 token 化**：通过 $B$ 参数控制粒度，$T$ 随之自适应变化的设计非常优雅
- **Label-supervised VQ**：将排序 loss 反向传播到 codebook，解决了 VQ 与下游任务对齐的核心问题
- **SIF-Mixer 的行列分解**：$O(L^2 T)$ 复杂度在保持效果的同时规避了 $O((LT)^2)$ 的计算爆炸

#### 需要深入理解的部分

- RVQ 在多层级（M=3）时的 codebook collapse 预防（论文 Appendix 有详细讨论）
- Cross Feature（$G_4$）的具体计算方式（ALS CF scores 每日更新，需关注时效性）
- Target Token 与历史 Token Sample 的 codebook 空间对齐机制（$\mathcal{L}_{\text{align}}$ 的具体实现）

### 6.4 可借鉴点

- **HGAQ 分层量化思路**：在需要对高维异构特征做高效存储/检索的场景（如特征 store、Embedding Cache）可直接借鉴
- **Token 均质化设计**：解决不同来源 token 的表征不对称问题，适用于多模态推荐
- **分解注意力**：Row/Column 分解 $N$ 维注意力矩阵的设计，可迁移到其他需要建模多维度交互的场景

### 6.5 批判性思考

- **离线→在线 gap**：离线 GAUC 提升 +0.88%，但在线 CTR 提升 +2.03%，说明离线指标低估了 sample-level 特征的实际价值（可能因为离线指标无法捕捉实时性信号）
- **冷启动改进的归因**：冷启动用户（$L<10$）仍有 +0.53% 改善，但论文将其归因于 Target Token 投影对齐，这部分论证不够细致
- **Codebook 个数与泛化性**：仅在美团本地生活场景验证，在其他场景（如电商 Amazon）的泛化性待验证（论文 Appendix 在公开数据集有初步验证但未在正文重点呈现）

---

## 我的笔记

%% 用户阅读后手动补充 %%

---

## 相关论文

- [[HSTU - 线性注意力推荐系统]] - SIF 的直接对比基线（都做序列拓宽，但 SIF 用完整 Raw Sample）
- [[HyFormer - 混合注意力统一推荐模型]] - SIF 的最强基线（统一架构但 item-level token）
- [[TIGER - 生成式检索]] - RVQ 在推荐检索中的应用，SIF 在排序侧应用 RVQ
- [[IAT - 实例级注意力]] - 两阶段压缩历史交互为统一嵌入，SIF 分解更细且用 HGAQ

---

## 外部资源

- [arXiv 论文链接](https://arxiv.org/abs/2604.15650)
- [arXiv PDF](https://arxiv.org/pdf/2604.15650)

> [!tip] 关键启示
> SIF 的核心洞察是：训练日志中存储的每个历史交互的完整 Raw Sample 从未被充分利用——瓶颈不在数据，而在表征方式。通过 HGAQ 将 Raw Sample 量化为 Token Sample，在 237× 存储压缩的同时实现了比 dense embedding 更好的效果，证明"可学习性 > 信息量"的 AI 设计原则。

> [!warning] 注意事项
> - SIF 的离线 GAUC 提升（+0.88%）显著低于在线 CTR 提升（+2.03%），说明离线指标低估了 sample-level 特征的实际价值
> - Codebook collapse 预防（EMA updates + random restart + entropy regularization）是 HGAQ 成功的关键，论文 Appendix 数据显示无预防策略会导致 -0.14% AUC 损失

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐——搜推系统工业实践者必读，学术价值与工程价值兼备，在 sample-level tokenization 方向具有里程碑意义。
