---
paper_id: "[arXiv:2311.04635](https://arxiv.org/abs/2311.04635)"
title: Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction
authors: Fangye Wang, Hansu Gu, Dongsheng Li, Tun Lu, Peng Zhang, Ning Gu
institution: Fudan University, Microsoft Research Asia
pushlication: CIKM 2023 (ACM International Conference on Information and Knowledge Management), 2023-10-21
tags:
  - CTR预测
  - 特征交叉
  - 门控机制
  - Cross-Network
  - 维度优化
  - 可解释性
  - 推荐系统
quality_score: 7.5/10
link:
  - "[Github](https://github.com/anonctr/GDCN)"
  - "[PDF](https://arxiv.org/pdf/2311.04635)"
date: 2026-04-24
---

## 一、研究背景与动机

### 1.1 领域现状

CTR（点击率）预测是推荐系统和在线广告中的重要组成部分，其目标是估计用户点击推荐物品或广告的概率。准确的 CTR 预测能够为平台带来可观的收入增长并提升用户体验。目前主流的 CTR 模型通常由特征嵌入层（Feature Embedding）、特征交互层（Feature Interaction）和预测层（Prediction）三个部分组成，大量研究工作聚焦于设计更有效的特征交互架构。从早期的 LR、FM 等低阶交互方法，到 DCN、xDeepFM、AutoInt 等能够捕获高阶显式交互的模型，领域已有长足进展。

### 1.2 现有方法的局限性

论文指出当前方法面临三个主要挑战：

- **高阶交互性能退化**：多数模型在交互层数增加时性能反而下降。当交互层加深时，交互组合呈指数增长，其中包含大量无用交互，引入噪声导致性能下滑。已有工作（如 xDeepFM、DCN-V2、AutoInt 等）在超参分析中均确认，交互阶数超过 2~3 阶后性能开始衰退。
- **缺乏可解释性**：多数方法通过 DNN 进行隐式交互或对交互赋予等权重，难以区分哪些交互是关键的。虽然部分方法（如 AutoInt、DCAP）尝试借助 Self-Attention 给出解释，但注意力机制倾向于融合信息，对高阶交互难以提供有说服力的解释。
- **Embedding 层参数冗余**：多数模型要求每个 field 使用相同维度的 embedding，但不同 field 的信息容量差异较大（如 "gender" 仅有 $O(2)$ 个特征，而 "item_id" 可达 $O(10^6)$），统一维度导致大量冗余参数。DCN/DCN-V2 虽采用经验公式 $d_f = |E_f|^{0.25}$ 分配不同维度，但该公式只考虑特征数量，忽略了 field 的实际重要性。

### 1.3 本文解决方案概述

针对上述三个问题，本文提出 Gated Deep Cross Network（GDCN）模型和 Field-level Dimension Optimization（FDO）方法。GDCN 在 DCN-V2 的 Cross Network 基础上引入信息门控（Information Gate），在每一阶交叉层对特征交互进行自适应筛选，从而在高阶交互中过滤噪声、保留有用信息，同时提供模型级和实例级的可解释性。FDO 则通过对训练后的 embedding 矩阵进行 PCA 分析，根据各 field 的固有重要性学习压缩的独立维度，在较大程度上减少 embedding 参数的同时保持模型性能。

## 二、解决方案

### 2.1 核心思想

GDCN 的核心观察是：随着 Cross Network 层数的加深，特征交互组合呈指数增长，但其中大量交互对最终预测并无正向贡献，反而引入噪声。因此，需要在每一阶交叉层引入一个"软门控"来自适应地放大重要交互、抑制无关交互。

具体来说，GDCN 在 DCN-V2 的 Cross Layer 基础上增加了一个 Information Gate，由 sigmoid 函数生成的门控值（0~1 之间）对每一阶的交叉特征进行 element-wise 加权。这使得模型能在更深的层数下持续受益于高阶交互，而不会因噪声交互导致性能退化。

### 2.2 整体架构

GDCN 由三个主要组件构成：Embedding Layer、Gated Cross Network（GCN）和 Deep Neural Network（DNN）。GCN 与 DNN 可以通过两种方式组合：堆叠结构（GDCN-S）和并行结构（GDCN-P）。

![[gdcn_all0509.png|800]]

> 图1：GDCN 的整体架构。(a) GDCN-S 为堆叠结构，embedding 向量先经过 GCN 再送入 DNN；(b) GDCN-P 为并行结构，embedding 向量同时送入 GCN 和 DNN，输出拼接后进行预测。图中 $\otimes$ 表示公式(1)中的门控交叉操作。

#### 各模块详细说明

**模块1：Embedding Layer**
- **功能**：将高维稀疏的输入特征转换为低维稠密的 embedding 向量
- **输入**：多 field 的 one-hot 向量表示
- **输出**：拼接后的 embedding 向量 $\mathbf{c}_0 = [\mathbf{e}_1 \| \cdots \| \mathbf{e}_F] \in \mathbb{R}^D$
- **关键特点**：与多数 CTR 模型要求所有 field 的 embedding 维度相同不同，GDCN 允许各 field 使用不同维度，为后续 FDO 提供了基础

**模块2：Gated Cross Network（GCN）**
- **功能**：显式捕获有界阶的特征交叉，并通过信息门控筛选重要交互
- **核心公式**：

$$\mathbf{c}_{l+1} = \underbrace{\mathbf{c}_0 \odot (\mathbf{W}_l^{(c)} \times \mathbf{c}_l + \mathbf{b}_l)}_{\text{Feature Crossing}} \odot \underbrace{\sigma(\mathbf{W}_l^{(g)} \times \mathbf{c}_l)}_{\text{Information Gate}} + \mathbf{c}_l$$

其中 $\mathbf{c}_0$ 是 embedding 层输出的一阶特征，$\mathbf{c}_l$ 是第 $l$ 层的输出，$\mathbf{W}_l^{(c)}$ 和 $\mathbf{W}_l^{(g)} \in \mathbb{R}^{D \times D}$ 分别是交叉矩阵和门控矩阵，$\sigma(\cdot)$ 是 sigmoid 函数。

![[gdcn_layer0425.png|800]]

> 图2：门控交叉层（Gated Cross Layer）的可视化。$\odot$ 表示 Hadamard 乘积（逐元素乘），$\times$ 表示矩阵乘法。Feature Crossing 部分计算一阶特征 $\mathbf{c}_0$ 与第 $l$ 阶特征的交互，生成第 $(l+2)$ 阶的交叉特征；Information Gate 通过 sigmoid 生成门控值，自适应地放大重要特征、抑制不重要的特征；最终加上残差连接 $\mathbf{c}_l$。

- **两个核心组件**：
  1. **Feature Crossing**：计算 $\mathbf{c}_0$ 与 $\mathbf{c}_l$ 的 bit-wise 交互，输出包含第 $(l+2)$ 阶交叉特征的向量。交叉矩阵 $\mathbf{W}_l^{(c)}$ 反映了不同 field 之间在第 $(l+1)$ 阶的固有交互重要性。
  2. **Information Gate**：对 Feature Crossing 的输出进行 element-wise 的门控过滤。门控值通过 sigmoid 函数生成（范围 0~1），>0.5 表示该交叉特征被认为是重要的（放大），<0.5 则被抑制。随着层数增加，每一层的门控机制逐层筛选下一阶的交叉特征，有效控制信息流。

**模块3：Deep Neural Network（DNN）**
- **功能**：建模隐式特征交互
- **结构**：标准的多层全连接网络，每层为 $\mathbf{h}_{l+1} = f(\mathbf{W}_l \mathbf{h}_l + \mathbf{b}_l)$，其中 $f(\cdot)$ 通常为 ReLU
- **作用**：在 GCN 显式捕获的交互基础上进一步学习隐式的高阶特征交互

**组合方式**：
- **GDCN-S（堆叠）**：$\mathbf{c}_0 \to \text{GCN} \to \mathbf{c}_{L_c} \to \text{DNN} \to \mathbf{c}_{final} = \mathbf{h}_{L_d}$
- **GDCN-P（并行）**：$\mathbf{c}_0$ 同时送入 GCN 和 DNN，输出拼接 $\mathbf{c}_{final} = [\mathbf{c}_{L_c} \| \mathbf{h}_{L_d}]$

**预测与训练**：最终点击概率 $\hat{y}_i = \sigma(\mathbf{w}_{logit} \mathbf{c}_{final})$，损失函数为标准的二元交叉熵（LogLoss）：

$$\mathcal{L}_{ctr} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)$$

### 2.3 与 DCN-V2 的关系

GDCN 是 DCN-V2 的泛化形式。当信息门控被移除或所有门控值设为 1 时，GDCN 退化为 DCN-V2。两者都使用了门控机制，但设计目的不同：DCN-V2 借鉴 MMoE 思想将交叉矩阵分解为多个小的"专家"子空间，用门控函数组合这些专家，主要目的是减少非 embedding 参数；而 GDCN 利用门控来选择重要的交叉特征，使模型在更深的层数下仍能有效利用高阶交互信息，并提供动态的、基于实例的可解释性。

### 2.4 Field-level Dimension Optimization（FDO）

FDO 方法的目标是根据各 field 的固有重要性，为每个 field 分配压缩后的独立维度，从而减少 embedding 层的冗余参数。

**具体流程**：
1. 使用固定维度（如 $d=16$）训练一个完整模型，生成各 field 的 embedding 表
2. 对每个 field 的 embedding 表执行 PCA，计算奇异值（按降序排列）
3. 根据指定的信息保留比例（如 95%），找到使累积信息超过阈值的最小维度 $k$ 作为该 field 的优化维度
4. 使用优化后的 field 维度重新训练模型

![[dim_align_eng2.png|519]]

> 图3：FDO 泛化框架的结构示意图。对于要求所有 field 维度相同的模型（如 xDeepFM、AutoInt 等），在 Embedding Layer 之后添加一个维度对齐层（Dimension Alignment Layer），将不同维度的 embedding 对齐到相同维度后送入后续交互层。

**参数分析**：设每个 field 的维度为 $d_f$，加权平均维度 $\overline{D} = \frac{\sum_{f=1}^F d_f |E_f|}{T}$，算术平均维度 $\overline{K} = \frac{\sum_{f=1}^F d_f}{F}$。以 Criteo 数据集为例，固定维度 16 时 embedding 参数为 $16T$；使用 FDO（95% 信息保留）后，加权平均维度降至 5.92，embedding 参数仅为原来的 37%。同时，$\overline{K}$ 的减小也自然地降低了 GCN 中交叉矩阵和门控矩阵的参数量。

## 三、实验结果

### 3.1 数据集

| 数据集 | 正样本比例 | 训练集 | 验证集 | 测试集 | 特征数 | Field数 |
|--------|-----------|--------|--------|--------|--------|---------|
| Criteo | 26% | 36,672K | 4,584K | 4,584K | 1,086K | 39 |
| Avazu | 17% | 32,343K | 4,043K | 4,043K | 1,544K | 23 |
| Malware | 50% | 7,137K | 892K | 892K | 976K | 81 |
| Frappe | 33% | 231K | 29K | 29K | 5K | 10 |
| ML-tag | 33% | 1,605K | 201K | 201K | 90K | 3 |

### 3.2 实验设置

**实现细节**：基于 PyTorch 实现，使用 Adam 优化器（默认 lr=0.001），Reduce-LR-On-Plateau 调度器（patience=3 时学习率缩小 10 倍），early stopping（patience=5），batch size=4096，默认 embedding 维度 16。DNN 结构统一为 3 层 400-400-400，dropout=0.5。GCN 默认 3 层门控交叉层。

**显著性检验**：每个方法在单张 NVIDIA TITAN V GPU 上运行 10 次取平均，双尾 t 检验确认统计显著性（p<0.01）。

#### 3.2.1 基线方法

论文对比了四类共 20+ 个代表性方法：一阶方法（LR），二阶方法（FM, FwFM, DIFM, FmFM），高阶方法（CN, CIN, AutoInt, AFN, CN-V2, IPNN, OPNN, FiBiNet, FINT, SerMaskNet），以及并行/集成方法（WDL, DeepFM, DCN, xDeepFM, AutoInt+, AFN+, DCN-V2, NON, FED, ParaMaskNet）。

#### 3.2.2 评估指标

- **AUC**（Area Under ROC）：越高越好
- **Logloss**（Binary Cross Entropy）：越低越好

在 CTR 预测领域，0.001 级别的 AUC/Logloss 改善通常被认为是有意义的提升。

### 3.3 实验结果与分析

#### 堆叠模型对比（Table 2）

| 模型类别       | 模型         | Criteo AUC | Criteo LL  | Avazu AUC  | Avazu LL   | Malware AUC | Frappe AUC | ML-tag AUC | Avg DAUC  | Avg DLL    |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- | ---------- | --------- | ---------- |
| High-order | SerMaskNet | 0.8141     | 0.4379     | 0.7891     | 0.3746     | 0.7440      | 0.9804     | 0.9602     | --        | --         |
| High-order | CN-V2      | 0.8140     | 0.4380     | 0.7893     | 0.3745     | 0.7383      | 0.9803     | 0.9549     | -0.26%    | 3.16%      |
| **Ours**   | **GCN**    | **0.8154** | **0.4367** | **0.7903** | **0.3742** | **0.7445**  | **0.9820** | **0.9619** | **0.14%** | **-0.45%** |
| **Ours**   | **GDCN-S** | **0.8158** | **0.4364** | **0.7905** | **0.3739** | **0.7456**  | **0.9838** | **0.9645** | **0.28%** | **-2.70%** |

GCN 仅建模显式多项式特征交互，就在所有堆叠基线中取得了较优的结果，验证了信息门控对高阶交互筛选的有效性。GDCN-S 在 GCN 上叠加 DNN 后进一步提升，在五个数据集上全面超越已有方法。

#### 并行模型对比（Table 3）

| 模型 | Criteo AUC | Avazu AUC | Malware AUC | Frappe AUC | ML-tag AUC | Avg DAUC | Avg DLL |
|------|-----------|-----------|-------------|-----------|-----------|---------|---------|
| DCN-V2 | 0.8144 | 0.7898 | 0.7433 | 0.9810 | 0.9603 | 0.11% | -2.67% |
| FED | 0.8141 | 0.7891 | 0.7436 | 0.9811 | 0.9610 | 0.12% | -1.99% |
| **GDCN-P** | **0.8161** | **0.7909** | **0.7462** | **0.9852** | **0.9663** | **0.46%** | **-5.57%** |

GDCN-P 在所有并行模型中表现较优，Rel.Imp 在 AUC 上达到 0.0011~0.0053，在 Logloss 上达到 -0.0010~-0.0129。

#### 结果分析

以上实验结果可以归纳为三点：其一，高阶模型通常优于低阶模型，同时建模显式与隐式交互的堆叠结构（如 OPNN、SerMaskNet）表现更好；其二，GCN 仅凭显式交互就超越了包含 DNN 的堆叠基线，说明信息门控对于过滤无效交互、保留有价值的高阶交叉信息具有较为明确的效果；其三，GDCN-P 整体表现优于 GDCN-S，表明并行结构在融合显式与隐式信息时可能更具优势。

### 消融实验

#### 深层高阶交叉实验

![[depth_auc_ll0421.png|800]]

> 图4：六个模型在 Criteo 数据集上随交叉层数增加的 AUC 和 Logloss 变化。可以看到 AFN、FINT、CN-V2、DNN、IPNN 等模型在层数超过 2~4 后性能开始下降，而 GCN 的性能随层数增加（1到8）持续提升。

这一实验是论文中较为关键的消融之一。五个对比模型均在交叉深度超过一定阈值后出现性能衰退，而 GCN 由于信息门控的存在，能够在每一阶过滤无用交叉特征，持续从更深的高阶交互中获取收益。

![[deeper_layers0523.png|800]]

> 图5：GCN 和 GDCN-P 在 Criteo 和 Malware 数据集上将交叉层数从 1 增加到 16 的实验结果。两者性能均随层数增加而提升，且在达到平台期（GCN 约 8~10 层、GDCN-P 约 4~6 层）后保持稳定而非下降。

#### 可解释性分析

**静态模型可解释性**：GCN 中的交叉矩阵 $\mathbf{W}^{(c)}$ 可以被分解为 block-wise 形式，每个块 $W_{i,j} \in \mathbb{R}^{d \times d}$ 反映第 $i$ 个和第 $j$ 个 field 之间一阶交互的重要性（Frobenius 范数越大表示交互越重要）。

![[w2_gcn.png|277]]
![[w2_dyn_gcn.png|263]]

> 图6：GCN 和 GCN-FDO(95%) 在 Criteo 数据集上第 1 层门控交叉层学到的 block-wise 交叉矩阵热力图。颜色越深（红色）表示 field 间交互越强。两者的 cosine similarity 为 0.89，表明 FDO 应用前后所捕获的 field 交互模式具有较好的一致性。

**动态实例可解释性**：Information Gate 为每个输入实例生成独立的门控向量，可在 bit-level 和 field-level 展示各交叉特征的重要性。

![[678_long_n.png|800]]

![[678_short_n.png|800]]

> 图7：两个随机实例在第 1~3 层门控交叉层的门控值可视化。(a) Bit-wise 门控向量（维度 39x16=624）；(b) Field-wise 门控向量（维度 39）。红色（>0.5）表示重要特征，蓝色（<0.5）表示不重要特征。低阶层中重要特征（红色）较多，高阶层中多数特征变为中性（白色）或不重要（蓝色），与"高阶交互中有用交叉特征逐渐减少"的直觉一致。

![[average_100w_short.png|800]]

> 图8：100 万个实例的平均 field-wise 门控向量可视化。从统计角度验证了 fields {#20, #23, #24} 较为重要，fields {#11, #27, #30} 相对不重要，以及随层数增加重要交叉特征数量的递减趋势。

#### FDO 有效性分析

| 信息保留比例 | $\overline{D}$ | $\overline{K}$ | 参数量 | 占比 | AUC | Logloss |
|-------------|----------------|----------------|--------|------|-----|---------|
| Full (16) | 16 | 16 | 19.73M | 100% | 0.8154 | 0.4367 |
| 98% | 7.56 | 9.54 | 9.04M | 45.8% | 0.8157 | 0.4365 |
| 95% | 5.92 | 7.87 | 7.00M | 35.5% | 0.8157 | 0.4365 |
| 90% | 4.99 | 5.92 | 5.80M | 29.4% | 0.8156 | 0.4365 |
| 80% | 3.98 | 4.85 | 4.54M | 23.0% | 0.8155 | 0.4366 |
| 70% | 2.94 | 3.69 | 3.32M | 16.8% | 0.8152 | 0.4371 |
| Full (8) | 8 | 8 | 9.27M | 47.9% | 0.8151 | 0.4371 |

在 Criteo 数据集上，保留 80% 信息时仅需 23% 的参数量即可达到与全模型相当的性能；保留 80%~98% 时性能甚至略优于全模型，这是因为 FDO 去除了冗余维度信息。相比之下，直接将固定维度从 16 降至 8 虽然参数量类似，但性能有所下降。

![[params_times0523.png|800]]

> 图9：Criteo 数据集上各模型的参数量与每 epoch 训练时间对比。多数模型参数量在 18M~20M 之间，应用 FDO 后 GCN-FDO 和 GDCN-P-FDO 仅需约 5M 参数即可达到相当甚至更优的性能，同时训练速度更快。

#### Field 维度与重要性的关联

![[relation_eng.png|600]]

> 图10：平均 field 重要性与 FDO 学到的 field 维度之间的散点图。两者的 Pearson 相关系数为 0.82（p < 1e-9），说明 FDO 学到的维度与 field 的实际重要性之间存在较为明确的正相关关系。例如 fields {#16, #25} 包含的特征数量多，但其学到的维度和重要性均偏低；而若使用经验公式，这两个 field 将被赋予较长的维度。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文结论部分主要总结了 GDCN 在高阶交互建模、可解释性和参数效率三方面的贡献，但未明确列出具体的未来工作方向。结合论文的 FDO 泛化框架（Appendix A），作者隐含地指出该方法可以拓展到更多 CTR 模型中使用。

### 4.2 基于分析的未来方向

1. **方向1：自适应门控机制**
   - 动机：当前信息门控在每一层使用固定的 sigmoid 函数，门控的"锐度"不可调节
   - 可能的方法：引入温度参数控制 sigmoid 的锐度，或使用 learnable 的门控函数（如 Gumbel-Softmax）
   - 预期成果：在不同层和不同数据集上可能获得更灵活的特征选择能力
   - 挑战：额外超参数的引入可能增加调参成本

2. **方向2：序列特征建模**
   - 动机：GDCN 主要针对 tabular 数据的 CTR 预测，未考虑用户行为序列信息
   - 可能的方法：将 GCN 与序列建模模块（如 DIEN、SIM）结合，在序列表征上进行门控交叉
   - 预期成果：在包含用户行为序列的场景中获得更好的效果
   - 挑战：序列特征的引入会增加 embedding 维度和计算量

3. **方向3：FDO 的自动化与在线学习**
   - 动机：当前 FDO 需要先训练完整模型再用 PCA 计算维度，流程较为繁琐
   - 可能的方法：设计端到端的维度搜索方法（如 NAS-based 或可微分维度选择），在训练过程中自动学习每个 field 的维度
   - 预期成果：减少 FDO 的两阶段训练开销，实现一次训练即可获得优化维度
   - 挑战：可微分维度搜索的离散化问题，以及搜索空间的高效遍历

### 4.3 改进建议

1. **改进1：Gate 结构轻量化**
   - 当前问题：每层 GCN 的信息门引入了与交叉矩阵同等规模的参数矩阵 $\mathbf{W}^{(g)} \in \mathbb{R}^{D \times D}$，参数量接近翻倍
   - 改进方案：采用低秩分解或共享部分参数的方式压缩门控矩阵
   - 预期效果：在保持门控筛选能力的同时减少非 embedding 参数

2. **改进2：多目标场景适配**
   - 当前问题：论文仅针对单目标（CTR）场景进行实验，未涉及多目标预估
   - 改进方案：将 GCN 与 MMoE/PLE 等多任务框架结合，在 expert 层面引入门控交叉
   - 预期效果：在 CVR、CTCVR 等多目标场景中获得更好的特征交互建模

## 五、 我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**7.5/10** - 在 DCN-V2 基础上的改进工作，核心创新点（信息门控 + FDO）设计合理且实验验证充分，对工业界 CTR 模型有较好的实用参考价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 6/10 | 信息门控机制本身并非新概念，但将其与 Cross Network 结合并提供可解释性分析有一定新意；FDO 方法基于 PCA 的思路较为直接 |
| 技术质量 | 8/10 | 方法推导清晰，GCN 公式与 DCN-V2 的关系分析到位，FDO 的参数分析严谨 |
| 实验充分性 | 8/10 | 五个数据集、20+基线对比、消融实验、深度分析、可解释性可视化、FDO 兼容性分析，实验覆盖面广 |
| 写作质量 | 7/10 | 整体结构清晰，但部分段落（如 Section 3 的 GCN 描述）偏冗长，可进一步精简 |
| 实用性 | 8/10 | GCN 结构简洁易于工程落地，FDO 可直接应用于现有模型的 embedding 压缩，实用价值较高 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- **信息门控的设计理念**：通过 sigmoid 门控在每一阶交叉层动态筛选有用的交叉特征，避免高阶交叉引入噪声导致性能下降。这一思路可以推广到其他显式特征交互方法中。
- **FDO 维度优化**：基于训练后 embedding 矩阵的 PCA 奇异值分析来确定每个 field 的合适维度，方法简单但效果可观（23% 参数即可达到同等性能）。这种后验维度选择方法值得在实际系统中尝试。
- **Field 维度与 Field 重要性的强相关**：Pearson 相关系数达到 0.82（p < 1e-9），说明 FDO 学到的维度本身就反映了 field 的预测重要性。

#### 5.2.2 需要深入理解的部分

- GCN 在深层交叉时的稳定性机制：为什么加了门控后性能不会随层数增加而下降？门控值在高阶层趋近于 0.5（中性）的现象背后的数学原理。
- FDO 与端到端维度搜索方法（如 AutoDim）的对比：FDO 的两阶段方法在工业场景中的实际可行性如何。

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DCN-V2|DCN-V2]] - GDCN 的直接前驱工作，GCN 是 CN-V2 的门控扩展，当门控值全为 1 时退化为 CN-V2
- [[DCN|DCN]] - Cross Network 的初始版本，提出了显式有界阶特征交叉的思路
- [[MaskNet|MaskNet]] - 同为 CTR 模型，采用 mask 机制进行特征选择，是本文实验中的强基线

### 6.2 背景相关
- [[DeepFM|DeepFM]] - 经典的并行结构 CTR 模型（FM + DNN），是 GDCN-P 并行结构的基础参照
- [[xDeepFM|xDeepFM]] - 提出 CIN 结构建模显式高阶交互，与 GCN 在目标上类似但方法不同
- [[AutoInt|AutoInt]] - 基于 Self-Attention 的特征交互方法，提供了注意力机制的可解释性
- [[FmFM|FmFM]] - 提出了 field-level 维度优化的启发性思想，FDO 方法受其启发

### 6.3 后续工作
- [[FinalMLP|FinalMLP]] - 后续 CTR 模型工作，关注特征交互的不同范式
- [[AutoDim|AutoDim]] - 端到端的 embedding 维度搜索方法，与 FDO 在维度优化方向上可形成对比

## 外部资源
- [GitHub 代码仓库](https://github.com/anonctr/GDCN)
- [CIKM 2023 会议论文集](https://dl.acm.org/doi/10.1145/3583780.3615089)
- [arXiv 论文页面](https://arxiv.org/abs/2311.04635)

> [!tip] 关键启示
> 在显式特征交叉网络中引入信息门控，可以在不牺牲性能的前提下有效利用更深层的高阶交叉信息，同时为模型提供了实例级别的动态可解释性。

> [!warning] 注意事项
> - FDO 需要先训练一个完整维度的模型，再通过 PCA 分析确定各 field 维度，增加了训练流程的复杂度
> - 论文实验主要在公开数据集上进行，工业级大规模数据上的效果有待验证
> - 门控矩阵 $\mathbf{W}^{(g)}$ 与交叉矩阵规模相同，在 field 数量多或 embedding 维度大时，参数增量不可忽视

> [!success] 推荐指数
> 推荐阅读。作为 DCN-V2 的改进工作，GDCN 在保持公式简洁的同时兼顾了性能提升和可解释性，FDO 方法对 embedding 参数压缩有较好的实用价值，适合搜推系统从业者参考。
