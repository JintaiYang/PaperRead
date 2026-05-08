---
title: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
short_name: "MTGR"
year: 2025
venue: "KDD 2025 / arXiv 2025"
authors: "Ruidong Han, Bin Yin, Shangyu Chen, et al."
affiliation: "美团"
direction: "生成式召回"
tags:
  - 召回论文
  - MTGR
  - 生成式推荐
  - HSTU
  - DLRM
  - 统一架构
  - 论文笔记
paper_info: "[[MTGR]]"
quality_score: "8.5/10"
---

# MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

> **Ruidong Han, Bin Yin, Shangyu Chen, et al.** | 美团 | KDD 2025 / arXiv 2025

## 一、研究背景与动机

### 1.1 领域现状

生成式推荐（Generative Recommendation, GR）正在成为推荐系统的新范式。Meta 的 HSTU 验证了 Scaling Law，快手的 OneRec 首次在工业界超越级联架构。但在实际落地中，从 DLRM 转向纯 GR 架构面临一个关键问题：DLRM 积累的特征工程经验（尤其是交叉特征）如何保留？

### 1.2 现有方法的局限性

HSTU 和 OneRec 采用纯序列化表示，要求将所有输入转换为 token 序列，这意味着 DLRM 中精心设计的交叉特征（如 user×item 交叉、context×item 交叉等）需要被放弃。但在美团外卖等场景中，交叉特征对效果至关重要（如"用户位置×商家距离×时间段"的交叉对外卖推荐非常关键）。直接移除交叉特征会导致效果下降。

### 1.3 本文解决方案概述

MTGR（Meituan Generative Recommendation）提出融合 HSTU 和 DLRM 两种范式的方案：在 HSTU 的序列建模框架中保留 DLRM 的交叉特征能力。具体通过 token 化的方式将交叉特征嵌入序列中，并提出 Group-Layer Norm 和 Dynamic Masking 等技术保证训练稳定性。

## 二、解决方案

### 2.1 核心思想

"鱼和熊掌兼得"——既享受 HSTU 的序列建模能力和 Scaling Law，又保留 DLRM 的交叉特征优势。

### 2.2 特征 Token 化

将 DLRM 的所有特征（包括交叉特征）转换为 token 序列的一部分。每个曝光物品对应的 token 包括：物品 ID embedding、物品属性 embedding、以及交叉特征 embedding（user×item 的特征交叉结果也被编码为 token）。

**用户级压缩**：DLRM 每个曝光一条样本，MTGR 将一个用户一天的所有曝光压缩为一条样本（用户级序列），配合 JaggedTensor 稀疏化存储，去除 padding 操作。

### 2.3 Group-Layer Norm

在 HSTU 的 Layer Norm 基础上，将特征分组进行归一化。不同类型的特征（行为序列特征 vs 交叉特征）有不同的分布，分组归一化可以避免相互干扰，提升训练稳定性。

### 2.4 Dynamic Masking

训练时动态调整注意力掩码策略，在因果注意力的基础上增加了对交叉特征 token 的特殊处理，使模型既能利用序列信息又能利用交叉信息。

### 2.5 基于 TorchRec 的训练框架

使用 TorchRec 框架实现分布式训练，支持大规模 embedding 表的分片和高效通信。设计了三种模型尺寸（MTGR-small、MTGR-middle、MTGR-large）验证 Scaling Law。

## 三、实验结果

### 3.1 数据集

美团外卖推荐的全量线上数据。

### 3.2 实验结果与分析

**Scaling Law 验证**：三种尺寸的模型（small→middle→large）效果逐步提升，离线和在线指标均呈幂律关系。

**在线效果**：MTGR-large 在美团外卖核心业务中，CTR 提升 +1.31%，推理成本降低 12%（得益于用户级压缩减少了样本数）。保留交叉特征的 MTGR 比去掉交叉特征的纯 HSTU 效果好约 +0.5% CTR。

**DLRM 特征的价值**：消融实验证实，交叉特征在美团外卖场景中贡献了显著的增量（尤其是位置和时间相关的交叉特征）。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**8.5/10** - MTGR 提供了一条务实的生成式推荐落地路径：不是推倒重来，而是在新架构中保留已有积累。这种"渐进式演进"的策略对大多数工业团队更具参考价值。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 融合思路务实，但技术创新增量相对有限 |
| 技术质量 | 9/10 | Group-Layer Norm、Dynamic Masking 设计合理，工程细节翔实 |
| 实验充分性 | 9/10 | 三种模型尺寸的 Scaling Law 验证 + 大规模在线部署 |
| 写作质量 | 8/10 | 结构清晰，工程细节丰富 |
| 实用性 | 9/10 | 对从 DLRM 渐进转型到 GR 的团队极具参考价值 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- [[HSTU|HSTU (2024)]] - MTGR 的基础架构
- DLRM (Naumov et al., 2019) - MTGR 保留的特征工程范式

### 6.2 同方向
- [[OneRec|OneRec (2025)]] - 快手的纯生成式方案，与 MTGR 的融合方案形成对比

### 6.3 技术组件
- TorchRec - MTGR 的分布式训练框架
- JaggedTensor - 稀疏化存储方案

## 外部资源

- [arXiv](https://arxiv.org/abs/2505.18654)
- [美团技术博客](https://tech.meituan.com/2025/05/19/meituan-generative-recommendation.html)
- [知乎解读](https://zhuanlan.zhihu.com/p/1936360207417578169)

> [!tip] 关键启示
> MTGR 给出了一条现实可行的"DLRM → GR"过渡路线：不需要放弃已有的特征工程积累，而是将其融入新的生成式架构中。对于有大量交叉特征积累的团队（如外卖、电商等需要强业务特征的场景），MTGR 的融合方案比纯 HSTU/OneRec 更具落地可行性。同时，MTGR 再次验证了推荐领域 Scaling Law 的存在。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。DLRM 向生成式推荐渐进转型的最佳实践。
