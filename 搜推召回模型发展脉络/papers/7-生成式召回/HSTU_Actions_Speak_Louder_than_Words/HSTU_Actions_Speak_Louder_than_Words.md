---
title: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
short_name: "HSTU"
year: 2024
venue: "ICML 2024"
authors: "Jiaqi Zhai, Lucy Liao, Xing Liu, et al."
affiliation: "Meta"
direction: "生成式召回"
tags:
  - 召回论文
  - HSTU
  - 生成式推荐
  - Scaling Law
  - Transformer
  - 论文笔记
paper_info: "[[HSTU]]"
quality_score: "9.5/10"
---

# HSTU: Actions Speak Louder than Words

> **Jiaqi Zhai, Lucy Liao, Xing Liu, et al.** | Meta | ICML 2024

## 一、研究背景与动机

### 1.1 领域现状

NLP 领域的 Transformer + Scaling Law 带来了 GPT 系列的巨大成功。但推荐系统领域长期停留在 DLRM（Deep Learning Recommendation Model）范式：稀疏特征 embedding + 特征交叉 + 全连接层。DLRM 的架构设计限制了模型规模的增长——增加参数并不能持续带来效果提升（没有 scaling law）。

### 1.2 现有方法的局限性

DLRM 的核心问题：异构的特征空间（数值特征、类别特征、序列特征混合处理），手工设计的特征交叉（DCN、DeepFM 等），以及模型规模无法有效扩展。TIGER 虽然引入了生成式范式，但停留在学术实验，未在工业规模验证。

### 1.3 本文解决方案概述

Meta 提出 HSTU（Hierarchical Sequential Transduction Unit），将推荐问题统一为序列转导任务。核心思想：把用户的所有行为（点击、购买、浏览等）编码为一个统一的行为 token 序列，用一个大规模 Transformer 处理这个序列来预测下一个行为。HSTU 首次在工业级推荐系统中验证了 Scaling Law。

## 二、解决方案

### 2.1 核心思想

**统一表示**：将推荐系统的所有输入统一为"行为 token 序列"。每个行为 token = (物品 ID, 行为类型, 时间戳) 的 embedding 组合。去掉 DLRM 的异构特征设计。

**生成式建模**：用 causal Transformer（类似 GPT）对行为序列建模，预测下一个行为（next-action prediction）。这将召回和排序统一为一个生成任务。

### 2.2 HSTU 架构

**Hierarchical Design**：
- Pointwise aggregation layer：对每个位置的多个特征进行聚合
- 修改的 Self-Attention：针对推荐数据特点优化注意力机制（去掉 softmax，改用线性注意力变体），支持超长序列（8192+）
- 对比标准 Transformer 在 8192 长度序列上实现 5.3-15.2x 加速

**M-FALCON**：新的推理方法，利用推荐数据的特性（大部分历史不变，只有最近几步是新增的）做增量推理，减少冗余计算。实现 1.5-2.99x 的推理加速。

### 2.3 Scaling Law

HSTU 首次在推荐领域验证了类似 LLM 的 Scaling Law：模型效果随参数量和训练数据量的增加呈幂律增长。从百万参数扩展到万亿参数，效果持续提升且未见饱和。这是推荐系统领域的里程碑发现。

### 2.4 统一召排

HSTU 可以同时用于召回和排序。召回时，生成式预测候选物品；排序时，对候选物品评估交互概率。不再需要分阶段的级联架构。

## 三、实验结果

### 3.1 数据集

Meta 内部多个平台（Instagram、Facebook）的大规模生产数据。公开数据集：ML-1M、ML-20M、Amazon Reviews。

### 3.2 实验结果与分析

在 Meta 内部平台部署后，相比 DLRM 基线：Instagram Reels 的推荐指标显著提升，Facebook Feed 的参与度指标提升。万亿参数规模的 HSTU 效果持续优于较小规模。M-FALCON 使得万亿参数模型的在线 serving 成为可能。在公开数据集上也全面超越 SASRec、BERT4Rec 等基线。

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.5/10** - HSTU 是推荐系统领域的重磅论文，首次在工业级证明了推荐的 Scaling Law，并提出了替代 DLRM 的统一生成式架构。可能是推荐系统十年来最重要的架构变革。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 10/10 | 统一序列表示 + 推荐 Scaling Law 的发现是革命性贡献 |
| 技术质量 | 9/10 | 架构设计精细，M-FALCON 推理优化实用 |
| 实验充分性 | 10/10 | 万亿参数工业部署 + 公开数据集 + 详细的 scaling 分析 |
| 写作质量 | 9/10 | 逻辑清晰，贡献明确 |
| 实用性 | 9/10 | 已在 Meta 大规模部署，开源了代码 |

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接前驱
- DLRM (Naumov et al., 2019) - HSTU 要替代的经典范式
- [[TIGER|TIGER (2023)]] - 学术界的生成式召回先驱

### 6.2 理论联系
- GPT (Radford et al., 2018) - 自回归生成的思想来源
- Scaling Laws for Neural Language Models (Kaplan et al., 2020) - LLM scaling law 的启发

### 6.3 同方向后续
- [[OneRec|OneRec (2025)]] - 快手跟进的工业级生成式推荐
- [[MTGR|MTGR (2025)]] - 美团融合 HSTU 和 DLRM 的方案

## 外部资源

- [arXiv](https://arxiv.org/abs/2402.17152)
- [ICML Page](https://proceedings.mlr.press/v235/zhai24a.html)
- [GitHub](https://github.com/meta-recsys/generative-recommenders)
- [知乎解读](https://zhuanlan.zhihu.com/p/1929683285354713447)

> [!tip] 关键启示
> HSTU 最深刻的发现是推荐系统存在 Scaling Law。这意味着推荐系统可以像 LLM 一样，通过增加模型规模和数据量持续提升效果。这一发现可能引发整个推荐系统行业从 DLRM 向生成式架构的全面转型。HSTU 的"统一序列表示"思想也启示我们：与其精心设计各种特征和交叉方式，不如把一切都表示为 token 序列，让大模型自己学习。

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 必读论文。推荐系统架构范式变革的里程碑。
