---
title: "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
short_name: "OneTrans"
year: 2025
venue: "WWW 2026"
authors: "Zhaoqi Zhang, Haolei Pei, Jun Guo, et al."
affiliation: "ByteDance / NTU"
direction: "Scaling-Law"
tags:
  - 精排论文
  - OneTrans
  - 统一架构
  - Transformer
  - Scaling-Law
contribution: "统一 Tokenizer 将特征交互与序列建模统一到单一 Transformer，GMV +5.68%"
industry: "字节跳动"
link: "https://arxiv.org/abs/2510.26104"
note_link: "[[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling]]"
---

# OneTrans

**OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender**

Zhaoqi Zhang, Haolei Pei, Jun Guo, et al. (ByteDance / NTU, WWW 2026)

## 核心贡献

统一 Tokenizer 将特征交互与序列建模统一到单一 Transformer，通过 Mixed Parameterization（S-token 共享参数、NS-token 独立参数）处理异质特征，Pyramid Stack 逐层压缩序列，Cross-Request KV Cache 实现高效推理。线上 A/B 人均 GMV +5.68%，p99 延迟反而降低。

## 工业落地

字节跳动电商（Feeds / Mall 场景）

## 详细笔记

[[OneTrans_Unified_Feature_Interaction_and_Sequence_Modeling|OneTrans 详细论文笔记]]

## 链接

- [arXiv](https://arxiv.org/abs/2510.26104)
- [PDF](https://arxiv.org/pdf/2510.26104)
