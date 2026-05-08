---
title: "RankMixer: Scaling Up Ranking Models in Industrial Recommenders"
short_name: "RankMixer"
year: 2025
venue: "arXiv 2025"
authors: "Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, et al."
affiliation: "字节跳动"
direction: "Scaling-Law"
tags:
  - 精排论文
  - RankMixer
  - Token-Mixing
  - Sparse-MoE
  - 硬件感知
contribution: "模型参数扩大 70 倍不增推理延迟，MFU 从 4.5%→45%，1B 全量部署 Active Day +0.29%"
industry: "字节跳动抖音推荐+广告全量上线"
link: "https://arxiv.org/abs/2507.15551"
note_link: "[[RankMixer_Scaling_Up_Ranking_Models_in_Industrial_Recommenders]]"
---

# RankMixer

**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**

Jie Zhu, Zhifang Fan, et al. (ByteDance, 2025)

## 核心贡献

Multi-head Token Mixing（无参数跨 token 交叉）+ Per-token FFN（参数隔离特征子空间建模）+ Sparse MoE（DTSI + ReLU Routing），实现参数与计算量解耦，1B 参数全量部署抖音推荐。

## 工业落地

字节跳动抖音推荐+广告全量上线，Active Day +0.29%，Duration +1.08%，ADVV +3.90%

## 详细笔记

[[RankMixer_Scaling_Up_Ranking_Models_in_Industrial_Recommenders|RankMixer 详细分析笔记]]

## 链接

- [arXiv](https://arxiv.org/abs/2507.15551)
- [PDF](https://arxiv.org/pdf/2507.15551)
