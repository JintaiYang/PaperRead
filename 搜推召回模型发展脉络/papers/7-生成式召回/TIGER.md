---
title: "Recommender Systems with Generative Retrieval"
short_name: "TIGER"
year: 2023
venue: "NeurIPS 2023"
authors: "Shashank Rajput, Nikhil Mehta, Anima Singh, et al."
affiliation: "Google"
direction: "生成式召回"
tags:
  - 召回论文
  - TIGER
  - 生成式召回
  - Semantic ID
  - RQ-VAE
contribution: "开创推荐系统生成式召回范式，提出 Semantic ID 概念（RQ-VAE 量化），Transformer 自回归生成推荐，Beauty NDCG@5 +29%"
industry: "Google（实验验证）"
link: "https://arxiv.org/abs/2305.05065"
note_link: "[[TIGER_Recommender_Systems_with_Generative_Retrieval]]"
---

# TIGER

**Recommender Systems with Generative Retrieval**

Shashank Rajput, Nikhil Mehta, Anima Singh, et al. (Google, NeurIPS 2023)

## 核心贡献

开创推荐系统生成式召回范式：用 RQ-VAE 将物品内容特征量化为层次化 Semantic ID，训练 Encoder-Decoder Transformer 自回归生成推荐物品的 Semantic ID。在 Amazon 三个数据集上全面超越 SASRec/S3-Rec 等基线（NDCG@5 最高 +29%），并展示了冷启动推荐和多样性控制两项新能力。

## 工业落地

Google（实验验证），启发了 HSTU/OneRec/MTGR 等工业级后续工作

## 链接

- [论文链接](https://arxiv.org/abs/2305.05065)
- [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)
