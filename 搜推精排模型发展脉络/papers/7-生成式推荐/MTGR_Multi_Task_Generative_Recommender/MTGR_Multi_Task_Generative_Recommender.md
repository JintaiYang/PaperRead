---
paper_id: "[arXiv:2505.18654](https://arxiv.org/abs/2505.18654)"
title: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
authors: "Meituan Recommendation Team, Jiang Fei, Yu Lei, et al."
institution: "美团"
pushlication: "KDD 2025 / arXiv 2025-05-19"
tags:
  - 精排论文
  - MTGR
  - HSTU
  - 生成式推荐
  - Scaling-Law
  - DLRM融合
  - 混合式架构
  - 工业落地
  - 外卖推荐
quality_score: "9.0/10"
link:
  - "[PDF](https://arxiv.org/pdf/2505.18654)"
  - "[美团技术博客](https://tech.meituan.com/2025/05/19/meituan-generative-recommendation.html)"
date: "2025-05-19"
---

## 一、研究背景与动机

### 1.1 领域现状

2025 年，Meta 的 HSTU/GR 框架已经证明了生成式推荐的巨大潜力（万亿参数、12.4%+ 在线提升）。然而，HSTU 完全抛弃了传统 DLRM 的交叉特征，这在某些业务场景中并非最优选择。美团外卖推荐场景经过近十年面向交易目标的迭代，已经形成了成熟的 DLRM 体系，积累了大量有价值的交叉特征。

### 1.2 现有方法的局限性

**DLRM 的迭代瓶颈**：美团外卖 DLRM 的迭代路径可分为两个阶段——2018-2022 年的 Scaling Cross Module 阶段（PLE、MoE、DCN、PEPNet），以及 2023 年开始的 Scaling User Module 阶段。两种路径都存在根本性局限：Scaling 依赖 MLP 捕捉特征共现关系，注意力机制使用较少，优化空间有限。

**生成式推荐落地的三种路径**：工业界形成了三种技术路径。生成式架构（Meta GR、快手 OneRec）完全抛弃交叉特征，对低点击率高复购率的外卖业务过于困难。堆叠式架构（阿里 LUM、字节 HLLM）需要多阶段串行优化，迭代成本极高。混合式架构兼容两类技术优势，是 MTGR 选择的方向。

![[MTGR_fig1_page3.png|800]]

> 图1：三种生成式推荐落地路径的对比。MTGR 采用混合式架构，保留 DLRM 的交叉特征体系同时引入 HSTU 的 Scaling 能力。

### 1.3 本文解决方案概述

MTGR 采用混合式架构，基于 HSTU 架构进行统一序列编码，同时保留与 DLRM 完全一致的特征体系（包括交叉特征）。通过用户粒度的数据压缩、Group LayerNorm、动态混合掩码等技术，以及训练推理引擎的深度优化，实现了 65 倍 FLOPs 提升下训练成本持平、推理成本降 12%、外卖首页订单量 +1.22% 的工业化落地。

## 二、解决方案

### 2.1 核心思想

MTGR 的核心设计哲学是"两全其美"——既享受 HSTU 的 Scaling 红利，又保留交叉特征的信息增益。论文用消融实验直接证明了交叉特征的关键性：完全去掉交叉特征后 CTCVR GAUC 显著下降。

### 2.2 整体架构

![[MTGR_fig2_page3.png|800]]

> 图2：MTGR 的整体架构。将样本信息 Token 化后，通过 HSTU 架构进行统一编码，采用动态混合掩码保证因果性。

#### 各模块详细说明

**模块1：数据与特征——对齐 DLRM + 用户粒度压缩**

MTGR 的特征体系与 DLRM 完全一致，包含四类信息：User Profile（用户画像）、Context（时间、地点、场景等环境特征）、User Behaviour Sequence（用户历史点击/曝光序列）、以及 Target Item（待打分商家信息，包含大量统计交叉特征）。

训练数据采用用户粒度压缩：传统 DLRM 对用户的 N 个曝光存 N 行样本，存在大量重复编码；MTGR 将同一用户的 N 行压缩为一行，采用稀疏化存储配合 JaggedTensor 和变长 HSTU 算子，完全消除 padding 操作。

**模块2：输入信息 Token 化**

将样本信息拆分为 Token：User Profile 的每个特征表示为一个 Token；用户行为序列中的每个 Item，将其 ItemID 与多个 side info 的 Embedding 拼接后通过非线性映射组装为一个 Token；每个曝光候选则将 ItemID、side info、交叉特征、时空 Context 拼接组成一个 Token。

![[MTGR_fig3_page4.png|800]]

> 图3：MTGR 的 Token 化策略和 Group LayerNorm 设计。不同类型的 Token 使用不同的 LayerNorm 参数。

**模块3：Group LayerNorm**

不同于 LLM 中 Token 语义相对统一，MTGR 的 Token 可分为多种类型（用户画像、行为序列、候选 Item）。对每层输入做额外 LayerNorm 以保证深层训练稳定性，并将 LayerNorm 扩展为 Group LayerNorm——对不同类别的 Token 使用不同的 LayerNorm 参数，实现不同语义空间的 Token 对齐。

**模块4：动态混合掩码**

![[MTGR_fig4_page5.png|800]]

> 图4：动态混合掩码策略。历史静态特征不加掩码，当日实时行为采用 Causal Mask，曝光样本的可见范围根据时间动态确定。

Token 被分为三个部分——历史静态特征（User Profile & Sequence）、当日实时行为（Real-Time Sequence）、曝光样本（Targets）。对历史静态特征不加掩码（完全可见），对当日实时行为采用 Causal Mask，曝光样本的可见范围根据其发生时间动态确定。这种设计在发挥 HSTU 作为 Encoder 的学习能力的同时，保证了因果性。

**模块5：训练引擎 MTGR-Training**

基于 TorchRec 构建的高性能分布式训练引擎。关键优化包括：Fused Cutlass-based HSTU Kernel（借鉴 FlashAttention 思想，单算子性能相比 Triton 版本提升 2-3 倍）；变长序列负载均衡（根据实际序列长度调整每张卡的 batch size，保证 total_tokens 基本相同）。最终效果：65 倍计算复杂度的 GR 模型训练成本降至与 DLRM 持平。

**模块6：推理引擎 MTGR-Inference**

基于 TensorRT + Triton Inference Server 构建。关键优化：特征 H2D 优化（H2D 耗时从 7.5ms 降至 12μs）；CUDA Graph 优化（吞吐提升 13%）；FP16 计算（吞吐提升 50%）。MTGR 在推理阶段对候选数量不敏感——用户序列的计算在所有候选间共享，在线推理资源反而比 DLRM 节省 12%。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 场景 |
|--------|------|------|
| 美团外卖生产数据 | 近半年训练数据 | 首页、频道页、小程序 |

### 3.2 实验设置

#### 3.2.1 基线方法

- 美团外卖 DLRM 基准（经过近十年迭代的成熟系统）
- MTGR-small / MTGR-middle / MTGR-large（不同尺寸）

#### 3.2.2 评估指标

- **CTCVR GAUC**：离线核心指标
- **首页列表订单量**：在线核心业务指标
- **PV_CTR**：点击率

### 3.3 实验结果与分析

![[MTGR_fig5_page7.png|800]]

> 图5：MTGR 不同尺寸模型的 Scaling Law 验证。从 small 到 large 性能持续提升，展现出清晰的 Scaling 趋势。

#### Scaling Law 验证

设置了 MTGR-small、MTGR-middle、MTGR-large 三种尺寸。尽管 MTGR 仅使用近半年数据训练（DLRM 经历了超过 2 年学习），各尺寸的 MTGR 在离线和在线指标上均大幅超越 DLRM 基准。离线最大实验了 22 层、$d_{model}=1024$、137.87 GFLOPs（约 160 倍 DLRM）的超大模型，取得了更高的离线结果。

#### 在线 A/B 测试

MTGR-large 对比 DLRM 基准：离线 CTCVR GAUC **+2.88pp**，首页列表订单量 **+1.22%**（近两年单次优化最大收益），PV_CTR **+1.31%**。2025 年 4 月底在外卖首页、频道页、小程序等核心场景全量上线。

![[MTGR_fig6_page7.png|800]]

> 图6：MTGR 在线 A/B 测试结果和系统效率对比。

### 消融实验

- 交叉特征的去除导致性能显著下降，验证了保留 DLRM 特征体系的必要性
- Group LayerNorm 带来可观的离线增益
- 动态混合掩码相比传统因果掩码带来明显的性能提升
- 用户粒度数据压缩使训练效率大幅提升而不损失模型效果

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文提到将继续探索更大规模的模型（受推理性能限制，160 倍 DLRM 的超大模型未上线），以及更多场景的推广。

### 4.2 基于分析的未来方向

1. **方向1：更大规模模型的在线部署**
   - 动机：离线实验显示 160 倍 DLRM 的模型有更高性能，但受推理限制未上线
   - 可能的方法：结合 Sparse MoE 或更激进的 KV Cache 压缩
   - 预期成果：在不增加推理成本的前提下进一步提升模型容量

2. **方向2：端到端的特征学习**
   - 动机：当前仍依赖人工设计的交叉特征
   - 可能的方法：逐步用模型学习到的交互替代手工交叉特征
   - 预期成果：减少特征工程成本同时保持或提升效果

### 4.3 改进建议

1. **改进1：自适应 Group LayerNorm**
   - 当前问题：Token 分组是预定义的
   - 改进方案：让模型自动学习 Token 的分组方式
   - 预期效果：更灵活的语义空间对齐

## 五、我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - MTGR 成功解决了"如何在真实工业场景中落地生成式推荐"这一高难度工程问题，证明了生成式推荐可以兼容传统特征工程的信息优势。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 7/10 | 核心架构基于 HSTU，主要创新在 Group LayerNorm、动态混合掩码和系统优化层面 |
| 技术质量 | 8/10 | 数据压缩、掩码设计、训推引擎优化的技术链条完整，Fused HSTU Kernel 有深度 |
| 实验充分性 | 9/10 | 充分的消融实验、多尺寸 Scaling 验证和在线 A/B 测试 |
| 写作质量 | 8/10 | 从业务背景到技术方案到系统优化层层递进，叙述清晰 |
| 实用性 | 9/10 | 在美团核心外卖业务全量上线，订单量 +1.22% 是非常实在的收益 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- 混合式架构的设计哲学——不必完全抛弃传统 DLRM 的特征工程积累
- Group LayerNorm——解决异构 Token 的语义空间对齐问题
- 动态混合掩码——在 Encoder 能力和因果性之间取得平衡
- 用户粒度数据压缩——消除重复编码，大幅提升训练效率
- 推理对候选数量不敏感——资源反而比 DLRM 节省 12%

#### 5.2.2 需要深入理解的部分

- 交叉特征的具体贡献有多大？哪些交叉特征最关键？
- 动态混合掩码中，历史静态特征"完全可见"是否会引入信息泄露？
- 65 倍 FLOPs 的模型如何做到训练成本与 DLRM 持平？随机长度采样的具体策略

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[HSTU_Actions_Speak_Louder_than_Words|HSTU]] - MTGR 的基础架构来源
- [[Wukong_Towards_a_Scaling_Law_for_Large_Scale_Recommendation|Wukong]] - 特征交互 Scaling Law

### 6.2 背景相关
- [[PLE_Progressive_Layered_Extraction|PLE]] - 美团外卖 DLRM 体系中使用的多任务架构
- [[DCN_V2_Improved_Deep_and_Cross_Network|DCN V2]] - DLRM 中的特征交叉模块

### 6.3 后续工作
- 更大规模 MTGR 的在线部署
- 端到端特征学习替代手工交叉特征

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/2505.18654)
- [美团技术博客](https://tech.meituan.com/2025/05/19/meituan-generative-recommendation.html)

> [!tip] 关键启示
> 生成式推荐不必完全抛弃传统 DLRM 的特征工程积累——混合式架构可以"两全其美"，既享受 HSTU 的 Scaling 红利，又保留交叉特征的信息增益。在低点击率高复购率的外卖场景中，交叉特征的价值不可替代。

> [!warning] 注意事项
> - MTGR 的核心创新偏工程导向，理论贡献相对有限
> - 仅在美团外卖场景验证，其他场景（如短视频、电商）的迁移效果未知
> - 160 倍 DLRM 的超大模型受推理限制未能上线，说明 Scaling 仍有工程瓶颈

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！MTGR 是生成式推荐工业落地的标杆案例，为"如何在已有成熟 DLRM 体系的基础上引入 GR"提供了完整的技术方案和系统优化经验。对于正在探索 GR 落地的推荐团队，MTGR 的混合式架构思路极具参考价值。
