---
paper_id: "[arXiv:1809.03672](https://arxiv.org/abs/1809.03672)"
title: "Deep Interest Evolution Network for Click-Through Rate Prediction"
authors: "Guorui Zhou, Na Mou, Ying Fan, et al."
institution: "阿里巴巴（阿里妈妈）"
pushlication: "AAAI 2019 2019-01-27"
tags:
  - 精排论文
  - DIEN
  - 兴趣演化
  - GRU
  - AUGRU
  - 序列建模
  - 辅助损失
  - CTR预估
quality_score: "9.0/10"
link:
  - "[Github](https://github.com/mouna99/dien)"
  - "[PDF](https://arxiv.org/pdf/1809.03672)"
date: "2019-01-27"
---

## 一、研究背景与动机

### 1.1 领域现状

2019 年前后，CTR 预估领域的用户行为建模已经从简单的 Pooling 进化到了 attention 机制（DIN），但仍然将行为序列视为无序集合。DIN 通过 target-aware attention 实现了候选 item 对用户历史行为的局部激活，但每条行为与候选 item 的相关性是独立计算的，行为之间的时序依赖关系被完全忽略。

### 1.2 现有方法的局限性

论文指出了两个核心问题：

- **行为 ≠ 兴趣**：大多数现有方法直接用行为的 Embedding 来表示兴趣，但行为只是兴趣的外在表现。用户点击了一件红色连衣裙，背后的兴趣可能是"夏季着装"、"约会服饰"或"红色偏好"中的任何一种
- **兴趣是动态演化的**：用户的兴趣不是静止的，而是随时间持续变化的。DIN 的 attention 只能看到独立的行为点，无法捕捉"用户先对连衣裙感兴趣，然后兴趣转向搭配的手包，最后到首饰"这种连续的兴趣演化轨迹

### 1.3 本文解决方案概述

DIEN 将用户兴趣建模从"静态加权"升级为"动态演化"，包含两个核心模块：兴趣抽取层（GRU + 辅助损失）从行为序列中提取隐含兴趣状态序列；兴趣演化层（AUGRU）建模与候选 item 相关的兴趣演化过程。

## 二、解决方案

### 2.1 核心思想

DIEN 的核心洞察是用户兴趣具有两个特性：**时序演化性**（前一个兴趣状态直接影响下一个行为的发生）和**目标相关性**（预测是否点击某广告时，只需关注与该广告相关的兴趣演化路径）。这两个特性分别对应了兴趣抽取层和兴趣演化层的设计。

### 2.2 整体架构

![[DIEN_fig1_page4.png|800]]

> 图1：DIEN 的整体架构。底层为行为层（Behavior Layer），中间为兴趣抽取层（Interest Extractor Layer，使用 GRU + 辅助损失），顶层为兴趣演化层（Interest Evolving Layer，使用 AUGRU），最终输出与候选 item 相关的用户兴趣演化表示。

整体架构分为三层：

- **行为层**：将行为序列通过 Embedding 转化为向量序列 $\{\mathbf{b}(1), \mathbf{b}(2), \ldots, \mathbf{b}(T)\}$
- **兴趣抽取层**：使用 GRU 从行为序列中提取隐含兴趣状态序列 $\{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T\}$
- **兴趣演化层**：使用 AUGRU 建模与目标 item 相关的兴趣演化过程，输出最终表示 $\mathbf{h}'_T$

#### 各模块详细说明

**模块1：兴趣抽取层（Interest Extractor Layer）**

- **功能**：从行为序列中提取每个时间步的隐含兴趣状态
- **输入**：行为 Embedding 序列 $\{\mathbf{b}(1), \ldots, \mathbf{b}(T)\}$
- **输出**：兴趣状态序列 $\{\mathbf{h}_1, \ldots, \mathbf{h}_T\}$
- **关键技术**：GRU + 辅助损失

GRU 的更新方程：

$$\mathbf{u}_t = \sigma(\mathbf{W}^u \mathbf{i}_t + \mathbf{U}^u \mathbf{h}_{t-1} + \mathbf{b}^u)$$
$$\mathbf{r}_t = \sigma(\mathbf{W}^r \mathbf{i}_t + \mathbf{U}^r \mathbf{h}_{t-1} + \mathbf{b}^r)$$
$$\mathbf{h}_t = (1 - \mathbf{u}_t) \odot \mathbf{h}_{t-1} + \mathbf{u}_t \odot \tilde{\mathbf{h}}_t$$

**辅助损失**（Auxiliary Loss）为每个时间步的隐藏状态提供密集监督：

$$L_{aux} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T} \left[\log \sigma(\mathbf{h}_t^i, \mathbf{e}_b^i[t+1]) + \log(1 - \sigma(\mathbf{h}_t^i, \hat{\mathbf{e}}_b^i[t+1]))\right]$$

核心思想：每一步的隐藏状态 $\mathbf{h}_t$ 应当能预测下一步实际发生的行为（正样本），同时区分于未发生的行为（负样本，从同时刻曝光未点击的商品中采样）。

**模块2：兴趣演化层（Interest Evolving Layer）**

- **功能**：建模与目标 item 相关的兴趣演化过程
- **输入**：兴趣状态序列 $\{\mathbf{h}_1, \ldots, \mathbf{h}_T\}$ + 候选广告 Embedding $\mathbf{e}_a$
- **输出**：最终兴趣演化表示 $\mathbf{h}'_T$
- **关键技术**：AUGRU（Attention Update Gate GRU）

论文探索了三种方案的渐进式设计：

| 方案 | 做法 | 问题 |
|------|------|------|
| AIGRU | 用 attention 缩放 GRU 输入：$\mathbf{i}'_t = \mathbf{h}_t \cdot a_t$ | 零输入仍会改变隐藏状态 |
| AGRU | 用 attention 替代更新门：$\mathbf{h}'_t = (1-a_t)\mathbf{h}'_{t-1} + a_t\tilde{\mathbf{h}}'_t$ | 标量替代向量，丢失维度粒度 |
| **AUGRU** | 用 attention 调制更新门：$\tilde{\mathbf{u}}'_t = a_t \cdot \mathbf{u}'_t$ | **最优方案** |

AUGRU 的核心公式：

$$\tilde{\mathbf{u}}'_t = a_t \cdot \mathbf{u}'_t$$
$$\mathbf{h}'_t = (1 - \tilde{\mathbf{u}}'_t) \odot \mathbf{h}'_{t-1} + \tilde{\mathbf{u}}'_t \odot \tilde{\mathbf{h}}'_t$$

其中 $a_t = \frac{\exp(\mathbf{h}_t \mathbf{W} \mathbf{e}_a)}{\sum_{j=1}^{T} \exp(\mathbf{h}_j \mathbf{W} \mathbf{e}_a)}$。AUGRU 同时保持了向量级别的门控精度（来自 $\mathbf{u}'_t$），又通过标量 $a_t$ 实现了对无关兴趣的全局抑制。

## 三、实验结果

### 3.1 数据集

| 数据集 | 规模 | 数据类型 |
|--------|------|----------|
| Amazon Books | 用户图书评论 | 公开数据集 |
| Amazon Electronics | 电子产品评论 | 公开数据集 |
| Alibaba | ~2300万样本子集 | 工业数据集 |
| 全量生产数据 | 数十亿样本 | 在线 A/B 测试 |

### 3.2 实验设置

#### 3.2.1 基线方法

- BaseModel（Embedding & MLP）
- Wide & Deep
- PNN
- DIN
- Two Layer GRU Attention（简单的 GRU + attention 方案）

#### 3.3.2 评估指标

- **AUC**：离线数据集上的 ROC-AUC
- **在线 CTR**：线上 A/B 实验中的点击率

### 3.3 实验结果与分析

DIEN 在所有数据集上均达到最优结果。值得注意的是 Two Layer GRU Attention 的效果甚至不如 DIN，这说明简单地在 GRU 上叠加 attention 并不能有效建模兴趣演化，辅助损失和 AUGRU 的设计缺一不可。

![[DIEN_fig2_page6.png|800]]

> 图2：不同模型在公开数据集上的学习曲线对比。DIEN 收敛更快且最终效果更优。

#### 结果分析

DIEN 相比 DIN 的提升来源于两方面：辅助损失确保了每一步兴趣提取的质量（密集监督 > 稀疏监督），AUGRU 实现了与目标相关的兴趣演化路径建模（有序演化 > 无序加权）。

### 消融实验

#### 实验设计

论文分别消融了辅助损失和兴趣演化模块的三种方案。

#### 消融结果和分析

- **去除辅助损失**：效果显著下降，验证了辅助损失对兴趣提取质量的关键作用
- **AUGRU vs AGRU vs AIGRU**：AUGRU > AGRU > AIGRU，验证了渐进式设计的合理性
- **辅助损失的额外好处**：有效缓解了长序列训练中的梯度消失问题

### 实验结果图

![[DIEN_fig3_page7.png|800]]

> 图3：兴趣演化的可视化。(a) AUGRU 的隐藏状态随时间的变化，可以看到与候选 item 相关的兴趣被逐步强化；(b) 不同兴趣演化方案的对比效果。

### 在线 A/B 测试

DIEN 在淘宝展示广告系统的在线 A/B 测试中取得了 **CTR 提升 20.7%** 的显著效果。论文指出 GRU 的序列计算无法并行是在线部署的主要挑战，团队通过优化 GRU kernel 和行为序列截断等工程手段实现了可接受的推理延迟。

## 四、未来工作建议

### 4.1 作者建议的未来工作

论文在结论中提到，未来的研究方向包括探索更高效的序列建模方法来替代 GRU（解决推理效率问题），以及将兴趣演化的思想扩展到更长的行为序列。

### 4.2 基于分析的未来方向

1. **方向1：超长行为序列的高效建模**
   - 动机：AUGRU 的计算复杂度为 O(T)，且无法并行，当行为序列很长时推理延迟不可接受
   - 可能的方法：两阶段检索架构，先粗筛再精排（即后来的 SIM）
   - 预期成果：支持数万级行为序列
   - 挑战：粗筛阶段的信息损失控制

2. **方向2：用 Transformer 替代 GRU**
   - 动机：GRU 无法并行计算，且长距离依赖建模能力有限
   - 可能的方法：Multi-head Self-Attention + 位置编码（即后来的 BST）
   - 预期成果：更好的并行性和长距离建模能力
   - 挑战：Self-Attention 的 O(T²) 复杂度

### 4.3 改进建议

1. **改进1：多粒度兴趣演化**
   - 当前问题：DIEN 在单一粒度（商品 ID 级别）建模兴趣演化
   - 改进方案：同时在品类、店铺等多个粒度建模演化
   - 预期效果：更全面的兴趣演化表示

2. **改进2：会话级兴趣分割**
   - 当前问题：DIEN 将所有行为视为一条连续序列
   - 改进方案：按会话分割行为序列，会话内和会话间分别建模（即 DSIN 的思路）
   - 预期效果：更好地处理兴趣的跳变

## 五、 我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**9.0/10** - DIEN 将用户兴趣建模从 DIN 的"静态 attention"推进到"动态演化"，AUGRU 的设计优雅而有效，辅助损失的思想影响深远。

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | 9/10 | "行为≠兴趣"的洞察深刻，AUGRU 将 attention 嵌入 GRU 门控的设计优雅 |
| 技术质量 | 9/10 | 三种方案的渐进式设计体现严谨思考，辅助损失既有理论动机又有实践价值 |
| 实验充分性 | 8/10 | 多数据集 + 在线 A/B + 消融实验完备，但兴趣演化的可解释性分析较简略 |
| 写作质量 | 8/10 | 从 DIN 局限性出发逐步引出设计动机，技术方案层层递进 |
| 实用性 | 9/10 | 20.7% 的在线 CTR 提升令人印象深刻，开启了序列模型在 CTR 中的深入应用 |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

- AUGRU 将 attention 嵌入 GRU 门控的设计——比"先 RNN 后 attention"或"先 attention 后 RNN"更优雅
- 辅助损失为序列模型中间状态提供密集监督的思想，在后续工作中被广泛采用
- "行为 ≠ 兴趣"的区分——行为是可观测的，兴趣是隐含的，需要专门模块提取

#### 5.2.2 需要深入理解的部分

- 为什么 AIGRU 效果最差？因为 GRU 中零输入仍会通过更新门改变隐藏状态
- 辅助损失如何缓解梯度消失？为每一步提供了直接的梯度信号，避免了梯度需要从最后一步反向传播到第一步
- AUGRU vs DIN attention 的本质区别：AUGRU 是"有记忆"的序列演化，DIN 是"无记忆"的独立加权

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[DIN|DIN]] - 前序工作，首次引入 target-aware attention
- [[SIM|SIM]] - 后续工作，解决超长行为序列的效率问题

### 6.2 背景相关
- [[Wide_and_Deep|Wide & Deep]] - Embedding & MLP 范式的代表
- Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder" - GRU 的原始论文

### 6.3 后续工作
- [[SIM|SIM]] - CIKM 2020，两阶段检索架构
- DSIN (IJCAI 2019) - 会话级兴趣分割
- BST (DLP-KDD 2019) - Transformer 替代 GRU

## 外部资源

- [arXiv 论文页面](https://arxiv.org/abs/1809.03672)
- [GitHub 代码](https://github.com/mouna99/dien)

> [!tip] 关键启示
> 行为（behavior）和兴趣（interest）不是同一回事——行为是兴趣的外在表现，兴趣是隐含的、动态演化的。DIEN 通过 GRU+辅助损失提取隐含兴趣，通过 AUGRU 建模与目标相关的兴趣演化路径，将序列建模从"无序加权"升级为"有序演化"。

> [!warning] 注意事项
> - GRU 的序列计算无法并行，在线推理延迟是主要瓶颈
> - 辅助损失的负样本采样策略（从曝光未点击中采样）对效果有较大影响
> - AUGRU 的 attention 仍然是全局 softmax 归一化的，与 DIN 的不归一化设计不同

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ 强烈推荐！DIEN 是 DIN 系列的关键演进，将用户兴趣建模从静态推进到动态，AUGRU 的设计堪称经典。理解 DIEN 是理解整个序列建模技术路线（DIN → DIEN → SIM → BST → HSTU）的关键环节。
