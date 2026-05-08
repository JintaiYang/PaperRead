---
name: paper-analyze
description: 深度分析单篇论文，生成详细笔记和评估，图文并茂 / Deep analyze a single paper, generate detailed notes with images
allowed-tools: Read, Write, Bash, WebFetch
---

# 目标
对特定论文进行深度分析，生成全面笔记，评估质量和价值，并更新知识库。

# 工作流程

## 实现脚本

### 步骤0：初始化环境

```bash
# 创建工作目录
mkdir -p /tmp/paper_analysis
cd /tmp/paper_analysis

# 设置变量（从环境变量 OBSIDIAN_VAULT_PATH 读取，或让用户指定）
PAPER_ID="[PAPER_ID]"
VAULT_ROOT="${OBSIDIAN_VAULT_PATH}"
PAPERS_DIR="${VAULT_ROOT}/Papers"
```

### 步骤1：识别论文

### 1.1 解析论文标识符

接受输入格式：
- arXiv ID："2402.12345"
- 完整ID："arXiv:2402.12345"
- 论文标题："论文标题"
- 文件路径：直接路径到现有笔记

### 1.2 检查现有笔记

1. **搜索已有笔记**
   - 按arXiv ID在`Papers/`目录中搜索
   - 按标题匹配
   - 如果找到，读取该笔记

2. **读取论文笔记**
   - 如果找到，返回完整内容

## 步骤2：获取论文内容

### 2.1 下载PDF并提取源码

```bash
# 下载PDF
curl -L "https://arxiv.org/pdf/[PAPER_ID]" -o /tmp/paper_analysis/[PAPER_ID].pdf

# 下载源码包（包含TeX和图片）
curl -L "https://arxiv.org/e-print/[PAPER_ID]" -o /tmp/paper_analysis/[PAPER_ID].tar.gz
tar -xzf /tmp/paper_analysis/[PAPER_ID].tar.gz -C /tmp/paper_analysis/
```

### 2.2 提取论文元数据

```bash
# 使用curl获取arXiv页面
curl -s "https://arxiv.org/abs/[PAPER_ID]" > /tmp/paper_analysis/arxiv_page.html

# 提取关键信息（使用通用正则，适用于任何论文）
TITLE=$(grep -oP '<title>\K[^<]*' /tmp/paper_analysis/arxiv_page.html | head -1)
AUTHORS=$(grep -oP 'citation_author" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | paste -sd ', ')
DATE=$(grep -oP 'citation_date" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | head -1)
```

### 2.3 读取TeX源码内容

```bash
# 读取各章节内容
cat /tmp/paper_analysis/1-introduction.tex > /tmp/paper_analysis/intro.txt
cat /tmp/paper_analysis/2-joint-optimization.tex > /tmp/paper_analysis/methods.txt
cat /tmp/paper_analysis/3-agent-swarm.tex > /tmp/paper_analysis/agent_swarm.txt
cat /tmp/paper_analysis/5-eval.tex > /tmp/paper_analysis/eval.txt
```

## 步骤2.1 从arXiv获取

1. **获取论文元数据**
   - 使用WebFetch访问arXiv API
   - 查询参数：`id_list=[arXiv ID]`
   - 提取：标题、作者、摘要、发布日期、类别、链接、PDF链接

2. **获取PDF内容和图片**
   - 使用WebFetch获取PDF
   - **重要**：提取论文中的所有图片
   - 保存图片到`Papers/[领域]/[论文标题]/images/`
   - 生成图片索引：`images/index.md`

### 2.2 从Hugging Face获取（如果适用）

1. **获取论文详情**
   - 使用WebFetch访问Hugging Face
   - 提取：标题、作者、摘要、标签、点赞、下载

## 步骤3：执行深度分析

### 3.1 分析摘要

1. **提取关键概念**
   - 识别主要研究问题
   - 列出关键术语和概念
   - 注明技术领域

2. **总结研究目标**
   - 要解决的问题是什么？
   - 提出的解决方案方法是什么？
   - 主要贡献是什么？

3. **生成中文翻译**
   - 将英文摘要翻译成流畅的中文
   - 使用适当的技术术语

### 3.2 分析方法论

1. **识别核心方法**
   - 主要算法或方法
   - 技术创新点
   - 与现有方法的区别

2. **分析方法结构**
   - 方法组件及其关系
   - 数据流或处理流水线
   - 关键参数或配置

3. **评估方法新颖性**
   - 这个方法有什么独特之处？
   - 与现有方法相比如何？
   - 有什么关键创新？

### 3.3 分析实验

1. **提取实验设置**
   - 使用的数据集
   - 对比基线方法
   - 评估指标
   - 实验环境

2. **提取结果**
   - 关键性能数字
   - 与基线的对比
   - 消融研究（如果有）

3. **评估实验严谨性**
   - 实验是否全面？
   - 评估是否公平？
   - 基线是否合适？

### 3.4 生成洞察

1. **研究价值**
   - 理论贡献
   - 实际应用
   - 领域影响

2. **局限性**
   - 论文中提到的局限性
   - 潜在弱点
   - 有什么假设可能不成立？

3. **未来工作**
   - 作者建议的后续研究
   - 有什么自然的扩展？
   - 有什么改进空间？

4. **与相关工作对比**
   - 搜索相关历史论文
   - 与相似论文相比如何？
   - 补充了什么空白？
   - 属于哪个研究路线

### 3.5 公式输出规范（Markdown LaTeX）

1. **统一格式**
   - 行内公式使用 `$...$`
   - 块级公式使用 `$$...$$` 并单独成行

2. **避免不可渲染写法**
   - 不要用三反引号代码块包裹需要渲染的公式
   - 不要使用纯文本伪公式替代 LaTeX

3. **推荐写法**
   - 行内示例：模型目标是最小化 `$L(\theta)$`
   - 块级示例：
     `$$\theta^* = \arg\min_\theta L(\theta)$$`

4. **复杂公式**
   - 多行或推导型公式统一使用块级 `$$...$$`
   - 保持符号与原论文一致，避免自行改写符号语义

## 步骤3：转换图片为高清 PNG 并复制到目标目录

```bash
# 1. 复制原始文件到目标目录
cp /tmp/paper_analysis/Figures/*.{pdf,png,jpg,jpeg} "PAPERS_DIR/[DOMAIN]/[PAPER_TITLE]/images/" 2>/dev/null

# 2. 使用 PyMuPDF 将所有 PDF 图片转换为高清 PNG（4x zoom，宽度 ≥ 800px）
python3 -c "
import fitz, os, glob
img_dir = 'PAPERS_DIR/[DOMAIN]/[PAPER_TITLE]/images'
for pdf_path in glob.glob(os.path.join(img_dir, '*.pdf')):
    png_path = pdf_path.replace('.pdf', '.png')
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = 4.0
    while True:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if pix.width >= 800 or zoom >= 8.0:
            break
        zoom += 1.0
    pix.save(png_path)
    doc.close()
    print(f'Converted: {os.path.basename(pdf_path)} -> {os.path.basename(png_path)} ({pix.width}x{pix.height})')
"

# 3. 验证所有 PNG 宽度 >= 800px
ls "PAPERS_DIR/[DOMAIN]/[PAPER_TITLE]/images/"*.png
```

**⚠️ 关键**：笔记中只引用 `.png` 文件，不引用 `.pdf`。

## 步骤4：生成综合论文笔记

### 4.1 确定笔记路径和领域

```bash
# 根据论文内容确定领域（搜推系统/智能体/大模型/多模态技术/强化学习_LLM_Agent等）
# 推断规则：
# - 如果提到"agent/swarm/multi-agent/orchestration" → 智能体
# - 如果提到"vision/visual/image/video" → 多模态技术
# - 如果提到"reinforcement learning/RL" → 强化学习_LLM_Agent
# - 如果提到"language model/LLM/MoE" → 大模型


PAPERS_DIR="${VAULT_ROOT}/Papers"
DOMAIN="[推断的领域]"
PAPER_TITLE="[论文标题，空格替换为下划线]"
NOTE_PATH="${PAPERS_DIR}/${DOMAIN}/${PAPER_TITLE}.md"
IMAGES_DIR="${PAPERS_DIR}/${DOMAIN}/${PAPER_TITLE}/images"
INDEX_PATH="${IMAGES_DIR}/index.md"
```

### 4.2 使用Python生成笔记（正确处理Obsidian格式）

```bash
# 调用外部脚本生成笔记
python "scripts/generate_note.py" --paper-id "[PAPER_ID]" --title "[论文标题]" --authors "[作者]" --domain "[领域]" --language "$LANGUAGE"
```

### 4.3 使用obsidian-markdown skill生成最终笔记

当分析完成后，调用obsidian-markdown skill来确保格式正确，然后手动补充详细内容。

## 步骤5：更新知识图谱

### 5.1 读取现有图谱

```bash
GRAPH_PATH="${PAPERS_DIR}/../PaperGraph/graph_data.json"
cat "$GRAPH_PATH" 2>/dev/null || echo "{}"
```

### 5.2 生成图谱节点和边

```bash
# 调用外部脚本更新知识图谱
python "scripts/update_graph.py" --paper-id "[PAPER_ID]" --title "[论文标题]" --domain "[领域]" --score [评分] --language "$LANGUAGE"
```

## 步骤4：生成综合论文笔记

"""
以下是完整的论文笔记模板。生成笔记时，**必须严格按照此结构填充内容**，不要额外加入其他没有列举的section，每个 section 都应包含实质性分析内容（而非占位符）， 。

写作风格要求

0. **信息真实**： 笔记信息来自于原文，不能推测，更不能捏造，需要真实可靠
1. **段落式写作为主**：优先使用自然语言段落进行深入分析和讨论，而非简单的要点罗列，论文内容需要有条理，可以一条一条列举，不要大段大段文字去陈诉
2. **列表辅助关键信息**：数据集统计、实验结果、评分等结构化信息用表格或列表展示
3. **公式融入上下文**：核心公式用 `$$...$$` 独立展示，然后用自然语言解释每个符号的含义
4. **图文并茂**：论文中的**所有重要图**都要引用（架构图、方法细节图、实验结果图、可视化分析图、参数分析图），每张图配说明文字
5. **具体有洞察力**：分析要用具体数字支撑（如"AUC 提升 0.0017"、"参数压缩至 23%"），避免泛泛而谈
6. **分批次写入**：笔记内容较长时（通常 400+ 行），先创建文件写入 frontmatter 和前几节，再用 string_replace 或追加方式逐节填充
"""

### 4.1 笔记结构

```markdown
---
paper_id: "[arXiv:XXXX.XXXXX / DOI](链接)"
title: "原论文标题" #不要修改
authors: "作者1, 作者2, 作者3" #如果超过3个作者，加上et al.
institution: "[从作者推断或查看论文]"
pushlication: "[期刊/论文] [发布时间:YYYY-MM-DD]"
tags:
  - [方法标签-无空格]  # 标签名不能有空格，空格替换为-
  - [KeyWords] # 从论文关键词获取，需要翻译成中文，空格替换为-
# ⚠️ 标签名格式规则
# Obsidian的tag名称不能包含空格，如有空格需用短横线(-)连接
# 例如：
#   "Agent Swarm" → "Agent-Swarm"
#   "Visual Agentic" → "Visual-Agentic"
#   "MoonViT-3D" → "MoonViT-Three-D"
#
# Python脚本(scripts/generate_note.py)会自动处理标签名中的空格
# 将所有tag.replace(' ', '-')移除空格
quality_score: "[X.X]/10"
link:
  - "[Github](链接)"
  - "[PDF](链接)"
date: "YYYY-MM-DD"
---

## 一、研究背景与动机

### 1.1 领域现状
[详细描述该研究领域当前的发展状况]

### 1.2 现有方法的局限性
[深入分析现有方法存在的问题, 信息来自原论文]

### 1.3 本文解决方案概述
[清晰、准确地概述论文解决方案，详细内容在下面解决方案里面展开]

## 二、解决方案

### 2.1 核心思想 
[用通俗易懂的语言解释方法的核心思想，让非专业人士也能理解]

### 2.2 整体架构
[描述方法的整体架构，包括主要组件和它们之间的关系]


**架构图选择原则**：
1. **优先使用论文中的现成图** - 如果论文PDF中有架构图/流程图/方法图，转换为高清PNG后插入
2. **仅在无图时创建Canvas** - 当论文没有合适的架构图时，才用JSON Canvas自行绘制

**方式1：插入论文中的图（优先）**
```
![[arch.png|800]]

> 图1：[架构描述，包括图中各个部分的含义和它们之间的关系]
```
**⚠️ 重要**：必须引用 `.png` 格式图片，禁止直接引用 `.pdf`。所有从arXiv提取的PDF图片必须先用 PyMuPDF 以 4x zoom 转换为高清 PNG（宽度 ≥ 800px），再在笔记中引用。详见 `extract-paper-images` skill 中的转换规范。

**方式2：创建Canvas架构图（论文无图时使用）**
调用 `json-canvas` skill 创建 `.canvas` 文件，然后嵌入：
```
![[论文标题_Architecture.canvas|1200|400]]
```

Canvas 创建步骤：
1. 调用 `json-canvas` skill
2. 使用 `--create --file "路径/架构图.canvas"` 参数
3. 创建节点和连接，使用不同颜色区分层级
4. 保存后在markdown中嵌入引用

**文本图表示例**（当无法插入图片或创建Canvas时的最后备选）：
```
输入 → [模块1] → [模块2] → [模块3] → 输出
         ↓         ↓         ↓
       [子模块]  [子模块]  [子模块]
```

#### 各模块详细说明

**模块1：[模块名称]**
- **功能**：[该模块的主要功能]
- **输入**：[输入数据/信息]
- **输出**：[输出数据/信息]
- **处理流程**：
  1. [步骤1详细描述]
  2. [步骤2详细描述]
  3. [步骤3详细描述]
- **关键技术**：[使用的关键技术或算法]
- **数学公式**：[如果有重要的数学公式]
   行内示例：损失函数为 $L(\theta)$。
   块级示例：
   $$\theta^* = \arg\min_\theta L(\theta)$$

**模块2：[模块名称]**
- **功能**：[该模块的主要功能]
- **输入**：[输入数据/信息]
- **输出**：[输出数据/信息]
- **处理流程**：
  1. [步骤1详细描述]
  2. [步骤2详细描述]
  3. [步骤3详细描述]
- **关键技术**：[使用的关键技术或算法]

**模块3：[模块名称]**
[类似格式]

### 方法架构图
[选择最适合的方式展示架构]

**选择原则**：
1. **优先使用论文中的架构图** - 如果论文中有合适的方法架构图、流程图或系统图，直接插入
2. **仅在无图时创建Canvas** - 当论文没有相关架构图时，才用JSON Canvas自行绘制

**方式1：插入论文中的图（优先）**
```
![[method_detail.png|800]]

> 图X：[架构描述，包括图中各个部分的含义和它们之间的关系]
```
**⚠️ 重要**：必须引用 `.png` 格式，禁止引用 `.pdf`。

**方式2：创建Canvas架构图（论文无图时使用）**
```
![[论文标题_Architecture.canvas|1200|400]]
```
调用`json-canvas` skill创建，支持：
- 彩色节点（颜色1-6或自定义hex）
- 带标签的箭头连接
- 节点分组和层级结构
- Markdown文本渲染

**注意**：Canvas只作为补充手段，不要替换论文中原有的架构图。论文中的图通常更准确、更权威。



## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 类别数 | 数据类型 |
|--------|--------|----------|--------|----------|
| 数据集1 | X万 | Y维 | Z类 | [类型] |
| 数据集2 | X万 | Y维 | Z类 | [类型] |

### 3.2 实验设置
[列出实验环境，实现细节，超参数设计等]

#### 3.2.1 基线方法
[列出所有对比的基线方法，并简要说明]


#### 3.3.2 评估指标
[列出所有评估指标，并解释每个指标的含义]


### 3.3 实验结果与分析

| 方法 | 数据集1-指标1 | 数据集1-指标2 | 数据集2-指标1 | 数据集2-指标2 | 平均排名 |
|------|---------------|---------------|---------------|---------------|----------|
| 基线1 | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | N |
| 基线2 | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | N |
| 基线3 | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | X.X±Y.Y | N |
| **本文方法** | **X.X±Y.Y** | **X.X±Y.Y** | **X.X±Y.Y** | **X.X±Y.Y** | **N** |

> 注：±后的数字表示标准差，**粗体**表示最优结果

#### 结果分析
[对主实验结果的详细分析]

### 消融实验

#### 实验设计
[消融实验的设计思路]

#### 消融结果和分析

### 实验结果图
[插入论文中的实验结果图]

![[experiment_results.png|800]]

> 图2：[图描述]
**⚠️ 重要**：必须引用 `.png` 格式，禁止引用 `.pdf`。

## 四、未来工作建议

### 4.1 作者建议的未来工作

[从论文结论部分提取，1 段]

### 4.2 基于分析的未来方向

[编号列表，每个方向包含：动机、可能的方法、预期成果、挑战]

1. **方向1：[方向名称]**
   - 动机：[...]
   - 可能的方法：[具体方法建议]
   - 预期成果：[...]
   - 挑战：[...]

2. **方向2：[方向名称]**
   [类似格式]

### 4.3 改进建议

[编号列表，每条包含：当前问题、改进方案、预期效果]

1. **改进1：[改进名称]**
   - 当前问题：[...]
   - 改进方案：[...]
   - 预期效果：[...]

## 五、 我的综合评价

### 5.1 价值评分

#### 5.1.1 总体评分
**[X.X]/10** - [评分理由简述]

#### 5.1.2 分项评分

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | [X]/10 | [详细理由] |
| 技术质量 | [X]/10 | [详细理由] |
| 实验充分性 | [X]/10 | [详细理由] |
| 写作质量 | [X]/10 | [详细理由] |
| 实用性 | [X]/10 | [详细理由] |

### 5.2 重点关注

#### 5.2.1 值得关注的技术点

#### 5.2.2 需要深入理解的部分

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[相关论文1]] - [关系描述：改进/扩展/对比等]
- [[相关论文2]] - [关系描述]

### 6.2 背景相关
- [[背景论文1]] - [关系描述]
- [[背景论文2]] - [关系描述]
   
### 6.3 后续工作
- [[后续论文1]] - [关系描述]
- [[后续论文2]] - [关系描述]

## 外部资源
[可列举一些相关的视频、博客、项目等的链接]

> [!tip] 关键启示
> [论文最重要的启示，用一句话总结核心思想]

> [!warning] 注意事项
> - [注意事项1]
> - [注意事项2]
> - [注意事项3]

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ [推荐指数和简要理由，如：强烈推荐阅读！这是XX领域的里程碑论文]
```

## 步骤5：更新知识图谱

### 5.1 添加或更新节点

1. **读取图谱数据**
   - 文件路径：`$OBSIDIAN_VAULT_PATH/PaperGraph/graph_data.json`

2. **添加或更新该论文的节点**
   - 包含分析元数据：
     - quality_score
     - tags
     - domain
     - analyzed: true

3. **创建到相关论文的边**
   - 对每篇相关论文，创建边
   - 边类型：
     - `improves`：改进关系
     - `related`：一般关系
   - 权重：基于相似度（0.3-0.8）

4. **更新时间戳**
   - 设置`last_updated`为当前日期

5. **保存图谱**
   - 写入更新的graph_data.json

## 步骤6：展示分析摘要

### 6.1 输出格式

```markdown
## 论文分析完成！

**论文**：[[论文标题]] (arXiv:XXXX.XXXXX)

**分析状态**：✅ 已生成详细笔记
**笔记位置**：[[Papers/领域/YYYY-MM-DD-arXiv-ID.md]]

---

**综合评分**：[X.X/10]

**分项评分**：
- 创新性：[X/10]
- 技术质量：[X/10]
- 实验充分性：[X/10]
- 写作质量：[X/10]
- 实用性：[X/10]

**突出亮点**：
- [亮点1]
- [亮点2]
- [亮点3]

**主要优势**：
- [优势1]
- [优势2]

**主要局限**：
- [局限1]
- [局限2]

**相关论文**（N篇）：
- [[相关论文1]] - [关系]
- [[相关论文2]] - [关系]
- [[相关论文3]] - [关系]

**技术路线**：
本文属于[技术路线]，主要关注[子方向]。

---

**快速操作**：
- 点击笔记链接查看详细分析
- 使用`/paper-search`搜索更多相关论文
- 打开Graph View查看论文关系
- 根据分析决定深入研究或跳过

**建议**：
- [基于分析的具体建议1]
- [基于分析的具体建议2]
```

## 重要规则

- **保留用户现有笔记** - 不要覆盖手动笔记
- **使用全面分析** - 涵盖方法论、实验、价值评估
- **根据 `$LANGUAGE` 设置选择语言** - `"en"` 用英文写笔记，`"zh"` 用中文写笔记（section headers、content 都要匹配）
- **引用相关工作** - 建立连接到现有知识库
- **客观评分** - 使用一致的评分标准
- **更新知识图谱** - 维护论文间关系
- **图文并茂** - 论文中的所有图都要用上（核心架构图、方法图、实验结果图等）
- **优雅处理错误** - 如果一个源失败则继续
- **管理token使用** - 全面但不超出token限制

### Obsidian 格式规则（必须遵守！）

1. **图片嵌入**：**必须使用** `![[filename.png|800]]`，**禁止使用** `![alt](path%20encoded)`
   - Obsidian 不支持 URL 编码路径（`%20`, `%26` 等不工作）
   - Obsidian 会自动在 vault 中搜索文件名，无需写完整路径
2. **Wikilink 必须用 display alias**：`[[File_Name|Display Title]]`，禁止 bare `[[File_Name]]`
   - 下划线文件名直接显示会很丑
3. **不要用 `---` 作为"无数据"占位符**：使用 `--` 代替（`---` 会被 Obsidian 解析为分隔线）
4. **机构/Affiliation 提取**：从 arXiv 源码包的 `.tex` 文件提取 `\author`/`\affiliation` 字段；若不可用，标 `--`

### 双语 Section Headers 对照表

根据 `$LANGUAGE` 设置选择对应语言的 section header：

| Chinese (`zh`) | English (`en`) |
|---|---|
| 核心信息 | Core Information |
| 摘要翻译 | Abstract & Translation |
| 研究背景与动机 | Research Background & Motivation |
| 研究问题 | Research Problem |
| 方法概述 | Method Overview |
| 实验结果 | Experimental Results |
| 深度分析 | In-Depth Analysis |
| 与相关论文对比 | Comparison with Related Work |
| 技术路线定位 | Technical Roadmap |
| 未来工作建议 | Future Work |
| 我的综合评价 | Assessment |
| 我的笔记 | My Notes |
| 相关论文 | Related Papers |
| 外部资源 | External Resources |

## 分析标准

### 评分细则（0-10分制）

**创新性**：
- 9-10分：新颖突破、新范式
- 7-8分：显著改进或组合
- 5-6分：次要贡献、已知或已确立
- 3-4分：增量改进
- 1-2分：已知或已确立

**技术质量**：
- 9-10分：严谨的方法论、合理的方法
- 7-8分：良好的方法、次要问题
- 5-6分：可接受的方法、有问题的方法
- 3-4分：有问题的方法、差的方法
- 1-2分：差的方法

**实验充分性**：
- 9-10分：全面的实验、强基线
- 7-8分：良好的实验、充分的基线
- 5-6分：可接受的实验、部分基线
- 3-4分：有限的实验、差基线
- 1-2分：差的实验或没有基线

**写作质量**：
- 9-10分：清晰、组织良好
- 7-8分：总体清晰、次要问题
- 5-6分：可理解、部分不清晰
- 3-4分：难以理解、混乱
- 1-2分：差写作

**实用性**：
- 9-10分：高实用影响、可直接应用
- 7-8分：良好实用潜力
- 5-6分：中等实用价值
- 3-4分：有限实用性、理论性仅
- 1-2分：低实用性、理论性仅

### 关系类型定义

- `improves`：对相关工作的明显改进
- `extends`：扩展或建立在相关工作之上
- `compares`：直接对比，可能更好/更差在什么方面
- `follows`：同一研究路线的后续工作
- `cites`：引用（如果有引用数据可用）
- `related`：一般概念关系
```

## 错误处理

- **论文未找到**：检查ID格式，建议搜索
- **arXiv掉线**：使用缓存或稍后重试，在输出中注明局限性
- **PDF解析失败**：回退到摘要，注明局限性
- **相关论文未找到**：说明缺乏上下文
- **图谱更新失败**：继续但不更新图谱

## 使用说明

当用户调用 `/paper-analyze [论文ID]` 时：

### 快速执行（推荐）

使用以下bash脚本一键执行完整流程：

```bash
#!/bin/bash

# 变量设置
PAPER_ID="$1"
TITLE="${2:-待定标题}"
AUTHORS="${3:-Kimi Team}"
DOMAIN="${4:-其他}"

# 执行完整流程
python "scripts/generate_note.py" --paper-id "$PAPER_ID" --title "$TITLE" --authors "$AUTHORS" --domain "$DOMAIN" --language "$LANGUAGE" --language "$LANGUAGE" || \
    echo "笔记生成脚本执行失败"

# 提取图片
# 调用 extract-paper-images skill
# /extract-paper-images "$PAPER_ID" "$DOMAIN" "$TITLE" || \
#     echo "图片提取失败"
```

### 手动分步执行（用于调试）

#### 步骤0：初始化环境
```bash
# 创建工作目录
mkdir -p /tmp/paper_analysis
cd /tmp/paper_analysis
```

#### 步骤1：识别论文
```bash
# 搜索已有笔记
find "${VAULT_ROOT}/Papers" -name "*${PAPER_ID}*" -type f
```

#### 步骤2：获取论文内容
```bash
# 下载PDF和源码（见步骤2.1、2.2、2.3）

# 或者从已有数据读取
cat /tmp/paper_analysis/{1-introduction,2-joint-optimization,3-agent-swarm,5-eval}.tex
```

#### 步骤3：复制图片
```bash
# 使用extract-paper-images skill
/extract-paper-images "$PAPER_ID" "$DOMAIN" "$TITLE"
```

#### 步骤4：生成笔记
```bash
# 使用外部脚本生成笔记
python "scripts/generate_note.py" --paper-id "$PAPER_ID" --title "$TITLE" --authors "$AUTHORS" --domain "$DOMAIN" --language "$LANGUAGE"
```

#### 步骤5：更新图谱
```bash
# 使用外部脚本更新知识图谱
python "scripts/update_graph.py" --paper-id "$PAPER_ID" --title "$TITLE" --domain "$DOMAIN" --score 8.8 --language "$LANGUAGE"
```

#### 步骤6：使用obsidian-markdown skill修复格式

分析完成后，调用`/obsidian-markdown`来确保frontmatter格式正确，然后手动补充详细内容。

### 完整工作流程示例

**场景1：分析arXiv论文（有网络访问）**
```bash
# 一键执行
bash run_full_analysis.sh 2602.02276 "Kimi K2.5: Visual Agentic Intelligence" "Kimi Team" "智能体"
```

**场景2：分析本地PDF（无网络访问）**
```bash
# 手动上传PDF
cp /path/to/local.pdf /tmp/paper_analysis/[ID].pdf

# 执行分析（跳过步骤2的下载）
python3 run_paper_analysis.py [ID] [TITLE] [AUTHORS] [DOMAIN] --local-pdf /tmp/paper_analysis/[ID].pdf
```

### 注意事项

1. **frontmatter格式（重要）**：所有字符串值必须用双引号包围
   ```yaml
   ---
   paper_id: "[arXiv:XXXX.XXXXX / DOI](链接)"
   title: "论文标题"
   authors: "[作者1, 作者2, 作者3] #如果超过3个作者，加上et al."
   institution: "[从作者推断或查看论文]"
   pushlication: "[期刊/论文] [发布时间:YYYY-MM-DD]"
   quality_score: "[X.X]/10"
   link:
     - "[Github](链接)"
     - "[PDF](链接)"
   date: "YYYY-MM-DD"
   ---
   ```
   **Obsidian对YAML格式要求严格，缺少引号会导致frontmatter无法正常显示！**

2. **link 属性必须用 YAML 列表**：每个链接单独一项，不要写成一个字符串包含多个链接（Obsidian 只会渲染第一个）
   ```yaml
   link:
     - "[Github](https://github.com/xxx)"
     - "[PDF](https://arxiv.org/pdf/xxx)"
   ```
3. **图片嵌入**：**必须使用 Obsidian wikilink 语法** `![[filename.png|800]]`
   - **禁止使用** `![alt](path%20encoded)` — URL 编码在 Obsidian 中不工作
   - Obsidian 会自动搜索 vault 中的文件名，无需写完整路径
   - 从arXiv提取的图片可能是 `.pdf` 或 `.png` 格式
4. **wikilinks**：必须使用 display alias `[[File_Name|Display Title]]`，禁止 bare `[[File_Name]]`
5. **领域推断**：根据论文内容自动推断
6. **相关论文**：在笔记中引用 `[[path/to/note|Paper Title]]`
