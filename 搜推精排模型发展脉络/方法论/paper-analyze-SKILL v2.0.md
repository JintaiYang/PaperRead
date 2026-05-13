---
name: paper-analyze
version: "2.0"
description: "深度分析单篇学术论文并生成 Obsidian 格式笔记。当用户提供 arXiv ID（如 2402.12345）、论文 URL（arxiv.org/abs/...）、论文标题、或说"分析论文""读论文""paper analyze""帮我看这篇论文"时触发。输出包含结构化笔记（背景/方法/实验/评价）、图片提取、知识图谱更新。"
allowed-tools: Read, Write, Bash, WebFetch
last-updated: "2026-05-13"
quality-rating: "A"
---

# paper-analyze · 单篇论文深度分析

> **目标**：对特定论文进行深度分析，生成图文并茂的 Obsidian 格式笔记，评估质量和价值，并更新知识图谱。

## 不适用场景

| 场景 | 应该用 |
|------|--------|
| 仅提取论文中的图片（不需要完整分析） | `extract-paper-images` |
| 搜索已有论文笔记库中的内容 | `paper-search` |
| 生成每日论文推荐列表 | `start-my-day` |
| 搜索顶会论文列表（CVPR/NeurIPS 等） | `conf-papers` |
| 批量分析多篇论文 | 多次调用本 skill |
| 非学术论文的文档阅读 | 直接对话或 `pdf` skill |

## 前置检查

执行分析前，按顺序验证以下条件：

1. **Vault 路径**：检查环境变量 `OBSIDIAN_VAULT_PATH` 是否设置且目录存在
   - 未设置 → 提示用户：「请设置 OBSIDIAN_VAULT_PATH 环境变量指向你的 Obsidian vault 根目录」
   - 目录不存在 → 报错退出
2. **网络连通性**：尝试访问 `https://arxiv.org`
   - 不可达 → 提示用户提供本地 PDF，走离线分析流程
3. **PyMuPDF 可用性**：检查 `python3 -c "import fitz"`
   - 不可用 → 警告图片转换将跳过，笔记中不含图片引用
4. **论文标识符有效性**：验证输入的 arXiv ID/URL/标题格式合法
   - 格式无效 → 提示用户提供正确格式

## 实体提取规则

从用户输入中自动提取论文标识符：

| 输入格式 | 提取规则 | 示例 |
|----------|----------|------|
| 纯数字 ID | 匹配 `\d{4}\.\d{4,5}(v\d+)?` | `2402.12345` |
| 带前缀 ID | 去掉 `arXiv:` 前缀 | `arXiv:2402.12345` |
| arXiv abs URL | 提取路径末段 | `https://arxiv.org/abs/2402.12345` |
| arXiv pdf URL | 提取路径末段，去 `.pdf` | `https://arxiv.org/pdf/2402.12345.pdf` |
| HuggingFace URL | 提取 `/papers/` 后的 ID | `https://huggingface.co/papers/2402.12345` |
| 论文标题 | 用标题搜索现有笔记或 arXiv API | `"Attention Is All You Need"` |
| 本地文件路径 | 直接读取 | `/path/to/note.md` |

## 执行流程

### 步骤1：识别论文 `[freedom: low]`

1. 解析用户输入，提取论文标识符（按上方实体提取规则）
2. 在 `$OBSIDIAN_VAULT_PATH/Papers/` 目录中搜索已有笔记
   - 找到 → 读取并展示，询问是否需要更新
   - 未找到 → 继续步骤2

### 步骤2：获取论文内容 `[freedom: low]`

1. **获取元数据**：访问 arXiv API 或页面，提取标题、作者、摘要、日期、类别
2. **下载源码包**：`curl -L "https://arxiv.org/e-print/[ID]"` 获取 TeX 源码和图片
3. **下载 PDF**：`curl -L "https://arxiv.org/pdf/[ID]"` 作为备选
4. **读取 TeX 内容**：解析各章节 `.tex` 文件获取全文

### 步骤3：提取并转换图片 `[freedom: low]`

1. 从源码包中复制图片到目标目录：`Papers/[领域]/[论文标题]/images/`
2. 用 PyMuPDF 将 PDF 格式图片转为高清 PNG（4x zoom，宽度≥800px）
3. 生成图片索引

**⚠️ 关键**：笔记中只引用 `.png` 文件，不引用 `.pdf`。

### 步骤4：执行深度分析 `[freedom: medium]`

按以下维度系统分析论文：

1. **摘要分析**：提取关键概念、研究目标、主要贡献，生成中文翻译
2. **方法论分析**：识别核心方法、技术创新点、与现有方法区别、方法结构
3. **实验分析**：提取数据集、基线方法、评估指标、关键结果、消融研究
4. **洞察生成**：研究价值、局限性、未来工作、与相关工作对比

分析深度可根据论文复杂度和长度灵活调整，但每个维度至少需要覆盖。

### 步骤5：生成论文笔记 `[freedom: low]`

1. 确定领域分类（搜推系统/智能体/大模型/多模态技术/强化学习等）
2. 按笔记模板结构生
3. 成完整笔记（见下方「笔记模板」节）
4. 插入论文图片引用（使用 Obsidian wikilink 语法 `![[filename.png|800]]`）
5. 写入文件：`Papers/[领域]/[论文标题].md`
6. 内容较长时（400+ 行），分批写入：先 frontmatter + 前几节，再逐节追加

**领域推断规则**：
- agent/swarm/multi-agent/orchestration → 智能体
- vision/visual/image/video → 多模态技术
- reinforcement learning/RL → 强化学习_LLM_Agent
- language model/LLM/MoE/transformer → 大模型
- recommendation/search/ranking/CTR → 搜推系统

### 步骤6：更新知识图谱 `[freedom: medium]`

1. 读取图谱文件：`$OBSIDIAN_VAULT_PATH/PaperGraph/graph_data.json`
2. 添加/更新论文节点（含 quality_score, tags, domain）
3. 创建到相关论文的边（类型：improves/extends/compares/follows/related）
4. 保存图谱

图谱更新可根据是否找到相关论文灵活处理，未找到相关论文时仅添加孤立节点。

### 步骤7：展示分析摘要 `[freedom: medium]`

输出格式：

```markdown
## 论文分析完成！

**论文**：[[论文标题]] (arXiv:XXXX.XXXXX)
**笔记位置**：[[Papers/领域/论文标题.md]]
**综合评分**：[X.X/10]

**分项评分**：创新性 X / 技术质量 X / 实验 X / 写作 X / 实用性 X

**突出亮点**：[2-3 个关键亮点]
**主要局限**：[1-2 个主要局限]
**技术路线**：本文属于[技术路线]，主要关注[子方向]。

**建议**：[基于分析的具体建议]
```

## 错误处理

| 阶段 | 异常 | 处理 | 降级方案 |
|------|------|------|----------|
| 前置检查 | OBSIDIAN_VAULT_PATH 未设置 | 提示用户设置 | 询问用户指定输出目录 |
| 前置检查 | PyMuPDF 不可用 | 警告并继续 | 跳过图片转换，笔记中不含图片 |
| 步骤2 | arXiv 不可达/论文未找到 | 检查 ID 格式，建议搜索 | 用户提供本地 PDF → 用 WebFetch 或 PDF skill 提取文本 |
| 步骤2 | 源码包下载失败（无 TeX） | 尝试 HTML 版本 | 仅从 PDF 提取文本，标注 `[⚠️ 无源码，基于PDF提取]` |
| 步骤3 | PDF 图片转换失败 | 跳过失败的图片 | 笔记中标注 `[图片转换失败]`，不中断流程 |
| 步骤4 | 论文内容过短/不完整 | 基于可用内容分析 | 在笔记中标注 `[⚠️ 信息不完整，部分基于摘要]` |
| 步骤5 | 文件写入失败 | 检查权限和路径 | 输出笔记内容到终端，用户手动保存 |
| 步骤6 | 图谱文件损坏/格式错误 | 创建新图谱文件 | 跳过图谱更新，不影响笔记生成 |

## 重要规则

- **保留用户现有笔记** — 不要覆盖手动笔记内容
- **图片格式** — 必须使用 `![[filename.png|800]]`，禁止 `![alt](path)`
- **Wikilink** — 必须使用 display alias `[[File_Name|Display Title]]`
- **语言** — 根据 `$LANGUAGE` 环境变量选择中文/英文（默认中文）
- **客观评分** — 使用评分细则进行一致评分
- **渐进写入** — 笔记较长时分批写入，避免超出 token 限制
- **信息真实** — 笔记信息来自原文，不能推测，更不能捏造

---

# 笔记模板

生成笔记时，**必须严格按照此结构填充内容**，每个 section 都应包含实质性分析内容（而非占位符）。

## 写作风格要求

1. **信息真实**：笔记信息来自于原文，不能推测，更不能捏造
2. **段落式写作为主**：优先使用自然语言段落进行深入分析和讨论，论文内容需要有条理
3. **列表辅助关键信息**：数据集统计、实验结果、评分等结构化信息用表格或列表展示
4. **公式融入上下文**：核心公式用 `$$...$$` 独立展示，然后用自然语言解释每个符号的含义
5. **图文并茂**：论文中的**所有重要图**都要引用（架构图、方法细节图、实验结果图），每张图配说明文字
6. **具体有洞察力**：分析要用具体数字支撑（如"AUC 提升 0.0017"），避免泛泛而谈
7. **分批次写入**：笔记内容较长时（400+ 行），先创建文件写入 frontmatter 和前几节，再用 string_replace 逐节填充

## Frontmatter 格式

```yaml
---
paper_id: "[arXiv:XXXX.XXXXX / DOI](链接)"
title: "原论文标题"
authors: "作者1, 作者2, 作者3 et al."
institution: "[从作者推断或查看论文]"
publication: "[期刊/会议] [发布时间:YYYY-MM-DD]"
tags:
  - [方法标签-无空格]
  - [KeyWords翻译成中文-空格替换为-]
quality_score: "[X.X]/10"
link:
  - "[Github](链接)"
  - "[PDF](链接)"
date: "YYYY-MM-DD"
---
```

**格式规则**：
- 所有字符串值用双引号包围
- link 属性用 YAML 列表，每个链接单独一项
- tag 名称不能包含空格，空格用短横线(-) 连接

## 笔记正文结构

```markdown
## 一、研究背景与动机

### 1.1 领域现状
[详细描述该研究领域当前的发展状况]

### 1.2 现有方法的局限性
[深入分析现有方法存在的问题, 信息来自原论文]

### 1.3 本文解决方案概述
[清晰、准确地概述论文解决方案]

## 二、解决方案

### 2.1 核心思想
[用通俗易懂的语言解释方法的核心思想]

### 2.2 整体架构
[描述方法的整体架构，包括主要组件和它们之间的关系]

**架构图选择原则**：
1. **优先使用论文中的现成图** - 转换为高清PNG后插入
2. **仅在无图时创建Canvas** - 用JSON Canvas自行绘制

插入方式：`![[arch.png|800]]`
> 图1：[架构描述]

**⚠️ 重要**：必须引用 `.png` 格式图片，禁止直接引用 `.pdf`。

#### 各模块详细说明

**模块N：[模块名称]**
- **功能**：[该模块的主要功能]
- **输入**：[输入数据/信息]
- **输出**：[输出数据/信息]
- **处理流程**：[步骤描述]
- **关键技术**：[使用的关键技术或算法]
- **数学公式**：
  $$\theta^* = \arg\min_\theta L(\theta)$$

## 三、实验结果

### 3.1 数据集

| 数据集 | 样本数 | 特征维度 | 类别数 | 数据类型 |
|--------|--------|----------|--------|----------|
| 数据集1 | X万 | Y维 | Z类 | [类型] |

### 3.2 实验设置
[实验环境、实现细节、超参数设计]

#### 3.2.1 基线方法
[列出所有对比的基线方法]

#### 3.2.2 评估指标
[列出所有评估指标]

### 3.3 实验结果与分析

| 方法 | 数据集1-指标1 | 数据集1-指标2 | 平均排名 |
|------|---------------|---------------|----------|
| 基线1 | X.X±Y.Y | X.X±Y.Y | N |
| **本文方法** | **X.X±Y.Y** | **X.X±Y.Y** | **N** |

#### 结果分析
[对主实验结果的详细分析]

### 3.4 消融实验
[消融实验的设计思路和结果分析]

### 3.5 实验结果图

![[experiment_results.png|800]]
> 图2：[图描述]

## 四、未来工作建议

### 4.1 作者建议的未来工作
[从论文结论部分提取]

### 4.2 基于分析的未来方向

1. **方向1：[方向名称]**
   - 动机：[...]
   - 可能的方法：[具体方法建议]
   - 预期成果：[...]
   - 挑战：[...]

### 4.3 改进建议

1. **改进1：[改进名称]**
   - 当前问题：[...]
   - 改进方案：[...]
   - 预期效果：[...]

## 五、我的综合评价

### 5.1 价值评分

**[X.X]/10** - [评分理由简述]

| 评分维度 | 分数 | 评分理由 |
|----------|------|----------|
| 创新性 | [X]/10 | [详细理由] |
| 技术质量 | [X]/10 | [详细理由] |
| 实验充分性 | [X]/10 | [详细理由] |
| 写作质量 | [X]/10 | [详细理由] |
| 实用性 | [X]/10 | [详细理由] |

### 5.2 重点关注

#### 值得关注的技术点
#### 需要深入理解的部分

## 我的笔记

%% 用户可以在这里添加个人阅读笔记 %%

## 六、相关论文

### 6.1 直接相关
- [[相关论文1|显示名]] - [关系描述]

### 6.2 背景相关
- [[背景论文1|显示名]] - [关系描述]

### 6.3 后续工作
- [[后续论文1|显示名]] - [关系描述]

## 外部资源
[相关的视频、博客、项目等]

> [!tip] 关键启示
> [论文最重要的启示]

> [!warning] 注意事项
> - [注意事项1]

> [!success] 推荐指数
> ⭐⭐⭐⭐⭐ [推荐指数和理由]
```

## 公式输出规范

1. 行内公式使用 `$...$`
2. 块级公式使用 `$$...$$` 并单独成行
3. 不要用三反引号代码块包裹需要渲染的公式
4. 保持符号与原论文一致

## Obsidian 格式规则

1. **图片嵌入**：**必须使用** `![[filename.png|800]]`，**禁止使用** `![alt](path%20encoded)`
2. **Wikilink 必须用 display alias**：`[[File_Name|Display Title]]`
3. **不要用 `---` 作为"无数据"占位符**：使用 `--` 代替
4. **机构提取**：从 arXiv 源码包的 `.tex` 文件提取 `\author`/`\affiliation` 字段

## 双语 Section Headers 对照表

| Chinese (`zh`) | English (`en`) |
|---|---|
| 研究背景与动机 | Research Background & Motivation |
| 解决方案 | Solution |
| 实验结果 | Experimental Results |
| 未来工作建议 | Future Work |
| 我的综合评价 | Assessment |
| 我的笔记 | My Notes |
| 相关论文 | Related Papers |
| 外部资源 | External Resources |

## 评分细则（0-10分制）

| 维度 | 9-10 | 7-8 | 5-6 | 3-4 | 1-2 |
|------|------|-----|-----|-----|-----|
| 创新性 | 新范式 | 显著改进 | 次要贡献 | 增量改进 | 已知方法 |
| 技术质量 | 严谨方法论 | 良好、次要问题 | 可接受 | 有问题 | 差 |
| 实验充分性 | 全面+强基线 | 良好+充分基线 | 部分基线 | 有限实验 | 无基线 |
| 写作质量 | 清晰组织好 | 总体清晰 | 可理解 | 难理解 | 差 |
| 实用性 | 可直接应用 | 良好潜力 | 中等 | 有限 | 仅理论 |

---

# 使用示例

## 快速执行（推荐）

```bash
#!/bin/bash
PAPER_ID="$1"
TITLE="${2:-待定标题}"
AUTHORS="${3:-Unknown}"
DOMAIN="${4:-其他}"

# 执行完整流程
python "scripts/generate_note.py" --paper-id "$PAPER_ID" --title "$TITLE" --authors "$AUTHORS" --domain "$DOMAIN" --language "$LANGUAGE" || \
    echo "笔记生成脚本执行失败"
```

## 手动分步执行（用于调试）

### 步骤1：初始化环境

```bash
mkdir -p /tmp/paper_analysis
cd /tmp/paper_analysis
```

### 步骤2：获取论文内容

```bash
# 下载PDF和源码
curl -L "https://arxiv.org/pdf/${PAPER_ID}" -o /tmp/paper_analysis/${PAPER_ID}.pdf
curl -L "https://arxiv.org/e-print/${PAPER_ID}" -o /tmp/paper_analysis/${PAPER_ID}.tar.gz
tar -xzf /tmp/paper_analysis/${PAPER_ID}.tar.gz -C /tmp/paper_analysis/
```

### 步骤3：提取元数据

```bash
curl -s "https://arxiv.org/abs/${PAPER_ID}" > /tmp/paper_analysis/arxiv_page.html
TITLE=$(grep -oP '<title>\K[^<]*' /tmp/paper_analysis/arxiv_page.html | head -1)
AUTHORS=$(grep -oP 'citation_author" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | paste -sd ', ')
DATE=$(grep -oP 'citation_date" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | head -1)
```

### 步骤4：提取图片

```bash
cp /tmp/paper_analysis/Figures/*.{pdf,png,jpg,jpeg} "${IMAGES_DIR}/" 2>/dev/null

# 用 PyMuPDF 将 PDF 图片转为高清 PNG（4x zoom，宽度≥800px）
python3 -c "
import fitz, os, glob
img_dir = '${IMAGES_DIR}'
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
```

### 步骤5：生成笔记

```bash
python "scripts/generate_note.py" --paper-id "$PAPER_ID" --title "$TITLE" --authors "$AUTHORS" --domain "$DOMAIN" --language "$LANGUAGE"
```

### 步骤6：更新图谱

```bash
python "scripts/update_graph.py" --paper-id "$PAPER_ID" --title "$TITLE" --domain "$DOMAIN" --score 8.8 --language "$LANGUAGE"
```

## 场景示例

**场景1：分析 arXiv 论文（有网络）**

```bash
bash run_full_analysis.sh 2602.02276 "Paper Title" "Author1, Author2" "智能体"
```

**场景2：分析本地 PDF（无网络）**

```bash
cp /path/to/local.pdf /tmp/paper_analysis/[ID].pdf
# 跳过下载步骤，直接执行分析
```

## 图片转换规范

从 arXiv 提取的图片可能是 PDF 格式，必须转换为 PNG 后再在笔记中引用：

- 使用 PyMuPDF (fitz) 以 4x zoom 转换
- 确保输出宽度 ≥ 800px
- 笔记中只引用 `.png` 文件，不引用 `.pdf`
- 图片存放目录：`Papers/[领域]/[论文标题]/images/`

---

# 脚本资源

| 脚本 | 用途 | 参数 |
|------|------|------|
| `scripts/generate_note.py` | 生成笔记骨架 | `--paper-id --title --authors --domain --language --vault` |
| `scripts/update_graph.py` | 更新知识图谱 | `--paper-id --title --domain --score --related --vault --language` |

脚本位于：`~/.catpaw/skills/paper-analyze/scripts/`

---

# 改版记录

## v2.0 (2026-05-13) — better-skill 审查后重构

基于 better-skill 10维度审查（原评分 C 级），全面重构为 A 级：

**主要改进**：

1. **Description 优化**：从简单描述扩展为 WHAT+WHEN+triggers 三段式，触发词覆盖完整
2. **边界定义**：新增6条排除场景表格，明确 what NOT to do
3. **前置检查**：新增4项 preflight checks（vault路径/网络/PyMuPDF/ID格式），每项含失败处理
4. **实体提取**：新增7种输入格式的自动解析规则表
5. **流程重构**：从冗长867行压缩至158行 SKILL.md + 2个 reference 文件，用 freedom 标注各步骤灵活度
6. **错误处理**：新增8场景 × 4列（阶段/异常/处理/降级）的完整错误表
7. **渐进披露**：核心流程在 SKILL.md，模板和示例拆至 references/ 目录
8. **评分细则独立**：5维度评分标准从正文抽取到模板文件，评分有据可依
