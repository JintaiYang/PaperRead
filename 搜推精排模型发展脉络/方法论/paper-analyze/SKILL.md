---
name: paper-analyze
description: "深度分析单篇学术论文并生成 Obsidian 格式笔记。当用户提供 arXiv ID（如 2402.12345）、论文 URL（arxiv.org/abs/...）、论文标题、或说"分析论文""读论文""paper analyze""帮我看这篇论文"时触发。输出包含结构化笔记（背景/方法/实验/评价）、图片提取、知识图谱更新。"
allowed-tools: Read, Write, Bash, WebFetch
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
2. 按 [references/note-template.md](references/note-template.md) 中的模板结构生成完整笔记
3. 插入论文图片引用（使用 Obsidian wikilink 语法 `![[filename.png|800]]`）
4. 写入文件：`Papers/[领域]/[论文标题].md`
5. 内容较长时（400+ 行），分批写入：先 frontmatter + 前几节，再逐节追加

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
- **客观评分** — 使用 [references/note-template.md](references/note-template.md) 中的评分细则
- **渐进写入** — 笔记较长时分批写入，避免超出 token 限制

## 脚本资源

| 脚本 | 用途 | 参数 |
|------|------|------|
| `scripts/generate_note.py` | 生成笔记骨架 | `--paper-id --title --authors --domain --language --vault` |
| `scripts/update_graph.py` | 更新知识图谱 | `--paper-id --title --domain --score --related --vault --language` |

详细使用示例见 [references/usage-examples.md](references/usage-examples.md)。
