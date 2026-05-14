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

### 步骤2：识别论文

```bash
# 搜索已有笔记
find "${VAULT_ROOT}/Papers" -name "*${PAPER_ID}*" -type f
```

### 步骤3：获取论文内容

```bash
# 下载PDF和源码
curl -L "https://arxiv.org/pdf/${PAPER_ID}" -o /tmp/paper_analysis/${PAPER_ID}.pdf
curl -L "https://arxiv.org/e-print/${PAPER_ID}" -o /tmp/paper_analysis/${PAPER_ID}.tar.gz
tar -xzf /tmp/paper_analysis/${PAPER_ID}.tar.gz -C /tmp/paper_analysis/
```

### 步骤4：提取元数据

```bash
curl -s "https://arxiv.org/abs/${PAPER_ID}" > /tmp/paper_analysis/arxiv_page.html
TITLE=$(grep -oP '<title>\K[^<]*' /tmp/paper_analysis/arxiv_page.html | head -1)
AUTHORS=$(grep -oP 'citation_author" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | paste -sd ', ')
DATE=$(grep -oP 'citation_date" content="\K[^"]*' /tmp/paper_analysis/arxiv_page.html | head -1)
```

### 步骤5：提取图片

```bash
# 使用 extract-paper-images skill
# 或手动复制并转换
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

### 步骤6：生成笔记

```bash
python "scripts/generate_note.py" --paper-id "$PAPER_ID" --title "$TITLE" --authors "$AUTHORS" --domain "$DOMAIN" --language "$LANGUAGE"
```

### 步骤7：更新图谱

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
