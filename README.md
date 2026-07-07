# PaddleOCR-VL 企业级文档解析 API 服务

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.2.1-brightgreen)](https://github.com/PaddlePaddle/Paddle)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.6.0-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-yellow)](LICENSE)

基于 PaddleOCR-VL 1.6 + PP-DocLayoutV3 的企业级文档解析 REST API，支持 **模型路由架构**：先版面分类、再动态调度模型，在大文档上实现 3-10 倍提速。

**响应采用双层结构设计**：保留原始机械精度（elements → bbox + 置信度）的同时，提供可直接消费的可读层（markdown / text），让客户端按需选择。

---

## 目录

- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [Docker 容器化部署](#docker-容器化部署生产推荐)
  - [部署踩坑全记录](#部署踩坑全记录12-个坑)
- [API 文档](#api-文档)
  - [Response 数据结构详解](#response-数据结构详解)
  - [Element 模型](#element-元素模型)
  - [响应字段分类说明](#响应字段分类说明)
  - [include 参数——按需返回字段](#include-参数按需返回字段)
  - [客户端最佳实践](#客户端最佳实践)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [路由架构详解](#路由架构详解)
- [安全说明](#安全说明)
- [常见问题](#常见问题)
- [License](#license)

---

## 架构概览

### 核心思想：Layout 先行的模型路由

```
                    ┌──────────────────────────┐
                    │    客户端 (任意 HTTP)     │
                    └──────────┬───────────────┘
                               │ POST /v1/ocr
                               ▼
                    ┌──────────────────────────┐
                    │   FastAPI + API Key      │
                    │   鉴权 / CORS / 日志     │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    OCREngine             │
                    │    (专用工作线程)         │
                    │    ┌──────────────┐      │
                    │    │  Task Queue  │      │
                    │    └──────┬───────┘      │
                    └──────────┬───────────────┘
                               │
                    ┌──────────▼───────────────┐
                    │    ModelRouter           │
                    │    ┌────────────────┐    │
                    │    │ PP-DocLayoutV3 │    │  ← 版面检测
                    │    │ (轻量，~0.5s)  │    │
                    │    └───────┬────────┘    │
                    │            ▼             │
                    │    ┌────────────────┐    │
                    │    │  路由决策      │    │
                    │    │ simple/complex │    │
                    │    └───┬────┬───────┘    │
                    │   ┌────▼┐ ┌─▼────────┐  │
                    │   │轻量  │ │PaddleOCR │  │
                    │   │OCR   │ │-VL 1.6  │  │
                    │   │~0.5s │ │~3-8s/页 │  │
                    │   └────┬┘ └───┬──────┘  │
                    └────────┴──────┴──────────┘
                               │  ╔═══双层输出═══╗
                    ┌──────────▼──▼──▼──────────┐
                    │   原始层       可读层      │
                    │   elements     markdown   │
                    │   (bbox+conf)  text       │
                    └───────────────────────────┘
```

### 路由逻辑

每页先跑 PP-DocLayoutV3 版面检测，根据检测到的区块类型决定处理路径：

| 版面特征 | 路由目标 | 速度 | 适用场景 |
|---------|---------|------|---------|
| 仅含 `text`, `paragraph_title`, `doc_title`, `header`, `footer` 等 | **PaddleOCR(det+rec)** | ~0.5s/页 | 纯文字条款、正文页面 |
| 含 `table`, `chart`, `formula`, `seal`, `image`, `figure_title` 等 | **PaddleOCR-VL** | ~3-8s/页 | 表格、复杂版面、扫描件 |
| 未检测到任何版面元素 | **PaddleOCR-VL**（安全回退） | ~3-8s/页 | 未知类型页面 |

### 性能预期

| 文档类型 | 纯 VLM 模式 | 路由模式 | 提速 |
|---------|------------|---------|------|
| 100 页纯文字 | ~600s (10 分钟) | ~60s | **10x** |
| 100 页混合文档 (70% 文字 + 30% 复杂) | ~600s | ~200s | **3x** |
| 20 页表格文档 | ~120s | ~60s | **2x** |
| 单页纯文字图片 | ~6s | ~1s | **6x** |

---

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA (推荐 16GB+ 显存) |
| CUDA | 11.8+ |
| Python | 3.10+ |
| 硬盘 | 10GB+ 模型缓存空间 + 5GB 项目空间 |

### 一键启动（本地）

```bash
cd paddleOCR-VL
bash start.sh
```

启动过程：

1. **检测端口** — 确认 8086 可用
2. **检查依赖** — PaddlePaddle 3.2.1 + PaddleOCR 3.6.0 + FastAPI
3. **检查 GPU** — 自动检测 NVIDIA 显卡
4. **检查模型缓存** — 首次启动自动下载模型（需联网）
5. **加载三个模型**（按顺序）：
   - PP-DocLayoutV3（版面检测，~0.5s 加载）
   - PaddleOCR(det+rec)（轻量 OCR，~1s 加载）
   - PaddleOCR-VL 1.6（VLM 模型，~30s 加载）
6. **预热推理** — 每个模型跑一次空白推理触发 JIT 编译
7. **服务就绪** — `http://0.0.0.0:8086`

启动成功的日志结尾：

```
模型路由引擎全部加载完成
Uvicorn running on http://0.0.0.0:8086
```

### 从零安装（非 Docker）

手动搭建（不依赖 start.sh）：

```bash
# 1. 创建虚拟环境
python3.10 -m venv .venv
source .venv/bin/activate

# 2. 安装 PaddlePaddle（GPU 版）
# ⚠️ PaddlePaddle 3.x GPU wheel 不在 PyPI 上，必须从官方索引站安装：
#    -i https://www.paddlepaddle.org.cn/packages/stable/cuXXX/
# PyPI 上的 paddlepaddle / paddlepaddle-gpu 都是 CPU-only。
# CUDA 12.6:
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# CUDA 11.8:
# pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# CPU only（从 PyPI 装）:
# pip install paddlepaddle==3.2.1

# 3. 安装其他依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 修改配置（特别是 API_KEYS）

# 5. 启动服务
python -m app.main
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8086/health

# 模型信息（无需鉴权）
curl http://localhost:8086/v1/models
```

---

## Docker 容器化部署（生产推荐）

项目提供 `docker/` 配置，支持三种 VLM 后端。**推荐使用 llama.cpp GGUF 后端**（性能最佳）。

### 全容器化（推荐）

所有组件在 Docker 中运行，**宿主机无需额外启动 Python 进程**：

```bash
cd docker
# 确保 GGUF 模型文件已下载（见下方说明）
docker compose build
docker compose up -d
# 验证
curl http://localhost:8086/health
```

### 下载 GGUF 模型

```bash
# 方式一：使用 modelscope（推荐）
pip install modelscope
modelscope download --model Aid003/PaddleOCR-VL-1.6-GGUF \
  PaddleOCR-VL-1.6-GGUF.gguf \
  PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  --local_dir ~/.paddlex/official_models/ppocr-vl-gguf

# 方式二：使用 wget
mkdir -p ~/.paddlex/official_models/ppocr-vl-gguf
wget -c https://www.modelscope.cn/models/Aid003/PaddleOCR-VL-1.6-GGUF/resolve/master/PaddleOCR-VL-1.6-GGUF.gguf \
  -O ~/.paddlex/official_models/ppocr-vl-gguf/PaddleOCR-VL-1.6-GGUF.gguf
wget -c https://www.modelscope.cn/models/Aid003/PaddleOCR-VL-1.6-GGUF/resolve/master/PaddleOCR-VL-1.6-GGUF-mmproj.gguf \
  -O ~/.paddlex/official_models/ppocr-vl-gguf/PaddleOCR-VL-1.6-GGUF-mmproj.gguf
```

### 后端切换

| 后端 | 配置 | VLM 推理框架 | 模型格式 | 说明 |
|------|------|-------------|---------|------|
| **llamacpp** | `VLM_BACKEND=llamacpp` （默认） | llama.cpp | GGUF 量化 | ✅ **推荐**，速度快、显存低 |
| fastdeploy | `VLM_BACKEND=fastdeploy` | FastDeploy 2.3 | Paddle 原生 | 旧方案，逐步迁移 |
| native | `VLM_BACKEND=native` | PaddleOCRVL | Paddle 原生 | 单进程模式 |

### 仅 VLM 加速（逐步迁移）

宿主机 FastAPI 不变，只替换 VLM 推理后端：

```bash
cd docker
docker compose up -d llama-ocr
# 修改 .env: VLM_BACKEND=llamacpp, LLAMACPP_URL=http://localhost:8118
# 重启宿主机 FastAPI 服务
```

详细部署文档见 [`docker/README.md`](docker/README.md)。

---

## ⚠️ 部署注意事项

几个最容易卡住的点，记住这些就够了：

| # | 关键点 | 正确做法 |
|---|--------|---------|
| 1 | **PaddlePaddle GPU wheel 不在 PyPI 上** | 必须从 `-i https://www.paddlepaddle.org.cn/packages/stable/cu126/` 装 `paddlepaddle-gpu==3.2.1`。PyPI 上的 `paddlepaddle` 装到的是 CPU 版 |
| 2 | **docker-compose v1 GPU 配置** | 用 `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=0`，`deploy.resources` 在 v1 中无效 |
| 3 | **Docker COPY 丢失 symlink** | 复制 .so 文件用 `cp -a` 保留 symlink 链，Dockerfile 中 COPY 所有版本化文件 |
| 4 | **llama.cpp 版本要求** | 必须用最新版（含 PR [#18825](https://github.com/ggml-org/llama.cpp/pull/18825)），`--branch` 会拉到旧版 |

---

## API 文档

### 鉴权

在请求头中添加 `X-API-Key`：

```bash
X-API-Key: your-api-key-here
```

也支持 `Authorization: Bearer <token>` 格式。

鉴权白名单（无需 Key）：
- `GET /health`
- `GET /v1/models`
- `GET /docs`、`GET /openapi.json`、`GET /redoc`

### 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康状态、GPU、模型加载状态 |
| GET | `/v1/models` | 模型版本、设备、能力列表 |
| POST | `/v1/ocr` | 单文件 OCR（自动识别图片/PDF），支持 `include` 参数 |
| POST | `/v1/ocr/batch` | 批量 OCR（最多 20 个文件），每项独立支持 `include` |

### POST /v1/ocr 请求

```json
{
  "image": "<文件路径 | URL | Base64编码>",
  "filename": "文件名.pdf",
  "page_size": 20,
  "mode": "routing",
  "include": ["markdown", "elements"]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 是 | 支持三种格式：**本地路径** / **URL** / **Base64** |
| `filename` | string | 否 | 文件名（用于日志和调试） |
| `page_size` | int | 否 | PDF 分页大小（默认 20，-1 不分批整份处理） |
| `mode` | string | 否 | 路由模式：`"routing"`（默认）版面分类后路由到专业引擎；`"vlm"` 全部使用 PaddleOCR-VL；`"table_pp"` 强制走 PPStructure 表格管线 |
| `include` | string[] | 否 | 按需返回字段列表，见 [include 参数](#include-参数按需返回字段) |

### POST /v1/ocr 响应——完整示例

```json
{
  "code": 0,
  "message": "success",
  "request_id": "a1b2c3d4",
  "data": {
    "index": 0,
    "filename": "文档.pdf",
    "file_type": "pdf",
    "success": true,
    "total_pages": 5,
    "markdown": "<!-- page 1 | route: light_ocr -->\n# 标题\n\n正文...\n\n---\n<!-- page 2 | route: table -->\n| 名称 | 数量 |\n|------|------|\n| A    | 10   |",
    "text": "标题\n\n正文...\n\n名称  数量\nA     10",
    "pages": [
      {
        "page": 1,
        "markdown": "# 标题\n\n正文...",
        "text": "标题\n\n正文...",
        "elements": [
          {
            "id": "p1_e0",
            "type": "paragraph",
            "reading_order": 0,
            "bbox": [12.3, 45.6, 789.0, 56.7],
            "confidence": 0.95,
            "content": {"text": "标题"}
          }
        ],
        "layout_blocks": [
          {"label": "text", "score": 0.98, "bbox": [10.0, 40.0, 800.0, 60.0]}
        ],
        "route": "light_ocr",
        "timing_ms": 523
      }
    ],
    "route_summary": {"light_ocr": 3, "table": 1, "vlm": 1, "error": 0},
    "total_timing_ms": 4523
  }
}
```

```

---

## Response 数据结构详解

> **设计哲学**: 信息不丢失，提供不同层级的服务效果。
>
> 原始识别信息（位置、置信度、结构化拆分）被完整保留在 `elements` 中；
> 经过修复和编排的可读内容（markdown / text）则让客户端无需处理底层细节就能直接消费。

### 双层结构全景

```
响应
├── 文件级 (data)
│   ├── markdown ── 所有页合并的可读 Markdown
│   ├── text     ── 所有页合并的纯文本
│   ├── pages ── 逐页结果
│   │   └── 每页:
│   │       ├── elements      ← ★ 原始层 (bbox + 置信度 + 结构化)
│   │       ├── layout_blocks ←   版面位置标注
│   │       ├── markdown      ←   可读层 (页码锚点分隔)
│   │       ├── text          ←   可读层 (纯文本)
│   │       ├── route         ←   路由标识
│   │       └── error_detail  ←   错误详情 (仅 route=error 时)
│   ├── route_summary ── 路由统计
│   └── total_timing_ms ── 总耗时
```

### 两条消费路径

| 路径 | 数据源 | 精度 | 适用场景 |
|------|--------|------|---------|
| **原始层** | `elements[]` | 精确 bbox + 置信度 | LLM 检索增强、高精度位置定位、表格结构化提取 |
| **可读层** | `markdown` / `text` | 已修复编排 | 直接展示、全文搜索、简单文本消费 |

两条路径信息一致，只是精度和格式不同。**不丢失任何信息**，客户端按需取用。

---

### Element 元素模型

每个 `element` 是页内的一个原子内容块：

```json
{
  "id": "p1_e0",
  "type": "paragraph",
  "reading_order": 0,
  "bbox": [12.3, 45.6, 789.0, 56.7],
  "confidence": 0.95,
  "content": {"text": "正文内容..."}
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 元素ID，格式 `p{page}_e{index}`，可在跨页引用中唯一标识 |
| `type` | string | 元素类型: `paragraph` / `table` / `formula` / `chart` / `seal` / `page_text` |
| `reading_order` | int | 页内阅读顺序，从 0 开始。`markdown` 即按此顺序拼接 |
| `bbox` | float[4] \| null | `[x1, y1, x2, y2]` 绝对像素坐标（相对于该页渲染图）。**vlm 路由为 null**——见下方"路由精度诚实"说明 |
| `confidence` | float \| null | 置信度。来源因路由而异——见下方 |
| `content` | dict | 按 type 区分结构的内容，详见下 |


#### content 结构按 type 区分

| type | content 格式 | 示例 |
|------|-------------|------|
| `paragraph` | `{"text": "..."}` | `{"text": "这是正文段落"}` |
| `table` | `{"html": "..."}` | `{"html": "\<table\>\<tr\>\<td\>数据\</td\>\</tr\>\</table\>"}` |
| `formula` | `{"text": "..."}` | `{"text": "E = mc²"}` |
| `chart` | `{"text": "..."}` | `{"text": "图表描述文字"}` |
| `seal` | `{"text": "..."}` | `{"text": "中华人民共和国财政部"}` |
| `page_text` | `{"text": "..."}` | VLM 路由兜底，整页文本 |

#### bbox / confidence 来源（路由精度诚实）

不同路由的 bbox 和 confidence 精度不同，服务端不做虚假精确：

| 路由 | bbox | confidence | 说明 |
|------|------|-----------|------|
| `light_ocr` | ✅ **真实 bbox**（每行检测框外接矩形） | ✅ **PP-OCR 逐行置信度** | 最精确 |
| `table` | ✅ **真实 bbox**（表格在渲染图中的位置） | ✅ **表格引擎置信度** | 仅返回大表框，非单元格级 |
| `vlm` | ⛔ **null** | ⛔ **null** | VLM 模型当前是整页推理，不具备逐元素定位能力 |

> ✅ **非虚假精确**：VLM 路由的 `bbox=null` 是诚实的精度声明，而非设计缺陷。
> 客户端收到 `null` 时，应知道该内容无法归属到具体子区域。

---

### 响应字段分类说明

为了便于客户端按需消费，响应字段分为三个层级：

#### 类别 A：核心元数据（始终返回，不受 `include` 影响）

```
code, message, request_id
data.index, data.filename, data.file_type, data.success, data.total_pages
data.pages[].page, data.pages[].route, data.pages[].timing_ms
data.route_summary, data.total_timing_ms
data.pages[].error_detail
```

#### 类别 B：内容字段（受 `include` 控制）

| 字段 | 说明 | 默认行为 |
|------|------|---------|
| `markdown` | 文件级 + 每页的 Markdown 可读内容 | ✅ 默认返回 |
| `text` | 文件级 + 每页的纯文本可读内容 | ✅ 默认返回 |
| `elements` | 结构化元素列表（原始层） | ✅ 默认返回 |
| `layout_blocks` | 版面检测位置标注 | ✅ 默认返回 |
| `hallucination_warnings` | VLM 幻觉检测警告 | ❌ 默认不返回（数据量大） |

---

### include 参数——按需返回字段

对大流量场景，可以用 `include` 参数裁剪响应体，减少网络开销。

**只返回可读层**（最简单的消费方式）：
```json
{"include": ["markdown", "text"]}
```

**只返回原始层 + 位置**（客户端自己做编排）：
```json
{"include": ["elements", "layout_blocks"]}
```

**只返回 markdown**（极致精简）：
```json
{"include": ["markdown"]}
```

**不传 include 则返回全部字段**（默认行为）。

---

### 客户端最佳实践

根据场景选择不同的消费策略：

#### 场景 1：仅需全文展示（RAG、知识库入库）

最直接的方式——消费文件级 `markdown`：

```python
import requests

resp = requests.post("http://localhost:8086/v1/ocr",
    json={"image": "document.pdf", "page_size": 20, "include": ["markdown"]},
    headers={"X-API-Key": "your-key"})
data = resp.json()["data"]
full_markdown = data["markdown"]          # 所有页合并的 markdown
```

> `markdown` 已按阅读顺序编排好，每页开头有 `<!-- page N | route: xxx -->`
> 锚点注释，正则可定位到具体页。

#### 场景 2：需要精确位置信息（文档 QA、LLM 检索增强）

结合 `elements` 的 bbox 做位置感知检索：

```python
resp = requests.post("http://localhost:8086/v1/ocr",
    json={"image": "contract.pdf"},
    headers={"X-API-Key": "your-key"})
data = resp.json()["data"]

# 遍历每页的元素，带位置信息
for page in data["pages"]:
    for el in page.get("elements") or []:
        if el["type"] == "paragraph" and el["bbox"]:
            x1, y1, x2, y2 = el["bbox"]
            print(f"Page {page['page']}, pos=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}):")
            print(f"  {el['content']['text'][:80]}")
        elif el["type"] == "table":
            html = el["content"]["html"]
            print(f"Page {page['page']}: table ({len(html)} chars HTML)")
```

#### 场景 3：表格结构化提取

`elements` 中 `type=table` 的 `content.html` 是完整 HTML 表格：

```python
from bs4 import BeautifulSoup

for page in data["pages"]:
    for el in page.get("elements") or []:
        if el["type"] != "table":
            continue
        soup = BeautifulSoup(el["content"]["html"], "html.parser")
        rows = soup.find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            print("\t".join(cells))
```

同时注意 `page["text"]` 中该表格的文本表示也已用 `\t` 和 `\n`
分隔保留——不依赖 HTML 解析也能读取表格内容。

#### 场景 4：过滤低置信度结果（精细化控制）

```python
for page in data["pages"]:
    for el in page.get("elements") or []:
        if el["confidence"] and el["confidence"] < 0.5:
            continue  # 跳过低置信度
        # 处理高置信度元素
```

#### 场景 5：PDF 多页文档——按页处理

```python
for page in data["pages"]:
    print(f"\n=== Page {page['page']} ({page['route']}) ===")
    # 查看渲染图中的位置坐标系
    for el in page.get("elements") or []:
        if el["bbox"]:
            # 绝对像素坐标，相对于该页渲染图
            x1, y1, x2, y2 = el["bbox"]
            area = (x2 - x1) * (y2 - y1)
            print(f"  [{el['type']}] area={area:.0f}px², conf={el['confidence']}")
```

#### 场景 6：API 错误处理

```python
resp = requests.post("http://localhost:8086/v1/ocr", ...)
result = resp.json()

if result["code"] != 0:
    print(f"请求失败: {result['message']}")
elif not result["data"]["success"]:
    print(f"处理失败: {result['data']['error']}")
else:
    # 检查是否有页处理出错
    for page in result["data"]["pages"]:
        if page["route"] == "error":
            print(f"Page {page['page']} 失败: {page['error_detail']}")
        else:
            print(f"Page {page['page']}: {page['route']} ({page['timing_ms']}ms)")

    print(f"路由分布: {result['data']['route_summary']}")
    print(f"总耗时: {result['data']['total_timing_ms']}ms")
```

#### 场景 7：通过 `include` 做带宽优化（移动端 / 高并发）

```python
# 只需要全文索引（最轻量）
light_resp = requests.post("http://localhost:8086/v1/ocr",
    json={"image": "doc.pdf", "include": ["markdown"]})

# 需要位置信息时再取完整数据
full_resp = requests.post("http://localhost:8086/v1/ocr",
    json={"image": "doc.pdf", "include": ["markdown", "elements", "layout_blocks"]})
```

---

### 调用示例

**Python:**

```python
import requests

API_URL = "http://your-server:8086/v1/ocr"
API_KEY = "your-api-key"

# 本地文件路径（默认返回完整字段）
resp = requests.post(API_URL, 
    json={"image": "/path/to/文档.pdf", "page_size": 20},
    headers={"X-API-Key": API_KEY})
result = resp.json()
print(result["data"]["markdown"])          # 可读层
print(result["data"]["pages"][0]["elements"])  # 原始层

# Base64（只取可读层，优化带宽）
import base64
with open("发票.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post(API_URL,
    json={"image": b64, "filename": "发票.pdf", "include": ["markdown"]},
    headers={"X-API-Key": API_KEY})

# 远程 URL（取 elements 做位置感知）
resp = requests.post(API_URL,
    json={"image": "https://example.com/document.pdf", "include": ["markdown", "elements"]},
    headers={"X-API-Key": API_KEY})
```

**cURL:**

```bash
curl -X POST http://localhost:8086/v1/ocr \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image": "/tmp/doc.pdf", "page_size": 20}'

# 带 include 参数
curl -X POST http://localhost:8086/v1/ocr \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"image": "/tmp/doc.pdf", "include": ["markdown", "elements"]}'
```

---

## 配置说明

编辑 `.env` 文件（从 `.env.example` 复制）：

```env
PORT=8086                           # 监听端口
HOST=0.0.0.0                        # 监听地址
API_KEYS=your-api-key-here          # API 鉴权密钥
DEVICE=gpu                          # gpu | cpu
DEVICE_ID=0                         # GPU 设备号
ROUTING_ENABLED=True                # 启用路由加速
ROUTING_COMPLEX_THRESHOLD=0.3       # 复杂区域占比阈值
LAYOUT_CONFIDENCE_THRESHOLD=0.3     # Layout 检测置信度
PDF_PAGE_SIZE=20                    # PDF 分页批处理大小
REQUEST_TIMEOUT=300                 # 请求超时（秒）
VLM_BACKEND=llamacpp                # VLM 后端
LLAMACPP_URL=http://localhost:8118  # llama.cpp 地址
```

完整配置项见 [`.env.example`](.env.example)。

---

## 项目结构

```
paddleOCR-VL/
├── app/                              # 核心应用代码
│   ├── __init__.py                   # 空包标记
│   ├── config.py                     # 配置加载（.env → Settings 类）
│   ├── auth.py                       # API Key 鉴权中间件
│   ├── models.py                     # Pydantic 请求/响应模型（Element, OCRResultPage 等）
│   ├── main.py                       # FastAPI 应用 + 端点 + 生命周期
│   ├── ocr_service.py                # OCREngine 工作线程 + PDF 处理（含双层输出组装）
│   ├── router.py                     # ★ 模型路由引擎（核心）
│   │   ├── PageClassifier            # PP-DocLayoutV3 版面检测
│   │   ├── LightweightOCREngine      # PaddleOCR(det+rec)
│   │   ├── TableRecognitionEngine    # PP-StructureV3 表格识别
│   │   ├── FastDeployClient          # FastDeploy HTTP 客户端
│   │   └── ModelRouter               # 路由决策 + 模型调度
│   └── llama_client.py               # llama.cpp GGUF HTTP 客户端
├── docker/                           # Docker 部署配置
│   ├── docker-compose.yml            # 多服务编排
│   ├── Dockerfile.llama              # llama.cpp 推理镜像
│   ├── Dockerfile.app                # FastAPI 路由镜像
│   ├── README.md                     # Docker 快速指南
│   ├── DEPLOY.md                     # ★ 部署踩坑 12 条
│   └── llama-proxy.py                # Cherry Studio 流式代理
├── .env                              # 环境配置（不提交 git）
├── .env.example                      # 环境配置模板
├── .gitignore                        # Git 忽略规则
├── requirements.txt                  # Python 依赖
├── start.sh                          # 一键启动脚本
└── README.md                         # 本文档
```

---

## 路由架构详解

### 实现原理

`app/router.py` 包含三个核心类：

| 类 | 职责 |
|----|------|
| `PageClassifier` | 封装 PP-DocLayoutV3，版面检测 → 区块标签列表 |
| `LightweightOCREngine` | 封装 PaddleOCR(det+rec)，纯文字页快速识别 |
| `ModelRouter` | 管理三模型生命周期，路由决策，统一输出格式 |

所有模型在同一工作线程串行加载调用，规避 PaddlePaddle 线程安全问题。

### 处理流程

```
请求 → _decode_input() → 判断图片/PDF
  ├── 图片 → ModelRouter.process_with_route()
  │         ├── classify() → 版面分析
  │         ├── decide_route() → light_ocr / table / vlm
  │         └── 调用对应模型 → 双层输出
  │               ├── 原始层: elements[] (bbox + confidence)
  │               └── 可读层: markdown + text
  │
  └── PDF → 渲染 → 逐页路由 → 收集 → 排序 → 合并 → 双层输出
```

### 路由开关

- `ROUTING_ENABLED=True`（默认）→ 路由加速
- `ROUTING_ENABLED=False` → 纯 VLM 模式

---

## 安全说明

> ⚠️ **重要**: `.env` 文件包含 API Key 等敏感信息，**切勿提交到 Git 仓库**。

1. **使用 `.env.example` 模板**创建你的 `.env`，不要直接编辑 `.env.example`
2. **生产环境使用强密码**替换 `API_KEYS`
3. **定期轮换 Key**，限制每个 Key 的使用范围
4. **网络隔离**：llama-server (port 8118) 无内置鉴权，建议内网使用或配合 nginx
5. **默认 Key 风险**：`config.py` 含 fallback 默认值，仅用于开发，生产务必覆盖

---

## GitHub 仓库准备

本项目已准备好发布到 GitHub：

```bash
# 1. 在 GitHub 创建新仓库（不勾选 README/.gitignore/LICENSE）

# 2. 本地初始化
git init
git add .
git status   # 确认 .gitignore 生效
git commit -m "Initial commit: PaddleOCR-VL enterprise document parsing API

- FastAPI REST server with model routing architecture
- PP-DocLayoutV3 layout-based page routing
- PaddleOCR(det+rec) light OCR for simple pages
- PaddleOCR-VL 1.6 VLM for complex pages
- llama.cpp GGUF backend support
- PDF processing with batch page rendering
- Docker compose deployment (llama-ocr + paddleocr-app)
- 12 documented deployment pitfalls"

# 3. 推送
git remote add origin https://github.com/<your-org>/paddleOCR-VL.git
git branch -M main
git push -u origin main
```

### 推送前检查清单

- [ ] `.env` 不被跟踪（`.gitignore` 中已排除）
- [ ] `docs/plan/` 不被跟踪（内部架构文档）
- [ ] 无 `.so`、`llama-cli`、`llama-server` 等二进制被跟踪
- [ ] 无 `__pycache__/`、`.venv/` 被跟踪
- [ ] 无 `cuda-keyring_*.deb` 被跟踪
- [ ] 无 `docker/llama.cpp-src.old/` 被跟踪
- [ ] 无 `docker/paddle-wheels/paddle-packages.tar.gz` 被跟踪

---

## 常见问题

### 首次请求为什么慢？

首次请求需要 PaddlePaddle JIT 编译，服务启动时已自动预热，仍建议启动后等 30-60 秒再发送请求。

### 支持哪些文件格式？

| 格式 | 支持 | 说明 |
|------|------|------|
| PNG/JPEG/WebP/GIF/TIFF | ✅ | 魔术字节自动识别 |
| PDF | ✅ | 自动检测、分页处理、结果合并 |
| PPT/DOCX | ❌ | 需要客户端先转 PDF |

### 如何关闭路由？

`.env` 中设置 `ROUTING_ENABLED=False` 后重启服务。

### GPU 显存占用？

- PaddleOCR-VL: ~2-3GB
- PP-DocLayoutV3: ~200MB
- PaddleOCR(det+rec): ~100MB
- 总计: ~3-4GB

### 和 Ollama 冲突吗？

会竞争 GPU 显存。建议：
- PaddleOCR-VL 单独使用一张 GPU
- 或通过 `CUDA_VISIBLE_DEVICES` 隔离

---

## License

本项目基于 PaddleOCR 开源模型构建，仅供学习和研究使用。
