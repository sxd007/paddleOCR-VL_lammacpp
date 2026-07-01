# PaddleOCR-VL 企业级文档解析 API 服务

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.2.1-brightgreen)](https://github.com/PaddlePaddle/Paddle)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.6.0-orange)](https://github.com/PaddlePaddle/PaddleOCR)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-yellow)](LICENSE)

基于 PaddleOCR-VL 1.6 + PP-DocLayoutV3 的企业级文档解析 REST API，支持 **模型路由架构**：先版面分类、再动态调度模型，在大文档上实现 3-10 倍提速。

---

## 目录

- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [Docker 容器化部署](#docker-容器化部署生产推荐)
  - [部署踩坑全记录](#部署踩坑全记录12-个坑)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [路由架构详解](#路由架构详解)
- [安全说明](#安全说明)
- [GitHub 仓库准备](#github-仓库准备)
- [常见问题](#常见问题)
- [License](#license)

---

## 架构概览

### 核心思想：Layout 先行的模型路由

```
                    ┌──────────────────────┐
                    │  客户端 (任意 HTTP)   │
                    └──────────┬───────────┘
                               │ POST /v1/ocr
                               ▼
                    ┌──────────────────────┐
                    │  FastAPI + API Key   │
                    │  鉴权 / CORS / 日志  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   OCREngine          │
                    │   (专用工作线程)      │
                    │   ┌──────────────┐   │
                    │   │  Task Queue  │   │
                    │   └──────┬───────┘   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ModelRouter        │
                    │   ┌────────────────┐ │
                    │   │ PP-DocLayoutV3 │ │  ← 版面检测
                    │   │ (轻量，~0.5s)  │ │
                    │   └───────┬────────┘ │
                    │           ▼          │
                    │   ┌────────────────┐ │
                    │   │  路由决策      │ │
                    │   │ simple/complex │ │
                    │   └───┬────┬───────┘ │
                    │  ┌────▼┐ ┌─▼────────┐│
                    │  │轻量  │ │PaddleOCR ││
                    │  │OCR   │ │-VL 1.6  ││
                    │  │~0.5s │ │~3-8s/页 ││
                    │  └────┬┘ └───┬──────┘│
                    └───────┴───────┴────────┘
                               │
                    ┌──────────▼───────────┐
                    │   统一 Markdown/Text  │
                    └──────────────────────┘
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
| POST | `/v1/ocr` | 单文件 OCR（自动识别图片/PDF） |
| POST | `/v1/ocr/batch` | 批量 OCR（最多 20 个文件） |

### POST /v1/ocr 请求

```json
{
  "image": "<文件路径 | URL | Base64编码>",
  "filename": "文件名.pdf",
  "page_size": 20
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 是 | 支持三种格式：**本地路径** / **URL** / **Base64** |
| `filename` | string | 否 | 文件名（用于日志和调试） |
| `page_size` | int | 否 | PDF 分页大小（默认 20，-1 不分批整份处理） |

### POST /v1/ocr 响应

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
    "markdown": "--- 第 1 页 --- [light_ocr]\n# 标题\n\n正文...\n\n--- 第 2 页 --- [vlm]\n## 表格数据\n\n[...]",
    "text": "标题\n\n正文...",
    "pages": [
      {"page": 1, "markdown": "# 标题\n\n正文...", "text": "正文...", "route": "light_ocr"},
      {"page": 2, "markdown": "## 表格数据\n\n...", "text": "表格数据\n\n...", "route": "vlm"}
    ]
  }
}
```

### 调用示例

**Python:**

```python
import requests

API_URL = "http://your-server:8086/v1/ocr"
API_KEY = "your-api-key"

# 本地文件路径
resp = requests.post(API_URL, 
    json={"image": "/path/to/文档.pdf", "page_size": 20},
    headers={"X-API-Key": API_KEY})
print(resp.json()["data"]["markdown"])

# Base64
import base64
with open("发票.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
resp = requests.post(API_URL,
    json={"image": b64, "filename": "发票.pdf"},
    headers={"X-API-Key": API_KEY})

# 远程 URL
resp = requests.post(API_URL,
    json={"image": "https://example.com/document.pdf"},
    headers={"X-API-Key": API_KEY})
```

**cURL:**

```bash
curl -X POST http://localhost:8086/v1/ocr \
  -H "X-API-Key: your-api-key" \
  -d '{"image": "/tmp/doc.pdf", "page_size": 20}'
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
│   ├── models.py                     # Pydantic 请求/响应模型
│   ├── main.py                       # FastAPI 应用 + 端点 + 生命周期
│   ├── ocr_service.py                # OCREngine 工作线程 + PDF 处理
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
  │         └── 调用对应模型 → 统一输出
  │
  └── PDF → 渲染 → 逐页路由 → 收集 → 排序 → 合并
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
