# PaddleOCR-VL + llama.cpp GGUF 部署文档

> 基于 PPOCR-VL1.6-llama.cpp 项目，使用 llama.cpp 替换 FastDeploy 作为 VLM 推理后端。

## 架构概览

```
┌─────────────────────────────────────────────────┐
│                   docker-compose                 │
│                                                  │
│  ┌─────────────────┐       ┌──────────────────┐ │
│  │   llama-ocr      │       │  paddleocr-app    │ │
│  │  (llama-server)  │◄──────│  (FastAPI 服务)   │ │
│  │                  │ HTTP  │                   │ │
│  │  port 8118       │       │  port 8086        │ │
│  │  GPU: RTX 4090   │       │  GPU: RTX 4090    │ │
│  └────────┬─────────┘       └────────┬──────────┘ │
│           │                          │            │
│           ▼                          ▼            │
│  ┌──────────────────────────────────────────┐     │
│  │        /home/alpha/.paddlex/             │     │
│  │  PPOCR GGUF 模型 + PaddleX 缓存         │     │
│  └──────────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

### 组件职责

| 服务 | 镜像 | 功能 |
|------|------|------|
| `llama-ocr` | `docker-llama-ocr` | PaddleOCR-VL-1.6 GGUF 推理（llama.cpp server） |
| `paddleocr-app` | `docker-paddleocr-app` | FastAPI 路由: PP-DocLayoutV3 版面检测 + PP-OCRv5/VL 识别 |

### 调用流程

```
用户请求 -> FastAPI (8086)
  +-> Layout 版面检测 (PaddleX)
  +-> 简单页 -> PP-OCRv5 (轻量识别)
  +-> 复杂页 -> llama.cpp -> PaddleOCR-VL GGUF (全面解析)
  +-> 合并结果 -> 返回 Markdown
```

---

## 部署步骤

### 1. 环境要求

| 组件 | 要求 |
|------|------|
| OS | Ubuntu 20.04+ (22.04 已验证) |
| GPU | NVIDIA RTX 4090 24GB (推荐) |
| 驱动 | >= 570.133.07 |
| CUDA | >= 12.6 (推荐 12.8) |
| Docker | 24.0+ + docker-compose v1 |
| 磁盘 | >= 50GB 剩余空间 |

### 2. 宿主机编译 llama.cpp（预编译二进制）

```bash
cd docker
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp-src
cd llama.cpp-src
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release -j8
```

> **重要**: 必须使用最新版 llama.cpp（含 PR #18825 的 paddleocr 架构支持），禁用 `--branch` 参数。

### 3. 准备二进制文件到 Docker context

```bash
cd docker
# 复制二进制
cp llama.cpp-src/build/bin/llama-server .
cp llama.cpp-src/build/bin/llama-cli .
# 复制所有 .so 文件（含版本化 symlink）
cd llama.cpp-src/build/bin
for so in libllama-server-impl.so libllama-common.so* libmtmd.so* \
           libllama.so* libggml.so* libggml-base.so* \
           libggml-cpu.so* libggml-cuda.so*; do
  cp -a "$so" ../../../../
done
cd ../../..
```

### 4. 下载 GGUF 模型

```bash
mkdir -p /home/alpha/.paddlex/official_models/ppocr-vl-gguf
# 放入以下两个文件（来自 PPOCR-VL1.6-llama.cpp 项目）:
#   PaddleOCR-VL-1.6-GGUF.gguf (主模型)
#   PaddleOCR-VL-1.6-GGUF-mmproj.gguf (视觉投影)
```

### 5. Docker 构建与启动

```bash
cd docker
docker-compose build
docker-compose up -d
```

### 6. 验证

```bash
docker ps --filter name=paddleocr
curl http://localhost:8086/health
curl http://localhost:8118/health
```

---

## 部署踩坑全记录

## ⚠️ 部署注意事项

| # | 关键点 | 正确做法 |
|---|--------|---------|
| 1 | **PaddlePaddle GPU wheel 不在 PyPI 上** | `pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/`。PyPI 上的 `paddlepaddle`（~180 MB）和 `paddlepaddle-gpu`（最高 2.6.2）都是 CPU-only |
| 2 | **docker-compose v1 GPU 配置** | `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=0`，`deploy.resources` 在 v1 中无效 |
| 3 | **Docker COPY 丢失 symlink** | `cp -a` 保留 symlink，COPY 所有版本化 .so 文件 |
| 4 | **llama.cpp 版本要求** | 克隆最新版（不带 `--branch`），需含 PR [#18825](https://github.com/ggml-org/llama.cpp/pull/18825) 的 paddleocr 架构支持 |
| 5 | **构建上下文过大** | 添加 `.dockerignore` 排除 `llama.cpp-src/`，改用预编译二进制模式 |

---

## API 调用示例

### 单张图片 OCR

```bash
curl -X POST http://localhost:8086/ocr \
  -H "Authorization: Bearer sk-paddleocr-vl-prod-2026" \
  -F "file=@/path/to/document.jpg"
```

### 多页 PDF OCR

```bash
curl -X POST http://localhost:8086/ocr/batch \
  -H "Authorization: Bearer sk-paddleocr-vl-prod-2026" \
  -F "files=@/path/to/doc1.pdf"
```

### 直接调用 llama.cpp API

```bash
curl -X POST http://localhost:8118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr-vl",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请完整提取图片中的文字"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }],
    "max_tokens": 4096,
    "temperature": 0
  }'
```

---

## 维护命令

```bash
# 查看日志
docker-compose logs -f llama-ocr
docker-compose logs -f paddleocr-app

# 重启服务
docker-compose restart llama-ocr

# 完整停止
docker-compose down

# 清理旧镜像
docker image prune -f
```
