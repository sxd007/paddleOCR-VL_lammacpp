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

共 12 个坑，按出现顺序排列：

### 坑1: CUDA Toolkit 未安装
**现象**: `CMake Error: CUDA Toolkit not found`  
**解决**: 安装 CUDA 12.8  

### 坑2: 80 并发编译导致系统卡死
**现象**: `docker build` 到 47% 时 SSH 断连、80核满载、风扇狂转  
**根因**: `-j$(nproc)` = 80 并行，CPU 410W + GPU 450W 触发 PSU 保护  
**解决**: 限制并发 `-j8`

### 坑3: Docker context 过大 (5.2GB)
**现象**: 构建时 I/O 100%，系统无响应  
**根因**: 无 `.dockerignore`，整个 `llama.cpp-src/` 被打包  
**解决**: 添加 `.dockerignore` + 预编译二进制模式

### 坑4: 缺失共享库 libllama.so
**现象**: `error while loading shared libraries: libllama.so`  
**解决**: COPY 二进制时同时 COPY 所有 `.so` 文件

### 坑5: CUDA 库路径未配置
**现象**: `libcudart.so.12: cannot open shared object file`  
**解决**: 设置 `LD_LIBRARY_PATH` 包含 `/usr/local/cuda/targets/x86_64-linux/lib`

### 坑6: PaddlePaddle 3.x 包名变更
**现象**: `ERROR: Could not find a version paddlepaddle-gpu==3.2.1`  
**根因**: PaddlePaddle 3.x 统一包名为 `paddlepaddle`（不再分 cpu/gpu）  
**解决**: `paddlepaddle-gpu` -> `paddlepaddle`

### 坑7: 内部 H3C PyPI 镜像不可达
**现象**: pip 从 `172.22.1.36` 安装失败  
**解决**: 换用公开镜像 `https://pypi.mirrors.ustc.edu.cn/simple/`

### 坑8: docker-compose v1 GPU 配置不生效
**现象**: `libcuda.so.1: cannot open shared object file`  
**根因**: `deploy.resources` 在 docker-compose v1 中无效（仅 swarm 模式）  
**解决**: 改用 `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=0`

### 坑9: Docker COPY 丢失 symlink
**现象**: ldconfig 报 `is not a symbolic link`  
**根因**: 新版 llama.cpp 使用版本化 .so (如 `libggml.so.0 -> libggml.so.0.15.1`)  
**解决**: `cp -a` 保留 symlink，Dockerfile 中 COPY 所有版本化文件

### 坑10: llama.cpp 版本不支持 paddleocr 架构
**现象**: `unknown model architecture: 'paddleocr'`  
**根因**: 旧版未合入 PR [#18825](https://github.com/ggml-org/llama.cpp/pull/18825)  
**解决**: 克隆最新版（不带 `--branch`），2026-02 后的版本均支持

### 坑11: --mmproj 参数名变化
**现象**: `error: invalid argument: --mmproj`  
**解决**: 新版已支持 `--mmproj`（别名为 `-mm`）

### 坑12: --flash-attn 参数格式变化
**现象**: `error: unknown value for --flash-attn: '-b'`  
**说明**: 不同版本参数格式不同，需查阅 `--help`

| 版本 | 格式 |
|------|------|
| 旧版 (b5050) | `--flash-attn` (布尔) |
| 新版 (f3e1828) | `--flash-attn on` (取值 on/off/auto) |

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
