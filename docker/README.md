# PaddleOCR-VL Docker 部署指南

## 架构

```
用户请求 ──▶ paddleocr-app (FastAPI, 端口 8086)
                  │
                  ├─ PP-DocLayoutV3 (版面检测)
                  ├─ PP-OCRv5 (轻量文字识别)
                  └─ LlamaCppClient ──▶ llama-ocr (llama.cpp, 端口 8118)
                                           └─ PaddleOCR-VL-1.6-GGUF (VLM模型)
```

## 前置条件

- Docker + NVIDIA Container Toolkit
- GGUF 模型文件（首次部署需下载）

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

## 部署

```bash
# 构建并启动所有服务
docker compose up -d

# 查看启动日志
docker compose logs -f

# 检查服务状态
curl http://localhost:8086/health

# 停止服务
docker compose down
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLM_BACKEND` | `llamacpp` | VLM 后端类型：`llamacpp` / `fastdeploy` / `native` |
| `LLAMACPP_URL` | `http://llama-ocr:8118` | llama.cpp 服务地址 |
| `PORT` | `8086` | FastAPI 服务端口 |
| `ROUTING_ENABLED` | `True` | 是否启用版面路由（简单页→轻量OCR，复杂页→VLM） |
| `PDF_PAGE_SIZE` | `20` | PDF 分批处理每批页数 |

## 从 FastDeploy 迁移

如需切换回原有 FastDeploy 后端，修改 `.env`：

```bash
VLM_BACKEND=fastdeploy
# 并确保 fastdeploy-ocr 服务已在 docker-compose.yml 中定义
```

## API 调用示例

```bash
# 健康检查
curl http://localhost:8086/health

# 单图片 OCR（Base64）
curl -X POST http://localhost:8086/v1/ocr \
  -H "X-API-Key: sk-paddleocr-vl-prod-2026" \
  -H "Content-Type: application/json" \
  -d '{"image": "'"$(base64 -w0 /path/to/doc.png)"'"}'

# 远程 URL（含 PDF）
curl -X POST http://localhost:8086/v1/ocr \
  -H "X-API-Key: sk-paddleocr-vl-prod-2026" \
  -H "Content-Type: application/json" \
  -d '{"image": "https://example.com/report.pdf", "page_size": 10}'

# 批量处理
curl -X POST http://localhost:8086/v1/ocr/batch \
  -H "X-API-Key: sk-paddleocr-vl-prod-2026" \
  -H "Content-Type: application/json" \
  -d '{"images": [
    {"image": "'"$(base64 -w0 img1.png)"'"},
    {"image": "'"$(base64 -w0 img2.png)"'"}
  ]}'

# 直接调用 llama.cpp（纯 VLM，不走路由）
curl http://localhost:8118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr-vl",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "OCR:"},
      {"type": "image_url", "image_url": {
        "url": "data:image/png;base64,'"$(base64 -w0 doc.png)"'"
      }}
    ]}],
    "temperature": 0
  }'
```
