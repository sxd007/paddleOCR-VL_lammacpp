# ============================================================================
# PaddleOCR-VL FastAPI 服务 — 路由 + OCR 引擎
# ============================================================================
# 架构:
#   PP-DocLayoutV3 (版面检测) → 简单页: PP-OCRv5, 复杂页: PaddleOCR-VL
#   VLM 后端通过 HTTP 调用 llama.cpp (OpenAI 兼容接口)
#
# 构建: docker build -t paddleocr-app -f Dockerfile.app ..
# ============================================================================

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

LABEL maintainer="PaddleOCR-VL"
LABEL description="PaddleOCR-VL FastAPI 路由服务 + PP-OCRv5 + PaddleX"
LABEL version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- 系统依赖 (清华镜像加速) ------------------------------------------------
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    echo 'Acquire::By-Hash "no";' > /etc/apt/apt.conf.d/99-disable-by-hash && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# ---- Python 依赖 ----------------------------------------------------------
# 注意：PaddlePaddle 3.x 已统一包名为 paddlepaddle (不再分 cpu/gpu)
# 使用中科大 PyPI 镜像加速（pypi.mirrors.ustc.edu.cn）
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        -i https://pypi.mirrors.ustc.edu.cn/simple/ \
        --trusted-host pypi.mirrors.ustc.edu.cn \
        paddlepaddle==3.2.1 \
        paddleocr==3.6.0 \
        "paddlex[ocr]==3.6.1" \
        fastapi==0.136.3 \
        uvicorn==0.49.0 \
        pydantic==2.13.4 \
        pydantic_core==2.46.4 \
        python-multipart==0.0.32 \
        pypdfium2==5.9.0 \
        pillow==12.1.0 \
        numpy==2.2.6 \
        requests

# ---- 应用代码 --------------------------------------------------------------
WORKDIR /app

# 先复制 app 代码
COPY app/ /app/app/
# 复制 .env 配置
COPY docker/.env /app/.env

# ---- 健康检查 --------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8086/health || exit 1

# ---- 启动 ------------------------------------------------------------------
EXPOSE 8086

ENV VLM_BACKEND=llamacpp
ENV LLAMACPP_URL=http://llama-ocr:8118
ENV CUDA_VISIBLE_DEVICES=0

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8086", "--workers", "1"]
