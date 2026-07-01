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

# ---- Python 依赖（通用包：走中科镜像加速）-----------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        -i https://pypi.mirrors.ustc.edu.cn/simple/ \
        --trusted-host pypi.mirrors.ustc.edu.cn \
        fastapi==0.136.3 \
        uvicorn==0.49.0 \
        pydantic==2.13.4 \
        pydantic_core==2.46.4 \
        python-multipart==0.0.32 \
        pypdfium2==5.9.0 \
        pillow==12.1.0 \
        numpy==2.2.6 \
        requests

# ---- PaddlePaddle 全家桶（GPU 版：从 PaddlePaddle 官方包索引安装）--------------
# ⚠️ 关键说明（2024‑07 排查记录）：
#
#   1. PaddlePaddle 从 3.x 起，GPU 版 wheel 不再发布到 PyPI。
#      PyPI 上的 `paddlepaddle`（~180 MB）和 `paddlepaddle-gpu`（最高 2.6.2）
#      都是 CPU-only。千万别从 PyPI 装 PaddlePaddle 3.x——装到的一定是 CPU 版。
#
#   2. 真正的 GPU wheel（~1.9 GB）发布在 PaddlePaddle 官方的独立包索引站：
#        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
#      包名叫 `paddlepaddle-gpu`。这个索引站类似 PyTorch 的
#      `--extra-index-url https://download.pytorch.org/whl/cu126` 模式，
#      是另一个完整的 PyPI 兼容仓库，不是加速镜像。
#
#   3. CUDA 版本对应关系（宿主机 CUDA 版本查看：`nvidia-smi`）：
#        CUDA 11.8 → cu118     CUDA 12.6 → cu126     CUDA 12.9 → cu129
#      容器运行时 CUDA 版本必须 ≥ wheel 编译目标版本。
#      如果宿主机 CUDA > wheel 目标版本（如 host=12.8, wheel=12.6），
#      向下兼容，无需升级。
#
#   4. GPU wheel 要求 Compute Capability ≥ 7.5。
#      RTX 4090 (CC 8.9) / A100 (CC 8.0) / V100 (CC 7.0→不满足)均可。
#
#   验证方式（构建后）：
#     docker exec paddleocr-app python3 -c "
#     import paddle; print(paddle.is_compiled_with_cuda());
#     paddle.utils.run_check()
#     "
#   应输出 "PaddlePaddle works well on 1 GPU."
# ----------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# paddleocr / paddlex 从官方 PyPI 安装（它们本身无 GPU/CPU 之分，
# 依赖的 paddlepaddle-gpu 已在上一行装好，不会重复拉取 CPU 版）
RUN python3 -m pip install --no-cache-dir \
        paddleocr==3.6.0 \
        "paddlex[ocr]==3.6.1"

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
