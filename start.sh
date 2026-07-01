#!/bin/bash
# ================================================================
# PaddleOCR-VL 企业级API服务 — 一键启动脚本
# 支持离线部署模式（模型已预缓存于 ~/.paddlex/official_models/）
# ================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}      PaddleOCR-VL 企业级 API 服务 启动脚本${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ---- 检测虚拟环境 ----
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    echo -e "${RED}[错误] 未找到虚拟环境！请先创建虚拟环境。${NC}"
    echo "执行: python3 -m venv .venv"
    exit 1
fi

echo -e "${YELLOW}[1/5] 激活虚拟环境...${NC}"
source "${VENV_DIR}/bin/activate"
echo -e "${GREEN}  ✓ Python: $(which python) $(python --version 2>&1 | awk '{print $2}')${NC}"

# ---- 检查端口冲突 ----
PORT=${PORT:-8086}
echo -e "${YELLOW}[2/5] 检查端口 ${PORT} 是否可用...${NC}"
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo -e "${RED}[错误] 端口 ${PORT} 已被占用！${NC}"
    echo "当前占用该端口的进程:"
    ss -tlnp 2>/dev/null | grep ":${PORT} "
    echo ""
    echo "请修改 .env 文件中的 PORT 配置，或使用以下命令指定其他端口："
    echo "  PORT=8087 ./start.sh"
    exit 1
fi
echo -e "${GREEN}  ✓ 端口 ${PORT} 可用${NC}"

# ---- 检查依赖完整性 ----
echo -e "${YELLOW}[3/5] 检查依赖...${NC}"

if python -c "import paddle; print(paddle.__version__)" 2>/dev/null 1>&2; then
    PADDLE_VER=$(python -c "import paddle; print(paddle.__version__)")
    echo -e "${GREEN}  ✓ PaddlePaddle ${PADDLE_VER}${NC}"
else
    echo -e "${RED}  ✗ PaddlePaddle 未安装！请执行:${NC}"
    echo "     pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
    exit 1
fi

if python -c "import paddleocr; print(paddleocr.__version__)" 2>/dev/null 1>&2; then
    OCR_VER=$(python -c "import paddleocr; print(paddleocr.__version__)")
    echo -e "${GREEN}  ✓ PaddleOCR ${OCR_VER}${NC}"
else
    echo -e "${RED}  ✗ PaddleOCR 未安装！请执行:${NC}"
    echo "     pip install 'paddleocr[doc-parser]'"
    exit 1
fi

python -c "import fastapi, uvicorn" 2>/dev/null 1>&2 || {
    echo -e "${RED}  ✗ FastAPI / uvicorn 未安装！请执行:${NC}"
    echo "     pip install fastapi uvicorn[standard] python-multipart"
    exit 1
}
echo -e "${GREEN}  ✓ 所有依赖已就绪${NC}"

# ---- GPU 检测 ----
echo -e "${YELLOW}[4/5] 检查 GPU 状态...${NC}"
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}  ✓ GPU: ${GPU_INFO}${NC}"
else
    echo -e "${YELLOW}  ⚠ nvidia-smi 不可用，将使用 CPU 模式${NC}"
    export DEVICE=cpu
fi

# ---- 模型缓存检测 ----
echo -e "${YELLOW}[5/5] 检查模型缓存...${NC}"
MODEL_DIR="${PADDLEX_HOME:-$HOME/.paddlex}/official_models"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null | wc -l)" -gt 5 ]; then
    MODEL_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -type d | wc -l)
    echo -e "${GREEN}  ✓ 模型已缓存 ($((MODEL_COUNT - 1)) 个模型)$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{printf " (~%s)", $1}')${NC}"
    echo -e "${GREEN}  ✓ 支持离线运行${NC}"
else
    echo -e "${YELLOW}  ⚠ 模型尚未完全缓存。首次启动会自动下载（需联网）。${NC}"
    echo -e "${YELLOW}    如需离线运行，先在联网环境执行:${NC}"
    echo -e "${YELLOW}    paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --save_path /tmp/preload${NC}"
fi

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  启动服务...${NC}"
echo -e "${CYAN}  API 地址: http://0.0.0.0:${PORT}${NC}"
echo -e "${CYAN}  健康检查: http://localhost:${PORT}/health${NC}"
echo -e "${CYAN}  接口文档: http://localhost:${PORT}/docs${NC}"
echo -e "${CYAN}  API Key: 在请求头中添加 X-API-Key${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

exec python -m app.main
