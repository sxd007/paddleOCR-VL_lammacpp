"""
PaddleOCR-VL 企业级 REST API 服务
支持 API Key 鉴权、并发控制、请求日志、离线部署
"""

import os
import sys
import uuid
import time
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import settings
from .auth import APIKeyMiddleware
from .ocr_service import engine
from .models import (
    OCRRequest, OCRBatchRequest, OCRResponse, OCRBatchResponse,
    OCRResultItem, HealthResponse, ModelInfo,
)

# 日志配置
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paddleocr-vl")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("=" * 60)
    logger.info("PaddleOCR-VL Enterprise API Service 启动中...")
    logger.info(f"端口: {settings.PORT}, 设备: {settings.DEVICE}")
    logger.info(f"最大并发: {settings.MAX_CONCURRENT}, 超时: {settings.REQUEST_TIMEOUT}s")
    logger.info(f"API Keys 数量: {len(settings.API_KEYS)}")
    logger.info("=" * 60)

    # 启动专用工作线程加载模型
    logger.info("正在启动 PaddleOCR-VL 工作线程...")
    engine.start()
    logger.info("工作线程已启动，模型后台加载中")

    yield

    logger.info("服务关闭中...")

    logger.info("服务关闭中...")


app = FastAPI(
    title="PaddleOCR-VL Enterprise API",
    description="企业级文档解析服务 - 基于 PaddleOCR-VL 1.6",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key 鉴权中间件
app.add_middleware(APIKeyMiddleware)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({elapsed:.3f}s)"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未捕获异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"内部错误: {str(exc)}"},
    )


# ========== API 端点 ==========

@app.get("/health", response_model=HealthResponse, tags=["系统"])
def health_check():
    """健康检查"""
    import paddle
    gpu_available = paddle.is_compiled_with_cuda()
    gpu_name = None
    if gpu_available:
        try:
            gpu_name = paddle.device.cuda.get_device_name(settings.DEVICE_ID)
        except Exception:
            pass

    uptime_seconds = int(time.time() - _start_time)
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60

    return HealthResponse(
        status="ok" if engine.is_ready else "loading",
        version=engine.version,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        model_loaded=engine.is_ready,
        uptime=f"{hours}小时{minutes}分钟",
    )


@app.get("/v1/models", response_model=ModelInfo, tags=["系统"])
def get_model_info():
    """获取模型信息"""
    return ModelInfo(
        version=engine.version,
        device=f"gpu:{settings.DEVICE_ID}" if settings.DEVICE.startswith("gpu") else "cpu",
    )


@app.post("/v1/ocr", response_model=OCRResponse, tags=["OCR"])
def ocr_single(request: OCRRequest, req: Request):
    """单文件OCR识别（自动识别图片/PDF，PDF支持分页处理与合并）"""
    request_id = str(uuid.uuid4())[:8]

    if not engine.is_ready:
        return OCRResponse(
            code=503,
            message="模型正在加载中，请稍后重试",
            request_id=request_id,
        )

    try:
        result = engine.predict(request.image, page_size=request.page_size)

        page_items = None
        if result.get("pages"):
            from .models import OCRResultPage
            page_items = [
                OCRResultPage(
                    page=p["page"],
                    markdown=p.get("markdown"),
                    text=p.get("text"),
                    route=p.get("route"),
                    timing_ms=p.get("timing_ms"),
                    classification=p.get("classification"),
                    hallucination_warnings=p.get("hallucination_warnings"),
                )
                for p in result["pages"]
            ]

        return OCRResponse(
            code=0,
            message="success",
            data=OCRResultItem(
                index=0,
                filename=request.filename,
                file_type=result.get("file_type", "image"),
                success=True,
                total_pages=result.get("total_pages", 1),
                markdown=result["markdown"],
                text=result["text"],
                pages=page_items,
                route_summary=result.get("route_summary"),
                total_timing_ms=result.get("total_timing_ms"),
            ),
            request_id=request_id,
        )
    except ValueError as e:
        return OCRResponse(
            code=400,
            message=str(e),
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"OCR处理失败 [{request_id}]: {e}", exc_info=True)
        return OCRResponse(
            code=500,
            message=f"OCR处理失败: {str(e)}",
            data=OCRResultItem(
                index=0,
                filename=request.filename,
                success=False,
                error=str(e),
            ),
            request_id=request_id,
        )


@app.post("/v1/ocr/batch", response_model=OCRBatchResponse, tags=["OCR"])
def ocr_batch(request: OCRBatchRequest, req: Request):
    """批量OCR识别（最多20张）"""
    request_id = str(uuid.uuid4())[:8]

    if not engine.is_ready:
        return OCRBatchResponse(
            code=503,
            message="模型正在加载中，请稍后重试",
            request_id=request_id,
        )

    results: List[OCRResultItem] = []

    for idx, item in enumerate(request.images):
        try:
            result = engine.predict(item.image, page_size=item.page_size)

            page_items = None
            if result.get("pages"):
                from .models import OCRResultPage
                page_items = [
                    OCRResultPage(
                        page=p["page"],
                        markdown=p.get("markdown"),
                        text=p.get("text"),
                        route=p.get("route"),
                        timing_ms=p.get("timing_ms"),
                    )
                    for p in result["pages"]
                ]

            results.append(OCRResultItem(
                index=idx,
                filename=item.filename,
                file_type=result.get("file_type", "image"),
                success=True,
                total_pages=result.get("total_pages", 1),
                markdown=result["markdown"],
                text=result["text"],
                pages=page_items,
                route_summary=result.get("route_summary"),
                total_timing_ms=result.get("total_timing_ms"),
            ))
        except Exception as e:
            logger.error(f"批量OCR第{idx}项失败: {e}")
            results.append(OCRResultItem(
                index=idx,
                filename=item.filename,
                success=False,
                error=str(e),
            ))

    return OCRBatchResponse(
        code=0,
        message="success" if all(r.success for r in results) else "部分处理失败",
        data=results,
        request_id=request_id,
    )


def main():
    """主入口"""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        log_level=settings.LOG_LEVEL.lower(),
        timeout_keep_alive=600,  # 10分钟保活，支持长耗时OCR
    )


if __name__ == "__main__":
    main()
