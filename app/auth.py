"""
API Key Authentication Middleware
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .config import settings


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.api_keys = set(settings.API_KEYS)
        extra_keys = os.getenv("EXTRA_API_KEYS", "")
        if extra_keys:
            self.api_keys.update(k.strip() for k in extra_keys.split(",") if k.strip())

    async def dispatch(self, request: Request, call_next):
        # 健康检查和文档接口不需要鉴权
        if request.url.path in ("/health", "/v1/models", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # OPTIONS 请求（CORS预检）不需要鉴权
        if request.method == "OPTIONS":
            return await call_next(request)

        # 检查 API Key
        api_key = request.headers.get("X-API-Key") or ""
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"code": 401, "message": "缺少API Key。请在请求头中提供 X-API-Key 或使用 Bearer token。"},
            )

        if api_key not in self.api_keys:
            return JSONResponse(
                status_code=403,
                content={"code": 403, "message": "无效的API Key。"},
            )

        return await call_next(request)
