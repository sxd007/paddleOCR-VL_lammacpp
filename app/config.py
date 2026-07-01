"""
PaddleOCR-VL Enterprise API Service Configuration
"""

import os
from pathlib import Path
from typing import List


class Settings:
    def __init__(self):
        # 加载 .env 文件
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            self._load_dotenv(env_path)

        # 端口
        self.PORT: int = int(os.getenv("PORT", "8086"))

        # 主机
        self.HOST: str = os.getenv("HOST", "0.0.0.0")

        # API Keys
        api_keys_str: str = os.getenv("API_KEYS", "sk-paddleocr-vl-prod-2026")
        self.API_KEYS: List[str] = [k.strip() for k in api_keys_str.split(",") if k.strip()]

        # 设备配置
        self.DEVICE: str = os.getenv("DEVICE", "gpu").lower()
        self.DEVICE_ID: int = int(os.getenv("DEVICE_ID", "0"))

        # 模型参数
        self.USE_DOC_ORIENTATION_CLASSIFY: bool = os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "False").lower() == "true"
        self.USE_DOC_UNWARPING: bool = os.getenv("USE_DOC_UNWARPING", "False").lower() == "true"

        # 并发控制
        self.MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "4"))

        # 超时
        self.REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))

        # PDF 分页处理参数
        self.PDF_PAGE_SIZE: int = int(os.getenv("PDF_PAGE_SIZE", "20"))

        # 模型路由配置
        self.ROUTING_ENABLED: bool = os.getenv("ROUTING_ENABLED", "True").lower() == "true"
        self.ROUTING_COMPLEX_THRESHOLD: float = float(os.getenv("ROUTING_COMPLEX_THRESHOLD", "0.3"))
        self.LAYOUT_CONFIDENCE_THRESHOLD: float = float(os.getenv("LAYOUT_CONFIDENCE_THRESHOLD", "0.3"))

        # VLM 后端选择
        self.VLM_BACKEND: str = os.getenv("VLM_BACKEND", "native").lower()
        # FastDeploy 后端（原默认）
        self.FASTDEPLOY_URL: str = os.getenv("FASTDEPLOY_URL", "http://localhost:8185")
        self.FASTDEPLOY_MODEL: str = os.getenv("FASTDEPLOY_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6")
        self.FASTDEPLOY_API_KEY: str = os.getenv("FASTDEPLOY_API_KEY", "")
        # llama.cpp 后端（推荐，替换 FastDeploy）
        self.LLAMACPP_URL: str = os.getenv("LLAMACPP_URL", "http://localhost:8118")

        # 日志级别
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # PaddleX 模型缓存目录
        paddlex_home = os.getenv("PADDLEX_HOME", "")
        if paddlex_home:
            os.environ["PADDLEX_HOME"] = paddlex_home

    def _load_dotenv(self, path: Path):
        """简单 .env 文件加载（避免引入 python-dotenv 依赖）"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                # 环境变量优先，不覆盖已存在的环境变量
                if key not in os.environ:
                    os.environ[key] = value

    @property
    def device_kwargs(self) -> dict:
        """返回 PaddleOCRVL 初始化参数"""
        kwargs = {}
        if self.DEVICE == "cpu":
            kwargs["device"] = "cpu"
        else:
            kwargs["device"] = f"gpu:{self.DEVICE_ID}"
        if self.USE_DOC_ORIENTATION_CLASSIFY:
            kwargs["use_doc_orientation_classify"] = True
        if self.USE_DOC_UNWARPING:
            kwargs["use_doc_unwarping"] = True
        return kwargs


settings = Settings()
