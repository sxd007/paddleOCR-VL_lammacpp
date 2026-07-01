"""
llama.cpp HTTP 客户端 — PaddleOCR-VL-1.6 GGUF 推理

基于 llama.cpp 的 OpenAI 兼容 API 进行多模态推理。
替换 FastDeploy 作为 VLM 推理后端。

API 格式: POST /v1/chat/completions (OpenAI 兼容)
"""

import base64
import logging
import time
from typing import Optional

logger = logging.getLogger("paddleocr-vl")


class LlamaCppClient:
    """
    llama.cpp 推理客户端 — OpenAI 兼容 API。

    与 FastDeployClient 使用相同的 API 协议（/v1/chat/completions），
    因此接口设计一致，可平滑替换。
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8118",
        model_name: str = "paddleocr-vl",
    ):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.chat_url = f"{self.server_url}/v1/chat/completions"
        self._ready = False

    # ==================================================================
    # 生命周期
    # ==================================================================

    def check_health(self) -> bool:
        """检查 llama.cpp 服务是否就绪"""
        try:
            import requests
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def load(self):
        """验证服务连接"""
        logger.info(f"正在连接 llama.cpp 服务: {self.server_url}")
        if self.check_health():
            self._ready = True
            logger.info(f"llama.cpp 服务连接成功 (模型: {self.model_name})")
        else:
            logger.warning(f"llama.cpp 服务未就绪: {self.server_url} (稍后将重试)")

    def warmup(self):
        """发送空白预热请求，确保 GPU 显存分配就绪"""
        if not self._ready:
            return
        try:
            logger.info("llama.cpp 预热推理中...")
            import numpy as np
            from PIL import Image
            warmup_path = "/tmp/paddleocr_vl_llama_warmup.png"
            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(warmup_path)
            self.predict(warmup_path)
            import os
            os.remove(warmup_path)
            logger.info("llama.cpp 预热完成")
        except Exception as e:
            logger.warning(f"llama.cpp 预热未完成: {e}")

    # ==================================================================
    # 核心推理
    # ==================================================================

    # 官方任务前缀（PaddleOCR-VL-1.6-GGUF 专用）
    # 模型使用极短前缀触发特定元素识别能力
    PROMPTS = {
        "text": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "seal": "Seal Recognition:",
    }

    def predict(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        task_type: str = "text",
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> dict:
        """
        单图 VLM 推理 — 通过 HTTP 调用 llama.cpp。

        Args:
            image_path: 图片本地路径
            prompt: 自定义提示词（覆盖 task_type）
            task_type: 任务类型 -> text/table/formula/chart/seal
            max_tokens: 最大生成长度
            timeout: 请求超时秒数

        Returns:
            dict: {"markdown": str, "raw": dict}
        """
        import requests

        if prompt is None:
            prompt = self.PROMPTS.get(task_type, "OCR:")

        # 读取并编码图片
        with open(image_path, "rb") as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode()

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                resp = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                elapsed = time.time() - t0
                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # 兼容 str / dict 两种返回格式
                if isinstance(content, str):
                    md = content.strip()
                elif isinstance(content, dict):
                    md = content.get("text", content.get("content", str(content)))
                else:
                    md = str(content)

                logger.debug(f"llama.cpp 推理完成: {len(md)}字符, 耗时{elapsed:.1f}s")
                return {"markdown": md, "raw": data}

            except requests.exceptions.Timeout:
                last_error = f"llama.cpp 请求超时 ({timeout}s)"
                logger.warning(f"{last_error} (第{attempt + 1}次)")
            except requests.exceptions.ConnectionError as e:
                last_error = f"llama.cpp 连接失败: {e}"
                logger.warning(f"{last_error} (第{attempt + 1}次)")
            except Exception as e:
                last_error = f"llama.cpp API 调用失败: {e}"
                logger.warning(f"{last_error} (第{attempt + 1}次)")

            if attempt < max_retries:
                sleep_time = 2 ** attempt  # 指数退避：1s, 2s
                logger.info(f"等待 {sleep_time}s 后重试...")
                time.sleep(sleep_time)

        logger.error(f"llama.cpp 全部重试失败: {last_error}")
        raise RuntimeError(last_error)

    def predict_batch(
        self,
        image_paths: list,
        prompt: Optional[str] = None,
        task_type: str = "text",
        max_tokens: int = 4096,
        concurrency: int = 4,
    ) -> list:
        """
        批量推理 — 利用线程池并发调用，充分利用 llama.cpp 的 parallel 能力。

        Args:
            image_paths: 图片路径列表
            prompt: OCR 提示词
            task_type: 任务类型（按路由决定）
            max_tokens: 最大生成长度
            concurrency: 并发数

        Returns:
            list[dict]: 每个元素格式与 predict() 一致
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(image_paths)
        logger.info(
            f"llama.cpp 批量推理: {len(image_paths)}张, 并发{concurrency}"
        )

        def _predict_single(idx: int, path: str) -> tuple:
            try:
                result = self.predict(path, prompt=prompt, task_type=task_type, max_tokens=max_tokens)
                return idx, result, None
            except Exception as e:
                logger.error(f"llama.cpp 批量第{idx}项失败 ({path}): {e}")
                return idx, None, e

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_predict_single, idx, path)
                for idx, path in enumerate(image_paths)
            ]
            for future in as_completed(futures):
                idx, result, error = future.result()
                if error:
                    results[idx] = {"markdown": "", "text": "", "raw": None}
                else:
                    results[idx] = result

        return results

    @property
    def is_ready(self) -> bool:
        return self._ready
