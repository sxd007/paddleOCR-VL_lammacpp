"""
模型路由引擎 — Layout 先行的页面级路由。

根据 PP-DocLayoutV3 版面检测结果，动态选择：
  - 纯文字页 → PaddleOCR(det+rec) 轻量识别
  - 复杂页（含表格/图表/公式/印章等）→ PaddleOCR-VL 全面解析

所有模型在同一个线程中串行调用，规避 PaddlePaddle 线程安全问题。
"""

import base64
import logging
import os
import re
import time
from typing import Tuple

logger = logging.getLogger("paddleocr-vl")


# 导入 llama.cpp 客户端（延迟导入，避免启动依赖）
from .llama_client import LlamaCppClient
from .config import settings


# PP-DocLayoutV3 标签分类
SIMPLE_LABELS = {
    "text", "paragraph_title", "doc_title",
    "header", "footer", "number", "content", "aside_text",
    "abstract", "algorithm", "reference", "reference_content",
    "footnote", "vision_footnote",
}
COMPLEX_LABELS = {
    "table", "chart", "formula", "display_formula",
    "inline_formula", "seal", "image", "figure_title",
    "header_image", "footer_image", "spotting",
}


class PageClassifier:
    """PP-DocLayoutV3 版面检测 + 页面分类"""

    def __init__(self, device: str = "gpu:0",
                 threshold: float = 0.3,
                 nms: bool = True):
        self.device = device
        self.threshold = threshold
        self.nms = nms
        self._model = None

    def load(self):
        """加载 PP-DocLayoutV3 模型"""
        from paddlex import create_model
        logger.info("正在加载版面检测模型 PP-DocLayoutV3...")
        t0 = time.time()
        self._model = create_model("PP-DocLayoutV3")
        elapsed = time.time() - t0
        logger.info(f"PP-DocLayoutV3 加载完成，耗时: {elapsed:.1f}秒")

    def classify(self, image_path: str) -> dict:
        """
        运行版面检测并分类页面类型。

        Returns:
            dict: {
                "label": "simple" | "complex" | "empty",
                "blocks": [{label, score, coordinate}, ...],
                "detected_complex": [label, ...],
            }
        """
        if self._model is None:
            raise RuntimeError("PP-DocLayoutV3 未加载，请先调用 load()")

        results = list(self._model.predict(
            [image_path],
            threshold=self.threshold,
            layout_nms=self.nms,
        ))

        if not results:
            return {"label": "empty", "blocks": [], "detected_complex": []}

        boxes = results[0].get("boxes", [])
        detected_complex = []
        for box in boxes:
            label = box.get("label", "")
            if label in COMPLEX_LABELS:
                detected_complex.append(label)

        label = "complex" if detected_complex else "simple"

        return {
            "label": label,
            "blocks": boxes,
            "detected_complex": detected_complex,
        }

    def classify_batch(self, image_paths: list) -> list:
        """
        批量版面检测，返回每个图片的分类结果列表。

        Args:
            image_paths: 图片路径列表

        Returns:
            list[dict]: 每个元素与 classify() 返回格式一致
        """
        if self._model is None:
            raise RuntimeError("PP-DocLayoutV3 未加载，请先调用 load()")

        raw_results = list(self._model.predict(
            image_paths,
            threshold=self.threshold,
            layout_nms=self.nms,
        ))

        results = []
        for raw in raw_results:
            if not raw:
                results.append({"label": "empty", "blocks": [], "detected_complex": []})
                continue
            boxes = raw.get("boxes", []) if isinstance(raw, dict) else []
            detected_complex = [
                box.get("label", "") for box in boxes
                if box.get("label", "") in COMPLEX_LABELS
            ]
            label = "complex" if detected_complex else "simple"
            results.append({"label": label, "blocks": boxes, "detected_complex": detected_complex})

        # 确保结果数与输入数一致
        if len(results) != len(image_paths):
            while len(results) < len(image_paths):
                results.append({"label": "empty", "blocks": [], "detected_complex": []})

        return results

    def warmup(self):
        """预热推理"""
        try:
            import numpy as np
            from PIL import Image
            warmup_path = "/tmp/paddleocr_vl_layout_warmup.png"
            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(warmup_path)
            _ = self.classify(warmup_path)
            os.remove(warmup_path)
            logger.info("PP-DocLayoutV3 预热完成")
        except Exception as e:
            logger.warning(f"PP-DocLayoutV3 预热未完成: {e}")


class LightweightOCREngine:
    """PaddleOCR(det+rec) 轻量文字识别引擎"""

    def __init__(self, device: str = "gpu:0", lang: str = "ch"):
        self.device = device
        self.lang = lang
        self._ocr = None

    def load(self):
        """加载 PaddleOCR 轻量模型"""
        from paddleocr import PaddleOCR
        logger.info("正在加载轻量 OCR 模型 PaddleOCR(det+rec)...")
        t0 = time.time()
        self._ocr = PaddleOCR(
            lang=self.lang,
            ocr_version="PP-OCRv5",
            use_textline_orientation=False,
        )
        elapsed = time.time() - t0
        logger.info(f"PaddleOCR(det+rec) 加载完成，耗时: {elapsed:.1f}秒")

    def predict(self, image_path: str) -> dict:
        """
        对图片进行文字检测+识别，返回标准格式结果。

        Returns:
            dict: {markdown, text, raw}
        """
        if self._ocr is None:
            raise RuntimeError("轻量 OCR 未加载，请先调用 load()")

        t0 = time.time()
        raw_results = self._ocr.predict(image_path)
        elapsed = time.time() - t0
        logger.debug(f"轻量OCR推理耗时: {elapsed:.2f}秒")

        markdown = self._ocr_to_markdown(raw_results)
        text = self._strip_markdown(markdown)

        return {"markdown": markdown, "text": text, "raw": raw_results}

    @staticmethod
    def _ocr_to_markdown(ocr_results: list) -> str:
        """
        将 PaddleOCR(det+rec) 结果转换为 markdown。
        输入格式: [[(polygon, (text, confidence)), ...], ...]
        输出: 按行排序的段落文本
        """
        if not ocr_results or not ocr_results[0]:
            return ""

        lines = []
        for item in ocr_results[0]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, rec = item
            if isinstance(rec, (list, tuple)):
                text = rec[0] if rec else ""
                confidence = rec[1] if len(rec) > 1 else 1.0
            else:
                text, confidence = str(rec), 1.0

            if confidence < 0.3 or not text:
                continue

            if isinstance(box, (list, tuple)) and len(box) >= 4:
                ys = [p[1] if isinstance(p, (list, tuple)) else 0 for p in box[:4]]
                xs = [p[0] if isinstance(p, (list, tuple)) else 0 for p in box[:4]]
                center_y = sum(ys) / len(ys)
                center_x = sum(xs) / len(xs)
            else:
                center_y, center_x = 0, 0

            lines.append((center_y, center_x, text))

        if not lines:
            return ""

        lines.sort(key=lambda x: (x[0], x[1]))
        rows = []
        current_row_y = lines[0][0]
        current_row = []
        for center_y, center_x, text in lines:
            if abs(center_y - current_row_y) > 20:
                current_row.sort(key=lambda x: x[1])
                rows.append(" ".join(t for _, _, t in current_row))
                current_row = [(center_y, center_x, text)]
                current_row_y = center_y
            else:
                current_row.append((center_y, center_x, text))

        if current_row:
            current_row.sort(key=lambda x: x[1])
            rows.append(" ".join(t for _, _, t in current_row))

        result = "\n\n".join(rows)
        # 清理多余空格行
        return re.sub(r" +\n", "\n", result).strip()

    @staticmethod
    def _strip_markdown(markdown: str) -> str:
        text = re.sub(r"[#*_`\[\]()>|~-]", "", markdown)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def warmup(self):
        """预热推理"""
        try:
            import numpy as np
            from PIL import Image
            warmup_path = "/tmp/paddleocr_vl_ocr_warmup.png"
            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(warmup_path)
            _ = self.predict(warmup_path)
            os.remove(warmup_path)
            logger.info("PaddleOCR(det+rec) 预热完成")
        except Exception as e:
            logger.warning(f"PaddleOCR(det+rec) 预热未完成: {e}")


class TableRecognitionEngine:
    """PP-Structure 表格结构化识别引擎 — 输出 HTML 表格"""

    def __init__(self, device: str = "gpu:0", lang: str = "ch"):
        self.device = device
        self.lang = lang
        self._engine = None

    def load(self):
        """加载 PP-StructureV3 表格识别引擎"""
        from paddleocr import PPStructureV3
        logger.info("正在加载表格识别引擎 PP-StructureV3...")
        t0 = time.time()
        # PPStructureV3 用 lang 参数控制语言，默认启用表格识别
        self._engine = PPStructureV3(
            lang=self.lang,
            use_table_recognition=True,
            use_formula_recognition=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
        )
        elapsed = time.time() - t0
        logger.info(f"PP-StructureV3 加载完成，耗时: {elapsed:.1f}秒")

    def predict(self, image_path: str) -> dict:
        """
        对图片进行表格结构化识别。

        PP-StructureV3 返回 PaddleX pipeline 格式结果。
        递归搜索所有字段，提取 HTML 表格和文本内容。

        Returns:
            dict: {markdown, text, raw}
                markdown 中包含 Markdown/HTML 表格
        """
        if self._engine is None:
            raise RuntimeError("表格识别引擎未加载")

        t0 = time.time()
        raw_results = self._engine.predict(image_path)
        elapsed = time.time() - t0

        # 递归提取 HTML 表格和文本
        parts = []
        def _extract(obj, depth=0):
            if depth > 5 or obj is None:
                return
            if isinstance(obj, str):
                if obj.strip().startswith("<table") or "<tr>" in obj or "<td>" in obj:
                    parts.append(obj.strip())
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _extract(item, depth + 1)
                return
            if isinstance(obj, dict):
                # 直接命中的 key
                html = obj.get("html", "")
                if html and isinstance(html, str) and ("<table" in html or "<tr>" in html):
                    parts.append(html)
                    return
                txt = obj.get("text", "")
                if txt and isinstance(txt, str) and len(txt) > 5 and not parts:
                    parts.append(txt)
                md = obj.get("markdown", "")
                if md and isinstance(md, str) and len(md) > 5 and not parts:
                    parts.append(md)
                # 递归所有字段
                for v in obj.values():
                    _extract(v, depth + 1)

        _extract(raw_results)

        if parts:
            markdown = "\n\n".join(parts).strip()
        else:
            # 最后兜底：取字符串中所有 <table>...</table>
            text_repr = str(raw_results)
            tables = re.findall(r"<table[^>]*>.*?</table>", text_repr, re.DOTALL)
            if tables:
                markdown = "\n\n".join(tables)
            else:
                markdown = text_repr[:2000] if len(text_repr) > 2000 else text_repr

        text = re.sub(r"<[^>]+>", "", markdown).strip()
        return {"markdown": markdown, "text": text, "raw": raw_results}

    def predict_batch(self, image_paths: list, concurrency: int = 4) -> list:
        """批量表格识别 — 线程池并发"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(image_paths)

        def _predict_single(idx, path):
            try:
                return idx, self.predict(path), None
            except Exception as e:
                return idx, {"markdown": "", "text": ""}, e

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_predict_single, idx, path)
                       for idx, path in enumerate(image_paths)]
            for future in as_completed(futures):
                idx, result, _ = future.result()
                results[idx] = result
        return results

    def warmup(self):
        """预热推理"""
        try:
            import numpy as np
            from PIL import Image
            path = "/tmp/paddleocr_vl_table_warmup.png"
            Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(path)
            self.predict(path)
            os.remove(path)
            logger.info("PP-Structure 预热完成")
        except Exception as e:
            logger.warning(f"PP-Structure 预热未完成: {e}")


class FastDeployClient:
    """FastDeploy 2.3 HTTP 客户端 — 替代原生 PaddleOCR-VL 推理"""

    def __init__(self, server_url: str = "http://localhost:8185",
                 model_name: str = "PaddlePaddle/PaddleOCR-VL-1.6",
                 api_key: str = ""):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.chat_url = f"{self.server_url}/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._ready = False

    def check_health(self) -> bool:
        """检查 FastDeploy 服务是否就绪"""
        try:
            import requests
            resp = requests.get(f"{self.server_url}/health",
                                headers=self._headers, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def load(self):
        """验证连接（模型已在 FastDeploy 服务端加载）"""
        logger.info(f"正在连接 FastDeploy 服务: {self.server_url}")
        if self.check_health():
            self._ready = True
            logger.info(f"FastDeploy 服务连接成功 (模型: {self.model_name})")
        else:
            logger.warning(f"FastDeploy 服务未就绪: {self.server_url} (稍后将重试)")

    def warmup(self):
        """发送空白预热请求"""
        if not self._ready:
            return
        try:
            logger.info("FastDeploy 预热推理中...")
            import numpy as np
            from PIL import Image
            import io
            warmup_path = "/tmp/paddleocr_vl_fd_warmup.png"
            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(warmup_path)
            self.predict(warmup_path)
            os.remove(warmup_path)
            logger.info("FastDeploy 预热完成")
        except Exception as e:
            logger.warning(f"FastDeploy 预热未完成: {e}")

    def predict(self, image_path: str) -> dict:
        """单图 VLM 推理 — 通过 HTTP 调用 FastDeploy"""
        import requests

        with open(image_path, "rb") as f:
            img_data = f.read()
        img_b64 = base64.b64encode(img_data).decode()

        prompt = (
            "请完整提取图片中的所有文字内容，保持原始段落结构和排版层次。"
            "输出格式：纯markdown，标题用#，表格用html，段落间空行分隔。"
        )

        try:
            resp = requests.post(
                self.chat_url,
                headers=self._headers,
                json={
                    "model": self.model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # 解析返回结构兼容
            if isinstance(content, str):
                md = content.strip()
            elif isinstance(content, dict):
                md = content.get("text", content.get("content", str(content)))
            else:
                md = str(content)

            return {"markdown": md, "raw": data}

        except Exception as e:
            logger.error(f"FastDeploy API 调用失败: {e}")
            raise

    def predict_batch(self, image_paths: list) -> list:
        """批量推理 — 逐张调用（后续可优化为真正的 batch 请求）"""
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"FastDeploy 批量处理失败 ({path}): {e}")
                results.append({"markdown": "", "text": "", "raw": None})
        return results


class ModelRouter:
    """
    模型路由引擎 — 管理三个模型的生命周期和动态调度。

    处理流程:
        输入 → PP-DocLayoutV3 → 分类 → 轻量OCR / VLM → 统一输出

    VLM 后端可选:
      - native: 原生 PaddlePaddle 推理 (PaddleOCRVL)
      - fastdeploy: FastDeploy 2.3 HTTP API
    """

    def __init__(self, device: str = "gpu:0",
                 vlm_backend: str = "native",
                 fastdeploy_url: str = "http://localhost:8185",
                 fastdeploy_model: str = "PaddlePaddle/PaddleOCR-VL-1.6",
                 fastdeploy_api_key: str = "",
                 llamacpp_url: str = "http://localhost:8118"):
        self.device = device
        self.vlm_backend = vlm_backend
        self.classifier = PageClassifier(device=device)
        self.light_ocr = LightweightOCREngine(device=device)
        self._vl_pipeline = None
        self._orientation_model = None
        self._table_engine = None
        self._fastdeploy_client = None
        self._llama_client = None
        if vlm_backend == "fastdeploy":
            self._fastdeploy_client = FastDeployClient(
                server_url=fastdeploy_url,
                model_name=fastdeploy_model,
                api_key=fastdeploy_api_key,
            )
        elif vlm_backend == "llamacpp":
            self._llama_client = LlamaCppClient(
                server_url=llamacpp_url,
            )

    def load_all(self):
        """在工作线程中依次加载所有模型"""
        logger.info("=" * 50)
        logger.info("模型路由引擎加载中...")
        logger.info("=" * 50)

        # 1. 版面检测模型（最轻量，先加载）
        self.classifier.load()
        self.classifier.warmup()

        # 2. 轻量 OCR 模型
        self.light_ocr.load()
        self.light_ocr.warmup()

        # 3. VLM 模型（按后端选择）
        if self.vlm_backend == "fastdeploy":
            logger.info("VLM 后端: FastDeploy 2.3 (HTTP API)")
            self._fastdeploy_client.load()
        elif self.vlm_backend == "llamacpp":
            logger.info("VLM 后端: llama.cpp GGUF (HTTP API)")
            self._llama_client.load()
        else:
            logger.info("VLM 后端: 原生 PaddlePaddle (PaddleOCRVL)")
            self._load_vl_pipeline()

        # 4. 文档方向分类模型（轻量，仅在校正启用时加载）
        if settings.USE_DOC_ORIENTATION_CLASSIFY:
            self._load_orientation_model()

        # 5. 表格结构化识别引擎
        self._load_table_engine()

        logger.info("=" * 50)
        logger.info("模型路由引擎全部加载完成")
        logger.info("=" * 50)

    def _load_vl_pipeline(self):
        """加载 PaddleOCR-VL 管线"""
        from paddleocr import PaddleOCRVL
        from app.config import settings

        kwargs = settings.device_kwargs
        logger.info(f"正在加载 PaddleOCR-VL (参数: {kwargs})...")
        t0 = time.time()
        self._vl_pipeline = PaddleOCRVL(**kwargs)
        elapsed = time.time() - t0
        logger.info(f"PaddleOCR-VL 加载完成，耗时: {elapsed:.1f}秒")

    def _load_orientation_model(self):
        """加载 PP-DocOrientationClassifier 方向分类模型"""
        try:
            from paddlex import create_model
            logger.info("正在加载文档方向分类模型 PP-DocOrientationClassifier...")
            t0 = time.time()
            self._orientation_model = create_model("PP-LCNet_x1_0_doc_ori")
            elapsed = time.time() - t0
            logger.info(f"PP-DocOrientationClassifier 加载完成，耗时: {elapsed:.1f}秒")
        except Exception as e:
            logger.warning(f"PP-DocOrientationClassifier 加载失败（不影响主流程）: {e}")
            self._orientation_model = None

    def _preprocess_image(self, image_path: str) -> str:
        """
        通用图像预处理：方向校正。

        当 USE_DOC_ORIENTATION_CLASSIFY 启用时，检测图像旋转角度并回正。
        轻量方向分类模型 PP-DocOrientationClassifier 返回 class_id:
            0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°

        Args:
            image_path: 输入图片路径

        Returns:
            处理后的图片路径（可能和输入相同，表示无需校正）
        """
        if self._orientation_model is None:
            return image_path

        try:
            # 推理方向
            raw = list(self._orientation_model.predict(image_path))
            if not raw:
                return image_path

            result = raw[0] if isinstance(raw, list) else raw
            class_id = result.get("class_id", 0) if isinstance(result, dict) else 0

            # class_id: 0=0°, 1=90°CW, 2=180°, 3=270°CW
            if class_id == 0:
                return image_path

            angle_map = {1: -90, 2: -180, 3: -270}
            angle = angle_map.get(class_id, 0)
            logger.info(f"检测到图像旋转 {abs(angle)}°，自动校正")

            from PIL import Image
            img = Image.open(image_path)
            rotated = img.rotate(angle, expand=True)

            # 写回同目录，避免跨设备/容器路径问题
            base, ext = os.path.splitext(image_path)
            rotated_path = f"{base}_rot{abs(angle)}c{ext}"
            rotated.save(rotated_path)
            logger.info(f"方向校正后图片: {rotated_path}")
            return rotated_path

        except Exception as e:
            logger.warning(f"方向校正失败，使用原图: {e}")
            return image_path

    def warmup_vlm(self):
        """预热 VLM 推理"""
        if self.vlm_backend == "fastdeploy":
            self._fastdeploy_client.warmup()
        elif self.vlm_backend == "llamacpp":
            self._llama_client.warmup()
        else:
            logger.info("PaddleOCR-VL 预热推理中...")
            try:
                import numpy as np
                from PIL import Image
                warmup_path = "/tmp/paddleocr_vl_vlm_warmup.png"
                img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                img.save(warmup_path)
                t0 = time.time()
                _ = self._vl_pipeline.predict(warmup_path)
                elapsed = time.time() - t0
                os.remove(warmup_path)
                logger.info(f"PaddleOCR-VL 预热完成，耗时: {elapsed:.1f}秒")
            except Exception as e:
                logger.warning(f"PaddleOCR-VL 预热未完成: {e}")

    def _load_table_engine(self):
        """加载 PP-Structure 表格识别引擎"""
        try:
            self._table_engine = TableRecognitionEngine(device=self.device)
            self._table_engine.load()
            self._table_engine.warmup()
        except Exception as e:
            logger.warning(f"表格识别引擎加载失败，将回退到 VLM: {e}")
            self._table_engine = None

    def decide_route(self, classification: dict) -> Tuple[str, str]:
        """
        根据版面分类结果做路由决策。

        Returns:
            (route, reason): ("light_ocr"|"vlm", 原因说明)
        """
        label = classification.get("label", "empty")
        detected = classification.get("detected_complex", [])

        if "table" in detected:
            return "table", f"检测到表格"
        elif label == "complex":
            return "vlm", f"检测到复杂区域: {detected}"
        elif label == "empty":
            return "vlm", "未检测到版面元素，回退到 VLM"
        else:
            return "light_ocr", "纯文字页面"

    def predict_vlm(self, image_path: str) -> dict:
        """使用配置的 VLM 后端处理页面"""
        # 图像预处理：方向校正（如果启用）
        processed_path = self._preprocess_image(image_path)

        if self.vlm_backend == "fastdeploy":
            result = self._fastdeploy_client.predict(processed_path)
            md = result["markdown"]
            return {"markdown": md, "text": self._extract_text_from_vlm(md), "raw": result.get("raw")}
        elif self.vlm_backend == "llamacpp":
            result = self._llama_client.predict(processed_path)
            md = result["markdown"]
            return {"markdown": md, "text": self._extract_text_from_vlm(md), "raw": result.get("raw")}
        else:
            if self._vl_pipeline is None:
                raise RuntimeError("PaddleOCR-VL 未加载")
            result = self._vl_pipeline.predict(processed_path)
            md = self._extract_markdown(result)
            return {"markdown": md, "text": self._extract_text_from_vlm(md), "raw": result}

    def predict_vlm_batch(self, image_paths: list) -> list:
        """
        批量 VLM 预测。

        Args:
            image_paths: 图片路径列表

        Returns:
            list[dict]: 每个元素格式与 predict_vlm() 一致
        """
        if self.vlm_backend == "fastdeploy":
            return self._fastdeploy_client.predict_batch(image_paths)
        elif self.vlm_backend == "llamacpp":
            return self._llama_client.predict_batch(image_paths)
        else:
            if self._vl_pipeline is None:
                raise RuntimeError("PaddleOCR-VL 未加载")
            raw_results = self._vl_pipeline.predict(image_paths)
            if not isinstance(raw_results, list):
                raw_results = list(raw_results)
            if raw_results and isinstance(raw_results[0], list):
                raw_results = [r[0] if r else {} for r in raw_results]

            processed = []
            for result in raw_results:
                if not result:
                    processed.append({"markdown": "", "text": "", "raw": None})
                    continue
                md = self._extract_markdown(result)
                processed.append({
                    "markdown": md,
                    "text": self._extract_text_from_vlm(md),
                    "raw": result,
                })
            return processed

    def predict_light_ocr(self, image_path: str) -> dict:
        """使用轻量 OCR 处理页面"""
        return self.light_ocr.predict(image_path)

    def _upscale_image(self, image_path: str, scale: int = 4) -> str:
        """
        整图放大，让 VLM 看到更多文字细节。
        不依赖 bbox，避免方向校正后坐标失效的问题。
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            zoomed = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
            zpath = image_path.replace(".png", f"_up{scale}x.png").replace(".jpg", f"_up{scale}x.png")
            zoomed.save(zpath)
            logger.info(f"整图放大 {scale}x: {zpath} ({zoomed.width}x{zoomed.height})")
            return zpath
        except Exception as e:
            logger.warning(f"整图放大失败: {e}")
            return image_path

    def _zoom_table_region(self, image_path: str, bbox: list, scale: int = 3) -> str:
        """
        从图片中裁剪表格区域并放大，提升小字识别率。

        Args:
            image_path: 原图路径
            bbox: [x1, y1, x2, y2] 表格边界框（绝对值坐标）
            scale: 放大倍数，默认 3x

        Returns:
            裁剪放大后的图片路径
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            # 加 10% 边距
            pad_x = max(20, int((x2 - x1) * 0.1))
            pad_y = max(20, int((y2 - y1) * 0.1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img.width, x2 + pad_x)
            y2 = min(img.height, y2 + pad_y)
            cropped = img.crop((x1, y1, x2, y2))
            zoomed = cropped.resize((cropped.width * scale, cropped.height * scale), Image.LANCZOS)
            zoomed_path = image_path.replace(".png", "_tbl_zoom.png").replace(".jpg", "_tbl_zoom.png")
            zoomed.save(zoomed_path)
            logger.info(f"表格区域裁剪放大 {scale}x: {zoomed_path} ({zoomed.width}x{zoomed.height})")
            return zoomed_path
        except Exception as e:
            logger.warning(f"表格放大失败，使用原图: {e}")
            return image_path

    def predict_table(self, image_path: str, table_bbox: list = None) -> dict:
        """
        使用表格专用引擎识别页面。

        回退策略:
          1. 方向校正 → 表格引擎
          2. 表格引擎空结果 → 整图 4x 放大 → VLM
        """
        if self._table_engine is None:
            logger.warning("表格引擎不可用，回退到 VLM")
            return self.predict_vlm(image_path)
        # 先校正方向，再送表格引擎
        processed_path = self._preprocess_image(image_path)
        result = self._table_engine.predict(processed_path)
        md = (result.get("markdown") or "").strip()
        txt = (result.get("text") or "").strip()
        if not md and not txt:
            logger.info("表格引擎输出为空，4x 放大校正后图片再走 VLM")
            # 在已校正的图片上整图放大（不依赖 bbox，避免旋转后坐标失效）
            upscaled = self._upscale_image(processed_path, scale=4)
            return self.predict_vlm(upscaled)
        return result

    def predict_table_batch(self, image_paths: list, table_bboxes: dict = None) -> list:
        """批量表格识别 — 空结果 4x 放大回退到 VLM"""
        if self._table_engine is None:
            logger.warning("表格引擎不可用，逐页回退到 VLM")
            return [self.predict_vlm(p) for p in image_paths]
        processed = [self._preprocess_image(p) for p in image_paths]
        results = self._table_engine.predict_batch(processed)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {"markdown": "", "text": ""}
            md = (results[i].get("markdown") or "").strip()
            txt = (results[i].get("text") or "").strip()
            if not md and not txt:
                logger.info(f"表格批量第{i}页输出为空，4x 放大后走 VLM")
                upscaled = self._upscale_image(processed[i], scale=4)
                results[i] = self.predict_vlm(upscaled)
        return results

    def process_with_route(self, image_path: str, routing_enabled: bool = True) -> dict:
        """
        完整处理流程：分类 → 路由 → 识别。

        Returns:
            dict: 统一格式 {markdown, text, raw, route, route_reason, timing_ms}
        """
        t_start = time.time()

        if not routing_enabled:
            result = self.predict_vlm(image_path)
            result["route"] = "vlm"
            result["route_reason"] = "路由关闭"
            result["timing_ms"] = int((time.time() - t_start) * 1000)
            return result

        # 1. 版面分类
        classification = self.classifier.classify(image_path)
        t_classified = time.time()

        # 2. 路由决策
        route, reason = self.decide_route(classification)
        logger.info(f"路由决策: {route} — {reason}")

        # 提取表格框坐标（用于裁剪放大）
        table_bbox = None
        for block in classification.get("blocks", []):
            if block.get("label") == "table" and block.get("coordinate"):
                table_bbox = block["coordinate"]
                break

        # 3. 按路由处理
        if route == "table":
            result = self.predict_table(image_path, table_bbox=table_bbox)
        elif route == "vlm":
            result = self.predict_vlm(image_path)
        else:
            result = self.predict_light_ocr(image_path)
        t_done = time.time()

        result["route"] = route
        result["route_reason"] = reason
        result["classification"] = {
            "label": classification.get("label", "unknown"),
            "detected_complex": classification.get("detected_complex", []),
            "blocks": [
                {"label": b.get("label", ""), "score": round(b.get("score", 0), 3)}
                for b in classification.get("blocks", [])
            ],
        }
        result["timing_ms"] = int((t_done - t_start) * 1000)
        result["timing_breakdown"] = {
            "classification_ms": int((t_classified - t_start) * 1000),
            "inference_ms": int((t_done - t_classified) * 1000),
        }
        # 幻觉检测
        warnings = self._detect_hallucinations(result.get("markdown", ""))
        if warnings:
            result["hallucination_warnings"] = warnings
        return result

    # ========== VLM 结果解析 ==========

    @staticmethod
    def _extract_markdown(result) -> str:
        """
        从 VLM 结果中提取 markdown。
        PaddleOCR-VL 返回格式可能为 list[dict] 或 dict（带 parsing_res_list）。
        """
        try:
            # 归一化：如果是 list，取第一个元素
            if isinstance(result, list):
                result = result[0] if result else {}
            if not isinstance(result, dict):
                return str(result)

            # 从 parsing_res_list 解析结构化内容
            pl = result.get("parsing_res_list", [])
            if pl:
                parts = []
                for b in pl:
                    label = b.get("block_label", b.get("label", ""))
                    content = b.get("block_content", b.get("content", ""))
                    if not content:
                        continue
                    if label == "doc_title":
                        parts.append(f"# {content}\n")
                    elif label == "paragraph_title":
                        parts.append(f"\n## {content}\n")
                    elif label == "image":
                        parts.append(f"\n![{content}]()\n")
                    elif label == "table":
                        # 表格内容已经是 HTML，原样输出
                        parts.append(f"\n{content}\n")
                    else:
                        parts.append(content + "\n")
                return "\n".join(parts).strip()

            # 没有 parsing_res_list 时的兜底
            return result.get("res", result.get("text", str(result)))
        except Exception:
            return str(result)

    @staticmethod
    def _extract_text_from_vlm(markdown: str) -> str:
        text = re.sub(r"[#*_`\[\]()>|~-]", "", markdown)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _detect_hallucinations(markdown: str) -> list:
        """
        检测 VLM 常见幻觉模式。
        返回警告信息列表，为空表示无异常。

        检测维度:
          1. 同一行重复 3+ 次 → 典型 VLM 循环输出
          2. 大量电话号码（OCR 场景不合理）
          3. 同一 N-gram 重复 5+ 次
        """
        if not markdown:
            return []

        warnings = []

        # 1. 行级重复
        lines = [l.strip() for l in markdown.split("\n") if l.strip() and len(l.strip()) > 3]
        from collections import Counter
        line_counts = Counter(lines)
        for line, cnt in line_counts.most_common(5):
            if cnt >= 3:
                warnings.append(f"行重复: 「{line[:40]}」出现 {cnt} 次")

        # 2. 电话号码/微信号批量出现（正常文档不会超过 2 个）
        phones = re.findall(r"1[3-9]\d{9}", markdown)
        if len(phones) >= 3:
            warnings.append(f"疑似电话号码幻觉: 检测到 {len(phones)} 个手机号")
        wechats = re.findall(r"微信号[：:]?\d{5,}", markdown)
        if wechats:
            warnings.append(f"检测到微信号模式 ({len(wechats)} 处)，可能是水印幻觉")

        # 3. 字符级 N-gram 重复（同一段文字反复出现）
        for n in [8, 10, 15]:
            seen = set()
            dupes = 0
            for i in range(len(markdown) - n):
                chunk = markdown[i:i + n]
                if chunk in seen:
                    dupes += 1
                else:
                    seen.add(chunk)
                if dupes > 3:
                    warnings.append(f"内容片段重复: 检测到 {dupes} 处重复 (n-gram={n})")
                    break
            if dupes > 3:
                break

        return warnings
