"""
PaddleOCR-VL 模型服务封装
使用专用工作线程避免 PaddlePaddle 线程安全问题
支持图片/PDF输入、自动格式识别、PDF分页与结果合并
"""

import os
import io
import base64
import logging
import time
import re
import queue
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any

import numpy as np

from .config import settings
from .router import ModelRouter

logger = logging.getLogger("paddleocr-vl")


class OCREngine:
    """
    PaddleOCR-VL 引擎封装。
    在独立线程中运行，通过任务队列与主线程通信，
    避免 PaddlePaddle 与 asyncio 线程池的冲突。
    """

    def __init__(self):
        self._paddleocr_version = ""
        self._ready = False
        self._worker_thread: Optional[threading.Thread] = None
        self._task_queue: queue.Queue = queue.Queue(maxsize=10)
        self._result_queue: queue.Queue = queue.Queue()
        self._load_start_time: Optional[float] = None

    def start(self):
        """启动工作线程（在 lifespan 中调用）"""
        self._load_start_time = time.time()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="paddleocr-worker")
        self._worker_thread.start()

    def _worker_loop(self):
        """
        工作线程主循环。
        在专用线程中加载模型路由引擎并处理预测任务。
        """
        logger.info("工作线程已启动，正在加载模型路由引擎...")

        try:
            import paddle

            if settings.DEVICE.startswith("gpu"):
                if paddle.is_compiled_with_cuda():
                    gpu_name = paddle.device.cuda.get_device_name(settings.DEVICE_ID)
                    logger.info(f"使用 GPU: {gpu_name} (ID: {settings.DEVICE_ID})")
                else:
                    logger.warning("PaddlePaddle 未编译 CUDA 支持，将使用 CPU")
            else:
                logger.info("使用 CPU 模式")

            # 加载模型路由引擎
            device = settings.device_kwargs["device"]
            router = ModelRouter(
                device=device,
                vlm_backend=settings.VLM_BACKEND,
                fastdeploy_url=settings.FASTDEPLOY_URL,
                fastdeploy_model=settings.FASTDEPLOY_MODEL,
                fastdeploy_api_key=settings.FASTDEPLOY_API_KEY,
                llamacpp_url=settings.LLAMACPP_URL,
            )
            router.load_all()

            try:
                import paddleocr
                self._paddleocr_version = paddleocr.__version__
            except (ImportError, AttributeError):
                self._paddleocr_version = "unknown"

            self._ready = True
            elapsed = time.time() - self._load_start_time
            logger.info(f"模型路由引擎加载完成。PaddleOCR 版本: {self._paddleocr_version}, 总耗时: {elapsed:.1f}秒")

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            self._ready = False
            router = None

        # 任务处理循环（只要 service 在运行就不会退出）
        while True:
            try:
                task = self._task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if task is None:  # 退出信号
                break

            task_id, input_path, page_size, result_callback, error_callback, include, mode = task
            try:
                result = self._process_internal(router, input_path, page_size, include=include, mode=mode)
                result_callback(result)
            except Exception as e:
                error_callback(e)
            finally:
                self._task_queue.task_done()

    def _process_internal(self, router: ModelRouter, input_path: str, page_size: int,
                          include: list = None, mode: str = None) -> dict:
        """
        在工作者线程中执行实际的 OCR 处理。
        自动识别图片/PDF，并使用 ModelRouter 进行版面分类和路由。
        """
        file_path, file_type, _ = self._decode_input(input_path)
        logger.info(f"处理: 类型={file_type}, 路径={str(file_path)[:80]}...")

        if file_type == "pdf":
            return self._process_pdf(router, file_path, page_size, include=include, mode=mode)
        else:
            return self._process_image(router, file_path, include=include, mode=mode)

    def _process_image(self, router: ModelRouter, image_path: str, include: list = None,
                       mode: str = None) -> dict:
        """处理单张图片 — 通过路由引擎调度"""
        mode = mode or "vlm"
        result = router.process_with_route(image_path, mode=mode)

        route = result.get("route", "vlm")
        return {
            "markdown": result["markdown"],
            "text": result["text"],
            "file_type": "image",
            "total_pages": 1,
            "total_timing_ms": result.get("timing_ms", 0),
            "route_summary": {
                "light_ocr": 1 if route == "light_ocr" else 0,
                "table": 1 if route == "table" else 0,
                "vlm": 1 if route == "vlm" else 0,
                "error": 0,
            },
            "pages": [self._build_page_dict(1, result, include=include)],
            "raw": result.get("raw"),
        }

    def _process_pdf(self, router: ModelRouter, pdf_path: str, page_size: int = 20,
                     include: list = None, mode: str = None) -> dict:
        """
        处理 PDF 文件 — 两阶段批量优化版。

        阶段 1：渲染分块内所有页面 + 批量版面分类
        阶段 2：按路由分组 → 轻量OCR / VLM批量推理 → 排序合并
        """
        import pypdfium2 as pdfium

        logger.info(f"加载PDF: {pdf_path}")
        pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf)
        logger.info(f"PDF总页数: {total_pages}")

        if total_pages == 0:
            raise ValueError("PDF文件为空")

        pdf_mode = mode or "vlm"

        all_results = []
        total_light = 0
        total_table = 0
        total_vlm = 0
        total_errors = 0
        t_pdf_start = time.time()
        temp_dir = Path("/tmp/paddleocr_vl_pages")
        temp_dir.mkdir(parents=True, exist_ok=True)

        if page_size == -1:
            chunks = [(0, total_pages)]
        else:
            chunks = [(start, min(start + page_size, total_pages))
                      for start in range(0, total_pages, page_size)]

        logger.info(f"分页处理: {total_pages}页, {len(chunks)}批, 每批{page_size if page_size > 0 else total_pages}页")

        for chunk_idx, (page_start, page_end) in enumerate(chunks):
            logger.info(f"批次 {chunk_idx + 1}/{len(chunks)} (页 {page_start + 1}-{page_end})")

            # === 阶段 1：渲染本批所有页面到临时文件 ===
            chunk_entries = []  # [(页码, 图片路径), ...]
            t_chunk_start = time.time()
            rotation_angles = {}
            for page_num in range(page_start, page_end):
                page = pdf[page_num]
                bitmap = page.render(scale=settings.PDF_DPI_DEFAULT / 72)
                pil_image = bitmap.to_pil()
                img_path = temp_dir / f"chunk_{chunk_idx}_p{page_num}.png"
                pil_image.save(str(img_path))
                # orientation correction BEFORE layout classification
                processed_path, angle = router._preprocess_image(str(img_path))
                rotation_angles[page_num + 1] = angle
                chunk_entries.append((page_num + 1, processed_path))
            t_render = time.time()

            # 用于传递版面分类信息到逐页结果（路由关闭时留空）
            cls_by_page = {}

            if pdf_mode == "vlm":
                # mode="vlm"：全部走 VLM，跳过分类和路由
                light_entries = []
                table_entries = []
                vlm_entries = chunk_entries[:]
                for page_num, _ in chunk_entries:
                    cls_by_page[page_num] = {
                        "route": "vlm", "detected_complex": [], "layout_blocks": [],
                    }
            elif settings.ROUTING_ENABLED:
                # === 阶段 2a：批量版面分类 ===
                image_paths = [p for _, p in chunk_entries]
                try:
                    classifications = router.classifier.classify_batch(image_paths)
                except Exception as e:
                    logger.warning(f"批量分类失败，退化到逐页分类: {e}")
                    classifications = [router.classifier.classify(p) for p in image_paths]
                t_classify = time.time()

                # === 阶段 2b：按路由分组 ===
                light_entries = []
                table_entries = []
                vlm_entries = []
                cls_by_page = {}
                for (page_num, img_path), cls in zip(chunk_entries, classifications):
                    route, reason = router.decide_route(cls)
                    logger.info(f"  第 {page_num} 页路由: {route} — {reason}")
                    cls_by_page[page_num] = {
                        "route": route,
                        "detected_complex": cls.get("detected_complex", []),
                        "layout_blocks": [
                            {
                                "label": b.get("label", ""),
                                "score": round(b.get("score", 0), 3),
                                "bbox": b.get("coordinate"),
                            }
                            for b in cls.get("blocks", [])
                        ],
                    }
                    if route == "light_ocr":
                        light_entries.append((page_num, img_path))
                    elif route == "table":
                        table_entries.append((page_num, img_path))
                    else:
                        vlm_entries.append((page_num, img_path))

                # === 阶段 2c：轻量 OCR（逐页，本身很快） ===
                if light_entries:
                    logger.info(f"  轻量OCR处理 {len(light_entries)} 页...")
                    t0 = time.time()
                for page_num, img_path in light_entries:
                    t_page = time.time()
                    try:
                        result = router.predict_light_ocr(img_path)
                        page_timing = int((time.time() - t_page) * 1000)
                        all_results.append({
                            "page": page_num, "markdown": result["markdown"],
                            "text": result["text"], "route": "light_ocr", "raw": result.get("raw"),
                            "elements": result.get("elements"),
                            "timing_ms": page_timing,
                            "classification": cls_by_page.get(page_num),
                        })
                        total_light += 1
                    except Exception as e:
                        logger.error(f"  第 {page_num} 页轻量OCR失败: {e}")
                        all_results.append({
                            "page": page_num, "markdown": "", "text": "",
                            "route": "error", "error": str(e), "raw": None, "timing_ms": 0,
                            "classification": cls_by_page.get(page_num),
                        })
                        total_errors += 1
                t_light = time.time()
                if light_entries:
                    logger.info(f"  轻量OCR完成 ({len(light_entries)} 页, {t_light - t0:.1f}s)")

                # === 阶段 2d：VLM 批量推理 ===
                if vlm_entries:
                    logger.info(f"  VLM处理 {len(vlm_entries)} 页...")
                    t0 = time.time()
                    vlm_paths = [p for _, p in vlm_entries]
                    try:
                        vlm_results = router.predict_vlm_batch(vlm_paths)
                        t_vlm_done = time.time()
                        avg_ms = int((t_vlm_done - t0) / max(len(vlm_paths), 1) * 1000)
                        for (page_num, _), result in zip(vlm_entries, vlm_results):
                            all_results.append({
                                "page": page_num, "markdown": result["markdown"],
                                "text": result["text"], "route": "vlm", "raw": result.get("raw"),
                                "elements": result.get("elements"),
                                "timing_ms": avg_ms,
                            })
                            total_vlm += 1
                        logger.info(f"  VLM批量完成 ({len(vlm_entries)} 页, {t_vlm_done - t0:.1f}s, 均 {avg_ms}ms/页)")
                    except Exception as e:
                        logger.warning(f"  VLM批量失败，退化到逐页处理: {e}")
                        for page_num, img_path in vlm_entries:
                            t_page = time.time()
                            try:
                                result = router.predict_vlm(img_path)
                                page_timing = int((time.time() - t_page) * 1000)
                                all_results.append({
                                    "page": page_num, "markdown": result["markdown"],
                                    "text": result["text"], "route": "vlm", "raw": result.get("raw"),
                                    "elements": result.get("elements"),
                                    "timing_ms": page_timing,
                                })
                                total_vlm += 1
                            except Exception as e2:
                                logger.error(f"  第 {page_num} 页VLM失败: {e2}")
                                all_results.append({
                                    "page": page_num, "markdown": "", "text": "",
                                    "route": "error", "error": str(e2), "raw": None, "timing_ms": 0,
                                })
                                total_errors += 1

                # === 阶段 2e：表格批量推理 ===
                if table_entries:
                    logger.info(f"  表格识别处理 {len(table_entries)} 页...")
                    t0 = time.time()
                    # 表格页用 PDF_DPI_TABLE 从矢量源重新渲染（比插值放大更清晰）
                    scale_ratio = settings.PDF_DPI_TABLE / settings.PDF_DPI_DEFAULT
                    table_image_paths = []
                    table_bboxes = {}
                    for idx, (page_num, _) in enumerate(table_entries):
                        page = pdf[page_num - 1]
                        bitmap = page.render(scale=settings.PDF_DPI_TABLE / 72)
                        pil_image = bitmap.to_pil()
                        img_path = temp_dir / f"chunk_{chunk_idx}_p{page_num}_hdpi.png"
                        pil_image.save(str(img_path))
                        # reuse angle from low-DPI orientation — don't re-run model
                        angle = rotation_angles.get(page_num, 0)
                        if angle != 0:
                            angle_map = {90: -90, 180: -180, 270: -270}
                            rotated_hdpi = router._rotate_image(str(img_path), angle_map.get(angle, 0))
                            table_image_paths.append((page_num, rotated_hdpi))
                        else:
                            table_image_paths.append((page_num, str(img_path)))
                        # bbox 坐标从低 DPI 空间缩放到高 DPI 空间
                        cls = cls_by_page.get(page_num, {})
                        for block in cls.get("layout_blocks", []):
                            if block.get("label") == "table" and block.get("bbox"):
                                table_bboxes[idx] = [
                                    int(v * scale_ratio) for v in block["bbox"]
                                ]
                                break
                    try:
                        table_results = router.predict_table_batch(
                            [p for _, p in table_image_paths],
                            table_bboxes=table_bboxes,
                        )
                        t_table_done = time.time()
                        avg_ms = int((t_table_done - t0) / max(len(table_image_paths), 1) * 1000)
                        for (page_num, _), result in zip(table_image_paths, table_results):
                            all_results.append({
                                "page": page_num, "markdown": result["markdown"],
                                "text": result["text"], "route": "table", "raw": result.get("raw"),
                                "elements": result.get("elements"),
                                "timing_ms": avg_ms,
                                "classification": cls_by_page.get(page_num),
                            })
                            total_table += 1
                        logger.info(f"  表格批量完成 ({len(table_entries)} 页, {t_table_done - t0:.1f}s, 均 {avg_ms}ms/页)")
                    except Exception as e:
                        logger.warning(f"  表格批量失败，退化到逐页处理: {e}")
                        for idx, (page_num, img_path) in enumerate(table_image_paths):
                            t_page = time.time()
                            try:
                                pg_bbox = table_bboxes.get(idx)
                                result = router.predict_table(img_path, table_bbox=pg_bbox)
                                page_timing = int((time.time() - t_page) * 1000)
                                all_results.append({
                                    "page": page_num, "markdown": result["markdown"],
                                    "text": result["text"], "route": "table", "raw": result.get("raw"),
                                    "elements": result.get("elements"),
                                    "timing_ms": page_timing,
                                    "classification": cls_by_page.get(page_num),
                                })
                                total_table += 1
                            except Exception as e2:
                                logger.error(f"  第 {page_num} 页表格识别失败: {e2}")
                                all_results.append({
                                    "page": page_num, "markdown": "", "text": "",
                                    "route": "error", "error": str(e2), "raw": None, "timing_ms": 0,
                                    "classification": cls_by_page.get(page_num),
                                })
                                total_errors += 1
            else:
                # 路由关闭：所有页面走 VLM 批量
                image_paths = [p for _, p in chunk_entries]
                logger.info(f"  VLM批量处理 {len(chunk_entries)} 页（路由关闭）...")
                t0 = time.time()
                try:
                    vlm_results = router.predict_vlm_batch(image_paths)
                    t_vlm_done = time.time()
                    avg_ms = int((t_vlm_done - t0) / max(len(image_paths), 1) * 1000)
                    for (page_num, _), result in zip(chunk_entries, vlm_results):
                        all_results.append({
                            "page": page_num, "markdown": result["markdown"],
                            "text": result["text"], "route": "vlm", "raw": result.get("raw"),
                            "elements": result.get("elements"),
                            "timing_ms": avg_ms,
                            "classification": cls_by_page.get(page_num),
                        })
                        total_vlm += 1
                    logger.info(f"  VLM批量完成 ({len(chunk_entries)} 页, {t_vlm_done - t0:.1f}s, 均 {avg_ms}ms/页)")
                except Exception as e:
                    logger.warning(f"  VLM批量失败，退化到逐页处理: {e}")
                    for page_num, img_path in chunk_entries:
                        t_page = time.time()
                        try:
                            result = router.predict_vlm(img_path)
                            page_timing = int((time.time() - t_page) * 1000)
                            all_results.append({
                                "page": page_num, "markdown": result["markdown"],
                                "text": result["text"], "route": "vlm", "raw": result.get("raw"),
                                "elements": result.get("elements"),
                                "timing_ms": page_timing,
                                "classification": cls_by_page.get(page_num),
                            })
                            total_vlm += 1
                        except Exception as e2:
                            logger.error(f"  第 {page_num} 页VLM失败: {e2}")
                            all_results.append({
                                "page": page_num, "markdown": "", "text": "",
                                "route": "error", "error": str(e2), "raw": None, "timing_ms": 0,
                                "classification": cls_by_page.get(page_num),
                            })
                            total_errors += 1

            # 批次耗时明细
            now = time.time()
            if settings.ROUTING_ENABLED:
                light_time = now - (locals().get("t_light", t_chunk_start))
                table_time = now - (locals().get("t_table_done", t_chunk_start))
                vlm_time = now - (locals().get("t_vlm_done", t_chunk_start))
                logger.info(
                    f"  批次 {chunk_idx + 1} 耗时明细: "
                    f"渲染={t_render - t_chunk_start:.1f}s | "
                    f"分类={locals().get('t_classify', t_chunk_start) - t_render:.1f}s | "
                    f"轻量OCR={light_time:.1f}s({len(light_entries)}页) | "
                    f"表格={table_time:.1f}s({len(table_entries)}页) | "
                    f"VLM={vlm_time:.1f}s({len(vlm_entries)}页) | "
                    f"合计={now - t_chunk_start:.1f}s"
                )

            # 清理本批临时文件
            for _, img_path in chunk_entries:
                try:
                    os.remove(str(img_path))
                except OSError:
                    pass

        pdf.close()

        # === 后处理：规范化字段名 + include 过滤 ===
        _FILTERABLE_KEYS = ("markdown", "text", "elements", "layout_blocks", "hallucination_warnings")
        for p in all_results:
            # 旧格式 classification → 拆解为 layout_blocks + detected_complex
            if "classification" in p:
                meta = p.pop("classification")
                if meta:
                    p.setdefault("layout_blocks", meta.get("layout_blocks"))
                    p.setdefault("detected_complex", meta.get("detected_complex"))
            # 确保 elements 存在
            if "elements" not in p:
                p["elements"] = []
            # 补充 error_detail
            if p.get("route") == "error" and "error_detail" not in p:
                p["error_detail"] = p.get("error", "unknown")
            # 按 include 参数过滤
            if include is not None:
                for key in list(p.keys()):
                    if key in _FILTERABLE_KEYS and key not in include:
                        del p[key]

        all_results.sort(key=lambda r: r["page"])

        # 路由汇总
        t_total = time.time() - t_pdf_start
        logger.info(
            f"PDF处理完成: {total_pages}页, "
            f"轻量OCR={total_light}页, 表格={total_table}页, VLM={total_vlm}页, "
            f"错误={total_errors}页, 总耗时={t_total:.1f}s"
        )

        # 合并（含页码锚点）
        merged_md = []
        merged_txt = []
        for p in all_results:
            pmd = (p.get("markdown") or "").strip()
            ptxt = (p.get("text") or "").strip()
            route = p.get("route", "")
            if pmd or ptxt:
                header = f"---\n<!-- page {p['page']} | route: {route} -->"
                merged_md.append(f"{header}\n{pmd}")
                merged_txt.append(f"{header}\n{ptxt}")
            # 将 elements 中的 ID 补充为全局格式 p{page}_e{index}
            for elem in (p.get("elements") or []):
                if "id" in elem and not elem["id"].startswith("p"):
                    elem["id"] = f"p{p['page']}_{elem['id']}"

        total_ms = sum(p.get("timing_ms", 0) for p in all_results)
        return {
            "markdown": "\n\n".join(merged_md).strip(),
            "text": "\n\n".join(merged_txt).strip(),
            "file_type": "pdf",
            "total_pages": len(all_results),
            "total_timing_ms": total_ms,
            "route_summary": {"light_ocr": total_light, "table": total_table, "vlm": total_vlm, "error": total_errors},
            "pages": all_results,
            "raw": [p.get("raw") for p in all_results],
        }

    # ========== 页面构建工具 ==========

    @staticmethod
    def _build_page_dict(page_num: int, result: dict,
                         layout_blocks: list = None,
                         include: list = None) -> dict:
        """
        从路由结果构建 OCRResultPage 级字典，按 include 参数控制输出字段。

        Args:
            page_num: 页码
            result: 路由结果 dict，应有 markdown, text, elements, route, timing_ms 等
            layout_blocks: 单独传入的 layout_blocks（PDF批量路径使用 cls_by_page 的）
            include: 字段过滤列表

        Returns:
            dict: 符合 OCRResultPage 结构的字典
        """
        include = include or ["markdown", "text", "elements", "layout_blocks", "hallucination_warnings"]
        page = {
            "page": page_num,
            "route": result.get("route"),
            "error_detail": result.get("error_detail"),
            "timing_ms": result.get("timing_ms", 0),
        }
        if "markdown" in include:
            page["markdown"] = result.get("markdown")
        if "text" in include:
            page["text"] = result.get("text")
        if "elements" in include:
            elements = result.get("elements")
            if elements:
                # 补全 element ID 为全局格式
                for elem in elements:
                    if "id" in elem and not elem["id"].startswith("p"):
                        elem["id"] = f"p{page_num}_{elem['id']}"
            page["elements"] = elements
        if "layout_blocks" in include:
            page["layout_blocks"] = layout_blocks if layout_blocks is not None else result.get("layout_blocks")
        if "hallucination_warnings" in include:
            page["hallucination_warnings"] = result.get("hallucination_warnings")
        return page

    # ========== 公共接口 ==========

    def predict(self, image_input: str, page_size: Optional[int] = None,
                timeout: int = 600, include: Optional[List[str]] = None,
                mode: str = None) -> dict:
        """
        提交 OCR 预测任务并等待结果。

        Args:
            image_input: 图片/PDF的URL、本地路径或Base64编码
            page_size: PDF分页大小（None使用默认值20，-1不拆分）
            timeout: 等待超时秒数
            include: 按需返回字段列表，不传则全部返回
            mode: 路由模式（"routing"/"vlm"/"table_pp"），None 使用服务端默认

        Returns:
            dict: {markdown, text, file_type, total_pages, pages, raw}
        """
        if not self._ready:
            raise RuntimeError("模型正在加载中，请稍后重试")

        ps = page_size if page_size is not None else settings.PDF_PAGE_SIZE
        result_container = []
        error_container = []

        def on_result(r):
            result_container.append(r)

        def on_error(e):
            error_container.append(e)

        self._task_queue.put(("predict", image_input, ps, on_result, on_error, include, mode))

        # 等待结果
        deadline = time.time() + timeout
        while time.time() < deadline:
            if error_container:
                raise error_container[0]
            if result_container:
                return result_container[0]
            time.sleep(0.1)

        raise TimeoutError(f"OCR处理超时 ({timeout}秒)")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def version(self) -> str:
        return self._paddleocr_version

    # ========== 工具方法 ==========

    @staticmethod
    def _is_pdf_by_bytes(data: bytes) -> bool:
        return data[:5] == b"%PDF-"

    @staticmethod
    def _guess_image_ext(data: bytes) -> str:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:2] == b"\xff\xd8":
            return ".jpg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:4] == b"GIF8":
            return ".gif"
        if data[:4] == b"\x89TIF":
            return ".tiff"
        return ".png"

    def _decode_input(self, raw_input: str) -> Tuple[str, str, str]:
        """解码输入，返回 (本地路径, 文件类型, 扩展名)"""
        if raw_input.startswith(("http://", "https://")):
            is_pdf = raw_input.lower().split("?")[0].endswith(".pdf")
            return raw_input, "pdf" if is_pdf else "image", ".pdf" if is_pdf else ""

        # 太长或明显是 base64 的字符串，跳过 Path.exists() 检查
        if not (len(raw_input) > 2000 or
                (len(raw_input) > 200 and re.match(r'^[A-Za-z0-9+/=]+$', raw_input[:200]))):
            if Path(raw_input).exists():
                ext = Path(raw_input).suffix.lower()
                return raw_input, "pdf" if ext == ".pdf" else "image", ext

        raw_data = raw_input
        if raw_data.startswith("data:application/pdf;base64,"):
            raw_data = raw_data[len("data:application/pdf;base64,"):]
        elif raw_data.startswith("data:"):
            raw_data = re.sub(r'^data:\w+/\w+;base64,', '', raw_data)

        is_b64 = len(raw_data) > 100 and bool(re.match(r'^[A-Za-z0-9+/=]+$', raw_data[:200]))
        if not is_b64:
            raise ValueError("无法解析输入: 不是URL、文件路径或Base64编码")

        try:
            file_bytes = base64.b64decode(raw_data)
        except Exception as e:
            raise ValueError(f"Base64解码失败: {e}")

        if self._is_pdf_by_bytes(file_bytes):
            ext = ".pdf"
            file_type = "pdf"
        else:
            ext = self._guess_image_ext(file_bytes)
            file_type = "image"

        temp_dir = Path("/tmp/paddleocr_vl_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"upload_{int(time.time()*1000)}_{np.random.randint(10000)}{ext}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        return str(temp_path), file_type, ext



# 全局单例
engine = OCREngine()
