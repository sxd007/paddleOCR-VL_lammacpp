"""
Pydantic models for request/response schemas
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    """单张图片/PDF OCR请求"""
    image: str = Field(
        ...,
        description="Base64编码的文件数据、文件URL或本地文件路径。自动识别图片(png/jpg)和PDF格式",
        example="data:image/png;base64,iVBORw0KGgo...",
    )
    filename: Optional[str] = Field(
        None,
        description="文件名（用于日志和调试）",
        example="document.pdf",
    )
    page_size: Optional[int] = Field(
        None,
        description="PDF分页处理时的每批页数。不传则使用服务端默认值(20)。-1表示不拆分整份处理",
        example=20,
    )
    use_doc_orientation_classify: Optional[bool] = Field(
        None,
        description="是否启用文档方向分类（覆盖全局配置）",
    )
    use_doc_unwarping: Optional[bool] = Field(
        None,
        description="是否启用文档矫正（覆盖全局配置）",
    )
    mode: Optional[str] = Field(
        None,
        description=(
            "路由模式。可选值:\n"
            '  "routing" (默认) — 版面分类后路由到专业引擎（表格→PPStructure，文字→PP-OCR，复杂→VLM）\n'
            '  "vlm" — 全部使用 PaddleOCR-VL（跳过分类和路由，适合复杂混合内容）\n'
            '  "table_pp" — 路由模式下强制走 PPStructure 表格管线（即使版面分类未检出表格）'
        ),
        example="routing",
    )
    include: Optional[List[str]] = Field(
        None,
        description=(
            "按需返回字段列表，不传则默认全部返回。"
            "可选值: markdown, text, elements, layout_blocks, hallucination_warnings。"
            "route/timing_ms/page/error_detail 等核心元数据始终返回，不受此参数影响。"
        ),
        example=["markdown", "elements"],
    )


class OCRBatchRequest(BaseModel):
    """批量OCR请求"""
    images: List[OCRRequest] = Field(
        ...,
        description="图片列表，最多20张",
        min_length=1,
        max_length=20,
    )


class Element(BaseModel):
    """单个文档结构元素（段落/表格/公式/图表/印章）"""
    id: str = Field(..., description="元素ID，格式 p{page}_e{index}，如 p3_e1")
    type: str = Field(
        ...,
        description=(
            "元素类型: "
            "paragraph(正文段落/行) / table / formula / chart / seal / page_text(vlm路由兜底整页文本)"
        ),
    )
    reading_order: int = Field(..., description="页内阅读顺序，从0开始")
    bbox: Optional[List[float]] = Field(
        None,
        description="[x1,y1,x2,y2]绝对像素坐标（相对于该页渲染图）。"
                    "vlm路由的page级别元素为null，表示该内容来自整页推理、无法归属到具体子区域",
    )
    confidence: Optional[float] = Field(
        None,
        description="置信度。table来自表格引擎(如有)；light_ocr来自PP-OCR逐行识别置信度；"
                    "vlm路由的元素为null(VLM当前不返回逐token置信度)",
    )
    content: dict = Field(
        ...,
        description=(
            "按type区分结构:\n"
            "  paragraph / page_text: {\"text\": str}\n"
            "  table: {\"html\": str}\n"
            "  formula: {\"text\": str}\n"
            "  chart / seal: {\"text\": str}"
        ),
    )


class OCRResultPage(BaseModel):
    """单页OCR结果"""
    page: int = Field(..., description="页码（从1开始）")
    markdown: Optional[str] = Field(None, description="该页的Markdown识别结果（由elements按reading_order派生拼接）")
    text: Optional[str] = Field(None, description="该页的纯文本识别结果（由elements派生）")
    elements: Optional[List[Element]] = Field(
        None, description="结构化元素列表。include不含'elements'时省略"
    )
    layout_blocks: Optional[List[dict]] = Field(
        None,
        description=(
            "版面检测(PP-DocLayoutV3)标注的区域位置: "
            "[{label, score, bbox}, ...]。注意：这是纯位置标注，不代表对应内容已被单独裁剪识别"
            "（vlm路由的公式/图表/印章区域即属此情况，具体内容请看该页elements里的page_text）。"
            "include不含'layout_blocks'时省略"
        ),
    )
    route: Optional[str] = Field(None, description="该页路由: light_ocr / table / vlm / error")
    error_detail: Optional[str] = Field(
        None,
        description="route=error时的具体原因: timeout / connection_error / hallucination_retry_failed / unknown",
    )
    timing_ms: Optional[int] = Field(None, description="该页处理耗时(毫秒)")
    hallucination_warnings: Optional[list] = Field(None, description="VLM幻觉检测警告列表")


class OCRResultItem(BaseModel):
    """单个文件OCR结果"""
    index: int = Field(..., description="序号")
    filename: Optional[str] = Field(None, description="文件名")
    file_type: str = Field("image", description="文件类型: image/pdf")
    success: bool = Field(..., description="是否成功")
    total_pages: int = Field(1, description="总页数（PDF多页时>1）")
    markdown: Optional[str] = Field(None, description="Markdown格式的完整识别结果（多页已合并，含页码锚点）")
    text: Optional[str] = Field(None, description="纯文本格式的完整识别结果")
    pages: Optional[List[OCRResultPage]] = Field(None, description="逐页识别结果")
    route_summary: Optional[dict] = Field(None, description="路由统计: {light_ocr: N, vlm: N, error: N}")
    total_timing_ms: Optional[int] = Field(None, description="总处理耗时(毫秒)")
    error: Optional[str] = Field(None, description="错误信息（如果失败）")


class OCRResponse(BaseModel):
    """单张OCR响应"""
    code: int = Field(0, description="状态码: 0=成功")
    message: str = Field("success", description="状态消息")
    data: Optional[OCRResultItem] = Field(None, description="识别结果")
    request_id: Optional[str] = Field(None, description="请求ID")


class OCRBatchResponse(BaseModel):
    """批量OCR响应"""
    code: int = Field(0, description="状态码: 0=成功")
    message: str = Field("success", description="状态消息")
    data: Optional[List[OCRResultItem]] = Field(None, description="识别结果列表")
    request_id: Optional[str] = Field(None, description="请求ID")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("ok", description="服务状态")
    version: str = Field("", description="PaddleOCR版本")
    gpu_available: bool = Field(False, description="GPU是否可用")
    gpu_name: Optional[str] = Field(None, description="GPU型号")
    model_loaded: bool = Field(False, description="模型是否已加载")
    uptime: Optional[str] = Field(None, description="运行时间")


class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field("PaddleOCR-VL-1.6", description="模型名称")
    version: str = Field("", description="PaddleOCR版本")
    description: str = Field("文档解析视觉语言模型", description="模型描述")
    capabilities: List[str] = Field(
        default=["文档解析", "版面分析", "表格识别", "公式识别", "文本识别"],
        description="支持的能力",
    )
    languages: int = Field(109, description="支持的语言数量")
    device: str = Field("gpu", description="运行设备")
    max_batch: int = Field(20, description="最大批处理数量")
