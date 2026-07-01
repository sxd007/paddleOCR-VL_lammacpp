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


class OCRBatchRequest(BaseModel):
    """批量OCR请求"""
    images: List[OCRRequest] = Field(
        ...,
        description="图片列表，最多20张",
        min_length=1,
        max_length=20,
    )


class OCRResultPage(BaseModel):
    """单页OCR结果"""
    page: int = Field(..., description="页码（从1开始）")
    markdown: Optional[str] = Field(None, description="该页的Markdown识别结果")
    text: Optional[str] = Field(None, description="该页的纯文本识别结果")
    route: Optional[str] = Field(None, description="该页路由: light_ocr / vlm / error")
    timing_ms: Optional[int] = Field(None, description="该页处理耗时(毫秒)")
    classification: Optional[dict] = Field(None, description="PP-DocLayoutV3 版面分类详情: {label, detected_complex, blocks}")
    hallucination_warnings: Optional[list] = Field(None, description="VLM 幻觉检测警告列表")


class OCRResultItem(BaseModel):
    """单个文件OCR结果"""
    index: int = Field(..., description="序号")
    filename: Optional[str] = Field(None, description="文件名")
    file_type: str = Field("image", description="文件类型: image/pdf")
    success: bool = Field(..., description="是否成功")
    total_pages: int = Field(1, description="总页数（PDF多页时>1）")
    markdown: Optional[str] = Field(None, description="Markdown格式的完整识别结果（多页已合并）")
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
