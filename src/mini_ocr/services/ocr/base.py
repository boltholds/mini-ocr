from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRBlockResult:
    text: str
    confidence: float | None
    bbox: list[list[float]]


@dataclass
class OCRPageResult:
    text: str
    blocks: list[OCRBlockResult]
    orientation: int = 0
    ocr_score: float = 0.0
    avg_confidence: float | None = None
    layout_type: str = "unknown"
    image_path: str | None = None


class OCRService:
    def recognize_page(self, image_path: Path) -> OCRPageResult:
        raise NotImplementedError
