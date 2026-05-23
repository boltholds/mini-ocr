from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import Document
from mini_ocr.services.image_preprocessor import ImagePreprocessor
from mini_ocr.services.observability import AgentTimer
from mini_ocr.services.ocr.paddleocr_service import PaddleOCRService
from mini_ocr.services.page_store import PageStore


class OCRPagesStage:
    """Runs OCR for rendered pages and stores text/blocks/page analysis."""

    def __init__(
        self,
        ocr=None,
        preprocessor: ImagePreprocessor | None = None,
        page_store: PageStore | None = None,
    ) -> None:
        self.ocr = ocr or PaddleOCRService()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.page_store = page_store or PageStore()

    def run(self, db: Session, document: Document) -> None:
        pages = self.page_store.list_pages(db, document)
        for page in pages:
            if page.ocr_status == "done":
                continue
            try:
                self.page_store.mark_page_running(db, page)
                image_path = Path(page.image_path)
                if settings.enable_image_preprocessing:
                    image_path = self.preprocessor.preprocess(image_path)
                with AgentTimer(
                    "ocr.page",
                    document_id=document.id,
                    page_number=page.page_number,
                    image_path=str(image_path),
                ) as trace:
                    result = self.ocr.recognize_page(image_path)
                    trace.set(
                        text_length=len(result.text or ""),
                        blocks_count=len(result.blocks),
                        orientation=result.orientation,
                        ocr_score=result.ocr_score,
                        avg_confidence=result.avg_confidence,
                        layout_type=result.layout_type,
                    )
                self.page_store.replace_page_ocr_result(db, page, result)
            except Exception as exc:
                self.page_store.mark_page_failed(db, page, exc)
