from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mini_ocr.services.document_registry import DocumentRegistry
from mini_ocr.services.document_status import DocumentStatusService
from mini_ocr.services.extraction_stage import ExtractItemsStage
from mini_ocr.services.ocr_stage import OCRPagesStage
from mini_ocr.services.page_store import PageStore
from mini_ocr.services.processing_orchestrator import DocumentProcessingOrchestrator
from mini_ocr.services.render_stage import RenderPagesStage

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from mini_ocr.models import Document


class ProcessingPipeline:
    """Application facade for registration + processing.

    Heavy details are delegated:
    - DocumentRegistry owns file hashing/storage + Document creation.
    - RenderPagesStage owns PDF rendering.
    - OCRPagesStage owns OCR execution.
    - ExtractItemsStage owns section detection + LangGraph extraction.
    - PageStore owns page/extraction cleanup and counters.
    - DocumentProcessingOrchestrator owns stage order and status transitions.
    """

    def __init__(
        self,
        registry: DocumentRegistry | None = None,
        status: DocumentStatusService | None = None,
        renderer: RenderPagesStage | None = None,
        ocr: OCRPagesStage | None = None,
        extractor: ExtractItemsStage | None = None,
        page_store: PageStore | None = None,
    ) -> None:
        self.registry = registry or DocumentRegistry()
        self.status = status or DocumentStatusService()
        self.render_stage = renderer or RenderPagesStage()
        self.ocr_stage = ocr or OCRPagesStage()
        self.extract_stage = extractor or ExtractItemsStage()
        self.page_store = page_store or PageStore()
        self.orchestrator = DocumentProcessingOrchestrator(
            status=self.status,
            renderer=self.render_stage,
            ocr=self.ocr_stage,
            extractor=self.extract_stage,
            page_store=self.page_store,
        )

    def register_document(self, db: "Session", src_path: Path, title: str | None = None) -> "Document":
        return self.registry.register(db, src_path, title=title)

    def process_document(self, db: "Session", document_id: str) -> "Document":
        from mini_ocr.models import Document

        document = db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        return self.orchestrator.process(db, document)
