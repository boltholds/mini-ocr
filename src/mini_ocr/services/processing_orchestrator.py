from __future__ import annotations

from typing import Protocol

from mini_ocr.services.observability import AgentTimer, get_logger


class StatusService(Protocol):
    def set_status(self, db, document, status: str, error_message: str | None = None) -> None: ...
    def mark_processed(self, db, document, failed_pages: int = 0) -> None: ...
    def mark_failed(self, db, document, exc: Exception | str) -> None: ...


class ProcessingStage(Protocol):
    def run(self, db, document): ...


class PageCounter(Protocol):
    def count_done_pages(self, db, document) -> int: ...
    def count_failed_pages(self, db, document) -> int: ...


class DocumentProcessingOrchestrator:
    """Pure orchestration: statuses, order, failure policy.

    DB models, OCR, rendering and extraction internals live behind injected
    services. That makes this class easy to test without a database.
    """

    def __init__(
        self,
        status: StatusService,
        renderer: ProcessingStage,
        ocr: ProcessingStage,
        extractor: ProcessingStage,
        page_store: PageCounter,
    ) -> None:
        self.status = status
        self.render_stage = renderer
        self.ocr_stage = ocr
        self.extract_stage = extractor
        self.page_store = page_store
        self.logger = get_logger("pipeline")

    def process(self, db, document):
        if document.status == "processed":
            return document

        try:
            self._run_render_stage(db, document)
            self._run_ocr_stage(db, document)

            done_pages = self.page_store.count_done_pages(db, document)
            if done_pages == 0:
                self.status.set_status(db, document, "failed", "OCR failed for all pages")
                return document

            extraction_errors = self._run_extraction_stage(db, document)
            if extraction_errors:
                document.error_message = "; ".join(extraction_errors[:3])
                db.commit()

            failed_pages = self.page_store.count_failed_pages(db, document)
            self.status.mark_processed(db, document, failed_pages=failed_pages)
            return document
        except Exception as exc:
            self.status.mark_failed(db, document, exc)
            raise

    def _run_render_stage(self, db, document) -> None:
        self.status.set_status(db, document, "rendering")
        with AgentTimer("pipeline.render_pages", document_id=document.id, title=document.title):
            self.render_stage.run(db, document)

    def _run_ocr_stage(self, db, document) -> None:
        self.status.set_status(db, document, "ocr_running")
        with AgentTimer("pipeline.ocr_pages", document_id=document.id, title=document.title):
            self.ocr_stage.run(db, document)

    def _run_extraction_stage(self, db, document) -> list[str]:
        self.status.set_status(db, document, "extracting")
        with AgentTimer("pipeline.extract_items", document_id=document.id, title=document.title):
            return list(self.extract_stage.run(db, document) or [])
