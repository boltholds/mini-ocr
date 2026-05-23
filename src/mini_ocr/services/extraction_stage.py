from __future__ import annotations

from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import Document
from mini_ocr.services.langgraph_workflow import LangGraphExtractionWorkflow
from mini_ocr.services.observability import AgentTimer, get_logger
from mini_ocr.services.page_store import PageStore
from mini_ocr.services.section_detector import SectionDetector


class ExtractItemsStage:
    """Detects target sections and delegates extraction to LangGraph workflow."""

    def __init__(self, detector: SectionDetector | None = None, page_store: PageStore | None = None) -> None:
        self.detector = detector or SectionDetector()
        self.page_store = page_store or PageStore()
        self.logger = get_logger("pipeline.extract")

    def run(self, db: Session, document: Document) -> list[str]:
        pages = self.page_store.list_done_pages(db, document)
        if not pages:
            return ["No OCR pages available for extraction"]

        # Re-run means replace extraction/validation output, but keep render/OCR cache.
        self.page_store.clear_extraction_output(db, document)

        valid_page_numbers = {p.page_number for p in pages}
        max_page_number = max(valid_page_numbers)
        page_texts = self.page_store.build_page_texts(db, pages)

        with AgentTimer("section_detector.detect", document_id=document.id, pages_count=len(page_texts)) as trace:
            candidates = self.detector.detect(page_texts)
            trace.set(raw_candidates_count=len(candidates))

        candidates = [
            c for c in candidates
            if c.page_from in valid_page_numbers
            and c.page_to in valid_page_numbers
            and c.page_from <= c.page_to <= max_page_number
        ]
        self.logger.info(
            "section candidates after filtering: document_id=%s candidates=%s",
            document.id,
            [f"{c.section_type}:{c.page_from}-{c.page_to}:{c.source}" for c in candidates],
        )
        if not candidates:
            return ["No target sections found"]

        if not settings.enable_langgraph_workflow:
            raise RuntimeError("ENABLE_LANGGRAPH_WORKFLOW=false is not supported in this build")

        workflow = LangGraphExtractionWorkflow(db, document)
        state = workflow.run(candidates)
        return list(state.get("errors") or [])
