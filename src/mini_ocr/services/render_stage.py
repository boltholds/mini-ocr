from __future__ import annotations

from sqlalchemy.orm import Session

from mini_ocr.models import Document
from mini_ocr.services.observability import get_logger
from mini_ocr.services.page_store import PageStore
from mini_ocr.services.pdf_renderer import PDFRenderer
from mini_ocr.services.pipeline_context import ProcessingContext


class RenderPagesStage:
    """Turns a PDF document into page images and persists DocumentPage rows."""

    def __init__(self, renderer: PDFRenderer | None = None, page_store: PageStore | None = None) -> None:
        self.renderer = renderer or PDFRenderer()
        self.page_store = page_store or PageStore()
        self.logger = get_logger("pipeline.render")

    def run(self, db: Session, document: Document) -> None:
        ctx = ProcessingContext(document)
        expected_count = self.renderer.page_count(ctx.pdf_path)

        if self.page_store.has_complete_render_cache(db, document, expected_count):
            self.logger.info(
                "render cache hit: document_id=%s pages=%s image_dir=%s",
                document.id,
                expected_count,
                ctx.page_image_dir,
            )
            return

        self.logger.info(
            "render cache miss: document_id=%s expected_pages=%s image_dir=%s",
            document.id,
            expected_count,
            ctx.page_image_dir,
        )
        image_paths = self.renderer.render(ctx.pdf_path, ctx.page_image_dir, force=True)
        if not image_paths:
            raise RuntimeError("PDF renderer produced no page images")
        if len(image_paths) != expected_count:
            raise RuntimeError(f"PDF renderer produced {len(image_paths)} images, expected {expected_count}")

        self.page_store.replace_rendered_pages(db, document, image_paths)
