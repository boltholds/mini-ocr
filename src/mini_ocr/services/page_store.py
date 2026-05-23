from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from mini_ocr.models import Document, DocumentPage, OCRBlock, PageAnalysis, ExtractedItem, ExtractionJob, ItemValidation
from mini_ocr.services.section_detector import PageText


class PageStore:
    """Persistence operations for rendered/OCR pages.

    Rendering and OCR services should not know how many tables need cleanup.
    This class owns that DB detail.
    """

    def list_pages(self, db: Session, document: Document) -> list[DocumentPage]:
        return (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id)
            .order_by(DocumentPage.page_number)
            .all()
        )

    def list_done_pages(self, db: Session, document: Document) -> list[DocumentPage]:
        return (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "done")
            .order_by(DocumentPage.page_number)
            .all()
        )

    def count_done_pages(self, db: Session, document: Document) -> int:
        return (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "done")
            .count()
        )

    def count_failed_pages(self, db: Session, document: Document) -> int:
        return (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "failed")
            .count()
        )

    def has_complete_render_cache(self, db: Session, document: Document, expected_count: int) -> bool:
        pages = self.list_pages(db, document)
        existing_count = len(pages)
        max_existing_page = max((p.page_number for p in pages), default=0)
        all_images_exist = all(p.image_path and Path(p.image_path).exists() for p in pages)
        return existing_count == expected_count and max_existing_page == expected_count and all_images_exist

    def replace_rendered_pages(self, db: Session, document: Document, image_paths: list) -> None:
        self.clear_all_page_dependent_data(db, document)
        for idx, image_path in enumerate(image_paths, start=1):
            db.add(DocumentPage(document_id=document.id, page_number=idx, image_path=str(image_path)))
        db.commit()

    def clear_all_page_dependent_data(self, db: Session, document: Document) -> None:
        db.query(OCRBlock).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(PageAnalysis).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ItemValidation).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractionJob).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractedItem).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(DocumentPage).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.commit()

    def clear_extraction_output(self, db: Session, document: Document) -> None:
        db.query(ItemValidation).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractionJob).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractedItem).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.commit()

    def replace_page_ocr_result(self, db: Session, page: DocumentPage, result) -> None:
        page.ocr_text = result.text
        page.ocr_status = "done"
        page.error_message = None

        db.query(OCRBlock).filter_by(page_id=page.id).delete()
        db.query(PageAnalysis).filter_by(page_id=page.id).delete()

        for block in result.blocks:
            db.add(OCRBlock(
                document_id=page.document_id,
                page_id=page.id,
                page_number=page.page_number,
                text=block.text,
                confidence=block.confidence,
                bbox={"points": block.bbox},
            ))

        db.add(PageAnalysis(
            document_id=page.document_id,
            page_id=page.id,
            page_number=page.page_number,
            orientation=result.orientation,
            ocr_score=result.ocr_score,
            avg_confidence=result.avg_confidence,
            layout_type=result.layout_type,
            text_length=len(result.text or ""),
            blocks_count=len(result.blocks),
        ))
        db.commit()

    def mark_page_running(self, db: Session, page: DocumentPage) -> None:
        page.ocr_status = "running"
        db.commit()

    def mark_page_failed(self, db: Session, page: DocumentPage, exc: Exception | str) -> None:
        page.ocr_status = "failed"
        page.error_message = str(exc)
        db.commit()

    def build_page_texts(self, db: Session, pages: list[DocumentPage]) -> list[PageText]:
        page_texts: list[PageText] = []
        for page in pages:
            analysis = db.query(PageAnalysis).filter_by(page_id=page.id).first()
            page_texts.append(PageText(
                page_number=page.page_number,
                text=page.ocr_text or "",
                layout_type=analysis.layout_type if analysis else None,
                ocr_score=analysis.ocr_score if analysis else None,
            ))
        return page_texts
