from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from mini_ocr.models import Document


class DocumentStatusService:
    """Small DB boundary for document lifecycle transitions."""

    def set_status(self, db: Session, document: Document, status: str, error_message: str | None = None) -> None:
        document.status = status
        document.error_message = error_message
        db.commit()

    def mark_processed(self, db: Session, document: Document, failed_pages: int = 0) -> None:
        if failed_pages:
            document.status = "processed_with_warnings"
            document.error_message = f"Processed with warnings: OCR failed for {failed_pages} page(s)"
        else:
            document.status = "processed"
            document.error_message = None
        db.commit()
        db.refresh(document)

    def mark_failed(self, db: Session, document: Document, exc: Exception | str) -> None:
        document.status = "failed"
        document.error_message = str(exc)
        db.commit()

    @contextmanager
    def stage(self, db: Session, document: Document, status: str) -> Iterator[None]:
        self.set_status(db, document, status)
        yield
