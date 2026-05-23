from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import Document
from mini_ocr.services.hash_utils import sha256_file


class DocumentRegistry:
    """Registers source files in storage and creates/reuses Document rows."""

    def register(self, db: Session, src_path: Path, title: str | None = None) -> Document:
        file_hash = sha256_file(src_path)
        existing = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing:
            return existing

        stored_path = self._copy_to_storage(src_path, file_hash)
        document = Document(
            title=title or src_path.name,
            file_path=str(stored_path),
            file_hash=file_hash,
            status="registered",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document

    def _copy_to_storage(self, src_path: Path, file_hash: str) -> Path:
        doc_dir = settings.storage_dir / file_hash
        doc_dir.mkdir(parents=True, exist_ok=True)
        stored_path = doc_dir / src_path.name
        if src_path.resolve() != stored_path.resolve():
            stored_path.write_bytes(src_path.read_bytes())
        return stored_path
