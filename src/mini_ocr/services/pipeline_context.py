from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_ocr.models import Document


@dataclass(frozen=True)
class ProcessingContext:
    document: Document

    @property
    def pdf_path(self) -> Path:
        return Path(self.document.file_path)

    @property
    def page_image_dir(self) -> Path:
        from mini_ocr.core.config import settings

        return settings.storage_dir / self.document.file_hash / "pages"
