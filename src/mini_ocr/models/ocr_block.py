import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class OCRBlock(Base):
    __tablename__ = "ocr_blocks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[str] = mapped_column(ForeignKey("document_pages.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    page = relationship("DocumentPage", back_populates="blocks")
