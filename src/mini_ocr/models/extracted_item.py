import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class ExtractedItem(Base):
    __tablename__ = "extracted_items"
    __table_args__ = (
        UniqueConstraint("document_id", "item_type", "key", "value", name="uq_extracted_item"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    item_type: Mapped[str] = mapped_column(String(32), index=True)  # abbreviation | term
    key: Mapped[str] = mapped_column(Text)
    value: Mapped[str] = mapped_column(Text)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    correction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    correction_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="auto")  # auto | needs_review | approved | rejected
    extractor: Mapped[str] = mapped_column(String(32), default="llm")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="items")
