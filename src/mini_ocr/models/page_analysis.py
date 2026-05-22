import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class PageAnalysis(Base):
    __tablename__ = "page_analyses"
    __table_args__ = (UniqueConstraint("page_id", name="uq_page_analysis_page"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[str] = mapped_column(ForeignKey("document_pages.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)

    orientation: Mapped[int] = mapped_column(Integer, default=0)
    ocr_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    layout_type: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    text_length: Mapped[int] = mapped_column(Integer, default=0)
    blocks_count: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    page = relationship("DocumentPage", back_populates="analysis")
