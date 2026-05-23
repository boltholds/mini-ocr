import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from mini_ocr.core.db import Base


class TermKnowledgeEntry(Base):
    __tablename__ = "term_knowledge_base"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    term: Mapped[str] = mapped_column(Text, index=True)
    definition: Mapped[str] = mapped_column(Text)
    source_document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    source_item_id: Mapped[str | None] = mapped_column(ForeignKey("extracted_items.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="confirmed")  # confirmed | approved | imported | auto
    embedding: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # lightweight fallback; pgvector can replace this column later
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
