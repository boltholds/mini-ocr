import uuid
from datetime import datetime
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class ItemValidation(Base):
    __tablename__ = "item_validations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    item_id: Mapped[str] = mapped_column(ForeignKey("extracted_items.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)

    agent_name: Mapped[str] = mapped_column(String(64), default="candidate_validation_agent")
    decision: Mapped[str] = mapped_column(String(32), default="needs_review")  # auto | needs_review | rejected
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    rag_evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    item = relationship("ExtractedItem")
