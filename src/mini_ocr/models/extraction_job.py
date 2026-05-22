import uuid
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from mini_ocr.core.db import Base


class ExtractionJob(Base):
    __tablename__ = "extraction_jobs"
    __table_args__ = (
        UniqueConstraint("document_id", "section_type", "input_text_hash", "prompt_version", "model_name", name="uq_extraction_job_cache"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    section_type: Mapped[str] = mapped_column(String(64), index=True)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_text_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_version: Mapped[str] = mapped_column(String(128))
    model_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(64), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
