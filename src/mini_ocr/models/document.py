import uuid
from datetime import datetime
from sqlalchemy import DateTime, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("file_hash", name="uq_documents_file_hash"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(512), index=True)
    file_path: Mapped[str] = mapped_column(Text)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), default="registered", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    items = relationship("ExtractedItem", back_populates="document", cascade="all, delete-orphan")
