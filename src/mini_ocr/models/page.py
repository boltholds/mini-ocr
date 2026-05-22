import uuid
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from mini_ocr.core.db import Base


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_document_page"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_status: Mapped[str] = mapped_column(String(64), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="pages")
    blocks = relationship("OCRBlock", back_populates="page", cascade="all, delete-orphan")
    analysis = relationship("PageAnalysis", back_populates="page", cascade="all, delete-orphan", uselist=False)
