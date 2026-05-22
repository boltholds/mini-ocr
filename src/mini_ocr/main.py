from fastapi import FastAPI
from mini_ocr.core.db import Base, engine
from mini_ocr.models import Document, DocumentPage, OCRBlock, ExtractedItem, ExtractionJob, PageAnalysis  # noqa: F401
from mini_ocr.api.documents import router as documents_router

app = FastAPI(title="OCR + LLM Document Extractor", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(documents_router)
