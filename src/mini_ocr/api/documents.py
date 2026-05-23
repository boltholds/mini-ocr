from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from mini_ocr.core.db import get_db
from mini_ocr.models import Document, DocumentPage, ExtractedItem, PageAnalysis, ItemValidation, TermKnowledgeEntry
from mini_ocr.schemas.document import DocumentOut, ItemOut, ValidationOut, KnowledgeEntryOut
from mini_ocr.services.pipeline import ProcessingPipeline
from mini_ocr.services.section_detector import PageText, SectionDetector
from mini_ocr.services.rag_store import RagStore

router = APIRouter(prefix="/documents", tags=["documents"])


def get_pipeline() -> ProcessingPipeline:
    # In real production this would be a singleton / background worker dependency.
    return ProcessingPipeline()


@router.post("/upload", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    pipeline: ProcessingPipeline = Depends(get_pipeline),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)

    document = pipeline.register_document(db, tmp_path, title=file.filename)
    return document


@router.post("/{document_id}/process", response_model=DocumentOut)
def process_document(
    document_id: str,
    db: Session = Depends(get_db),
    pipeline: ProcessingPipeline = Depends(get_pipeline),
):
    try:
        return pipeline.process_document(db, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/items", response_model=list[ItemOut])
def get_items(document_id: str, status: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ExtractedItem).filter(ExtractedItem.document_id == document_id)
    if status:
        query = query.filter(ExtractedItem.status == status)
    return query.order_by(ExtractedItem.item_type, ExtractedItem.key).all()


@router.get("/{document_id}/ocr-text")
def get_ocr_text(
    document_id: str,
    page: int | None = Query(default=None, ge=1),
    limit: int = Query(default=3000, ge=100, le=20000),
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    query = db.query(DocumentPage).filter(DocumentPage.document_id == document_id)
    if page is not None:
        query = query.filter(DocumentPage.page_number == page)

    pages = query.order_by(DocumentPage.page_number).all()
    return [
        {
            "page_number": p.page_number,
            "ocr_status": p.ocr_status,
            "text": (p.ocr_text or "")[:limit],
            "error_message": p.error_message,
            "analysis": _page_analysis_payload(db, p),
        }
        for p in pages
    ]


@router.get("/{document_id}/sections")
def get_sections(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id, DocumentPage.ocr_status == "done")
        .order_by(DocumentPage.page_number)
        .all()
    )
    page_texts = []
    for p in pages:
        analysis = db.query(PageAnalysis).filter_by(page_id=p.id).first()
        page_texts.append(PageText(
            page_number=p.page_number,
            text=p.ocr_text or "",
            layout_type=analysis.layout_type if analysis else None,
            ocr_score=analysis.ocr_score if analysis else None,
        ))

    candidates = SectionDetector().detect(page_texts)
    return [
        {
            "section_type": c.section_type,
            "page_from": c.page_from,
            "page_to": c.page_to,
            "score": c.score,
            "title": c.title,
            "source": c.source,
            "layout_type": c.layout_type,
            "text_preview": c.text[:1200],
        }
        for c in candidates
    ]


def _page_analysis_payload(db: Session, page: DocumentPage) -> dict | None:
    analysis = db.query(PageAnalysis).filter_by(page_id=page.id).first()
    if analysis is None:
        return None
    return {
        "orientation": analysis.orientation,
        "ocr_score": analysis.ocr_score,
        "avg_confidence": analysis.avg_confidence,
        "layout_type": analysis.layout_type,
        "text_length": analysis.text_length,
        "blocks_count": analysis.blocks_count,
    }


@router.get("/by-title/{title}/items", response_model=list[ItemOut])
def get_items_by_title(title: str, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.title == title).order_by(Document.created_at.desc()).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return db.query(ExtractedItem).filter(ExtractedItem.document_id == document.id).all()


@router.get("/items/{item_id}/validations", response_model=list[ValidationOut])
def get_item_validations(item_id: str, db: Session = Depends(get_db)):
    item = db.get(ExtractedItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return (
        db.query(ItemValidation)
        .filter(ItemValidation.item_id == item_id)
        .order_by(ItemValidation.created_at.desc())
        .all()
    )


@router.patch("/items/{item_id}/approve", response_model=ItemOut)
def approve_item(item_id: str, db: Session = Depends(get_db)):
    item = db.get(ExtractedItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.status = "approved"
    db.commit()
    db.refresh(item)
    RagStore().add_confirmed_item(db, item, status="approved")
    return item


@router.patch("/items/{item_id}/reject", response_model=ItemOut)
def reject_item(item_id: str, db: Session = Depends(get_db)):
    item = db.get(ExtractedItem, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    item.status = "rejected"
    db.commit()
    db.refresh(item)
    return item


@router.get("/kb/terms", response_model=list[KnowledgeEntryOut])
def get_knowledge_base(limit: int = Query(default=50, ge=1, le=500), db: Session = Depends(get_db)):
    return db.query(TermKnowledgeEntry).order_by(TermKnowledgeEntry.created_at.desc()).limit(limit).all()
