from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from mini_ocr.core.config import settings
from mini_ocr.models import Document, DocumentPage, OCRBlock, ExtractedItem, ExtractionJob, PageAnalysis
from mini_ocr.services.hash_utils import sha256_file, sha256_text
from mini_ocr.services.pdf_renderer import PDFRenderer
from mini_ocr.services.image_preprocessor import ImagePreprocessor
from mini_ocr.services.ocr.paddleocr_service import PaddleOCRService
from mini_ocr.services.section_detector import SectionDetector, PageText
from mini_ocr.services.langgraph_workflow import LangGraphExtractionWorkflow
from mini_ocr.services.observability import AgentTimer, get_logger


class ProcessingPipeline:
    def __init__(self) -> None:
        self.renderer = PDFRenderer()
        self.preprocessor = ImagePreprocessor()
        self.ocr = PaddleOCRService()
        self.detector = SectionDetector()
        self.logger = get_logger("pipeline")

    def register_document(self, db: Session, src_path: Path, title: str | None = None) -> Document:
        file_hash = sha256_file(src_path)
        existing = db.query(Document).filter(Document.file_hash == file_hash).first()
        if existing:
            return existing

        doc_dir = settings.storage_dir / file_hash
        doc_dir.mkdir(parents=True, exist_ok=True)
        stored_path = doc_dir / src_path.name
        if src_path.resolve() != stored_path.resolve():
            stored_path.write_bytes(src_path.read_bytes())

        document = Document(
            title=title or src_path.name,
            file_path=str(stored_path),
            file_hash=file_hash,
            status="registered",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document

    def process_document(self, db: Session, document_id: str) -> Document:
        document = db.get(Document, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        if document.status == "processed":
            return document

        try:
            document.status = "rendering"
            db.commit()
            with AgentTimer("pipeline.render_pages", document_id=document.id, title=document.title):
                self._render_pages(db, document)

            document.status = "ocr_running"
            db.commit()
            with AgentTimer("pipeline.ocr_pages", document_id=document.id, title=document.title):
                self._ocr_pages(db, document)

            done_pages = (
                db.query(DocumentPage)
                .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "done")
                .count()
            )
            if done_pages == 0:
                document.status = "failed"
                document.error_message = "OCR failed for all pages"
                db.commit()
                return document

            document.status = "extracting"
            db.commit()
            with AgentTimer("pipeline.extract_items", document_id=document.id, title=document.title):
                self._extract_items(db, document)

            failed_pages = (
                db.query(DocumentPage)
                .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "failed")
                .count()
            )
            document.status = "processed_with_warnings" if failed_pages else "processed"
            if failed_pages:
                document.error_message = f"Processed with warnings: OCR failed for {failed_pages} page(s)"
            else:
                document.error_message = None
            db.commit()
            db.refresh(document)
            return document
        except Exception as exc:
            document.status = "failed"
            document.error_message = str(exc)
            db.commit()
            raise

    def _render_pages(self, db: Session, document: Document) -> None:
        pdf_path = Path(document.file_path)
        image_dir = settings.storage_dir / document.file_hash / "pages"
        expected_count = self.renderer.page_count(pdf_path)

        existing_pages = (
            db.query(DocumentPage)
            .filter_by(document_id=document.id)
            .order_by(DocumentPage.page_number)
            .all()
        )
        existing_count = len(existing_pages)
        max_existing_page = max((p.page_number for p in existing_pages), default=0)
        all_images_exist = all(p.image_path and Path(p.image_path).exists() for p in existing_pages)

        # Do not trust stale page rows. During development or after renderer changes,
        # an old DB state can contain page_number values that do not belong to the
        # current PDF. That later produces impossible page ranges like 30-32 in a
        # 20-page document. Re-render if DB rows do not match the real PDF page count.
        if existing_count == expected_count and max_existing_page == expected_count and all_images_exist:
            self.logger.info(
                "render cache hit: document_id=%s pages=%s image_dir=%s",
                document.id,
                existing_count,
                image_dir,
            )
            return

        db.query(OCRBlock).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(PageAnalysis).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractionJob).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractedItem).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(DocumentPage).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.commit()

        self.logger.info(
            "render cache miss: document_id=%s expected_pages=%s existing_pages=%s image_dir=%s",
            document.id,
            expected_count,
            existing_count,
            image_dir,
        )
        image_paths = self.renderer.render(pdf_path, image_dir, force=True)
        if not image_paths:
            raise RuntimeError("PDF renderer produced no page images")
        if len(image_paths) != expected_count:
            raise RuntimeError(f"PDF renderer produced {len(image_paths)} images, expected {expected_count}")

        for idx, image_path in enumerate(image_paths, start=1):
            db.add(DocumentPage(document_id=document.id, page_number=idx, image_path=str(image_path)))
        db.commit()

    def _ocr_pages(self, db: Session, document: Document) -> None:
        pages = (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id)
            .order_by(DocumentPage.page_number)
            .all()
        )
        for page in pages:
            if page.ocr_status == "done":
                continue
            try:
                page.ocr_status = "running"
                db.commit()
                image_path = Path(page.image_path)
                if settings.enable_image_preprocessing:
                    image_path = self.preprocessor.preprocess(image_path)
                with AgentTimer(
                    "ocr.page",
                    document_id=document.id,
                    page_number=page.page_number,
                    image_path=str(image_path),
                ) as trace:
                    result = self.ocr.recognize_page(image_path)
                    trace.set(
                        text_length=len(result.text or ""),
                        blocks_count=len(result.blocks),
                        orientation=result.orientation,
                        ocr_score=result.ocr_score,
                        avg_confidence=result.avg_confidence,
                        layout_type=result.layout_type,
                    )
                page.ocr_text = result.text
                page.ocr_status = "done"
                page.error_message = None

                db.query(OCRBlock).filter_by(page_id=page.id).delete()
                db.query(PageAnalysis).filter_by(page_id=page.id).delete()

                for block in result.blocks:
                    db.add(OCRBlock(
                        document_id=document.id,
                        page_id=page.id,
                        page_number=page.page_number,
                        text=block.text,
                        confidence=block.confidence,
                        bbox={"points": block.bbox},
                    ))

                db.add(PageAnalysis(
                    document_id=document.id,
                    page_id=page.id,
                    page_number=page.page_number,
                    orientation=result.orientation,
                    ocr_score=result.ocr_score,
                    avg_confidence=result.avg_confidence,
                    layout_type=result.layout_type,
                    text_length=len(result.text or ""),
                    blocks_count=len(result.blocks),
                ))
                db.commit()
            except Exception as exc:
                page.ocr_status = "failed"
                page.error_message = str(exc)
                db.commit()

    def _extract_items(self, db: Session, document: Document) -> None:
        pages = (
            db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id, DocumentPage.ocr_status == "done")
            .order_by(DocumentPage.page_number)
            .all()
        )
        if not pages:
            document.error_message = "No OCR pages available for extraction"
            db.commit()
            return

        # Re-run means replace extraction/validation output, but keep OCR cache.
        from mini_ocr.models import ItemValidation
        db.query(ItemValidation).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractionJob).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.query(ExtractedItem).filter_by(document_id=document.id).delete(synchronize_session=False)
        db.commit()

        valid_page_numbers = {p.page_number for p in pages}
        max_page_number = max(valid_page_numbers)

        page_texts = []
        for p in pages:
            analysis = db.query(PageAnalysis).filter_by(page_id=p.id).first()
            page_texts.append(PageText(
                page_number=p.page_number,
                text=p.ocr_text or "",
                layout_type=analysis.layout_type if analysis else None,
                ocr_score=analysis.ocr_score if analysis else None,
            ))

        with AgentTimer("section_detector.detect", document_id=document.id, pages_count=len(page_texts)) as trace:
            candidates = self.detector.detect(page_texts)
            trace.set(raw_candidates_count=len(candidates))
        candidates = [
            c for c in candidates
            if c.page_from in valid_page_numbers
            and c.page_to in valid_page_numbers
            and c.page_from <= c.page_to <= max_page_number
        ]
        self.logger.info(
            "section candidates after filtering: document_id=%s candidates=%s",
            document.id,
            [f"{c.section_type}:{c.page_from}-{c.page_to}:{c.source}" for c in candidates],
        )
        if not candidates:
            document.error_message = "No target sections found"
            db.commit()
            return

        if settings.enable_langgraph_workflow:
            workflow = LangGraphExtractionWorkflow(db, document)
            state = workflow.run(candidates)
            if state.get("errors"):
                document.error_message = "; ".join(state["errors"][:3])
                db.commit()
            return

        # LangGraph is the default path. This branch is kept only as a clear
        # configuration failure instead of silently returning old behavior.
        raise RuntimeError("ENABLE_LANGGRAPH_WORKFLOW=false is not supported in this build")
