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
from mini_ocr.services.llm.extractor import HybridExtractor
from mini_ocr.schemas.extraction import ExtractionResult
from mini_ocr.services.extraction_validator import ExtractionValidator


class ProcessingPipeline:
    def __init__(self) -> None:
        self.renderer = PDFRenderer()
        self.preprocessor = ImagePreprocessor()
        self.ocr = PaddleOCRService()
        self.detector = SectionDetector()
        self.extractor = HybridExtractor()
        self.validator = ExtractionValidator()

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
            self._render_pages(db, document)

            document.status = "ocr_running"
            db.commit()
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
        existing_count = db.query(DocumentPage).filter_by(document_id=document.id).count()
        if existing_count > 0:
            return
        pdf_path = Path(document.file_path)
        image_dir = settings.storage_dir / document.file_hash / "pages"
        image_paths = self.renderer.render(pdf_path, image_dir)
        if not image_paths:
            raise RuntimeError("PDF renderer produced no page images")
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
                result = self.ocr.recognize_page(image_path)
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
        page_texts = []
        for p in pages:
            analysis = db.query(PageAnalysis).filter_by(page_id=p.id).first()
            page_texts.append(PageText(
                page_number=p.page_number,
                text=p.ocr_text or "",
                layout_type=analysis.layout_type if analysis else None,
                ocr_score=analysis.ocr_score if analysis else None,
            ))
        candidates = self.detector.detect(page_texts)
        if not candidates:
            document.error_message = "No target sections found"
            db.commit()
            return

        for candidate in candidates:
            input_hash = sha256_text(candidate.text + settings.prompt_version + settings.llm_model)
            job = (
                db.query(ExtractionJob)
                .filter_by(
                    document_id=document.id,
                    section_type=candidate.section_type,
                    input_text_hash=input_hash,
                    prompt_version=settings.prompt_version,
                    model_name=settings.llm_model,
                )
                .first()
            )
            if job and job.status == "done":
                continue
            if job is None:
                job = ExtractionJob(
                    document_id=document.id,
                    section_type=candidate.section_type,
                    page_from=candidate.page_from,
                    page_to=candidate.page_to,
                    input_text_hash=input_hash,
                    prompt_version=settings.prompt_version,
                    model_name=settings.llm_model,
                    status="running",
                )
                db.add(job)
            else:
                job.status = "running"
            db.commit()

            try:
                result = self.extractor.extract(candidate.text)
                self._save_result(db, document, result, candidate.page_from, candidate.page_to, candidate.section_type, candidate.text)
                job.status = "done"
                job.error_message = None
                db.commit()
            except Exception as exc:
                job.status = "failed"
                job.error_message = str(exc)
                db.commit()

    def _save_result(
        self,
        db: Session,
        document: Document,
        result: ExtractionResult,
        page_from: int,
        page_to: int,
        section_type: str,
        chunk_text: str,
    ) -> None:
        allowed_types = {"abbreviations": {"abbreviation"}, "terms": {"term"}, "mixed": {"abbreviation", "term"}}
        allowed = allowed_types.get(section_type, {"abbreviation", "term"})

        for item_type, entities in (("abbreviation", result.abbreviations), ("term", result.terms)):
            if item_type not in allowed:
                continue
            for entity in entities:
                decision = self.validator.validate(item_type, entity, chunk_text, section_type)
                if not decision.keep:
                    continue

                row = ExtractedItem(
                    document_id=document.id,
                    item_type=item_type,
                    key=entity.key.strip(),
                    value=entity.value.strip(),
                    source_text=(entity.source_text or "").strip() or None,
                    page_from=page_from,
                    page_to=page_to,
                    confidence=decision.confidence,
                    status=decision.status,
                    extractor=getattr(self.extractor, "extractor_name", "unknown"),
                )
                db.add(row)
                try:
                    db.commit()
                except IntegrityError:
                    db.rollback()
