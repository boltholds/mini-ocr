from pathlib import Path
import typer
from mini_ocr.core.db import Base, SessionLocal, engine
from mini_ocr.models import Document, ExtractedItem  # noqa: F401
from mini_ocr.services.pipeline import ProcessingPipeline

cli = typer.Typer()


@cli.command()
def init_db():
    Base.metadata.create_all(bind=engine)
    typer.echo("DB initialized")


@cli.command()
def process(path: Path):
    """Process one PDF file or all PDF files in directory."""
    Base.metadata.create_all(bind=engine)
    pipeline = ProcessingPipeline()
    db = SessionLocal()
    try:
        pdfs = [path] if path.is_file() else sorted(path.glob("*.pdf"))
        for pdf in pdfs:
            doc = pipeline.register_document(db, pdf)
            typer.echo(f"Processing {doc.title} [{doc.id}] status={doc.status}")
            pipeline.process_document(db, doc.id)
            typer.echo(f"Done {doc.title}")
    finally:
        db.close()


@cli.command("list-documents")
def list_documents():
    db = SessionLocal()
    try:
        for doc in db.query(Document).order_by(Document.created_at.desc()).all():
            typer.echo(f"{doc.id}\t{doc.title}\t{doc.status}")
    finally:
        db.close()


@cli.command("get-items")
def get_items(title: str):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.title == title).order_by(Document.created_at.desc()).first()
        if not doc:
            typer.echo("Document not found")
            raise typer.Exit(1)
        items = db.query(ExtractedItem).filter(ExtractedItem.document_id == doc.id).all()
        for item in items:
            typer.echo(f"[{item.item_type}] {item.key} — {item.value} (p.{item.page_from}-{item.page_to}, conf={item.confidence})")
    finally:
        db.close()


if __name__ == "__main__":
    cli()
