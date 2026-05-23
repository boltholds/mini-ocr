from pathlib import Path
import fitz


class PDFRenderer:
    def page_count(self, pdf_path: Path) -> int:
        doc = fitz.open(pdf_path)
        try:
            return int(doc.page_count)
        finally:
            doc.close()

    def render(self, pdf_path: Path, out_dir: Path, dpi: int = 200, force: bool = False) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
        paths: list[Path] = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        try:
            for i, page in enumerate(doc, start=1):
                image_path = out_dir / f"page_{i:04d}.png"
                if force or not image_path.exists():
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    pix.save(image_path)
                paths.append(image_path)
            return paths
        finally:
            doc.close()
