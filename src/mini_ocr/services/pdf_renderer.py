from pathlib import Path
import fitz


class PDFRenderer:
    def render(self, pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)
        paths: list[Path] = []
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            image_path = out_dir / f"page_{i:04d}.png"
            if not image_path.exists():
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(image_path)
            paths.append(image_path)
        return paths
