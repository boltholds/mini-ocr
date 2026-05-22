from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from mini_ocr.core.config import settings
from mini_ocr.services.ocr.base import OCRBlockResult, OCRPageResult, OCRService

# More stable on Windows CPU builds; harmless when the backend ignores it.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")


class PaddleOCRService(OCRService):
    """PaddleOCR wrapper with optional auto-orientation.

    The service supports PaddleOCR 2.x `.ocr()` output and PaddleOCR/PaddleX
    3.x `.predict()` output. When auto-rotation is enabled, each page is tested
    at 0/90/180/270 degrees and the best OCR result is selected by a generic
    score: text length, average confidence and letter ratio.
    """

    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is not installed. Install dependencies or use Docker image.") from exc

        try:
            self.ocr = PaddleOCR(
                lang=settings.ocr_lang,
                use_textline_orientation=True,
            )
            self._api = "v3"
        except TypeError:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ocr_lang,
                use_gpu=settings.ocr_use_gpu,
                show_log=False,
            )
            self._api = "v2"

    def recognize_page(self, image_path: Path) -> OCRPageResult:
        angles = [0, 90, 180, 270] if settings.enable_auto_rotation else [0]
        best: OCRPageResult | None = None
        rotated_paths: list[Path] = []

        for angle in angles:
            candidate_path = image_path if angle == 0 else _rotated_copy(image_path, angle)
            if angle != 0:
                rotated_paths.append(candidate_path)

            result = self._recognize_single(candidate_path)
            result.orientation = angle
            result.image_path = str(candidate_path)
            result.avg_confidence = _avg_confidence(result.blocks)
            result.layout_type = _classify_layout(result.blocks, result.text)
            result.ocr_score = _score_ocr_result(result.text, result.avg_confidence, result.blocks)

            if best is None or result.ocr_score > best.ocr_score:
                best = result

        # Keep rotated debug images if they were selected; remove unused ones.
        selected = Path(best.image_path) if best and best.image_path else image_path
        for path in rotated_paths:
            if path != selected:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

        return best or OCRPageResult(text="", blocks=[], image_path=str(image_path))

    def _recognize_single(self, image_path: Path) -> OCRPageResult:
        if hasattr(self.ocr, "predict"):
            try:
                raw = self.ocr.predict(str(image_path))
            except TypeError:
                raw = self.ocr.predict(input=str(image_path))
            return self._parse_v3_result(raw)

        raw = self.ocr.ocr(str(image_path), cls=True)
        return self._parse_v2_result(raw)

    def _parse_v2_result(self, raw: Any) -> OCRPageResult:
        blocks: list[OCRBlockResult] = []
        lines = raw[0] if raw and isinstance(raw, list) and raw[0] else []

        for line in lines:
            try:
                bbox = _to_plain(line[0])
                text = str(line[1][0]).strip()
                conf = float(line[1][1])
            except Exception:
                continue
            if text:
                blocks.append(OCRBlockResult(text=text, confidence=conf, bbox=bbox))

        return OCRPageResult(text="\n".join(block.text for block in blocks), blocks=blocks)

    def _parse_v3_result(self, raw: Any) -> OCRPageResult:
        blocks: list[OCRBlockResult] = []

        for item in raw or []:
            data = getattr(item, "json", None)
            if callable(data):
                data = data()
            if data is None:
                to_dict = getattr(item, "to_dict", None)
                if callable(to_dict):
                    data = to_dict()
            if data is None and isinstance(item, dict):
                data = item
            if not isinstance(data, dict):
                continue

            res = data.get("res", data)
            texts = res.get("rec_texts") or res.get("texts") or []
            scores = res.get("rec_scores") or res.get("scores") or []
            polys = res.get("rec_polys") or res.get("dt_polys") or res.get("rec_boxes") or res.get("boxes") or []

            if isinstance(texts, str):
                texts = [texts]

            for idx, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue
                conf = _safe_float(scores[idx] if idx < len(scores) else None)
                bbox = _to_plain(polys[idx] if idx < len(polys) else [])
                blocks.append(OCRBlockResult(text=text, confidence=conf, bbox=bbox))

        return OCRPageResult(text="\n".join(block.text for block in blocks), blocks=blocks)


def _rotated_copy(image_path: Path, angle: int) -> Path:
    out = image_path.with_name(f"{image_path.stem}_rot{angle}{image_path.suffix}")
    if out.exists():
        return out
    with Image.open(image_path) as img:
        img.rotate(angle, expand=True).save(out)
    return out


def _avg_confidence(blocks: list[OCRBlockResult]) -> float | None:
    values = [b.confidence for b in blocks if b.confidence is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _score_ocr_result(text: str, avg_confidence: float | None, blocks: list[OCRBlockResult]) -> float:
    stripped = "".join(ch for ch in text if not ch.isspace())
    if not stripped:
        return 0.0
    letters = sum(ch.isalpha() for ch in stripped)
    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in stripped)
    letter_ratio = letters / max(len(stripped), 1)
    cyr_ratio = cyr / max(letters, 1)
    return len(stripped) * 0.02 + len(blocks) * 0.08 + (avg_confidence or 0) * 5 + letter_ratio * 3 + cyr_ratio * 2


def _classify_layout(blocks: list[OCRBlockResult], text: str) -> str:
    if not blocks or len(text.strip()) < 20:
        return "empty_or_failed"

    avg_conf = _avg_confidence(blocks)
    if avg_conf is not None and avg_conf < 0.45:
        return "low_quality"

    xs: list[float] = []
    ys: list[float] = []
    for block in blocks:
        box = block.bbox or []
        try:
            x_values = [float(p[0]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
            y_values = [float(p[1]) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        except Exception:
            continue
        if x_values and y_values:
            xs.append(min(x_values))
            ys.append(min(y_values))

    if len(blocks) >= 12 and xs:
        width = max(xs) - min(xs) if len(xs) > 1 else 0
        # Several x-start clusters usually means columns/table-like layout.
        buckets = {round((x - min(xs)) / max(width, 1) * 6) for x in xs}
        if len(buckets) >= 3:
            return "table_like"

    if len(blocks) >= 8 and any(word in text.lower().replace("ё", "е") for word in ("термин", "определение", "сокращение", "обозначение")):
        return "table_like"

    if len(blocks) >= 6:
        return "plain_text"
    return "mixed_text_image"


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value
