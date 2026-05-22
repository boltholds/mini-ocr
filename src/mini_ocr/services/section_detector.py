from __future__ import annotations

from dataclasses import dataclass
from rapidfuzz import fuzz


ABBREVIATION_HEADERS = [
    "сокращения",
    "перечень сокращений",
    "список сокращений",
    "условные обозначения и сокращения",
    "обозначения и сокращения",
    "условные обозначения",
    "обозначения",
    "основные обозначения",
    "буквенные обозначения",
]

TERM_HEADERS = [
    "термины и определения",
    "термины, определения и сокращения",
    "основные термины и определения",
    "термины",
    "определения",
]

BAD_HEADER_CONTEXT = [
    "применение терминов",
    "терминов-синонимов",
    "терминов синонимов",
    "не допускается",
    "недопустим",
]

NEXT_SECTION_MARKERS = [
    "приложение",
    "библиография",
    "содержание",
    "перечень",
    "список",
]


@dataclass
class SectionCandidate:
    section_type: str
    text: str
    page_from: int
    page_to: int
    score: int
    title: str | None = None
    source: str = "header"
    layout_type: str | None = None


@dataclass
class PageText:
    page_number: int
    text: str
    layout_type: str | None = None
    ocr_score: float | None = None


class SectionDetector:
    """Finds candidate chunks without binding to a specific document type.

    The detector prefers explicit section headers, but also handles table-like
    pages that contain both "Термин" and "Определение" columns. Extraction is
    chunked by page/section instead of a large multi-page blob to reduce LLM
    hallucinations and preserve precise page links.
    """

    def __init__(self, threshold: int = 82, max_pages_after_header: int = 3) -> None:
        self.threshold = threshold
        self.max_pages_after_header = max_pages_after_header

    def detect(self, pages: list[tuple[int, str] | PageText]) -> list[SectionCandidate]:
        normalized_pages = [self._coerce_page(p) for p in pages]
        candidates: list[SectionCandidate] = []
        seen: set[tuple[str, int, int, str]] = set()

        # Explicit headers: split into small chunks from the header page onward.
        for section_type, headers in (("abbreviations", ABBREVIATION_HEADERS), ("terms", TERM_HEADERS)):
            for found in self._find_headers(normalized_pages, headers):
                page_idx, line_idx, score, matched_title = found
                section_pages = normalized_pages[page_idx: page_idx + self.max_pages_after_header]
                if not section_pages:
                    continue
                section_text = self._slice_from_header(section_pages, line_idx)
                key = (section_type, section_pages[0].page_number, section_pages[-1].page_number, "header")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SectionCandidate(
                        section_type=section_type,
                        text=section_text,
                        page_from=section_pages[0].page_number,
                        page_to=section_pages[-1].page_number,
                        score=score,
                        title=matched_title,
                        source="header",
                        layout_type=section_pages[0].layout_type,
                    )
                )

        # Generic table pages: one candidate per page when a page looks like a
        # term-definition table. This covers standards and scanned tables without
        # assuming a concrete document name.
        for page in normalized_pages:
            table_score = self._term_table_score(page)
            if table_score >= 75:
                key = ("terms", page.page_number, page.page_number, "table")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SectionCandidate(
                        section_type="terms",
                        text=f"--- page {page.page_number} ---\n{page.text}",
                        page_from=page.page_number,
                        page_to=page.page_number,
                        score=table_score,
                        title="term-definition table candidate",
                        source="table",
                        layout_type=page.layout_type,
                    )
                )

        # Generic abbreviation pages: one candidate per page if abbreviation
        # signals are strong enough.
        for page in normalized_pages:
            abbr_score = self._abbreviation_page_score(page)
            if abbr_score >= 78:
                key = ("abbreviations", page.page_number, page.page_number, "table")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SectionCandidate(
                        section_type="abbreviations",
                        text=f"--- page {page.page_number} ---\n{page.text}",
                        page_from=page.page_number,
                        page_to=page.page_number,
                        score=abbr_score,
                        title="abbreviation table candidate",
                        source="table",
                        layout_type=page.layout_type,
                    )
                )

        return sorted(candidates, key=lambda c: (c.page_from, c.section_type, -c.score))

    def _find_headers(self, pages: list[PageText], headers: list[str]) -> list[tuple[int, int, int, str]]:
        found: list[tuple[int, int, int, str]] = []
        for page_idx, page in enumerate(pages):
            for line_idx, line in enumerate(page.text.splitlines()):
                norm = self._normalize(line)
                if not self._looks_like_header_line(norm):
                    continue
                if any(bad in norm for bad in BAD_HEADER_CONTEXT):
                    continue

                scores = [(int(self._header_score(norm, header)), header) for header in headers]
                score, header = max(scores, key=lambda x: x[0])
                if score >= self.threshold:
                    found.append((page_idx, line_idx, int(score), header))
        return found

    def _slice_from_header(self, pages: list[PageText], header_line_idx: int) -> str:
        chunks: list[str] = []
        for idx, page in enumerate(pages):
            lines = page.text.splitlines()
            if idx == 0:
                lines = lines[header_line_idx:]
            else:
                # Stop if the next page starts with a strong unrelated marker.
                first_lines = " ".join(lines[:3]).lower().replace("ё", "е")
                if any(marker in first_lines for marker in NEXT_SECTION_MARKERS):
                    break
            chunks.append(f"--- page {page.page_number} ---\n" + "\n".join(lines))
        return "\n".join(chunks)

    def _term_table_score(self, page: PageText) -> int:
        norm = self._normalize(page.text)
        score = 0
        if "термин" in norm:
            score += 35
        if "определ" in norm:
            score += 35
        if page.layout_type == "table_like":
            score += 20
        if _has_repeated_numbered_rows(norm):
            score += 10
        if any(bad in norm for bad in BAD_HEADER_CONTEXT) and score < 80:
            score -= 25
        return max(0, min(score, 100))

    def _abbreviation_page_score(self, page: PageText) -> int:
        norm = self._normalize(page.text)
        score = 0
        if "сокращ" in norm:
            score += 45
        if "обознач" in norm:
            score += 35
        if "расшифров" in norm:
            score += 30
        if page.layout_type == "table_like":
            score += 15
        return max(0, min(score, 100))

    @staticmethod
    def _coerce_page(page: tuple[int, str] | PageText) -> PageText:
        if isinstance(page, PageText):
            return page
        return PageText(page_number=page[0], text=page[1])

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().replace("ё", "е")
        text = text.replace("—", "-").replace("–", "-")
        return " ".join(text.split())

    @staticmethod
    def _looks_like_header_line(norm: str) -> bool:
        if not norm:
            return False
        if len(norm) > 90:
            return False
        if " - " in norm or "-" in norm and len(norm.split()) > 3:
            return False
        return True

    @staticmethod
    def _header_score(norm: str, header: str) -> int:
        return int(max(
            fuzz.ratio(norm, header),
            fuzz.token_sort_ratio(norm, header),
            fuzz.token_set_ratio(norm, header),
        ))


def _has_repeated_numbered_rows(norm: str) -> bool:
    # OCR often flattens tables; this weak signal catches pages with numbered
    # rows like "1. ... 2. ... 3. ..." without tying to a specific standard.
    hits = sum(1 for marker in ("1.", "2.", "3.", "4.", "5.") if marker in norm)
    return hits >= 2
