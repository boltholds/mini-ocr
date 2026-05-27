from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict
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


class SectionCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    section_type: str
    text: str
    page_from: int
    page_to: int
    score: int
    title: str | None = None
    source: str = "header"
    layout_type: str | None = None


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    layout_type: str | None = None
    ocr_score: float | None = None


@dataclass(frozen=True)
class SectionDetectionContext:
    pages: Sequence[PageText]
    threshold: int
    max_pages_after_header: int


class SectionDetectionStrategy(Protocol):
    """Independent source of section candidates.

    Strategies are intentionally small: each one knows how to detect one kind
    of candidate and returns zero or more results. The detector only merges,
    deduplicates and sorts their output.
    """

    def detect(self, ctx: SectionDetectionContext) -> list[SectionCandidate]:
        ...


class CandidatePreservePolicy(Protocol):
    """Policy that protects a candidate from overlap pruning."""

    def should_preserve(self, candidate: SectionCandidate, ctx: SectionDetectionContext) -> bool:
        ...


class NumberedTermRowsPreservePolicy:
    """Keep windows that contain explicit numbered term rows.

    Some standards contain a compact term list formatted as rows like
    ``2.1. Term — definition``. A one-page table candidate may overlap that
    area, but the broader header window can still be necessary to keep recall
    for rows 2.1-2.9. Such windows are protected from table-overlap pruning.
    """

    min_rows: int = 2

    def should_preserve(self, candidate: SectionCandidate, ctx: SectionDetectionContext) -> bool:
        if candidate.section_type != "terms":
            return False
        return has_numbered_term_rows(candidate.text, min_rows=self.min_rows)


class CandidateSelectionPolicy(Protocol):
    """Post-processing policy for candidate lists.

    Detection strategies are intentionally recall-oriented. Selection policies
    make the final list less noisy: exact dedup, overlap pruning, and
    table-document specialization live here instead of inside strategies.
    """

    def apply(self, candidates: list[SectionCandidate], ctx: SectionDetectionContext) -> list[SectionCandidate]:
        ...


class ExactCandidateDedupPolicy:
    def apply(self, candidates: list[SectionCandidate], ctx: SectionDetectionContext) -> list[SectionCandidate]:
        result: list[SectionCandidate] = []
        seen: set[tuple[str, int, int, str]] = set()
        for candidate in candidates:
            key = candidate_identity(candidate)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
        return result


@dataclass(frozen=True)
class PreferTableCandidatesForSmallTableDocsPolicy:
    """Suppress broad header windows in small table-heavy documents.

    Old scanned standards often have one terminology table split across a few
    pages. Header windows like terms:3-5, terms:4-6, terms:6-8 overlap the
    same table pages and cause repeated LLM calls and duplicate entities.
    If the document is small and table strategies produced enough confident
    candidates, prefer those focused one-page candidates.
    """

    max_pages: int = 12
    min_table_candidates: int = 2
    min_table_score: int = 75
    preserve_policies: Sequence[CandidatePreservePolicy] = (NumberedTermRowsPreservePolicy(),)

    def apply(self, candidates: list[SectionCandidate], ctx: SectionDetectionContext) -> list[SectionCandidate]:
        if len(ctx.pages) > self.max_pages:
            return candidates
        confident_tables = [
            c for c in candidates
            if c.source == "table" and c.score >= self.min_table_score
        ]
        if len(confident_tables) < self.min_table_candidates:
            return candidates

        result: list[SectionCandidate] = []
        for candidate in candidates:
            if candidate.source != "header" or candidate.page_to <= candidate.page_from:
                result.append(candidate)
                continue
            if self._overlaps_any_table(candidate, confident_tables):
                if self._must_preserve(candidate, ctx):
                    result.append(candidate)
                continue
            result.append(candidate)
        return result

    def _overlaps_any_table(self, candidate: SectionCandidate, tables: Sequence[SectionCandidate]) -> bool:
        return any(
            not (table.page_to < candidate.page_from or table.page_from > candidate.page_to)
            for table in tables
        )

    def _must_preserve(self, candidate: SectionCandidate, ctx: SectionDetectionContext) -> bool:
        return any(policy.should_preserve(candidate, ctx) for policy in self.preserve_policies)


class SortCandidatesPolicy:
    def apply(self, candidates: list[SectionCandidate], ctx: SectionDetectionContext) -> list[SectionCandidate]:
        return sorted(candidates, key=lambda c: (c.page_from, c.section_type, -c.score, c.source))


class SectionDetector:
    """Find candidate chunks without binding to a specific document type.

    The detector is a small strategy runner now. Header detection, term-table
    pages and abbreviation-table pages are separate strategies, so new checks
    can be added without growing this class into a long list of if-statements.
    """

    def __init__(
        self,
        threshold: int = 82,
        max_pages_after_header: int = 3,
        strategies: Sequence[SectionDetectionStrategy] | None = None,
        selection_policies: Sequence[CandidateSelectionPolicy] | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_pages_after_header = max_pages_after_header
        self.strategies = list(strategies or default_section_detection_strategies())
        self.selection_policies = list(selection_policies or default_candidate_selection_policies())

    def detect(self, pages: list[tuple[int, str] | PageText]) -> list[SectionCandidate]:
        ctx = SectionDetectionContext(
            pages=[coerce_page(p) for p in pages],
            threshold=self.threshold,
            max_pages_after_header=self.max_pages_after_header,
        )

        candidates: list[SectionCandidate] = []
        for strategy in self.strategies:
            candidates.extend(strategy.detect(ctx))

        for policy in self.selection_policies:
            candidates = policy.apply(candidates, ctx)

        return candidates


class HeaderSectionStrategy:
    """Detect explicit section headers and slice a small page window."""

    def __init__(self, section_type: str, headers: Sequence[str]) -> None:
        self.section_type = section_type
        self.headers = tuple(headers)

    def detect(self, ctx: SectionDetectionContext) -> list[SectionCandidate]:
        candidates: list[SectionCandidate] = []
        for match in self._find_headers(ctx.pages, ctx.threshold):
            page_idx, line_idx, score, title = match
            section_pages = list(ctx.pages[page_idx: page_idx + ctx.max_pages_after_header])
            if not section_pages:
                continue
            candidates.append(
                SectionCandidate(
                    section_type=self.section_type,
                    text=slice_from_header(section_pages, line_idx),
                    page_from=section_pages[0].page_number,
                    page_to=section_pages[-1].page_number,
                    score=score,
                    title=title,
                    source="header",
                    layout_type=section_pages[0].layout_type,
                )
            )
        return candidates

    def _find_headers(self, pages: Sequence[PageText], threshold: int) -> list[tuple[int, int, int, str]]:
        found: list[tuple[int, int, int, str]] = []
        for page_idx, page in enumerate(pages):
            for line_idx, line in enumerate(page.text.splitlines()):
                norm = normalize_text(line)
                if not looks_like_header_line(norm):
                    continue
                if has_bad_header_context(norm):
                    continue

                score, header = best_header_match(norm, self.headers)
                if score >= threshold:
                    found.append((page_idx, line_idx, score, header))
        return found


class ScoredPageStrategy:
    """Base strategy for one-candidate-per-page detectors."""

    section_type: str
    source: str = "table"
    title: str
    min_score: int

    def detect(self, ctx: SectionDetectionContext) -> list[SectionCandidate]:
        candidates: list[SectionCandidate] = []
        for page in ctx.pages:
            score = self.score(page)
            if score < self.min_score:
                continue
            candidates.append(
                SectionCandidate(
                    section_type=self.section_type,
                    text=format_page_candidate_text(page),
                    page_from=page.page_number,
                    page_to=page.page_number,
                    score=score,
                    title=self.title,
                    source=self.source,
                    layout_type=page.layout_type,
                )
            )
        return candidates

    def score(self, page: PageText) -> int:
        raise NotImplementedError


class TermTablePageStrategy(ScoredPageStrategy):
    section_type = "terms"
    title = "term-definition table candidate"
    min_score = 75

    def score(self, page: PageText) -> int:
        norm = normalize_text(page.text)
        score = 0
        if "термин" in norm:
            score += 35
        if "определ" in norm:
            score += 35
        if page.layout_type == "table_like":
            score += 20
        if has_repeated_numbered_rows(norm):
            score += 10
        if has_bad_header_context(norm) and score < 80:
            score -= 25
        return clamp_score(score)


class AbbreviationPageStrategy(ScoredPageStrategy):
    section_type = "abbreviations"
    title = "abbreviation table candidate"
    min_score = 78

    def score(self, page: PageText) -> int:
        norm = normalize_text(page.text)
        score = 0
        if "сокращ" in norm:
            score += 45
        if "обознач" in norm:
            score += 35
        if "расшифров" in norm:
            score += 30
        if page.layout_type == "table_like":
            score += 15
        return clamp_score(score)


def default_section_detection_strategies() -> list[SectionDetectionStrategy]:
    return [
        HeaderSectionStrategy("abbreviations", ABBREVIATION_HEADERS),
        HeaderSectionStrategy("terms", TERM_HEADERS),
        TermTablePageStrategy(),
        AbbreviationPageStrategy(),
    ]


def default_candidate_selection_policies() -> list[CandidateSelectionPolicy]:
    return [
        ExactCandidateDedupPolicy(),
        PreferTableCandidatesForSmallTableDocsPolicy(),
        SortCandidatesPolicy(),
    ]


def coerce_page(page: tuple[int, str] | PageText) -> PageText:
    if isinstance(page, PageText):
        return page
    return PageText(page_number=page[0], text=page[1])


def candidate_identity(candidate: SectionCandidate) -> tuple[str, int, int, str]:
    return (candidate.section_type, candidate.page_from, candidate.page_to, candidate.source)


def format_page_candidate_text(page: PageText) -> str:
    return f"--- page {page.page_number} ---\n{page.text}"


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = text.replace("—", "-").replace("–", "-")
    return " ".join(text.split())


def looks_like_header_line(norm: str) -> bool:
    if not norm:
        return False
    if len(norm) > 90:
        return False
    if " - " in norm or ("-" in norm and len(norm.split()) > 3):
        return False
    return True


def has_bad_header_context(norm: str) -> bool:
    return any(bad in norm for bad in BAD_HEADER_CONTEXT)


def best_header_match(norm: str, headers: Sequence[str]) -> tuple[int, str]:
    scores = [(header_score(norm, header), header) for header in headers]
    return max(scores, key=lambda x: x[0])


def header_score(norm: str, header: str) -> int:
    return int(
        max(
            fuzz.ratio(norm, header),
            fuzz.token_sort_ratio(norm, header),
            fuzz.token_set_ratio(norm, header),
        )
    )


def slice_from_header(pages: Sequence[PageText], header_line_idx: int) -> str:
    chunks: list[str] = []
    for idx, page in enumerate(pages):
        lines = page.text.splitlines()
        if idx == 0:
            lines = lines[header_line_idx:]
        elif starts_with_next_section_marker(lines):
            break
        chunks.append(f"--- page {page.page_number} ---\n" + "\n".join(lines))
    return "\n".join(chunks)


def starts_with_next_section_marker(lines: Sequence[str]) -> bool:
    first_lines = " ".join(lines[:3]).lower().replace("ё", "е")
    return any(marker in first_lines for marker in NEXT_SECTION_MARKERS)


def has_repeated_numbered_rows(norm: str) -> bool:
    # OCR often flattens tables; this weak signal catches pages with numbered
    # rows like "1. ... 2. ... 3. ..." without tying to a specific standard.
    hits = sum(1 for marker in ("1.", "2.", "3.", "4.", "5.") if marker in norm)
    return hits >= 2


def has_numbered_term_rows(text: str, min_rows: int = 2) -> bool:
    """Return true for explicit term rows like ``2.1. Term — definition``.

    This is stronger than ``has_repeated_numbered_rows``. It intentionally
    looks for subsection-style numbering with at least one dot inside the
    number, because table classifiers should not preserve every arbitrary
    numbered list such as ``1. ... 2. ...``.
    """

    if not text:
        return False
    pattern = re.compile(r"(?m)(?:^|\n)\s*\d+(?:\.\d+)+\.\s*[А-ЯЁA-Zа-яё]")
    return len(pattern.findall(text)) >= min_rows


def clamp_score(score: int) -> int:
    return max(0, min(score, 100))
