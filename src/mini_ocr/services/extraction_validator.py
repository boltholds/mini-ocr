from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from mini_ocr.schemas.extraction import ExtractedEntity


BAD_TERM_KEYS = (
    "применение терминов",
    "применение терминов-синонимов",
    "терминов-синонимов",
    "синонимов не допускается",
    "не допускается",
    "недопустим",
    "документ",
)

BAD_VALUE_FRAGMENTS = (
    "не допускается",
    "см.",
)

PAGE_NOISE = (
    "ост",
    "стр",
    "страница",
    "группа",
)

ABBR_RE = re.compile(r"^[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9./-]{1,14}$")


@dataclass
class ValidationDecision:
    keep: bool
    confidence: float
    status: str
    reason: str | None = None


class ExtractionValidator:
    """Hard guardrails for LLM output.

    The LLM proposes candidates; this validator decides what is safe to persist.
    Rules are intentionally generic: no document-specific terms are hardcoded.
    """

    def validate(self, item_type: str, entity: ExtractedEntity, chunk_text: str, section_type: str) -> ValidationDecision:
        key = _compact(entity.key)
        value = _compact(entity.value)
        source = _compact(entity.source_text or "")
        chunk = _compact(chunk_text)
        confidence = float(entity.confidence or 0.5)

        if not key or not value:
            return ValidationDecision(False, 0.0, "rejected", "empty key/value")
        if not source:
            confidence = min(confidence, 0.49)
            source = f"{key} {value}"

        if item_type == "abbreviation":
            decision = self._validate_abbreviation(key, value, source, chunk, confidence, section_type)
        elif item_type == "term":
            decision = self._validate_term(key, value, source, chunk, confidence, section_type)
        else:
            return ValidationDecision(False, 0.0, "rejected", "unknown item_type")

        if not decision.keep:
            return decision

        # Grounding: source_text should be present or at least close to the OCR chunk.
        similarity = _best_similarity(source.lower(), chunk.lower())
        if source.lower() not in chunk.lower() and similarity < 0.72:
            confidence = min(decision.confidence, 0.49)
        else:
            confidence = decision.confidence

        status = "auto" if confidence >= 0.85 else "needs_review"
        return ValidationDecision(True, confidence, status, decision.reason)

    def _validate_abbreviation(self, key: str, value: str, source: str, chunk: str, confidence: float, section_type: str) -> ValidationDecision:
        if section_type not in {"abbreviations", "mixed"}:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation outside abbreviation section")
        if key not in source:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation key is not in source_text")
        if not ABBR_RE.match(key):
            return ValidationDecision(False, 0.0, "rejected", "key does not look like an abbreviation")
        if len(value.split()) > 18:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation value is too long")
        if _looks_like_noise(key) or _looks_like_noise(value):
            return ValidationDecision(False, 0.0, "rejected", "noise-like abbreviation")
        return ValidationDecision(True, min(confidence, 0.9), "auto")

    def _validate_term(self, key: str, value: str, source: str, chunk: str, confidence: float, section_type: str) -> ValidationDecision:
        key_lower = key.lower().replace("ё", "е")
        value_lower = value.lower().replace("ё", "е")

        if section_type == "abbreviations":
            return ValidationDecision(False, 0.0, "rejected", "term inside abbreviation section")
        if len(key) < 2 or len(value) < 5:
            return ValidationDecision(False, 0.0, "rejected", "too short")
        if len(key.split()) > 8:
            return ValidationDecision(False, 0.0, "rejected", "key is too long for a term")
        if any(bad in key_lower for bad in BAD_TERM_KEYS):
            return ValidationDecision(False, 0.0, "rejected", "service phrase is not a term")
        if any(bad in value_lower for bad in BAD_VALUE_FRAGMENTS) and len(value.split()) <= 4:
            return ValidationDecision(False, 0.0, "rejected", "service value is not a definition")
        if _looks_like_noise(key):
            return ValidationDecision(False, 0.0, "rejected", "noise-like term key")
        if key.lower() not in source.lower():
            confidence = min(confidence, 0.7)
        if value.lower() not in source.lower():
            confidence = min(confidence, 0.75)
        return ValidationDecision(True, min(confidence, 0.95), "auto")


def _compact(text: str) -> str:
    return " ".join((text or "").split())


def _looks_like_noise(text: str) -> bool:
    norm = text.lower().replace("ё", "е")
    if any(noise in norm for noise in PAGE_NOISE) and len(norm.split()) <= 3:
        return True
    letters = [ch for ch in norm if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in text if ch.isalpha()) / max(len([ch for ch in text if ch.isalpha()]), 1)
        # ALL CAPS multi-word headings are often section/table headers rather than terms.
        if upper_ratio > 0.85 and len(text.split()) > 3:
            return True
    if text.count(".") >= 3:
        return True
    return False


def _best_similarity(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    if len(needle) > len(haystack):
        needle, haystack = haystack, needle
    # Fast path for long chunks: compare against windows around likely first token.
    first = needle.split(" ", 1)[0]
    positions = [m.start() for m in re.finditer(re.escape(first), haystack)] if first else []
    if not positions:
        return SequenceMatcher(None, needle, haystack[: max(len(needle) * 2, 200)]).ratio()
    best = 0.0
    window_size = max(len(needle) * 2, 200)
    for pos in positions[:20]:
        window = haystack[max(0, pos - 30): pos + window_size]
        best = max(best, SequenceMatcher(None, needle, window).ratio())
    return best
