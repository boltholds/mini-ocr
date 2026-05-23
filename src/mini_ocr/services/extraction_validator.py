from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

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

FOREIGN_ALIAS_RE = re.compile(r"^[A-ZА-Я]?\s*[A-Z]\.\s+[A-Za-z].+")
MIXED_GARBAGE_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[-_/]).{5,}$")
ABBR_RE = re.compile(r"^[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9./-]{1,14}$")


@dataclass(frozen=True)
class ValidationDecision:
    keep: bool
    confidence: float
    status: str
    reason: str | None = None


@dataclass
class ValidationContext:
    item_type: str
    key: str
    value: str
    source: str
    chunk: str
    confidence: float
    section_type: str

    @classmethod
    def from_entity(
        cls,
        item_type: str,
        entity: ExtractedEntity,
        chunk_text: str,
        section_type: str,
    ) -> "ValidationContext":
        key = _compact(entity.key)
        value = _compact(entity.value)
        source = _compact(entity.source_text or "")
        confidence = float(entity.confidence or 0.5)
        if not source:
            confidence = min(confidence, 0.49)
            source = f"{key} {value}".strip()
        return cls(
            item_type=item_type,
            key=key,
            value=value,
            source=source,
            chunk=_compact(chunk_text),
            confidence=confidence,
            section_type=section_type,
        )

    def with_confidence(self, confidence: float) -> "ValidationContext":
        return ValidationContext(
            item_type=self.item_type,
            key=self.key,
            value=self.value,
            source=self.source,
            chunk=self.chunk,
            confidence=confidence,
            section_type=self.section_type,
        )


class ValidationRule(Protocol):
    def apply(self, ctx: ValidationContext) -> ValidationDecision | ValidationContext | None:
        """Return a rejection/acceptance decision, modified context, or None to continue."""


class ValidationStrategy(Protocol):
    def validate(self, ctx: ValidationContext) -> ValidationDecision:
        ...


class SequentialValidationStrategy:
    """Runs small validation rules one by one.

    A rule may reject early, mutate confidence via a new context, or return None.
    If no rule rejects, the entity is kept with the accumulated confidence.
    """

    def __init__(self, rules: list[ValidationRule]) -> None:
        self.rules = rules

    def validate(self, ctx: ValidationContext) -> ValidationDecision:
        for rule in self.rules:
            result = rule.apply(ctx)
            if result is None:
                continue
            if isinstance(result, ValidationDecision):
                return result
            ctx = result

        confidence = min(max(ctx.confidence, 0.0), 0.95)
        status = "auto" if confidence >= 0.85 else "needs_review"
        return ValidationDecision(True, confidence, status)


class ExtractionValidator:
    """Hard guardrails for LLM output.

    The LLM proposes candidates; this validator decides what is safe to persist.
    Validation is implemented as Strategy + sequential rule checks:
    item_type selects the strategy, the strategy runs small checks one by one.
    """

    def __init__(self) -> None:
        self.strategies: dict[str, ValidationStrategy] = {
            "abbreviation": SequentialValidationStrategy([
                EmptyKeyValueRule(),
                AbbreviationSectionRule(),
                AbbreviationKeyInSourceRule(),
                AbbreviationShapeRule(),
                AbbreviationValueLengthRule(),
                NoiseRejectRule("noise-like abbreviation"),
                GroundingRule(),
                MaxConfidenceRule(0.9),
            ]),
            "term": SequentialValidationStrategy([
                EmptyKeyValueRule(),
                TermSectionRule(),
                TermLengthRule(),
                ServiceTermRule(),
                ServiceValueRule(),
                ForeignAliasConfidenceRule(),
                OcrGarbageConfidenceRule(),
                NoiseConfidenceRule(),
                SourceContainsKeyConfidenceRule(),
                SourceContainsValueConfidenceRule(),
                GroundingRule(),
                MaxConfidenceRule(0.95),
            ]),
        }

    def validate(
        self,
        item_type: str,
        entity: ExtractedEntity,
        chunk_text: str,
        section_type: str,
    ) -> ValidationDecision:
        strategy = self.strategies.get(item_type)
        if strategy is None:
            return ValidationDecision(False, 0.0, "rejected", "unknown item_type")
        ctx = ValidationContext.from_entity(item_type, entity, chunk_text, section_type)
        return strategy.validate(ctx)


class EmptyKeyValueRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if not ctx.key or not ctx.value:
            return ValidationDecision(False, 0.0, "rejected", "empty key/value")
        return None


class AbbreviationSectionRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if ctx.section_type not in {"abbreviations", "mixed"}:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation outside abbreviation section")
        return None


class TermSectionRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if ctx.section_type == "abbreviations":
            return ValidationDecision(False, 0.0, "rejected", "term inside abbreviation section")
        return None


class AbbreviationKeyInSourceRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if ctx.key not in ctx.source:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation key is not in source_text")
        return None


class AbbreviationShapeRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if not ABBR_RE.match(ctx.key):
            return ValidationDecision(False, 0.0, "rejected", "key does not look like an abbreviation")
        return None


class AbbreviationValueLengthRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if len(ctx.value.split()) > 18:
            return ValidationDecision(False, 0.0, "rejected", "abbreviation value is too long")
        return None


class TermLengthRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if len(ctx.key) < 2 or len(ctx.value) < 5:
            return ValidationDecision(False, 0.0, "rejected", "too short")
        if len(ctx.key.split()) > 8:
            return ValidationDecision(False, 0.0, "rejected", "key is too long for a term")
        return None


class ServiceTermRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        key_lower = ctx.key.lower().replace("ё", "е")
        if any(bad in key_lower for bad in BAD_TERM_KEYS):
            return ValidationDecision(False, 0.0, "rejected", "service phrase is not a term")
        return None


class ServiceValueRule:
    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        value_lower = ctx.value.lower().replace("ё", "е")
        if any(bad in value_lower for bad in BAD_VALUE_FRAGMENTS) and len(ctx.value.split()) <= 4:
            return ValidationDecision(False, 0.0, "rejected", "service value is not a definition")
        return None


class NoiseRejectRule:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def apply(self, ctx: ValidationContext) -> ValidationDecision | None:
        if _looks_like_noise(ctx.key) or _looks_like_noise(ctx.value):
            return ValidationDecision(False, 0.0, "rejected", self.reason)
        return None


class ConfidenceCapRule:
    def __init__(self, predicate, cap: float) -> None:
        self.predicate = predicate
        self.cap = cap

    def apply(self, ctx: ValidationContext) -> ValidationContext | None:
        if self.predicate(ctx):
            return ctx.with_confidence(min(ctx.confidence, self.cap))
        return None


class ForeignAliasConfidenceRule(ConfidenceCapRule):
    def __init__(self) -> None:
        super().__init__(lambda ctx: _looks_like_foreign_alias(ctx.key), 0.49)


class OcrGarbageConfidenceRule(ConfidenceCapRule):
    def __init__(self) -> None:
        super().__init__(lambda ctx: _looks_like_ocr_garbage_key(ctx.key), 0.49)


class NoiseConfidenceRule(ConfidenceCapRule):
    def __init__(self) -> None:
        super().__init__(lambda ctx: _looks_like_noise(ctx.key), 0.49)


class SourceContainsKeyConfidenceRule(ConfidenceCapRule):
    def __init__(self) -> None:
        super().__init__(lambda ctx: ctx.key.lower() not in ctx.source.lower(), 0.7)


class SourceContainsValueConfidenceRule(ConfidenceCapRule):
    def __init__(self) -> None:
        super().__init__(lambda ctx: ctx.value.lower() not in ctx.source.lower(), 0.75)


class GroundingRule:
    def apply(self, ctx: ValidationContext) -> ValidationContext | None:
        similarity = _best_similarity(ctx.source.lower(), ctx.chunk.lower())
        if ctx.source.lower() not in ctx.chunk.lower() and similarity < 0.72:
            return ctx.with_confidence(min(ctx.confidence, 0.49))
        return None


class MaxConfidenceRule:
    def __init__(self, cap: float) -> None:
        self.cap = cap

    def apply(self, ctx: ValidationContext) -> ValidationContext:
        return ctx.with_confidence(min(ctx.confidence, self.cap))


def _compact(text: str) -> str:
    return " ".join((text or "").split())


def _looks_like_noise(text: str) -> bool:
    norm = text.lower().replace("ё", "е")
    if any(noise in norm for noise in PAGE_NOISE) and len(norm.split()) <= 3:
        return True
    letters = [ch for ch in norm if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in text if ch.isalpha()) / max(len([ch for ch in text if ch.isalpha()]), 1)
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


def _looks_like_foreign_alias(text: str) -> bool:
    return bool(FOREIGN_ALIAS_RE.match(text.strip()))


def _looks_like_ocr_garbage_key(text: str) -> bool:
    stripped = text.strip()
    if MIXED_GARBAGE_RE.match(stripped):
        return True

    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return True

    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters)
    latin = sum(ch.isascii() and ch.isalpha() for ch in letters)

    if latin > cyr and len(stripped) > 5:
        return True

    upper_ratio = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
    if upper_ratio > 0.85 and len(stripped) > 8:
        if cyr / max(len(letters), 1) < 0.8:
            return True
        if any(ch.isdigit() for ch in stripped) or "-" in stripped:
            return True

    weird = sum(1 for ch in stripped if not (ch.isalnum() or ch.isspace() or ch in "-().,/:;№\"'«»"))
    if weird >= 2:
        return True

    return False
