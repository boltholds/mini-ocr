from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from mini_ocr.utils.text import compact, letter_stats

FOREIGN_DOT_ALIAS_RE = re.compile(r"^[A-ZА-Я]\.?.*\s+[A-Za-z].*")
BAD_SYMBOL_RE = re.compile(r"[@#$%^*_+=<>|]")


@dataclass(frozen=True)
class TextFeatures:
    """Precomputed text characteristics shared by all text policies."""

    original: str | None
    text: str
    letters: int
    cyrillic: int
    latin: int
    upper_ratio: float
    digit_count: int
    weird_count: int
    words_count: int

    @classmethod
    def from_text(cls, value: str | None) -> "TextFeatures":
        text = compact(value)
        letters, cyrillic, latin = letter_stats(text)
        alpha = [ch for ch in text if ch.isalpha()]
        upper_ratio = sum(ch.isupper() for ch in alpha) / max(len(alpha), 1)
        digit_count = sum(ch.isdigit() for ch in text)
        weird_count = sum(ch.isdigit() or ch in "@#$%^*_+=<>|/" for ch in text)
        return cls(
            original=value,
            text=text,
            letters=letters,
            cyrillic=cyrillic,
            latin=latin,
            upper_ratio=upper_ratio,
            digit_count=digit_count,
            weird_count=weird_count,
            words_count=len(text.split()),
        )

    @property
    def has_letters(self) -> bool:
        return self.letters > 0

    @property
    def has_cyrillic(self) -> bool:
        return self.cyrillic > 0

    @property
    def has_latin(self) -> bool:
        return self.latin > 0

    @property
    def cyrillic_ratio(self) -> float:
        return self.cyrillic / max(self.letters, 1)


class TextPolicy(Protocol):
    """Reusable boolean policy over OCR/text values."""

    name: str
    reason: str

    def matches(self, text: str | None) -> bool:
        ...

    def matches_features(self, features: TextFeatures) -> bool:
        ...


@dataclass(frozen=True)
class BaseTextPolicy:
    name: str
    reason: str

    def matches(self, text: str | None) -> bool:
        return self.matches_features(TextFeatures.from_text(text))

    def matches_features(self, features: TextFeatures) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class MixedCyrillicLatinPolicy(BaseTextPolicy):
    name: str = "mixed_cyrillic_latin"
    reason: str = "Текст содержит смешение кириллицы и латиницы."

    def matches_features(self, features: TextFeatures) -> bool:
        return features.has_cyrillic and features.has_latin


@dataclass(frozen=True)
class LatinOrForeignPolicy(BaseTextPolicy):
    name: str = "latin_or_foreign"
    reason: str = "Ключ выглядит как латинский термин или иностранный эквивалент."

    def matches_features(self, features: TextFeatures) -> bool:
        if not features.has_letters:
            return False
        if features.has_latin and not features.has_cyrillic:
            return True
        return bool(FOREIGN_DOT_ALIAS_RE.match(features.text))


@dataclass(frozen=True)
class CleanCyrillicCapsPolicy(BaseTextPolicy):
    name: str = "clean_cyrillic_caps"
    reason: str = "Термин написан заглавными русскими буквами."
    min_cyrillic_ratio: float = 0.9
    min_upper_ratio: float = 0.85
    allowed_punctuation: str = " -().,/№«»\"'"

    def matches_features(self, features: TextFeatures) -> bool:
        if not features.has_letters or features.has_latin:
            return False
        if features.cyrillic_ratio < self.min_cyrillic_ratio:
            return False
        if features.digit_count > 0:
            return False
        allowed = set(self.allowed_punctuation)
        if any((not ch.isalnum()) and (ch not in allowed) for ch in features.text):
            return False
        return features.upper_ratio >= self.min_upper_ratio


@dataclass(frozen=True)
class CleanRussianTermPolicy(BaseTextPolicy):
    name: str = "clean_russian_term"
    reason: str = "Термин выглядит как читаемый русский термин; коррекция не требуется."
    min_cyrillic_ratio: float = 0.85
    max_words: int = 8
    caps_policy: TextPolicy = CleanCyrillicCapsPolicy()

    def matches_features(self, features: TextFeatures) -> bool:
        if not features.text or not features.has_letters or features.has_latin:
            return False
        if features.cyrillic_ratio < self.min_cyrillic_ratio:
            return False
        if features.words_count > self.max_words:
            return False
        if self.caps_policy.matches_features(features):
            return False
        if BAD_SYMBOL_RE.search(features.text):
            return False
        return True


@dataclass(frozen=True)
class OCRNoisyPolicy(BaseTextPolicy):
    name: str = "ocr_noisy"
    reason: str = "Ключ содержит признаки OCR-шума."
    mixed_policy: TextPolicy = MixedCyrillicLatinPolicy()

    def matches_features(self, features: TextFeatures) -> bool:
        if not features.text or not features.has_letters:
            return True
        if self.mixed_policy.matches_features(features):
            return True
        return features.upper_ratio > 0.75 or features.weird_count >= 2 or ("-" in features.text and len(features.text) <= 12)


MIXED_CYRILLIC_LATIN_TEXT_POLICY = MixedCyrillicLatinPolicy()
LATIN_OR_FOREIGN_TEXT_POLICY = LatinOrForeignPolicy()
CLEAN_CYRILLIC_CAPS_TEXT_POLICY = CleanCyrillicCapsPolicy()
CLEAN_RUSSIAN_TERM_TEXT_POLICY = CleanRussianTermPolicy(caps_policy=CLEAN_CYRILLIC_CAPS_TEXT_POLICY)
OCR_NOISY_TEXT_POLICY = OCRNoisyPolicy(mixed_policy=MIXED_CYRILLIC_LATIN_TEXT_POLICY)
