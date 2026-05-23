from __future__ import annotations

import re
from typing import Any

CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")
FOREIGN_DOT_ALIAS_RE = re.compile(r"^[A-ZА-Я]\.?\s+[A-Za-z].*")


def compact(text: str | None) -> str:
    return " ".join((text or "").split())


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return max(0.0, min(1.0, number))


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "нет"}:
        return None
    return text


def letter_stats(text: str | None) -> tuple[int, int, int]:
    letters = [ch for ch in (text or "") if ch.isalpha()]
    cyr = sum(("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in letters)
    lat = sum(ch.isascii() and ch.isalpha() for ch in letters)
    return len(letters), cyr, lat


def has_mixed_cyrillic_latin(text: str | None) -> bool:
    from mini_ocr.services.policies.text import MIXED_CYRILLIC_LATIN_TEXT_POLICY

    return MIXED_CYRILLIC_LATIN_TEXT_POLICY.matches(text)


def looks_latin_or_foreign(text: str | None) -> bool:
    from mini_ocr.services.policies.text import LATIN_OR_FOREIGN_TEXT_POLICY

    return LATIN_OR_FOREIGN_TEXT_POLICY.matches(text)


def is_clean_cyrillic_caps(text: str | None) -> bool:
    from mini_ocr.services.policies.text import CLEAN_CYRILLIC_CAPS_TEXT_POLICY

    return CLEAN_CYRILLIC_CAPS_TEXT_POLICY.matches(text)


def looks_clean_russian_term(text: str | None) -> bool:
    from mini_ocr.services.policies.text import CLEAN_RUSSIAN_TERM_TEXT_POLICY

    return CLEAN_RUSSIAN_TERM_TEXT_POLICY.matches(text)


def looks_ocr_noisy(text: str | None) -> bool:
    from mini_ocr.services.policies.text import OCR_NOISY_TEXT_POLICY

    return OCR_NOISY_TEXT_POLICY.matches(text)


def titlecase_cyrillic_caps(text: str) -> str:
    stripped = compact(text)
    if not is_clean_cyrillic_caps(stripped):
        return stripped
    return " ".join(part.capitalize() if part.isalpha() else part for part in stripped.split())
