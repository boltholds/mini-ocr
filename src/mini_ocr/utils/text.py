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
    _, cyr, lat = letter_stats(text)
    return cyr > 0 and lat > 0


def looks_latin_or_foreign(text: str | None) -> bool:
    stripped = compact(text)
    letters, cyr, lat = letter_stats(stripped)
    if not letters:
        return False
    if lat > 0 and cyr == 0:
        return True
    return bool(FOREIGN_DOT_ALIAS_RE.match(stripped))


def is_clean_cyrillic_caps(text: str | None) -> bool:
    stripped = compact(text)
    letters, cyr, lat = letter_stats(stripped)
    if letters == 0 or lat > 0:
        return False
    if cyr / max(letters, 1) < 0.9:
        return False
    if any(ch.isdigit() for ch in stripped):
        return False
    allowed_punct = set(" -().,/№«»\"'")
    if any((not ch.isalnum()) and (ch not in allowed_punct) for ch in stripped):
        return False
    return sum(ch.isupper() for ch in stripped if ch.isalpha()) / max(letters, 1) >= 0.85


def looks_clean_russian_term(text: str | None) -> bool:
    stripped = compact(text)
    if not stripped:
        return False
    letters, cyr, lat = letter_stats(stripped)
    if letters == 0 or lat > 0:
        return False
    if cyr / max(letters, 1) < 0.85:
        return False
    if len(stripped.split()) > 8:
        return False
    if is_clean_cyrillic_caps(stripped):
        return False
    if re.search(r"[@#$%^*_+=<>|]", stripped):
        return False
    return True


def looks_ocr_noisy(text: str | None) -> bool:
    stripped = compact(text)
    if not stripped:
        return True
    letters, cyr, lat = letter_stats(stripped)
    if letters == 0:
        return True
    if cyr > 0 and lat > 0:
        return True
    upper_ratio = sum(ch.isupper() for ch in stripped if ch.isalpha()) / max(letters, 1)
    weird = sum(ch.isdigit() or ch in "@#$%^*_+=<>|/" for ch in stripped)
    return upper_ratio > 0.75 or weird >= 2 or ("-" in stripped and len(stripped) <= 12)


def titlecase_cyrillic_caps(text: str) -> str:
    stripped = compact(text)
    if not is_clean_cyrillic_caps(stripped):
        return stripped
    return " ".join(part.capitalize() if part.isalpha() else part for part in stripped.split())
