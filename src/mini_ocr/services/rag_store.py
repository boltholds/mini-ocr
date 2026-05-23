from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections import Counter
from sqlalchemy.orm import Session

from mini_ocr.models import TermKnowledgeEntry, ExtractedItem
from mini_ocr.core.config import settings

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


@dataclass
class RagMatch:
    term: str
    definition: str
    score: float
    source_item_id: str | None = None


class RagStore:
    """Small PostgreSQL-backed RAG store.

    For the MVP we store a lightweight token-vector in JSON and compute cosine in
    Python. This keeps the project runnable on a plain local PostgreSQL.
    The table is intentionally isolated so it can be migrated to pgvector later
    without changing extractor/agent interfaces.
    """

    def retrieve(self, db: Session, query: str, top_k: int | None = None) -> list[RagMatch]:
        top_k = top_k or settings.rag_top_k
        query_vec = _text_vector(query)
        if not query_vec:
            return []

        rows = db.query(TermKnowledgeEntry).limit(2000).all()
        scored: list[RagMatch] = []
        for row in rows:
            row_vec = row.embedding or _text_vector(f"{row.term} {row.definition}")
            score = _cosine(query_vec, row_vec)
            if score <= 0:
                continue
            scored.append(RagMatch(
                term=row.term,
                definition=row.definition,
                score=round(score, 4),
                source_item_id=row.source_item_id,
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def add_confirmed_item(self, db: Session, item: ExtractedItem, status: str = "auto") -> None:
        if item.item_type != "term":
            return
        key = (item.key or "").strip()
        value = (item.value or "").strip()
        if not key or not value:
            return

        existing = (
            db.query(TermKnowledgeEntry)
            .filter(TermKnowledgeEntry.term == key, TermKnowledgeEntry.definition == value)
            .first()
        )
        if existing:
            return

        db.add(TermKnowledgeEntry(
            term=key,
            definition=value,
            source_document_id=item.document_id,
            source_item_id=item.id,
            status=status,
            embedding=_text_vector(f"{key} {value}"),
        ))
        db.commit()


def _tokens(text: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    clean = re.sub(r"\s+", " ", (text or "").lower().replace("ё", "е")).strip()
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i:i+n] for i in range(len(clean) - n + 1)]


def _text_vector(text: str) -> dict[str, float]:
    toks = _tokens(text)
    grams = _char_ngrams(text)
    counts = Counter(toks + grams)
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))
