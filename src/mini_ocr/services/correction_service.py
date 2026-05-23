from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import ExtractedItem, ItemValidation
from mini_ocr.services.agents.correction import CorrectionGraph, CorrectionState, CorrectionSuggestion
from mini_ocr.services.observability import AgentTimer
from mini_ocr.services.rag_store import RagMatch, RagStore


class CorrectionRepository:
    """Database/RAG boundary for OCR correction.

    Agents and LangGraph operate on plain CorrectionState/CorrectionSuggestion.
    This repository is the only place that knows how to read/write SQLAlchemy
    models, create ItemValidation records and add confirmed items to RAG.
    """

    agent_name = "langgraph_ocr_correction_workflow"

    def __init__(self, rag: RagStore | None = None) -> None:
        self.rag = rag or RagStore()

    def build_state(self, item: ExtractedItem, matches: list[dict[str, Any]]) -> CorrectionState:
        return {
            "item_id": item.id,
            "document_id": item.document_id,
            "key": item.key,
            "value": item.value,
            "source_text": item.source_text,
            "page_from": item.page_from,
            "page_to": item.page_to,
            "confidence": item.confidence,
            "status": item.status,
            "rag_matches": matches,
        }

    def retrieve_matches(self, db: Session, item: ExtractedItem) -> list[dict[str, Any]]:
        if not settings.enable_rag_validation:
            return []
        with AgentTimer("rag.retrieve_for_correction", document_id=item.document_id, item_id=item.id, key=item.key, top_k=settings.rag_top_k) as trace:
            matches = self.rag.retrieve(db, f"{item.key}\n{item.value}\n{item.source_text or ''}", settings.rag_top_k)
            trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
        return matches_payload(matches)

    def persist(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion, matches: list[dict[str, Any]]) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name=self.agent_name,
            decision=suggestion.status,
            confidence=suggestion.confidence,
            reason=suggestion.reason,
            normalized_key=suggestion.normalized_key,
            normalized_value=suggestion.normalized_value,
            rag_evidence={"matches": matches},
            payload={
                "key": item.key,
                "value": item.value,
                "strategy": suggestion.strategy,
                "orchestrator_reason": suggestion.orchestrator_reason,
            },
        ))
        db.commit()

    def apply(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion) -> None:
        item.normalized_key = suggestion.normalized_key
        item.normalized_value = suggestion.normalized_value
        item.correction_confidence = suggestion.confidence
        item.correction_reason = suggestion.reason
        item.correction_strategy = suggestion.strategy
        item.correction_status = suggestion.status
        item.correction_orchestrator_reason = suggestion.orchestrator_reason
        if item.status == "auto" and suggestion.confidence < 0.85:
            item.status = "needs_review"
        db.commit()


class OCRCorrectionWorkflow:
    """DB-facing correction service used by the extraction workflow."""

    def __init__(self, graph: CorrectionGraph | None = None, repository: CorrectionRepository | None = None) -> None:
        self.graph = graph or CorrectionGraph()
        self.repository = repository or CorrectionRepository()

    def normalize_item(self, db: Session, item: ExtractedItem) -> CorrectionSuggestion:
        matches = self.repository.retrieve_matches(db, item)
        state = self.repository.build_state(item, matches)
        suggestion = self.graph.run(state)
        self.repository.persist(db, item, suggestion, matches)
        self.repository.apply(db, item, suggestion)
        return suggestion


def matches_payload(matches: list[RagMatch]) -> list[dict[str, Any]]:
    return [
        {"term": m.term, "definition": m.definition[:400], "score": m.score, "source_item_id": m.source_item_id}
        for m in matches
    ]
