from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import ExtractedItem, ItemValidation
from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.rag_store import RagMatch, RagStore
from mini_ocr.services.observability import AgentTimer
from mini_ocr.utils.json_utils import loads_json_relaxed
from mini_ocr.utils.text import clamp_float


class ValidationDecision(BaseModel):
    decision: str = Field(description="auto | needs_review | rejected")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    normalized_key: str | None = None
    normalized_value: str | None = None


class CandidateValidationAgent:
    agent_name = "langchain_rag_validation_agent"

    def __init__(self) -> None:
        self.rag = RagStore()
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict validation agent for OCR extraction. "
                "Validate one already extracted candidate. Do not extract new terms. "
                "Return only JSON.",
            ),
            (
                "human",
                "Task: validate candidate term/abbreviation from OCR.\n\n"
                "Rules:\n"
                "- decision must be exactly one of: auto, needs_review, rejected.\n"
                "- auto only when the candidate is clearly grounded in source_text and the definition is clean.\n"
                "- needs_review when candidate may be real but OCR noise is high.\n"
                "- rejected for service phrases, empty/unrelated text, or hallucinations.\n"
                "- RAG matches are hints, not proof.\n\n"
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n"
                "{{\"decision\": \"auto|needs_review|rejected\", "
                "\"confidence\": 0.0, "
                "\"reason\": \"short reason\", "
                "\"normalized_key\": null, "
                "\"normalized_value\": null}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def validate_item(self, db: Session, item: ExtractedItem) -> ValidationDecision:
        matches = self._retrieve_matches(db, item)
        candidate = _candidate_payload(item)
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
                "rag_json": json.dumps(_matches_payload(matches), ensure_ascii=False),
            })
            data = loads_json_relaxed(content)
            decision = ValidationDecision.model_validate({
                "decision": _normalize_decision(data.get("decision")),
                "confidence": clamp_float(data.get("confidence"), default=0.5),
                "reason": str(data.get("reason") or "LangChain validation"),
                "normalized_key": data.get("normalized_key"),
                "normalized_value": data.get("normalized_value"),
            })
        except Exception as exc:
            decision = self._heuristic_decision(item, matches, f"LangChain validation failed: {exc}")

        self._persist_validation(db, item, decision, matches)
        self._apply_decision(db, item, decision)
        return decision

    def _retrieve_matches(self, db: Session, item: ExtractedItem) -> list[RagMatch]:
        if not settings.enable_rag_validation:
            return []
        with AgentTimer("rag.retrieve_for_validation", document_id=item.document_id, item_id=item.id, key=item.key, top_k=settings.rag_top_k) as trace:
            matches = self.rag.retrieve(db, f"{item.key}\n{item.value}\n{item.source_text or ''}", settings.rag_top_k)
            trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
            return matches

    def _heuristic_decision(self, item: ExtractedItem, matches: list[RagMatch], prefix: str = "") -> ValidationDecision:
        confidence = float(item.confidence or 0.5)
        if not item.key or not item.value:
            return ValidationDecision(decision="rejected", confidence=0.0, reason="Empty candidate")
        if len((item.value or "").strip()) < 8:
            return ValidationDecision(decision="rejected", confidence=0.2, reason=f"{prefix}; definition is too short")
        if matches and matches[0].score > 0.88:
            confidence = max(confidence, min(matches[0].score, 0.85))
        return ValidationDecision(decision="needs_review", confidence=confidence, reason=prefix or "heuristic validation")

    def _persist_validation(self, db: Session, item: ExtractedItem, decision: ValidationDecision, matches: list[RagMatch]) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name=self.agent_name,
            decision=decision.decision,
            confidence=decision.confidence,
            reason=decision.reason,
            normalized_key=decision.normalized_key,
            normalized_value=decision.normalized_value,
            rag_evidence={"matches": _matches_payload(matches)},
            payload=_candidate_payload(item),
        ))
        db.commit()

    def _apply_decision(self, db: Session, item: ExtractedItem, decision: ValidationDecision) -> None:
        if decision.decision == "rejected":
            item.status = "rejected"
            item.confidence = min(float(item.confidence or 0.5), decision.confidence)
        elif decision.decision == "auto" and decision.confidence >= settings.validation_auto_threshold:
            item.status = "auto"
            item.confidence = min(max(float(item.confidence or 0.5), decision.confidence), 0.95)
        else:
            item.status = "needs_review"
            item.confidence = min(float(item.confidence or decision.confidence or 0.5), decision.confidence)
        db.commit()
        if item.status == "auto":
            self.rag.add_confirmed_item(db, item, status="auto")


def _candidate_payload(item: ExtractedItem) -> dict[str, Any]:
    return {
        "item_type": item.item_type,
        "key": item.key,
        "value": item.value,
        "source_text": item.source_text,
        "page_from": item.page_from,
        "page_to": item.page_to,
        "current_confidence": item.confidence,
        "current_status": item.status,
        "normalized_key": getattr(item, "normalized_key", None),
        "normalized_value": getattr(item, "normalized_value", None),
        "correction_confidence": getattr(item, "correction_confidence", None),
    }


def _normalize_decision(value: Any) -> str:
    decision = str(value or "needs_review").strip().lower()
    return decision if decision in {"auto", "needs_review", "rejected"} else "needs_review"


def _matches_payload(matches: list[RagMatch]) -> list[dict[str, Any]]:
    return [
        {"term": m.term, "definition": m.definition[:400], "score": m.score, "source_item_id": m.source_item_id}
        for m in matches
    ]
