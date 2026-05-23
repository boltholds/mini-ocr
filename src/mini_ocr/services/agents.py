from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import ExtractedItem, ItemValidation
from mini_ocr.services.rag_store import RagStore, RagMatch


@dataclass
class AgentDecision:
    decision: str
    confidence: float
    reason: str
    normalized_key: str | None = None
    normalized_value: str | None = None
    rag_matches: list[dict[str, Any]] | None = None


class CandidateValidationAgent:
    """Agentic validator for already extracted candidates.

    It does not extract new terms from the document. It only checks a candidate
    against source_text and RAG evidence, then decides auto / needs_review / rejected.
    """

    def __init__(self) -> None:
        self.rag = RagStore()
        self.client: OpenAI | None = None
        if settings.enable_agent_validation:
            base_url = settings.llm_base_url or "http://localhost:11434/v1"
            api_key = settings.llm_api_key or "ollama"
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def validate_item(self, db: Session, item: ExtractedItem) -> AgentDecision:
        query = f"{item.key}\n{item.value}\n{item.source_text or ''}"
        matches = self.rag.retrieve(db, query, settings.rag_top_k) if settings.enable_rag_validation else []

        # Deterministic floor: empty or clearly non-grounded candidates stay review/reject
        if not item.key or not item.value:
            return AgentDecision("rejected", 0.0, "Empty candidate", rag_matches=_matches_payload(matches))

        if self.client is None:
            return self._heuristic_decision(item, matches)

        try:
            decision = self._llm_decision(item, matches)
        except Exception as exc:
            decision = self._heuristic_decision(item, matches)
            decision.reason = f"Heuristic fallback after agent error: {exc}. {decision.reason}"

        self._persist_validation(db, item, decision)

        # Agent can only downgrade automatically. A human/explicit endpoint may approve later.
        if decision.decision == "rejected":
            item.status = "rejected"
            item.confidence = min(item.confidence or decision.confidence, decision.confidence)
        elif decision.decision == "auto":
            item.status = "auto" if decision.confidence >= 0.85 else "needs_review"
            item.confidence = min(max(item.confidence or 0.5, decision.confidence), 0.95)
        else:
            item.status = "needs_review"
            item.confidence = min(item.confidence or decision.confidence, decision.confidence)
        db.commit()

        if item.status == "auto":
            self.rag.add_confirmed_item(db, item, status="auto")
        return decision

    def _llm_decision(self, item: ExtractedItem, matches: list[RagMatch]) -> AgentDecision:
        system = (
            "You are a strict validation agent for OCR document extraction. "
            "You do not extract new terms. You validate one candidate against its OCR source. "
            "OCR may confuse Russian and Latin letters. Do not reject only because the key has Latin-looking characters. "
            "Return only JSON."
        )
        user = {
            "task": "Validate candidate term/abbreviation extracted from OCR text.",
            "rules": [
                "decision must be one of: auto, needs_review, rejected",
                "auto only when key/value are clearly grounded in source_text and look like a term-definition pair",
                "needs_review when OCR noise is high but the candidate may be real",
                "rejected for service phrases, empty values, unrelated text, or obvious hallucination",
                "normalized_key may be provided when OCR distortion is likely, but do not invent it if uncertain",
                "use rag_matches as hints, not as proof",
            ],
            "candidate": {
                "item_type": item.item_type,
                "key": item.key,
                "value": item.value,
                "source_text": item.source_text,
                "page_from": item.page_from,
                "page_to": item.page_to,
                "current_confidence": item.confidence,
            },
            "rag_matches": _matches_payload(matches),
            "output_schema": {
                "decision": "auto|needs_review|rejected",
                "confidence": "0..1",
                "reason": "short explanation",
                "normalized_key": "string or null",
                "normalized_value": "string or null",
            },
        }
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
            temperature=0,
        )
        data = _loads_json_relaxed(response.choices[0].message.content or "{}")
        decision = str(data.get("decision") or "needs_review").strip().lower()
        if decision not in {"auto", "needs_review", "rejected"}:
            decision = "needs_review"
        confidence = _safe_float(data.get("confidence"), default=0.5)
        confidence = max(0.0, min(1.0, confidence))
        return AgentDecision(
            decision=decision,
            confidence=confidence,
            reason=str(data.get("reason") or "agent validation"),
            normalized_key=data.get("normalized_key"),
            normalized_value=data.get("normalized_value"),
            rag_matches=_matches_payload(matches),
        )

    def _heuristic_decision(self, item: ExtractedItem, matches: list[RagMatch]) -> AgentDecision:
        key = item.key or ""
        value = item.value or ""
        source = item.source_text or ""
        confidence = float(item.confidence or 0.5)
        reason = "heuristic validation"
        decision = "needs_review"

        if not source or key.lower() not in source.lower():
            confidence = min(confidence, 0.49)
            reason = "candidate is weakly grounded in source_text"
        if len(value) < 12:
            decision = "rejected"
            confidence = min(confidence, 0.25)
            reason = "definition is too short"
        if matches and matches[0].score > 0.88:
            confidence = max(confidence, min(0.85, matches[0].score))
            reason = f"similar term found in RAG: {matches[0].term}"

        return AgentDecision(decision, confidence, reason, rag_matches=_matches_payload(matches))

    def _persist_validation(self, db: Session, item: ExtractedItem, decision: AgentDecision) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name="candidate_validation_agent",
            decision=decision.decision,
            confidence=decision.confidence,
            reason=decision.reason,
            normalized_key=decision.normalized_key,
            normalized_value=decision.normalized_value,
            rag_evidence={"matches": decision.rag_matches or []},
            payload={
                "key": item.key,
                "value": item.value,
                "source_text": item.source_text,
                "page_from": item.page_from,
                "page_to": item.page_to,
            },
        ))
        db.commit()


def _matches_payload(matches: list[RagMatch]) -> list[dict[str, Any]]:
    return [
        {"term": m.term, "definition": m.definition[:400], "score": m.score, "source_item_id": m.source_item_id}
        for m in matches
    ]


def _loads_json_relaxed(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except Exception:
        return default
