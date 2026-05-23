from __future__ import annotations

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from mini_ocr.utils.text import clamp_float


class ValidationDecision(BaseModel):
    decision: str = Field(description="auto | needs_review | rejected")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    normalized_key: str | None = None
    normalized_value: str | None = None


class ValidationCandidate(BaseModel):
    """Typed candidate boundary for validation.

    It can be built directly from an ExtractedItem ORM object through
    model_validate(..., from_attributes=True), so validation code does not need
    handwritten _candidate_payload helpers.
    """

    model_config = ConfigDict(from_attributes=True)

    item_type: str
    key: str
    value: str
    source_text: str | None = None
    page_from: int | None = None
    page_to: int | None = None
    confidence: float | None = None
    status: str | None = None
    normalized_key: str | None = None
    normalized_value: str | None = None
    correction_confidence: float | None = None


class RagEvidence(BaseModel):
    term: str
    definition: str
    score: float
    source_item_id: str | None = None


class RagMatchLike(Protocol):
    term: str
    definition: str
    score: float
    source_item_id: str | None


def rag_evidence_from_matches(matches: Sequence[RagMatchLike]) -> list[RagEvidence]:
    return [
        RagEvidence(
            term=match.term,
            definition=(match.definition or "")[:400],
            score=float(match.score),
            source_item_id=match.source_item_id,
        )
        for match in matches
    ]


def normalize_decision(value: Any) -> str:
    decision = str(value or "needs_review").strip().lower()
    return decision if decision in {"auto", "needs_review", "rejected"} else "needs_review"


def decision_from_agent_payload(data: dict[str, Any]) -> ValidationDecision:
    return ValidationDecision.model_validate({
        "decision": normalize_decision(data.get("decision")),
        "confidence": clamp_float(data.get("confidence"), default=0.5),
        "reason": str(data.get("reason") or "LangChain validation"),
        "normalized_key": data.get("normalized_key"),
        "normalized_value": data.get("normalized_value"),
    })
