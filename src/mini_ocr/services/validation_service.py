from __future__ import annotations

from typing import Protocol, Sequence

from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import ExtractedItem, ItemValidation
from mini_ocr.services.validation_models import (
    RagMatchLike,
    ValidationCandidate,
    ValidationDecision,
    rag_evidence_from_matches,
)
from mini_ocr.services.validation_policies import (
    SequentialValidationDecisionPolicy,
    ValidationContext,
    DecisionApplicationPolicy,
    default_decision_application_policy,
    default_heuristic_validation_policy,
)
from mini_ocr.services.observability import AgentTimer
from mini_ocr.services.rag_store import RagStore


class RagRetriever(Protocol):
    def retrieve(self, db: Session, query: str, top_k: int | None = None) -> Sequence[RagMatchLike]:
        ...


class RagIndexer(Protocol):
    def add_confirmed_item(self, db: Session, item: ExtractedItem, status: str = "auto") -> None:
        ...


class ValidationRepository:
    """Persistence boundary for validation records and item status updates."""

    def __init__(
        self,
        retriever: RagRetriever | None = None,
        indexer: RagIndexer | None = None,
        application_policy: DecisionApplicationPolicy | None = None,
    ) -> None:
        rag = RagStore()
        self.retriever = retriever or rag
        self.indexer = indexer or rag
        self.application_policy = application_policy or default_decision_application_policy(settings.validation_auto_threshold)

    def retrieve_matches(self, db: Session, item: ExtractedItem) -> Sequence[RagMatchLike]:
        if not settings.enable_rag_validation:
            return []
        query = f"{item.key}\n{item.value}\n{item.source_text or ''}"
        with AgentTimer("rag.retrieve_for_validation", document_id=item.document_id, item_id=item.id, key=item.key, top_k=settings.rag_top_k) as trace:
            matches = list(self.retriever.retrieve(db, query, settings.rag_top_k))
            trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
            return matches

    def persist_validation(
        self,
        db: Session,
        item: ExtractedItem,
        agent_name: str,
        candidate: ValidationCandidate,
        decision: ValidationDecision,
        matches: Sequence[RagMatchLike],
    ) -> None:
        evidence = rag_evidence_from_matches(matches)
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name=agent_name,
            decision=decision.decision,
            confidence=decision.confidence,
            reason=decision.reason,
            normalized_key=decision.normalized_key,
            normalized_value=decision.normalized_value,
            rag_evidence={"matches": [match.model_dump() for match in evidence]},
            payload=candidate.model_dump(),
        ))
        db.commit()

    def apply_decision(self, db: Session, item: ExtractedItem, decision: ValidationDecision) -> None:
        self.application_policy.apply(item, decision)
        db.commit()
        if item.status == "auto":
            self.indexer.add_confirmed_item(db, item, status="auto")


class CandidateValidationService:
    """Application service that wires pure validation agent to DB and RAG."""

    def __init__(
        self,
        agent: CandidateValidationAgent | None = None,
        repository: ValidationRepository | None = None,
        fallback_policy: SequentialValidationDecisionPolicy | None = None,
    ) -> None:
        if agent is None:
            from mini_ocr.services.agents.validation import CandidateValidationAgent

            agent = CandidateValidationAgent()
        self.agent = agent
        self.repository = repository or ValidationRepository()
        self.fallback_policy = fallback_policy or default_heuristic_validation_policy()

    @property
    def agent_name(self) -> str:
        return self.agent.agent_name

    def validate_item(self, db: Session, item: ExtractedItem) -> ValidationDecision:
        matches = self.repository.retrieve_matches(db, item)
        candidate = ValidationCandidate.model_validate(item, from_attributes=True)
        try:
            decision = self.agent.validate(candidate, matches)
        except Exception as exc:
            ctx = ValidationContext.from_item(item, matches, reason_prefix=f"LangChain validation failed: {exc}")
            decision = self.fallback_policy.decide(ctx)

        self.repository.persist_validation(db, item, self.agent_name, candidate, decision, matches)
        self.repository.apply_decision(db, item, decision)
        return decision


# Backwards-compatible alias for old imports. It is intentionally the service
# now, not the pure LLM agent. New code should prefer CandidateValidationService.
CandidateValidationWorkflow = CandidateValidationService
