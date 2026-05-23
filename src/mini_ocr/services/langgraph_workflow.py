from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import Document, ExtractedItem, ExtractionJob
from mini_ocr.schemas.extraction import ExtractedEntity
from mini_ocr.services.agents.correction import OCRCorrectionWorkflow
from mini_ocr.services.agents.extraction import ExtractionAgent
from mini_ocr.services.agents.validation import CandidateValidationAgent
from mini_ocr.services.extraction_validator import ExtractionValidator
from mini_ocr.services.hash_utils import sha256_text
from mini_ocr.services.observability import AgentTimer, get_logger
from mini_ocr.services.section_detector import SectionCandidate
from mini_ocr.utils.text import looks_ocr_noisy


class WorkflowState(TypedDict):
    document_id: str
    candidates: list[dict[str, Any]]
    extracted: list[dict[str, Any]]
    saved_item_ids: list[str]
    errors: list[str]


class ExtractedCandidate(BaseModel):
    """Pydantic payload passed between LangGraph nodes.

    LangGraph state is JSON-like, but we should not maintain hand-written
    _to_dict/_from_dict pairs. Pydantic is the boundary: nodes receive dicts
    from the graph and immediately validate them into typed models.
    """

    item_type: str
    entity: ExtractedEntity
    page_from: int
    page_to: int
    section_type: str
    chunk_text: str
    extractor: str = "langchain_llm"


@dataclass(slots=True)
class WorkflowServices:
    db: Session
    document: Document
    extractor: ExtractionAgent
    validator: ExtractionValidator
    corrector: OCRCorrectionWorkflow | None
    validation_agent: CandidateValidationAgent | None
    logger: Any


class WorkflowNode(Protocol):
    def __call__(self, state: WorkflowState) -> WorkflowState:
        ...


class LangGraphExtractionWorkflow:
    """Document-level LangGraph workflow.

    This class only wires graph nodes. Node implementation lives in small
    command objects below; each node owns one stage and delegates its internal
    checks to policies/strategies.
    """

    def __init__(self, db: Session, document: Document) -> None:
        self.services = WorkflowServices(
            db=db,
            document=document,
            extractor=ExtractionAgent(),
            validator=ExtractionValidator(),
            corrector=OCRCorrectionWorkflow() if settings.enable_ocr_correction_agent else None,
            validation_agent=CandidateValidationAgent() if settings.enable_agent_validation else None,
            logger=get_logger("langgraph_workflow"),
        )
        self.graph = self._build_graph()

    def run(self, candidates: list[SectionCandidate]) -> WorkflowState:
        state: WorkflowState = {
            "document_id": self.services.document.id,
            "candidates": [candidate.model_dump() for candidate in candidates],
            "extracted": [],
            "saved_item_ids": [],
            "errors": [],
        }
        with AgentTimer(
            "langgraph.workflow",
            document_id=self.services.document.id,
            title=self.services.document.title,
            candidates_count=len(candidates),
        ) as trace:
            result = self.graph.invoke(state, config={"recursion_limit": 30})
            trace.set(
                extracted_count=len(result.get("extracted", [])),
                saved_count=len(result.get("saved_item_ids", [])),
                errors_count=len(result.get("errors", [])),
            )
            return result

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("extract", ExtractNode(self.services))
        graph.add_node("save", SaveNode(self.services))
        graph.add_node("normalize", NormalizeNode(self.services, default_normalization_policy()))
        graph.add_node("validate", ValidateNode(self.services))
        graph.set_entry_point("extract")
        graph.add_edge("extract", "save")
        graph.add_edge("save", "normalize")
        graph.add_edge("normalize", "validate")
        graph.add_edge("validate", END)
        return graph.compile()


class ExtractNode:
    def __init__(self, services: WorkflowServices) -> None:
        self.s = services

    def __call__(self, state: WorkflowState) -> WorkflowState:
        extracted = [ExtractedCandidate.model_validate(item) for item in state.get("extracted", [])]
        errors = list(state.get("errors", []))

        for candidate_data in state["candidates"]:
            candidate = SectionCandidate.model_validate(candidate_data)
            job = self._start_job(candidate)
            try:
                result = self._extract_candidate(candidate)
                extracted.extend(_result_to_candidates(result, candidate, self.s.extractor.extractor_name))
                job.status = "done"
                job.error_message = None
            except Exception as exc:
                job.status = "failed"
                job.error_message = str(exc)
                errors.append(f"candidate {candidate.page_from}-{candidate.page_to}: {exc}")
            finally:
                self.s.db.commit()

        state["extracted"] = [item.model_dump() for item in extracted]
        state["errors"] = errors
        return state

    def _start_job(self, candidate: SectionCandidate) -> ExtractionJob:
        input_hash = sha256_text(candidate.text + settings.prompt_version + settings.llm_model)
        job = (
            self.s.db.query(ExtractionJob)
            .filter_by(
                document_id=self.s.document.id,
                section_type=candidate.section_type,
                input_text_hash=input_hash,
                prompt_version=settings.prompt_version,
                model_name=settings.llm_model,
            )
            .first()
        )
        if job is None:
            job = ExtractionJob(
                document_id=self.s.document.id,
                section_type=candidate.section_type,
                page_from=candidate.page_from,
                page_to=candidate.page_to,
                input_text_hash=input_hash,
                prompt_version=settings.prompt_version,
                model_name=settings.llm_model,
                status="running",
            )
            self.s.db.add(job)
        else:
            job.status = "running"
            job.error_message = None
        self.s.db.commit()
        return job

    def _extract_candidate(self, candidate: SectionCandidate):
        with AgentTimer(
            "agent.extractor",
            document_id=self.s.document.id,
            section_type=candidate.section_type,
            page_from=candidate.page_from,
            page_to=candidate.page_to,
            text_chars=len(candidate.text or ""),
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ) as trace:
            result = self.s.extractor.extract(candidate)
            trace.set(abbreviations_count=len(result.abbreviations), terms_count=len(result.terms))
            return result


class SaveNode:
    def __init__(self, services: WorkflowServices) -> None:
        self.s = services

    def __call__(self, state: WorkflowState) -> WorkflowState:
        saved_item_ids = list(state.get("saved_item_ids", []))
        candidates = [ExtractedCandidate.model_validate(item) for item in state.get("extracted", [])]
        kept_count = 0
        skipped_count = 0

        with AgentTimer("workflow.save_node", document_id=self.s.document.id, extracted_count=len(candidates)) as trace:
            for candidate in candidates:
                decision = self.s.validator.validate(
                    candidate.item_type,
                    candidate.entity,
                    candidate.chunk_text,
                    candidate.section_type,
                )
                if not decision.keep:
                    skipped_count += 1
                    self.s.logger.info(
                        "deterministic validator skipped candidate: document_id=%s key=%r reason=%s",
                        self.s.document.id,
                        candidate.entity.key,
                        decision.reason,
                    )
                    continue

                row = _candidate_to_row(self.s.document.id, candidate, decision.confidence, decision.status)
                self.s.db.add(row)
                try:
                    self.s.db.commit()
                    saved_item_ids.append(row.id)
                    kept_count += 1
                except IntegrityError:
                    self.s.db.rollback()
                    skipped_count += 1

            trace.set(saved_count=kept_count, skipped_count=skipped_count)

        state["saved_item_ids"] = saved_item_ids
        return state


class NormalizeNode:
    def __init__(self, services: WorkflowServices, policy: NormalizationPolicy) -> None:
        self.s = services
        self.policy = policy

    def __call__(self, state: WorkflowState) -> WorkflowState:
        if self.s.corrector is None:
            self.s.logger.info("normalize node skipped: OCR correction agent disabled")
            return state

        attempted = 0
        normalized = 0
        item_ids = list(state.get("saved_item_ids", []))
        with AgentTimer("workflow.normalize_node", document_id=self.s.document.id, items_count=len(item_ids)) as trace:
            for item_id in item_ids:
                row = self.s.db.get(ExtractedItem, item_id)
                if row is None or not self.policy.should_normalize(row):
                    continue
                attempted += 1
                normalized += self._normalize_row(row)
            trace.set(attempted_count=attempted, normalized_count=normalized)
        return state

    def _normalize_row(self, row: ExtractedItem) -> int:
        with AgentTimer(
            "agent.ocr_correction",
            document_id=self.s.document.id,
            item_id=row.id,
            key=row.key,
            status=row.status,
            confidence=float(row.confidence or 0.0),
            model=settings.llm_model,
        ) as trace:
            suggestion = self.s.corrector.normalize_item(self.s.db, row)  # type: ignore[union-attr]
            changed = suggestion.normalized_key != row.key
            trace.set(
                normalized_key=suggestion.normalized_key,
                correction_confidence=suggestion.confidence,
                changed=changed,
            )
            return int(bool(suggestion.normalized_key))


class ValidateNode:
    def __init__(self, services: WorkflowServices) -> None:
        self.s = services

    def __call__(self, state: WorkflowState) -> WorkflowState:
        if self.s.validation_agent is None:
            self.s.logger.info("validation node skipped: validation agent disabled")
            return state

        validated = 0
        item_ids = list(state.get("saved_item_ids", []))
        with AgentTimer("workflow.validate_node", document_id=self.s.document.id, items_count=len(item_ids)) as trace:
            for item_id in item_ids:
                row = self.s.db.get(ExtractedItem, item_id)
                if row is None:
                    continue
                validated += self._validate_row(row)
            trace.set(validated_count=validated)
        return state

    def _validate_row(self, row: ExtractedItem) -> int:
        with AgentTimer(
            "agent.rag_validation",
            document_id=self.s.document.id,
            item_id=row.id,
            key=row.key,
            normalized_key=getattr(row, "normalized_key", None),
            status=row.status,
            confidence=float(row.confidence or 0.0),
            model=settings.llm_model,
        ) as trace:
            decision = self.s.validation_agent.validate_item(self.s.db, row)  # type: ignore[union-attr]
            trace.set(
                decision=decision.decision,
                validation_confidence=decision.confidence,
                reason=decision.reason[:240] if decision.reason else None,
            )
            return 1


class NormalizationStrategy(Protocol):
    def should_normalize(self, item: ExtractedItem) -> bool:
        ...


class NeedsReviewNormalizationStrategy:
    def should_normalize(self, item: ExtractedItem) -> bool:
        return item.status == "needs_review"


class LowConfidenceNormalizationStrategy:
    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def should_normalize(self, item: ExtractedItem) -> bool:
        return float(item.confidence or 0.0) < self.threshold


class OCRNoisyNormalizationStrategy:
    def should_normalize(self, item: ExtractedItem) -> bool:
        return looks_ocr_noisy(item.key)


class NormalizationPolicy:
    """OR-composition of independent normalization strategies."""

    def __init__(self, strategies: list[NormalizationStrategy]) -> None:
        self.strategies = strategies

    def should_normalize(self, item: ExtractedItem) -> bool:
        return any(strategy.should_normalize(item) for strategy in self.strategies)


def default_normalization_policy() -> NormalizationPolicy:
    return NormalizationPolicy([
        NeedsReviewNormalizationStrategy(),
        LowConfidenceNormalizationStrategy(threshold=0.75),
        OCRNoisyNormalizationStrategy(),
    ])


def _result_to_candidates(result: Any, candidate: SectionCandidate, extractor_name: str) -> list[ExtractedCandidate]:
    items: list[ExtractedCandidate] = []
    for item_type, entities in (("abbreviation", result.abbreviations), ("term", result.terms)):
        for entity in entities:
            items.append(
                ExtractedCandidate(
                    item_type=item_type,
                    entity=entity,
                    page_from=candidate.page_from,
                    page_to=candidate.page_to,
                    section_type=candidate.section_type,
                    chunk_text=candidate.text,
                    extractor=extractor_name,
                )
            )
    return items


def _candidate_to_row(document_id: str, candidate: ExtractedCandidate, confidence: float, status: str) -> ExtractedItem:
    entity = candidate.entity
    return ExtractedItem(
        document_id=document_id,
        item_type=candidate.item_type,
        key=entity.key.strip(),
        value=entity.value.strip(),
        source_text=(entity.source_text or "").strip() or None,
        page_from=candidate.page_from,
        page_to=candidate.page_to,
        confidence=confidence,
        status=status,
        extractor=candidate.extractor,
    )
