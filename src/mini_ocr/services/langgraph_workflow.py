from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
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


class LangGraphExtractionWorkflow:
    """Document-level LangGraph workflow.

    The heavy logic is intentionally delegated to small components:
    ExtractionAgent, ExtractionValidator, OCRCorrectionWorkflow, and
    CandidateValidationAgent. This file should contain orchestration only.
    """

    def __init__(self, db: Session, document: Document) -> None:
        self.db = db
        self.document = document
        self.extractor = ExtractionAgent()
        self.validator = ExtractionValidator()
        self.corrector = OCRCorrectionWorkflow() if settings.enable_ocr_correction_agent else None
        self.validation_agent = CandidateValidationAgent() if settings.enable_agent_validation else None
        self.logger = get_logger("langgraph_workflow")
        self.graph = self._build_graph()

    def run(self, candidates: list[SectionCandidate]) -> WorkflowState:
        state: WorkflowState = {
            "document_id": self.document.id,
            "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
            "extracted": [],
            "saved_item_ids": [],
            "errors": [],
        }
        with AgentTimer("langgraph.workflow", document_id=self.document.id, title=self.document.title, candidates_count=len(candidates)) as trace:
            result = self.graph.invoke(state, config={"recursion_limit": 30})
            trace.set(
                extracted_count=len(result.get("extracted", [])),
                saved_count=len(result.get("saved_item_ids", [])),
                errors_count=len(result.get("errors", [])),
            )
            return result

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("extract", self._extract_node)
        graph.add_node("save", self._save_node)
        graph.add_node("normalize", self._normalize_node)
        graph.add_node("validate", self._validate_node)
        graph.set_entry_point("extract")
        graph.add_edge("extract", "save")
        graph.add_edge("save", "normalize")
        graph.add_edge("normalize", "validate")
        graph.add_edge("validate", END)
        return graph.compile()

    def _extract_node(self, state: WorkflowState) -> WorkflowState:
        extracted = list(state.get("extracted", []))
        errors = list(state.get("errors", []))

        for candidate_data in state["candidates"]:
            candidate = _candidate_from_dict(candidate_data)
            input_hash = sha256_text(candidate.text + settings.prompt_version + settings.llm_model)
            job = self._get_or_create_job(candidate, input_hash)
            try:
                result = self._extract_candidate(candidate)
                for item_type, entities in (("abbreviation", result.abbreviations), ("term", result.terms)):
                    for entity in entities:
                        extracted.append({
                            "item_type": item_type,
                            "entity": entity.model_dump(),
                            "page_from": candidate.page_from,
                            "page_to": candidate.page_to,
                            "section_type": candidate.section_type,
                            "chunk_text": candidate.text,
                            "extractor": self.extractor.extractor_name,
                        })
                job.status = "done"
                job.error_message = None
            except Exception as exc:
                job.status = "failed"
                job.error_message = str(exc)
                errors.append(f"candidate {candidate.page_from}-{candidate.page_to}: {exc}")
            finally:
                self.db.commit()

        state["extracted"] = extracted
        state["errors"] = errors
        return state

    def _save_node(self, state: WorkflowState) -> WorkflowState:
        saved_item_ids = list(state.get("saved_item_ids", []))
        input_count = len(state.get("extracted", []))
        kept_count = 0
        skipped_count = 0

        with AgentTimer("workflow.save_node", document_id=self.document.id, extracted_count=input_count) as trace:
            for item in state.get("extracted", []):
                if not _item_allowed_in_section(item["item_type"], item["section_type"]):
                    skipped_count += 1
                    continue

                entity = ExtractedEntity.model_validate(item["entity"])
                decision = self.validator.validate(item["item_type"], entity, item["chunk_text"], item["section_type"])
                if not decision.keep:
                    skipped_count += 1
                    self.logger.info(
                        "deterministic validator skipped candidate: document_id=%s key=%r reason=%s",
                        self.document.id,
                        entity.key,
                        decision.reason,
                    )
                    continue

                row = ExtractedItem(
                    document_id=self.document.id,
                    item_type=item["item_type"],
                    key=entity.key.strip(),
                    value=entity.value.strip(),
                    source_text=(entity.source_text or "").strip() or None,
                    page_from=item["page_from"],
                    page_to=item["page_to"],
                    confidence=decision.confidence,
                    status=decision.status,
                    extractor=item.get("extractor", "langchain_llm"),
                )
                self.db.add(row)
                try:
                    self.db.commit()
                    saved_item_ids.append(row.id)
                    kept_count += 1
                except IntegrityError:
                    self.db.rollback()
                    skipped_count += 1
            trace.set(saved_count=kept_count, skipped_count=skipped_count)

        state["saved_item_ids"] = saved_item_ids
        return state

    def _normalize_node(self, state: WorkflowState) -> WorkflowState:
        if self.corrector is None:
            self.logger.info("normalize node skipped: OCR correction agent disabled")
            return state

        attempted = 0
        normalized = 0
        with AgentTimer("workflow.normalize_node", document_id=self.document.id, items_count=len(state.get("saved_item_ids", []))) as trace:
            for item_id in state.get("saved_item_ids", []):
                row = self.db.get(ExtractedItem, item_id)
                if row is None or not _should_normalize(row):
                    continue
                attempted += 1
                with AgentTimer("agent.ocr_correction", document_id=self.document.id, item_id=row.id, key=row.key, status=row.status, confidence=float(row.confidence or 0.0), model=settings.llm_model) as item_trace:
                    suggestion = self.corrector.normalize_item(self.db, row)
                    item_trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence, changed=suggestion.normalized_key != row.key)
                    if suggestion.normalized_key:
                        normalized += 1
            trace.set(attempted_count=attempted, normalized_count=normalized)
        return state

    def _validate_node(self, state: WorkflowState) -> WorkflowState:
        if self.validation_agent is None:
            self.logger.info("validation node skipped: validation agent disabled")
            return state

        validated = 0
        with AgentTimer("workflow.validate_node", document_id=self.document.id, items_count=len(state.get("saved_item_ids", []))) as trace:
            for item_id in state.get("saved_item_ids", []):
                row = self.db.get(ExtractedItem, item_id)
                if row is None:
                    continue
                with AgentTimer("agent.rag_validation", document_id=self.document.id, item_id=row.id, key=row.key, normalized_key=getattr(row, "normalized_key", None), status=row.status, confidence=float(row.confidence or 0.0), model=settings.llm_model) as item_trace:
                    decision = self.validation_agent.validate_item(self.db, row)
                    item_trace.set(decision=decision.decision, validation_confidence=decision.confidence, reason=decision.reason[:240] if decision.reason else None)
                    validated += 1
            trace.set(validated_count=validated)
        return state

    def _get_or_create_job(self, candidate: SectionCandidate, input_hash: str) -> ExtractionJob:
        job = (
            self.db.query(ExtractionJob)
            .filter_by(document_id=self.document.id, section_type=candidate.section_type, input_text_hash=input_hash, prompt_version=settings.prompt_version, model_name=settings.llm_model)
            .first()
        )
        if job is None:
            job = ExtractionJob(
                document_id=self.document.id,
                section_type=candidate.section_type,
                page_from=candidate.page_from,
                page_to=candidate.page_to,
                input_text_hash=input_hash,
                prompt_version=settings.prompt_version,
                model_name=settings.llm_model,
                status="running",
            )
            self.db.add(job)
        else:
            job.status = "running"
            job.error_message = None
        self.db.commit()
        return job

    def _extract_candidate(self, candidate: SectionCandidate):
        with AgentTimer("agent.extractor", document_id=self.document.id, section_type=candidate.section_type, page_from=candidate.page_from, page_to=candidate.page_to, text_chars=len(candidate.text or ""), model=settings.llm_model, timeout_seconds=settings.llm_timeout_seconds) as trace:
            result = self.extractor.extract(candidate)
            trace.set(abbreviations_count=len(result.abbreviations), terms_count=len(result.terms))
            return result


def _should_normalize(row: ExtractedItem) -> bool:
    return row.status == "needs_review" or float(row.confidence or 0.0) < 0.75 or looks_ocr_noisy(row.key)


def _item_allowed_in_section(item_type: str, section_type: str) -> bool:
    allowed_types = {"abbreviations": {"abbreviation"}, "terms": {"term"}, "mixed": {"abbreviation", "term"}}
    return item_type in allowed_types.get(section_type, {"abbreviation", "term"})


def _candidate_to_dict(candidate: SectionCandidate) -> dict[str, Any]:
    return {
        "section_type": candidate.section_type,
        "text": candidate.text,
        "page_from": candidate.page_from,
        "page_to": candidate.page_to,
        "score": candidate.score,
        "title": candidate.title,
        "source": candidate.source,
        "layout_type": candidate.layout_type,
    }


def _candidate_from_dict(data: dict[str, Any]) -> SectionCandidate:
    return SectionCandidate(
        section_type=data["section_type"],
        text=data["text"],
        page_from=int(data["page_from"]),
        page_to=int(data["page_to"]),
        score=int(data.get("score") or 0),
        title=data.get("title"),
        source=data.get("source") or "header",
        layout_type=data.get("layout_type"),
    )
