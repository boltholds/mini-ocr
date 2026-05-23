from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import Document, ExtractedItem, ExtractionJob, ItemValidation
from mini_ocr.schemas.extraction import ExtractedEntity, ExtractionResult
from mini_ocr.services.extraction_validator import ExtractionValidator
from mini_ocr.services.hash_utils import sha256_text
from mini_ocr.services.rag_store import RagMatch, RagStore
from mini_ocr.services.section_detector import SectionCandidate
from mini_ocr.services.llm.prompt import SYSTEM_PROMPT


class WorkflowState(TypedDict):
    document_id: str
    candidates: list[dict[str, Any]]
    extracted: list[dict[str, Any]]
    saved_item_ids: list[str]
    errors: list[str]


class ValidationDecision(BaseModel):
    decision: str = Field(description="auto | needs_review | rejected")
    confidence: float = Field(ge=0, le=1)
    reason: str
    normalized_key: str | None = None
    normalized_value: str | None = None


class OCRCorrectionDecision(BaseModel):
    normalized_key: str | None = None
    normalized_value: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str = "no correction"
    changed: bool = False


class LangChainExtractor:
    """LangChain-based structured extraction chain.

    Ollama's OpenAI-compatible endpoint is often more reliable with plain JSON
    instructions than with provider-specific JSON mode. We therefore parse JSON
    ourselves and validate it with Pydantic.
    """

    extractor_name = "langchain_llm"

    def __init__(self) -> None:
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "OCR fragment metadata:\n"
                "section_type={section_type}\n"
                "page_from={page_from}\n"
                "page_to={page_to}\n\n"
                "OCR text:\n{text}\n\n"
                "Return only a JSON object with keys 'abbreviations' and 'terms'.",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def extract(self, candidate: SectionCandidate) -> ExtractionResult:
        content = self.chain.invoke({
            "section_type": candidate.section_type,
            "page_from": candidate.page_from,
            "page_to": candidate.page_to,
            "text": candidate.text[:30000],
        })
        data = _loads_json_relaxed(content)
        result = ExtractionResult.model_validate(data)
        return self._ground_to_source(result, candidate.text)

    def _ground_to_source(self, result: ExtractionResult, source: str) -> ExtractionResult:
        normalized_source = _compact(source).lower()
        for group in (result.abbreviations, result.terms):
            for item in group:
                evidence = item.source_text or f"{item.key} {item.value}"
                grounded = _compact(evidence).lower() in normalized_source
                if not grounded:
                    item.confidence = min(item.confidence or 0.5, 0.49)
        return result


class LangChainOCRCorrectionAgent:
    """Context-aware OCR correction agent for already extracted candidates.

    This agent does not create new items. It only proposes normalized_key /
    normalized_value for an existing extracted item when the OCR distortion is
    likely and the correction is supported by source_text and optional RAG hints.
    The original key/value/source_text remain unchanged for auditability.
    """

    agent_name = "ocr_correction_agent"

    def __init__(self) -> None:
        self.rag = RagStore()
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an OCR correction agent for Russian technical standards. "
                "You correct OCR distortions in already extracted term candidates. "
                "Do not extract new terms. Do not change meaning. "
                "OCR may confuse Cyrillic and Latin letters, merge words, split words, and miss letters. "
                "Return only JSON.",
            ),
            (
                "human",
                "Task: propose OCR normalization for one extracted candidate.\n\n"
                "Rules:\n"
                "- Preserve the original key and value; only propose normalized_key / normalized_value.\n"
                "- Correct only obvious OCR distortions supported by source_text, value and RAG hints.\n"
                "- If correction is uncertain, return normalized_key=null and normalized_value=null.\n"
                "- Do not invent facts or definitions.\n"
                "- RAG matches are hints, not proof.\n\n"
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Return JSON with fields: normalized_key, normalized_value, confidence, reason, changed.",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def correct_item(self, db: Session, item: ExtractedItem) -> OCRCorrectionDecision:
        matches = self.rag.retrieve(
            db,
            f"{item.key}\n{item.value}\n{item.source_text or ''}",
            settings.rag_top_k,
        ) if settings.enable_rag_validation else []

        candidate = {
            "item_type": item.item_type,
            "key": item.key,
            "value": item.value,
            "source_text": item.source_text,
            "page_from": item.page_from,
            "page_to": item.page_to,
            "current_confidence": item.confidence,
            "current_status": item.status,
        }
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
                "rag_json": json.dumps(_matches_payload(matches), ensure_ascii=False),
            })
            data = _loads_json_relaxed(content)
            decision = OCRCorrectionDecision.model_validate({
                "normalized_key": _none_if_blank(data.get("normalized_key")),
                "normalized_value": _none_if_blank(data.get("normalized_value")),
                "confidence": _clamp_float(data.get("confidence"), default=0.0),
                "reason": str(data.get("reason") or "OCR correction"),
                "changed": bool(data.get("changed")),
            })
        except Exception as exc:
            decision = self._heuristic_correction(item, matches, f"OCR correction failed: {exc}")

        # Be conservative: only store correction when useful and sufficiently supported.
        if decision.confidence < 0.55:
            decision.changed = False
            if not decision.normalized_key and not decision.normalized_value:
                decision.reason = f"{decision.reason}; correction confidence below threshold"

        self._persist_correction(db, item, decision, matches)
        self._apply_correction(db, item, decision)
        return decision

    def _heuristic_correction(self, item: ExtractedItem, matches: list[RagMatch], reason: str) -> OCRCorrectionDecision:
        if matches and matches[0].score > 0.90:
            return OCRCorrectionDecision(
                normalized_key=matches[0].term,
                normalized_value=matches[0].definition,
                confidence=min(matches[0].score, 0.88),
                reason=f"{reason}; high-similarity RAG match found",
                changed=True,
            )
        return OCRCorrectionDecision(confidence=0.0, reason=reason, changed=False)

    def _persist_correction(self, db: Session, item: ExtractedItem, decision: OCRCorrectionDecision, matches: list[RagMatch]) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name=self.agent_name,
            decision="normalized" if decision.changed else "unchanged",
            confidence=decision.confidence,
            reason=decision.reason,
            normalized_key=decision.normalized_key,
            normalized_value=decision.normalized_value,
            rag_evidence={"matches": _matches_payload(matches)},
            payload={
                "key": item.key,
                "value": item.value,
                "source_text": item.source_text,
                "page_from": item.page_from,
                "page_to": item.page_to,
            },
        ))
        db.commit()

    def _apply_correction(self, db: Session, item: ExtractedItem, decision: OCRCorrectionDecision) -> None:
        if decision.changed:
            if decision.normalized_key:
                item.normalized_key = decision.normalized_key
            if decision.normalized_value:
                item.normalized_value = decision.normalized_value
            item.correction_confidence = decision.confidence
            item.correction_reason = decision.reason
            # A corrected item should usually be reviewed unless validation later accepts it.
            if item.status == "auto" and decision.confidence < 0.9:
                item.status = "needs_review"
        db.commit()


class LangChainCandidateValidationAgent:
    """RAG-assisted LangChain validation chain for one saved candidate."""

    agent_name = "langchain_rag_validation_agent"

    def __init__(self) -> None:
        self.rag = RagStore()
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict validation agent for OCR extraction. "
                "You validate one already extracted candidate. "
                "Do not extract new terms from the document. "
                "OCR may confuse Russian and Latin letters, so do not reject only because the key contains Latin-looking symbols. "
                "Return only JSON.",
            ),
            (
                "human",
                "Task: validate candidate term/abbreviation from OCR.\n\n"
                "Rules:\n"
                "- decision must be exactly one of: auto, needs_review, rejected.\n"
                "- auto only when the candidate is clearly grounded in source_text and the definition is clean.\n"
                "- needs_review when the candidate may be real but OCR noise is high.\n"
                "- rejected for service phrases, empty/unrelated text, or hallucinations.\n"
                "- If OCR distortion is likely, preserve original key and propose normalized_key only when obvious.\n"
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
        matches = self.rag.retrieve(
            db,
            f"{item.key}\n{item.value}\n{item.source_text or ''}",
            settings.rag_top_k,
        ) if settings.enable_rag_validation else []

        candidate = {
            "item_type": item.item_type,
            "key": item.key,
            "value": item.value,
            "source_text": item.source_text,
            "page_from": item.page_from,
            "page_to": item.page_to,
            "current_confidence": item.confidence,
            "current_status": item.status,
        }
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
                "rag_json": json.dumps(_matches_payload(matches), ensure_ascii=False),
            })
            data = _loads_json_relaxed(content)
            decision = ValidationDecision.model_validate({
                "decision": _normalize_decision(data.get("decision")),
                "confidence": _clamp_float(data.get("confidence"), default=0.5),
                "reason": str(data.get("reason") or "LangChain validation"),
                "normalized_key": data.get("normalized_key"),
                "normalized_value": data.get("normalized_value"),
            })
        except Exception as exc:
            decision = self._heuristic_decision(item, matches, f"LangChain validation failed: {exc}")

        self._persist_validation(db, item, decision, matches)
        self._apply_decision(db, item, decision)
        return decision

    def _heuristic_decision(self, item: ExtractedItem, matches: list[RagMatch], prefix: str = "") -> ValidationDecision:
        confidence = float(item.confidence or 0.5)
        decision = "needs_review"
        reason = prefix or "heuristic validation"
        if not item.key or not item.value:
            return ValidationDecision(decision="rejected", confidence=0.0, reason="Empty candidate")
        if not item.source_text:
            confidence = min(confidence, 0.45)
            reason = f"{reason}; source_text is missing"
        if len((item.value or '').strip()) < 8:
            return ValidationDecision(decision="rejected", confidence=0.2, reason=f"{reason}; definition is too short")
        if matches and matches[0].score > 0.88:
            confidence = max(confidence, min(matches[0].score, 0.85))
            reason = f"{reason}; similar confirmed term found: {matches[0].term}"
        return ValidationDecision(decision=decision, confidence=confidence, reason=reason)

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
            payload={
                "key": item.key,
                "value": item.value,
                "source_text": item.source_text,
                "page_from": item.page_from,
                "page_to": item.page_to,
            },
        ))
        db.commit()

    def _apply_decision(self, db: Session, item: ExtractedItem, decision: ValidationDecision) -> None:
        if decision.decision == "rejected":
            item.status = "rejected"
            item.confidence = min(float(item.confidence or 0.5), decision.confidence)
        elif decision.decision == "auto" and decision.confidence >= 0.85:
            item.status = "auto"
            item.confidence = min(max(float(item.confidence or 0.5), decision.confidence), 0.95)
        else:
            item.status = "needs_review"
            item.confidence = min(float(item.confidence or decision.confidence or 0.5), decision.confidence)
        db.commit()
        if item.status == "auto":
            self.rag.add_confirmed_item(db, item, status="auto")


class LangGraphExtractionWorkflow:
    """LangGraph orchestration for extraction + deterministic guardrails + RAG validation."""

    def __init__(self, db: Session, document: Document) -> None:
        self.db = db
        self.document = document
        self.extractor = LangChainExtractor()
        self.validator = ExtractionValidator()
        self.corrector = LangChainOCRCorrectionAgent() if settings.enable_ocr_correction_agent else None
        self.agent = LangChainCandidateValidationAgent() if settings.enable_agent_validation else None
        self.graph = self._build_graph()

    def run(self, candidates: list[SectionCandidate]) -> WorkflowState:
        state: WorkflowState = {
            "document_id": self.document.id,
            "candidates": [_candidate_to_dict(c) for c in candidates],
            "extracted": [],
            "saved_item_ids": [],
            "errors": [],
        }
        return self.graph.invoke(state)

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
        extracted: list[dict[str, Any]] = list(state.get("extracted", []))
        errors: list[str] = list(state.get("errors", []))

        for candidate_data in state["candidates"]:
            candidate = _candidate_from_dict(candidate_data)
            input_hash = sha256_text(candidate.text + settings.prompt_version + settings.llm_model)
            job = (
                self.db.query(ExtractionJob)
                .filter_by(
                    document_id=self.document.id,
                    section_type=candidate.section_type,
                    input_text_hash=input_hash,
                    prompt_version=settings.prompt_version,
                    model_name=settings.llm_model,
                )
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

            try:
                result = self.extractor.extract(candidate)
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
                self.db.commit()
            except Exception as exc:
                job.status = "failed"
                job.error_message = str(exc)
                self.db.commit()
                errors.append(f"candidate {candidate.page_from}-{candidate.page_to}: {exc}")

        state["extracted"] = extracted
        state["errors"] = errors
        return state

    def _save_node(self, state: WorkflowState) -> WorkflowState:
        saved_item_ids: list[str] = list(state.get("saved_item_ids", []))
        allowed_types = {"abbreviations": {"abbreviation"}, "terms": {"term"}, "mixed": {"abbreviation", "term"}}

        for item in state.get("extracted", []):
            item_type = item["item_type"]
            section_type = item["section_type"]
            allowed = allowed_types.get(section_type, {"abbreviation", "term"})
            if item_type not in allowed:
                continue

            entity = ExtractedEntity.model_validate(item["entity"])
            decision = self.validator.validate(item_type, entity, item["chunk_text"], section_type)
            if not decision.keep:
                continue

            row = ExtractedItem(
                document_id=self.document.id,
                item_type=item_type,
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
            except IntegrityError:
                self.db.rollback()

        state["saved_item_ids"] = saved_item_ids
        return state

    def _normalize_node(self, state: WorkflowState) -> WorkflowState:
        if self.corrector is None:
            return state
        for item_id in state.get("saved_item_ids", []):
            row = self.db.get(ExtractedItem, item_id)
            if row is None or row.status == "rejected":
                continue
            self.corrector.correct_item(self.db, row)
        return state


    def _validate_node(self, state: WorkflowState) -> WorkflowState:
        if self.agent is None:
            return state
        for item_id in state.get("saved_item_ids", []):
            row = self.db.get(ExtractedItem, item_id)
            if row is None:
                continue
            self.agent.validate_item(self.db, row)
        return state


def _build_chat_model() -> ChatOpenAI:
    provider = settings.llm_provider.lower().strip()
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key
    if provider == "ollama":
        base_url = base_url or "http://localhost:11434/v1"
        api_key = api_key or "ollama"
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set. For Ollama use LLM_API_KEY=ollama")
    return ChatOpenAI(model=settings.llm_model, api_key=api_key, base_url=base_url, temperature=0)


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


def _loads_json_relaxed(content: str) -> dict[str, Any]:
    content = (content or "").strip()
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


def _matches_payload(matches: list[RagMatch]) -> list[dict[str, Any]]:
    return [
        {"term": m.term, "definition": m.definition[:400], "score": m.score, "source_item_id": m.source_item_id}
        for m in matches
    ]


def _none_if_blank(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "нет"}:
        return None
    return text


def _normalize_decision(value: Any) -> str:
    decision = str(value or "needs_review").strip().lower()
    return decision if decision in {"auto", "needs_review", "rejected"} else "needs_review"


def _clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return max(0.0, min(1.0, number))


def _compact(text: str) -> str:
    return " ".join((text or "").split())
