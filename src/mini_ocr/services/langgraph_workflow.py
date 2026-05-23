from __future__ import annotations

import json
import re
from typing import Any, TypedDict, Literal

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
from mini_ocr.services.observability import AgentTimer, get_logger


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


class CorrectionSuggestion(BaseModel):
    normalized_key: str
    normalized_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str


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
        data = _sanitize_extraction_payload(_loads_json_relaxed(content))
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



class CorrectionRoute(BaseModel):
    strategy:  Literal["keep", "capitalizer", "corrector", "restorer", "skip"] = Field(description="keep | capitalizer | corrector | restorer | skip")
    confidence: float = Field(ge=0, le=1)
    reason: str


class CorrectionItemState(TypedDict):
    document_id: str
    item_id: str
    key: str
    value: str
    source_text: str | None
    page_from: int | None
    page_to: int | None
    item_type: str
    confidence: float | None
    status: str
    rag_matches: list[dict[str, Any]]
    correction_strategy: str | None
    correction_status: str | None
    orchestrator_reason: str | None
    orchestrator_confidence: float | None
    normalized_key: str | None
    normalized_value: str | None
    correction_confidence: float | None
    correction_reason: str | None


class CorrectionOrchestratorAgent:
    """LLM router that chooses which correction branch should run.

    The orchestrator does not correct text. It only decides which LangGraph
    conditional edge to follow: keep, capitalizer, corrector, restorer, or skip.
    """

    agent_name = "langchain_correction_orchestrator_agent"

    def __init__(self) -> None:
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты ИИ-оркестратор обработки OCR-термина. "
                "Ты НЕ исправляешь термин. Ты НЕ предлагаешь normalized_key. "
                "Ты выбираешь только одну стадию обработки. "
                ""
                "Главное правило: по умолчанию выбирай keep. "
                "Активную обработку выбирай только если есть явный признак, что она нужна. "
                ""
                "Доступные стадии: "
                ""
                "keep — термин выглядит как читаемый русский термин или словосочетание, "
                "исправление не требуется. Низкая OCR confidence сама по себе НЕ является причиной "
                "выбирать corrector или restorer. "
                ""
                "capitalizer — термин распознан корректно, но написан полностью заглавными "
                "русскими буквами. Меняется только регистр. "
                ""
                "corrector — термин слегка повреждён OCR, но его можно исправить по самому слову "
                "без восстановления по определению. "
                ""
                "restorer — термин сильно повреждён, но определение и RAG-контекст дают достаточно "
                "оснований восстановить термин. Это рискованная стадия. "
                ""
                "skip — термин написан латиницей как иностранный эквивалент, выглядит как код, "
                "мусор OCR или повреждён настолько, что восстановление будет ненадёжным. "
                ""
                "Примеры keep: "
                "Дефект; Дефекты; Глубина; Разрыв; Трещина; Складка; Складчатость; "
                "Рвань; Рваность; Рваный; Расслоение; Раскатанное окисление; "
                "Поверхность отслоения; Прямолинейный дефект поверхности. "
                ""
                "Примеры capitalizer: "
                "ПРОПЛАВЛЕНИЕ; МАТОВОСТЬ; ШЕРОХОВАТОСТЬ; ПИТОВАЯ ПОВЕРХНОСТЬ. "
                ""
                "НЕ выбирай capitalizer, если есть латиница, цифры, дефисы, смешение алфавитов "
                "или OCR-шум. Например: PAORBA-3EAD, ОСТATКИ OKAЛH-, ВИТТОВО. СЛЕД. "
                ""
                "Примеры corrector: "
                "Дефект поворхности; Трециня; Пероховатость; ОСТATКИ OKAЛH-. "
                ""
                "Примеры restorer: "
                "Косне трсшинн; НТЕНя; Морвгша; Дрккечания; Плеез единичная. "
                ""
                "Примеры skip: "
                "D. Rohraalzhaut; E. Tube rolling; F. Repliure de; P. Ronillob; PAORBA-3EAD; ПООС-. "
                ""
                "Жёсткие правила: "
                "Если термин выглядит нормальным русским словом или словосочетанием — выбирай keep. "
                "Если сомневаешься между keep и corrector — выбирай keep. "
                "Если сомневаешься между keep и restorer — выбирай keep. "
                "Если сомневаешься между restorer и skip — выбирай skip. "
                "Не выбирай restorer для нормальных русских терминов. "
                "Не выбирай corrector, если термин уже читаемый. "
                "Не выбирай capitalizer для слов, которые уже написаны нормальным регистром. "
                "Для латинских ключей вида 'D. ...', 'E. ...', 'F. ...', 'P. ...' всегда выбирай skip. "
                ""
                "Всегда отвечай только JSON. Без markdown. Без пояснений вне JSON. "
                "Причина должна быть на русском языке.",
            ),
            (
                "human",
                "Выбери стратегию обработки для одного OCR-кандидата.\n\n"
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n"
                "{{\"strategy\": \"keep|capitalizer|corrector|restorer|skip\", "
                "\"confidence\": 0.0, "
                "\"reason\": \"краткая причина на русском\"}}\n\n"
                "Напоминание: если key уже выглядит как нормальный русский термин, выбери keep."
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, state: CorrectionItemState) -> CorrectionRoute:
        candidate = _correction_candidate_payload(state)
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
                "rag_json": json.dumps(state.get("rag_matches") or [], ensure_ascii=False),
            })
            data = _loads_json_relaxed(content)
            route = CorrectionRoute.model_validate({
                "strategy": _normalize_correction_strategy(data.get("strategy")),
                "confidence": _clamp_float(data.get("confidence"), default=0.5),
                "reason": _sanitize_agent_reason(data.get("reason"), state.get("key"), state.get("key")),
            })
        except Exception as exc:
            route = _heuristic_correction_route(state, f"LLM-оркестратор недоступен: {exc}")

        # Deterministic safety net: foreign-looking terms must not enter restorer.
        key = state.get("key")
        if _looks_like_foreign_equivalent(key) and route.strategy != "skip":
            route.strategy = "skip"
            route.confidence = max(route.confidence, 0.85)
            route.reason = "Ключ выглядит как иностранный эквивалент, поэтому восстановление русского термина отключено."
        elif route.strategy == "capitalizer" and not _is_all_caps_cyrillic(key):
            route.strategy = "keep"
            route.confidence = max(route.confidence, 0.75)
            route.reason = "Термин не написан капсом; коррекция регистра не требуется."
        elif _looks_like_clean_russian_term(key) and route.strategy in {"corrector", "restorer", "skip"}:
            route.strategy = "keep"
            route.confidence = max(route.confidence, 0.75)
            route.reason = "Русский термин читаемый, явных OCR-искажений нет; коррекция не требуется."
        return route


class KeepCorrectionAgent:
    agent_name = "keep_correction_agent"

    def keep(self, state: CorrectionItemState) -> CorrectionSuggestion:
        key = (state.get("key") or "").strip()
        return CorrectionSuggestion(
            normalized_key=key,
            normalized_value=None,
            confidence=0.0,
            reason=_sanitize_agent_reason(
                state.get("orchestrator_reason") or "Термин выглядит читаемым; коррекция не требуется.",
                key,
                key,
            ),
        )


class CapitalizerCorrectionAgent:
    agent_name = "capitalizer_correction_agent"

    def normalize(self, state: CorrectionItemState) -> CorrectionSuggestion:
        key = (state.get("key") or "").strip()
        normalized = _normalize_capitalization(key)
        changed = normalized != key
        return CorrectionSuggestion(
            normalized_key=normalized or key,
            normalized_value=None,
            confidence=0.75 if changed else 0.0,
            reason="Термин распознан, изменён только регистр букв." if changed else "Изменение регистра не требуется.",
        )


class LightOCRCorrectorAgent:
    agent_name = "light_ocr_corrector_agent"

    def __init__(self) -> None:
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты корректор лёгких OCR-ошибок в русских технических терминах. "
                "Исправляй только сам термин по форме слова и типичным OCR-ошибкам. "
                "Не восстанавливай термин по определению"
                "Если исправление неочевидно, верни исходный ключ и confidence 0.0. "
                "Всегда отвечай только на русском языке и только JSON.",
            ),
            (
                "human",
                "Key: {key}\n"
                "Value preview: {value}\n"
                "Source preview: {source_text}\n\n"
                "Output schema:\n"
                "{{\"normalized_key\": \"string\", "
                "\"normalized_value\": null, "
                "\"confidence\": 0.0, "
                "\"reason\": \"краткая причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def correct(self, state: CorrectionItemState) -> CorrectionSuggestion:
        key = (state.get("key") or "").strip()
        try:
            content = self.chain.invoke({
                "key": key,
                "value": (state.get("value") or "")[:800],
                "source_text": (state.get("source_text") or "")[:1200],
            })
            data = _loads_json_relaxed(content)
            suggestion = CorrectionSuggestion.model_validate({
                "normalized_key": str(data.get("normalized_key") or key).strip() or key,
                "normalized_value": _clean_optional_text(data.get("normalized_value")),
                "confidence": _clamp_float(data.get("confidence"), default=0.45),
                "reason": _sanitize_agent_reason(data.get("reason"), data.get("normalized_key"), key),
            })
        except Exception as exc:
            normalized = _normalize_key_heuristic(key) or key
            suggestion = CorrectionSuggestion(
                normalized_key=normalized,
                normalized_value=None,
                confidence=0.55 if normalized != key else 0.0,
                reason=f"Эвристическая коррекция лёгкой OCR-ошибки после ошибки агента: {exc}",
            )
        return _post_filter_correction_suggestion(key, suggestion, strategy="corrector")


class DefinitionRestorerAgent:
    agent_name = "definition_restorer_agent"

    def __init__(self) -> None:
        self.llm = _build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты агент восстановления сильно повреждённого OCR русского технического термина. "
                "Используй определение, source_text и RAG-подсказки. "
                "Не запускайся для латинских иностранных эквивалентов. "
                "normalized_key должен быть коротким термином, а не фрагментом определения. "
                "Если восстановление ненадёжно, верни исходный ключ и confidence 0.0. "
                "Всегда отвечай только на русском языке и только JSON.",
            ),
            (
                "human",
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n"
                "{{\"normalized_key\": \"string\", "
                "\"normalized_value\": null, "
                "\"confidence\": 0.0, "
                "\"reason\": \"краткая причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def restore(self, state: CorrectionItemState) -> CorrectionSuggestion:
        key = (state.get("key") or "").strip()
        if _looks_like_foreign_equivalent(key):
            return _skip_suggestion(key, "Ключ выглядит как иностранный эквивалент, восстановление пропущено.")
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(_correction_candidate_payload(state), ensure_ascii=False),
                "rag_json": json.dumps(state.get("rag_matches") or [], ensure_ascii=False),
            })
            data = _loads_json_relaxed(content)
            suggestion = CorrectionSuggestion.model_validate({
                "normalized_key": str(data.get("normalized_key") or key).strip() or key,
                "normalized_value": _clean_optional_text(data.get("normalized_value")),
                "confidence": _clamp_float(data.get("confidence"), default=0.45),
                "reason": _sanitize_agent_reason(data.get("reason"), data.get("normalized_key"), key),
            })
        except Exception as exc:
            fallback = _fallback_restored_key_from_rag(key, state.get("rag_matches") or [])
            suggestion = CorrectionSuggestion(
                normalized_key=fallback or key,
                normalized_value=None,
                confidence=0.55 if fallback and fallback != key else 0.0,
                reason=f"Эвристическое восстановление после ошибки агента: {exc}",
            )
        return _post_filter_correction_suggestion(key, suggestion, strategy="restorer")


class SkipCorrectionAgent:
    agent_name = "skip_correction_agent"

    def skip(self, state: CorrectionItemState) -> CorrectionSuggestion:
        return _skip_suggestion(
            state.get("key") or "",
            state.get("orchestrator_reason") or "Коррекция пропущена по решению оркестратора.",
        )


class LangGraphOCRCorrectionWorkflow:
    """Per-item correction subgraph with conditional edges selected by an LLM orchestrator."""

    agent_name = "langgraph_ocr_correction_workflow"

    def __init__(self) -> None:
        self.rag = RagStore()
        self.orchestrator = CorrectionOrchestratorAgent()
        self.keeper = KeepCorrectionAgent()
        self.capitalizer = CapitalizerCorrectionAgent()
        self.corrector = LightOCRCorrectorAgent()
        self.restorer = DefinitionRestorerAgent()
        self.skipper = SkipCorrectionAgent()
        self.graph = self._build_graph()

    def normalize_item(self, db: Session, item: ExtractedItem) -> CorrectionSuggestion:
        if settings.enable_rag_validation:
            with AgentTimer(
                "rag.retrieve_for_correction",
                document_id=item.document_id,
                item_id=item.id,
                key=item.key,
                top_k=settings.rag_top_k,
            ) as trace:
                matches = self.rag.retrieve(db, f"{item.key}\n{item.value}\n{item.source_text or ''}", settings.rag_top_k)
                trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
        else:
            matches = []

        state: CorrectionItemState = {
            "document_id": item.document_id,
            "item_id": item.id,
            "key": item.key,
            "value": item.value,
            "source_text": item.source_text,
            "page_from": item.page_from,
            "page_to": item.page_to,
            "item_type": item.item_type,
            "confidence": item.confidence,
            "status": item.status,
            "rag_matches": _matches_payload(matches),
            "correction_strategy": None,
            "correction_status": None,
            "orchestrator_reason": None,
            "orchestrator_confidence": None,
            "normalized_key": None,
            "normalized_value": None,
            "correction_confidence": None,
            "correction_reason": None,
        }
        result = self.graph.invoke(state)
        suggestion = CorrectionSuggestion(
            normalized_key=result.get("normalized_key") or item.key,
            normalized_value=result.get("normalized_value"),
            confidence=_clamp_float(result.get("correction_confidence"), default=0.0),
            reason=result.get("correction_reason") or result.get("orchestrator_reason") or "Коррекция завершена.",
        )
        self._persist_correction(db, item, suggestion, matches, result)
        self._apply_correction(db, item, suggestion, result)
        return suggestion

    def _build_graph(self):
        graph = StateGraph(CorrectionItemState)
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("keep", self._keep_node)
        graph.add_node("capitalizer", self._capitalizer_node)
        graph.add_node("corrector", self._corrector_node)
        graph.add_node("restorer", self._restorer_node)
        graph.add_node("skip", self._skip_node)
        graph.add_node("post_filter", self._post_filter_node)
        graph.set_entry_point("orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            self._route_correction,
            {
                "keep": "keep",
                "capitalizer": "capitalizer",
                "corrector": "corrector",
                "restorer": "restorer",
                "skip": "skip",
            },
        )
        graph.add_edge("keep", "post_filter")
        graph.add_edge("capitalizer", "post_filter")
        graph.add_edge("corrector", "post_filter")
        graph.add_edge("restorer", "post_filter")
        graph.add_edge("skip", "post_filter")
        graph.add_edge("post_filter", END)
        return graph.compile()

    def _orchestrator_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer(
            "agent.correction_orchestrator",
            document_id=state["document_id"],
            item_id=state["item_id"],
            key=state["key"],
            model=settings.llm_model,
        ) as trace:
            route = self.orchestrator.route(state)
            trace.set(selected_strategy=route.strategy, confidence=route.confidence, reason=route.reason[:240])
        return {
            **state,
            "correction_strategy": route.strategy,
            "correction_status": "routed",
            "orchestrator_reason": route.reason,
            "orchestrator_confidence": route.confidence,
        }

    def _route_correction(self, state: CorrectionItemState) -> str:
        strategy = _normalize_correction_strategy(state.get("correction_strategy"))
        return strategy if strategy in {"keep", "capitalizer", "corrector", "restorer", "skip"} else "skip"

    def _keep_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer("agent.keep_correction", document_id=state["document_id"], item_id=state["item_id"], key=state["key"]) as trace:
            suggestion = self.keeper.keep(state)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence)
        return _state_with_suggestion(state, suggestion, correction_status="kept")

    def _capitalizer_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer("agent.capitalizer", document_id=state["document_id"], item_id=state["item_id"], key=state["key"]) as trace:
            suggestion = self.capitalizer.normalize(state)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence)
        return _state_with_suggestion(state, suggestion, correction_status="capitalized")

    def _corrector_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer("agent.light_ocr_corrector", document_id=state["document_id"], item_id=state["item_id"], key=state["key"], model=settings.llm_model) as trace:
            suggestion = self.corrector.correct(state)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence)
        return _state_with_suggestion(state, suggestion, correction_status="corrected" if suggestion.confidence > 0 else "unchanged")

    def _restorer_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer("agent.definition_restorer", document_id=state["document_id"], item_id=state["item_id"], key=state["key"], model=settings.llm_model) as trace:
            suggestion = self.restorer.restore(state)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence)
        return _state_with_suggestion(state, suggestion, correction_status="restored" if suggestion.confidence > 0 else "unrecoverable")

    def _skip_node(self, state: CorrectionItemState) -> CorrectionItemState:
        with AgentTimer("agent.skip_correction", document_id=state["document_id"], item_id=state["item_id"], key=state["key"]) as trace:
            suggestion = self.skipper.skip(state)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence)
        return _state_with_suggestion(state, suggestion, correction_status="skipped")

    def _post_filter_node(self, state: CorrectionItemState) -> CorrectionItemState:
        key = state.get("key") or ""
        suggestion = CorrectionSuggestion(
            normalized_key=state.get("normalized_key") or key,
            normalized_value=state.get("normalized_value"),
            confidence=_clamp_float(state.get("correction_confidence"), default=0.0),
            reason=state.get("correction_reason") or "Коррекция завершена.",
        )
        filtered = _post_filter_correction_suggestion(key, suggestion, strategy=state.get("correction_strategy") or "skip")
        status = state.get("correction_status") or "filtered"
        if filtered.confidence == 0.0 and filtered.normalized_key == key and status not in {"kept", "skipped", "unrecoverable"}:
            status = "rejected_by_guardrail"
        return _state_with_suggestion(state, filtered, correction_status=status)

    def _persist_correction(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion, matches: list[RagMatch], state: CorrectionItemState) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name=self.agent_name,
            decision=state.get("correction_status") or "normalized",
            confidence=suggestion.confidence,
            reason=suggestion.reason,
            normalized_key=suggestion.normalized_key,
            normalized_value=suggestion.normalized_value,
            rag_evidence={"matches": _matches_payload(matches)},
            payload={
                "key": item.key,
                "value": item.value,
                "source_text": item.source_text,
                "page_from": item.page_from,
                "page_to": item.page_to,
                "correction_strategy": state.get("correction_strategy"),
                "correction_status": state.get("correction_status"),
                "orchestrator_confidence": state.get("orchestrator_confidence"),
                "orchestrator_reason": state.get("orchestrator_reason"),
            },
        ))
        db.commit()

    def _apply_correction(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion, state: CorrectionItemState) -> None:
        item.normalized_key = suggestion.normalized_key
        item.normalized_value = suggestion.normalized_value
        item.correction_confidence = suggestion.confidence
        item.correction_reason = _sanitize_agent_reason(suggestion.reason, suggestion.normalized_key, item.key)
        item.correction_strategy = state.get("correction_strategy")
        item.correction_status = state.get("correction_status")
        item.correction_orchestrator_reason = state.get("orchestrator_reason")
        if suggestion.confidence < settings.correction_auto_threshold:
            item.status = "needs_review"
        elif item.status == "auto" and suggestion.confidence < 0.85:
            item.status = "needs_review"
        db.commit()
        db.refresh(item)


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
                "Всегда отвечай только на русском языке. "
                "Все поля reason/correction_reason должны быть на русском. "
                "Не используй английский, китайский или другой язык. "
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
        if settings.enable_rag_validation:
            with AgentTimer(
                "rag.retrieve_for_validation",
                document_id=item.document_id,
                item_id=item.id,
                key=item.key,
                top_k=settings.rag_top_k,
            ) as trace:
                matches = self.rag.retrieve(
                    db,
                    f"{item.key}\n{item.value}\n{item.source_text or ''}",
                    settings.rag_top_k,
                )
                trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
        else:
            matches = []

        candidate = {
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
        try:
            content = self.chain.invoke({
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
                "rag_json": json.dumps(_matches_payload(matches), ensure_ascii=False),
            })
            data = _loads_json_relaxed(content)
            decision = ValidationDecision.model_validate({
                "decision": _normalize_decision(data.get("decision")),
                "confidence": _clamp_float(data.get("confidence"), default=0.5),
                "reason": _sanitize_agent_reason(data.get("reason"), data.get("normalized_key"), item.key),
                "normalized_key": data.get("normalized_key"),
                "normalized_value": data.get("normalized_value"),
            })
            decision = _enforce_validation_policy(item, decision)
        except Exception as exc:
            decision = self._heuristic_decision(item, matches, f"LangChain validation failed: {exc}")
            decision = _enforce_validation_policy(item, decision)

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
        elif decision.decision == "auto" and decision.confidence >= settings.validation_auto_threshold:
            correction_confidence = getattr(item, "correction_confidence", None)
            if correction_confidence is not None and correction_confidence < settings.correction_auto_threshold:
                item.status = "needs_review"
            else:
                item.status = "auto"
            item.confidence = min(max(float(item.confidence or 0.5), decision.confidence), 0.95)
        else:
            item.status = "needs_review"
            item.confidence = min(float(item.confidence or decision.confidence or 0.5), decision.confidence)
        db.commit()
        db.refresh(item)
        if item.status == "auto":
            self.rag.add_confirmed_item(db, item, status="auto")


class LangGraphExtractionWorkflow:
    """LangGraph orchestration for extraction + deterministic guardrails + RAG validation."""

    def __init__(self, db: Session, document: Document) -> None:
        self.db = db
        self.document = document
        self.extractor = LangChainExtractor()
        self.validator = ExtractionValidator()
        self.normalizer = LangGraphOCRCorrectionWorkflow() if settings.enable_ocr_correction_agent else None
        self.agent = LangChainCandidateValidationAgent() if settings.enable_agent_validation else None
        self.logger = get_logger("langgraph_workflow")
        self.graph = self._build_graph()

    def run(self, candidates: list[SectionCandidate]) -> WorkflowState:
        state: WorkflowState = {
            "document_id": self.document.id,
            "candidates": [_candidate_to_dict(c) for c in candidates],
            "extracted": [],
            "saved_item_ids": [],
            "errors": [],
        }
        with AgentTimer(
            "langgraph.workflow",
            document_id=self.document.id,
            title=self.document.title,
            candidates_count=len(candidates),
        ) as trace:
            result = self.graph.invoke(state)
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
                payloads, retry_errors = self._extract_candidate_with_page_retry(candidate)
                extracted.extend(payloads)
                errors.extend(retry_errors)
                job.status = "done"
                job.error_message = "; ".join(retry_errors[:3]) if retry_errors else None
                self.db.commit()
            except Exception as exc:
                job.status = "failed"
                job.error_message = str(exc)
                self.db.commit()
                errors.append(f"candidate {candidate.page_from}-{candidate.page_to}: {exc}")

        state["extracted"] = extracted
        state["errors"] = errors
        return state

    def _extract_candidate_with_page_retry(self, candidate: SectionCandidate) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            return self._extract_one_candidate(candidate), []
        except Exception as first_exc:
            page_candidates = _split_candidate_by_page_markers(candidate)
            if len(page_candidates) <= 1:
                raise

            retry_errors: list[str] = [
                f"candidate {candidate.page_from}-{candidate.page_to} failed, retrying by page: {first_exc}"
            ]
            payloads: list[dict[str, Any]] = []
            with AgentTimer(
                "agent.extractor.retry_split",
                document_id=self.document.id,
                section_type=candidate.section_type,
                page_from=candidate.page_from,
                page_to=candidate.page_to,
                pages_count=len(page_candidates),
            ) as trace:
                for page_candidate in page_candidates:
                    try:
                        payloads.extend(self._extract_one_candidate(page_candidate))
                    except Exception as page_exc:
                        retry_errors.append(f"page {page_candidate.page_from}: {page_exc}")
                trace.set(extracted_payloads=len(payloads), retry_errors_count=len(retry_errors) - 1)

            if payloads:
                return payloads, retry_errors
            raise first_exc

    def _extract_one_candidate(self, candidate: SectionCandidate) -> list[dict[str, Any]]:
        with AgentTimer(
            "agent.extractor",
            document_id=self.document.id,
            section_type=candidate.section_type,
            page_from=candidate.page_from,
            page_to=candidate.page_to,
            text_chars=len(candidate.text or ""),
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        ) as trace:
            result = self.extractor.extract(candidate)
            trace.set(
                abbreviations_count=len(result.abbreviations),
                terms_count=len(result.terms),
            )

        payloads: list[dict[str, Any]] = []
        for item_type, entities in (("abbreviation", result.abbreviations), ("term", result.terms)):
            for entity in entities:
                if not (entity.key or "").strip() or not (entity.value or "").strip():
                    continue
                payloads.append({
                    "item_type": item_type,
                    "entity": entity.model_dump(),
                    "page_from": candidate.page_from,
                    "page_to": candidate.page_to,
                    "section_type": candidate.section_type,
                    "chunk_text": candidate.text,
                    "extractor": self.extractor.extractor_name,
                })
        return payloads

    def _save_node(self, state: WorkflowState) -> WorkflowState:
        saved_item_ids: list[str] = list(state.get("saved_item_ids", []))
        allowed_types = {"abbreviations": {"abbreviation"}, "terms": {"term"}, "mixed": {"abbreviation", "term"}}
        input_count = len(state.get("extracted", []))
        kept_count = 0
        skipped_count = 0

        with AgentTimer("workflow.save_node", document_id=self.document.id, extracted_count=input_count) as trace:
            for item in state.get("extracted", []):
                item_type = item["item_type"]
                section_type = item["section_type"]
                allowed = allowed_types.get(section_type, {"abbreviation", "term"})
                if item_type not in allowed:
                    skipped_count += 1
                    continue

                entity = ExtractedEntity.model_validate(item["entity"])
                decision = self.validator.validate(item_type, entity, item["chunk_text"], section_type)
                if not decision.keep:
                    skipped_count += 1
                    self.logger.info(
                        "deterministic validator skipped candidate: document_id=%s key=%r reason=%s",
                        self.document.id,
                        entity.key,
                        getattr(decision, "reason", None),
                    )
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
                    kept_count += 1
                    self.logger.info(
                        "saved extracted item: document_id=%s item_id=%s type=%s key=%r status=%s confidence=%.3f pages=%s-%s",
                        self.document.id,
                        row.id,
                        row.item_type,
                        row.key,
                        row.status,
                        float(row.confidence or 0.0),
                        row.page_from,
                        row.page_to,
                    )
                except IntegrityError:
                    self.db.rollback()
                    skipped_count += 1
            trace.set(saved_count=kept_count, skipped_count=skipped_count)

        state["saved_item_ids"] = saved_item_ids
        return state

    def _normalize_node(self, state: WorkflowState) -> WorkflowState:
        if self.normalizer is None:
            self.logger.info("normalize node skipped: OCR correction agent disabled")
            return state
        attempted = 0
        normalized = 0
        with AgentTimer("workflow.normalize_node", document_id=self.document.id, items_count=len(state.get("saved_item_ids", []))) as trace:
            for item_id in state.get("saved_item_ids", []):
                row = self.db.get(ExtractedItem, item_id)
                if row is None:
                    continue
                should_normalize = row.status == "needs_review" or float(row.confidence or 0.0) < 0.75 or _looks_ocr_noisy(row.key)
                self.logger.info(
                    "normalize candidate check: document_id=%s item_id=%s key=%r status=%s confidence=%.3f ocr_noisy=%s should_normalize=%s",
                    self.document.id,
                    row.id,
                    row.key,
                    row.status,
                    float(row.confidence or 0.0),
                    _looks_ocr_noisy(row.key),
                    should_normalize,
                )
                if should_normalize:
                    attempted += 1
                    with AgentTimer(
                        "agent.ocr_correction",
                        document_id=self.document.id,
                        item_id=row.id,
                        key=row.key,
                        status=row.status,
                        confidence=float(row.confidence or 0.0),
                        model=settings.llm_model,
                    ) as item_trace:
                        suggestion = self.normalizer.normalize_item(self.db, row)
                        item_trace.set(
                            normalized_key=suggestion.normalized_key,
                            correction_confidence=suggestion.confidence,
                            changed=suggestion.normalized_key != row.key,
                        )
                        if suggestion.normalized_key:
                            normalized += 1
            trace.set(attempted_count=attempted, normalized_count=normalized)
        return state

    def _validate_node(self, state: WorkflowState) -> WorkflowState:
        if self.agent is None:
            self.logger.info("validation node skipped: validation agent disabled")
            return state
        validated = 0
        with AgentTimer("workflow.validate_node", document_id=self.document.id, items_count=len(state.get("saved_item_ids", []))) as trace:
            for item_id in state.get("saved_item_ids", []):
                row = self.db.get(ExtractedItem, item_id)
                if row is None:
                    continue
                with AgentTimer(
                    "agent.rag_validation",
                    document_id=self.document.id,
                    item_id=row.id,
                    key=row.key,
                    normalized_key=getattr(row, "normalized_key", None),
                    status=row.status,
                    confidence=float(row.confidence or 0.0),
                    model=settings.llm_model,
                ) as item_trace:
                    decision = self.agent.validate_item(self.db, row)
                    item_trace.set(
                        decision=decision.decision,
                        validation_confidence=decision.confidence,
                        reason=decision.reason[:240] if decision.reason else None,
                    )
                    validated += 1
            trace.set(validated_count=validated)
        return state


def _correction_candidate_payload(state: CorrectionItemState) -> dict[str, Any]:
    return {
        "item_type": state.get("item_type"),
        "key": state.get("key"),
        "value": state.get("value"),
        "source_text": state.get("source_text"),
        "page_from": state.get("page_from"),
        "page_to": state.get("page_to"),
        "current_confidence": state.get("confidence"),
        "current_status": state.get("status"),
    }


def _normalize_correction_strategy(value: Any) -> str:
    strategy = str(value or "skip").strip().lower()
    aliases = {
        "capitalize": "capitalizer",
        "capitalise": "capitalizer",
        "spelling_corrector": "corrector",
        "ocr_corrector": "corrector",
        "correction": "corrector",
        "restore": "restorer",
        "definition_restore": "restorer",
        "definition_restorer": "restorer",
        "keep": "keep",
        "no_correction": "keep",
        "unchanged": "keep",
        "as_is": "keep",
        "none": "keep",
        "pass": "skip",
    }
    strategy = aliases.get(strategy, strategy)
    return strategy if strategy in {"keep", "capitalizer", "corrector", "restorer", "skip"} else "skip"


def _state_with_suggestion(state: CorrectionItemState, suggestion: CorrectionSuggestion, correction_status: str) -> CorrectionItemState:
    return {
        **state,
        "normalized_key": suggestion.normalized_key,
        "normalized_value": suggestion.normalized_value,
        "correction_confidence": suggestion.confidence,
        "correction_reason": suggestion.reason,
        "correction_status": correction_status,
    }


def _heuristic_correction_route(state: CorrectionItemState, reason: str = "Эвристический выбор стратегии.") -> CorrectionRoute:
    key = state.get("key") or ""
    value = state.get("value") or ""
    if _looks_like_foreign_equivalent(key):
        return CorrectionRoute(strategy="skip", confidence=0.9, reason=f"{reason} Ключ похож на иностранный эквивалент.")
    if _looks_unrecoverable_key(key):
        return CorrectionRoute(strategy="skip", confidence=0.75, reason=f"{reason} Ключ слишком повреждён для надёжного восстановления.")
    if _is_all_caps_cyrillic(key):
        return CorrectionRoute(strategy="capitalizer", confidence=0.8, reason=f"{reason} Термин написан заглавными буквами.")
    if _has_light_ocr_noise(key):
        return CorrectionRoute(strategy="corrector", confidence=0.65, reason=f"{reason} В ключе есть лёгкие OCR-искажения.")
    if _looks_ocr_noisy(key) and len(value) > 40:
        return CorrectionRoute(strategy="restorer", confidence=0.6, reason=f"{reason} Ключ повреждён, но определение может помочь восстановлению.")
    if _looks_like_clean_russian_term(key):
        return CorrectionRoute(strategy="keep", confidence=0.75, reason=f"{reason} Русский термин читаемый, коррекция не требуется.")
    return CorrectionRoute(strategy="skip", confidence=0.5, reason=f"{reason} Надёжная стратегия коррекции не найдена.")


def _looks_like_foreign_equivalent(key: str | None) -> bool:
    text = (key or "").strip()
    # Standards often contain foreign equivalents in adjacent columns: D. ..., E. ..., F. ...
    return bool(re.match(r"^[A-Z]\.?\s+[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\- ]+$", text))


def _looks_unrecoverable_key(key: str | None) -> bool:
    text = (key or "").strip()
    if not text:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if len(text) <= 5 and not any(("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in letters):
        return True
    if sum(ch.isdigit() for ch in text) >= 2 and sum(ch.isascii() for ch in text) / max(len(text), 1) > 0.4:
        return True
    if re.fullmatch(r"[A-ZА-Я0-9\-]{6,}", text) and sum(ch.isdigit() for ch in text) >= 1:
        return True
    return False


def _is_all_caps_cyrillic(key: str | None) -> bool:
    text = (key or "").strip()
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return False
    cyr = [ch for ch in letters if ("а" <= ch.lower() <= "я") or ch.lower() == "ё"]
    if len(cyr) / len(letters) < 0.75:
        return False
    return all(not ch.isalpha() or ch.upper() == ch for ch in text)


def _has_light_ocr_noise(key: str | None) -> bool:
    text = (key or "").strip()
    low = text.lower()
    patterns = ["поворх", "трец", "трсшин", "окалh", "��", "рвсполоб", "ориентаща"]
    return any(p in low for p in patterns) or _has_mixed_cyrillic_latin(text)


def _has_mixed_cyrillic_latin(text: str | None) -> bool:
    letters = [ch for ch in (text or "") if ch.isalpha()]
    if not letters:
        return False
    latin = sum(ch.isascii() and ch.isalpha() for ch in letters)
    cyr = sum(("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in letters)
    return latin > 0 and cyr > 0


def _normalize_capitalization(key: str) -> str:
    text = (key or "").strip()
    if not text:
        return text
    parts = [p for p in re.split(r"(\s+|-)", text.lower())]
    if not parts:
        return text.capitalize()
    for idx, part in enumerate(parts):
        if part and part[0].isalpha():
            parts[idx] = part[0].upper() + part[1:]
            break
    return "".join(parts)


def _post_filter_correction_suggestion(original_key: str, suggestion: CorrectionSuggestion, strategy: str) -> CorrectionSuggestion:
    normalized = (suggestion.normalized_key or "").strip()
    original = (original_key or "").strip()
    if not normalized:
        return _skip_suggestion(original, "Нормализация отклонена: агент вернул пустой термин.")

    # Keep/skip are allowed to return the original key safely.
    if strategy == "keep":
        return _keep_suggestion(original, suggestion.reason)
    if strategy == "skip":
        return _skip_suggestion(original, suggestion.reason)

    # A normalized key must be a term, not a definition fragment.
    if _looks_like_bad_normalized_key(original, normalized):
        return _skip_suggestion(
            original,
            "Нормализация отклонена: предложенный вариант похож на фрагмент определения или недостаточно подтверждён контекстом.",
        )

    if _looks_like_foreign_equivalent(original) and normalized != original:
        return _skip_suggestion(original, "Нормализация отклонена: исходный ключ выглядит как иностранный эквивалент.")

    suggestion.reason = _sanitize_agent_reason(suggestion.reason, normalized, original)
    suggestion.normalized_key = normalized
    suggestion.confidence = _clamp_float(suggestion.confidence, default=0.0)
    return suggestion


def _looks_like_bad_normalized_key(original_key: str, normalized_key: str | None) -> bool:
    nk = (normalized_key or "").strip()
    if not nk:
        return True
    if len(nk) > 80:
        return True
    if len(nk.split()) > 5:
        return True
    low = nk.lower()
    bad_fragments = [
        "представляет собой",
        "представляющий собой",
        "дефект поверхности",
        "поверхность отслоения",
        "образуется",
        "окислены",
        "содержит",
        "под ними",
        "вследствие",
    ]
    if any(fragment in low for fragment in bad_fragments):
        # Some real terms include "дефект поверхности"; allow only if original already had almost the same wording.
        if "дефект поверхности" in low and "дефект поверхности" in (original_key or "").lower() and len(nk.split()) <= 4:
            return False
        return True
    return False


def _keep_suggestion(key: str, reason: str | None = None) -> CorrectionSuggestion:
    original = (key or "").strip()
    return CorrectionSuggestion(
        normalized_key=original,
        normalized_value=None,
        confidence=0.0,
        reason=_sanitize_agent_reason(reason or "Термин сохранён без изменений; коррекция не требуется.", original, original),
    )


def _looks_like_clean_russian_term(key: str | None) -> bool:
    text = (key or "").strip()
    if not text:
        return False
    if _looks_like_foreign_equivalent(text) or _looks_unrecoverable_key(text):
        return False
    if _is_all_caps_cyrillic(text) or _has_light_ocr_noise(text) or _has_mixed_cyrillic_latin(text):
        return False
    if _looks_like_definition_fragment_key(text):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 3:
        return False
    cyr = sum(("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in letters)
    if cyr / max(len(letters), 1) < 0.85:
        return False
    # A normal term is usually short. Longer noun phrases are allowed if they do not look like a definition fragment.
    return len(text.split()) <= 4 and len(text) <= 60


def _looks_like_definition_fragment_key(key: str | None) -> bool:
    text = (key or "").strip().lower()
    if not text:
        return False
    if len(text) > 70 or len(text.split()) > 6:
        return True
    return any(token in text for token in [",", ";", "представ", "соедин", "образ", "вследствие", "котор", "отслоен"])

def _skip_suggestion(key: str, reason: str | None = None) -> CorrectionSuggestion:
    original = (key or "").strip()
    return CorrectionSuggestion(
        normalized_key=original,
        normalized_value=None,
        confidence=0.0,
        reason=_sanitize_agent_reason(reason or "Коррекция пропущена; исходный термин сохранён для ручной проверки.", original, original),
    )


def _fallback_restored_key_from_rag(key: str, rag_matches: list[dict[str, Any]]) -> str | None:
    if not rag_matches:
        return None
    first = rag_matches[0]
    try:
        score = float(first.get("score") or 0.0)
    except Exception:
        score = 0.0
    term = str(first.get("term") or "").strip()
    if score >= 0.78 and term and not _looks_like_bad_normalized_key(key, term):
        return term
    return None


def _build_chat_model() -> ChatOpenAI:
    provider = settings.llm_provider.lower().strip()
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key
    if provider == "ollama":
        base_url = base_url or "http://localhost:11434/v1"
        api_key = api_key or "ollama"
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set. For Ollama use LLM_API_KEY=ollama")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


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


def _normalize_decision(value: Any) -> str:
    decision = str(value or "needs_review").strip().lower()
    return decision if decision in {"auto", "needs_review", "rejected"} else "needs_review"


def _clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except Exception:
        number = default
    return max(0.0, min(1.0, number))


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "нет"}:
        return None
    return text


def _looks_ocr_noisy(text: str | None) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    latin = sum(ch.isascii() and ch.isalpha() for ch in letters)
    cyr = sum(("а" <= ch.lower() <= "я") or ch.lower() == "ё" for ch in letters)
    mixed = latin > 0 and cyr > 0
    upper_ratio = sum(ch.isupper() for ch in letters) / max(len(letters), 1)
    weird = sum(ch.isdigit() or ch in "@#$%^*_+=<>|/" for ch in text)
    return mixed or upper_ratio > 0.75 or weird >= 2 or "-" in text and len(text) <= 12


def _fallback_normalized_key(item: ExtractedItem, matches: list[RagMatch]) -> str:
    key = (item.key or "").strip()
    if matches and matches[0].score >= 0.78:
        return matches[0].term
    return _normalize_key_heuristic(key) or key


def _fallback_correction_confidence(item: ExtractedItem) -> float:
    key = item.key or ""
    if _looks_ocr_noisy(key):
        return 0.42
    return min(max(float(item.confidence or 0.5), 0.5), 0.7)


def _normalize_key_heuristic(key: str) -> str:
    text = (key or "").strip()
    if not text:
        return text
    # Common OCR confusions for Cyrillic/Latin and old scan noise. Conservative on purpose.
    replacements = str.maketrans({
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
        "O": "О", "P": "Р", "T": "Т", "X": "Х", "a": "а", "c": "с", "e": "е",
        "o": "о", "p": "р", "x": "х", "y": "у",
    })
    text = text.translate(replacements)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    known = {
        "ВИТТОВО СЛЕД": "Винтовой след",
        "ВИТТОВО. СЛЕД": "Винтовой след",
        "КОСНЕ ТРЕЩИНН": "Косые трещины",
        "КОСНЕ ТРСШИНН": "Косые трещины",
        "ОСТАТКИ ОКАЛН": "Остатки окалины",
        "ОСТАТКИ ОКАЛИ": "Остатки окалины",
        "ПЕРОХОВАТОСТЬ": "Шероховатость",
        "ПОВОРХНОСТИ": "поверхности",
        "ПОВОРХНОСТЬ": "поверхность",
        "ТРЕЦИНА": "Трещина",
    }
    normalized_upper = re.sub(r"[^А-ЯЁ ]+", "", text.upper()).strip()
    for bad, good in known.items():
        if bad in normalized_upper:
            return good
    if text.isupper() and len(text) > 2:
        return text.capitalize()
    return text



def _sanitize_extraction_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Drop malformed LLM rows before Pydantic validation.

    Local models sometimes return {"key": "...", "value": ""}. A single
    malformed row should not invalidate the whole extraction chunk.
    """
    cleaned: dict[str, Any] = {"abbreviations": [], "terms": []}
    for field in ("abbreviations", "terms"):
        rows = data.get(field) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip()
            value = str(row.get("value") or "").strip()
            if not key or not value:
                continue
            cleaned[field].append({
                "key": key,
                "value": value,
                "source_text": _clean_optional_text(row.get("source_text")),
                "confidence": _clamp_float(row.get("confidence"), default=0.5),
            })
    return cleaned


def _contains_cjk(text: str | None) -> bool:
    if not text:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _cyrillic_ratio(text: str | None) -> float:
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cyr = [ch for ch in letters if ("а" <= ch.lower() <= "я") or ch.lower() == "ё"]
    return len(cyr) / len(letters)


def _sanitize_agent_reason(reason: Any, normalized_key: Any = None, original_key: Any = None) -> str:
    text = str(reason or "").strip()
    if text and not _contains_cjk(text) and _cyrillic_ratio(text) >= 0.55:
        return text
    norm = str(normalized_key or "").strip()
    original = str(original_key or "").strip()
    if norm and original and norm != original:
        return "Предложена нормализация OCR-искажённого термина на основе контекста."
    return "Надёжная нормализация не найдена; исходный термин сохранён для ручной проверки."


def _enforce_validation_policy(item: ExtractedItem, decision: ValidationDecision) -> ValidationDecision:
    """LLM decision is advisory; thresholds are deterministic guardrails."""
    if decision.confidence < settings.validation_auto_threshold and decision.decision == "auto":
        decision.decision = "needs_review"
        decision.reason = f"{_sanitize_agent_reason(decision.reason, decision.normalized_key, item.key)}; auto понижен до needs_review из-за низкой уверенности."
    else:
        decision.reason = _sanitize_agent_reason(decision.reason, decision.normalized_key, item.key)

    correction_confidence = getattr(item, "correction_confidence", None)
    if decision.decision == "auto" and correction_confidence is not None and correction_confidence < settings.correction_auto_threshold:
        decision.decision = "needs_review"
        decision.reason = f"{decision.reason}; auto понижен до needs_review из-за низкой уверенности OCR-нормализации."
    return decision


def _split_candidate_by_page_markers(candidate: SectionCandidate) -> list[SectionCandidate]:
    """Split a multi-page OCR chunk into single-page candidates using markers.

    SectionDetector formats chunks as "--- page N ---\n...". When a large
    chunk times out, retrying one page at a time is cheaper and usually enough.
    """
    text = candidate.text or ""
    matches = list(re.finditer(r"(?im)^---\s*page\s+(\d+)\s*---\s*$", text))
    if len(matches) <= 1:
        return []

    result: list[SectionCandidate] = []
    for idx, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if not page_text:
            continue
        result.append(SectionCandidate(
            section_type=candidate.section_type,
            text=page_text,
            page_from=page_number,
            page_to=page_number,
            score=candidate.score,
            title=candidate.title,
            source=f"{candidate.source}:retry_page",
            layout_type=candidate.layout_type,
        ))
    return result

def _compact(text: str) -> str:
    return " ".join((text or "").split())
