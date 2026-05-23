from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from mini_ocr.core.config import settings
from mini_ocr.models import ExtractedItem, ItemValidation
from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.observability import AgentTimer
from mini_ocr.services.rag_store import RagMatch, RagStore
from mini_ocr.utils.json_utils import loads_json_relaxed
from mini_ocr.utils.text import (
    clean_optional_text,
    clamp_float,
    is_clean_cyrillic_caps,
    looks_clean_russian_term,
    looks_latin_or_foreign,
    titlecase_cyrillic_caps,
)


class CorrectionRoute(BaseModel):
    strategy: Literal["keep", "capitalizer", "corrector", "restorer", "skip"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class CorrectionSuggestion(BaseModel):
    normalized_key: str
    normalized_value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    strategy: str
    status: str
    orchestrator_reason: str | None = None


class CorrectionState(TypedDict, total=False):
    item_id: str
    document_id: str
    key: str
    value: str
    source_text: str | None
    page_from: int | None
    page_to: int | None
    confidence: float | None
    status: str | None
    rag_matches: list[dict[str, Any]]
    route: dict[str, Any]
    suggestion: dict[str, Any]


class CorrectionOrchestratorAgent:
    def __init__(self) -> None:
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Ты ИИ-оркестратор обработки OCR-термина. "
                "Ты НЕ исправляешь термин и НЕ предлагаешь normalized_key. "
                "Ты выбираешь только одну стадию обработки. "
                "Главное правило: по умолчанию выбирай keep. "
                "Активную обработку выбирай только если есть явный признак, что она нужна. "
                "Стадии: keep — читаемый термин, исправление не требуется; "
                "capitalizer — чистый русский термин полностью заглавными буквами, меняется только регистр; "
                "corrector — лёгкая OCR-ошибка, исправимая по самому слову; "
                "restorer — сильное повреждение, но определение и RAG дают основания восстановить термин; "
                "skip — иностранный эквивалент, код, OCR-мусор или ненадёжный кандидат. "
                "Если key на латинице или в основном латиницей — выбирай skip, не keep. "
                "Если key выглядит нормальным русским словом/словосочетанием — выбирай keep. "
                "Если сомневаешься между keep и corrector/restorer — выбирай keep. "
                "Если сомневаешься между restorer и skip — выбирай skip. "
                "Для EN, IDT, MOD и латинских терминов выбирай skip. "
                "Верни только JSON без markdown. Причина на русском.",
            ),
            (
                "human",
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n"
                "{{\"strategy\": \"keep|capitalizer|corrector|restorer|skip\", "
                "\"confidence\": 0.0, "
                "\"reason\": \"краткая причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def choose(self, candidate: dict[str, Any], matches: list[dict[str, Any]]) -> CorrectionRoute:
        # Cheap deterministic exits keep the LLM from over-processing obvious cases.
        key = str(candidate.get("key") or "").strip()
        if looks_latin_or_foreign(key):
            return CorrectionRoute(strategy="skip", confidence=0.9, reason="Ключ выглядит как латинский термин или иностранный эквивалент.")
        if looks_clean_russian_term(key):
            return CorrectionRoute(strategy="keep", confidence=0.9, reason="Термин выглядит как читаемый русский термин; коррекция не требуется.")
        if is_clean_cyrillic_caps(key):
            return CorrectionRoute(strategy="capitalizer", confidence=0.85, reason="Термин написан заглавными русскими буквами.")

        content = self.chain.invoke({
            "candidate_json": json.dumps(candidate, ensure_ascii=False),
            "rag_json": json.dumps(matches, ensure_ascii=False),
        })
        data = loads_json_relaxed(content)
        route = CorrectionRoute.model_validate({
            "strategy": str(data.get("strategy") or "skip").strip().lower(),
            "confidence": clamp_float(data.get("confidence"), default=0.5),
            "reason": str(data.get("reason") or "Маршрут выбран агентом."),
        })
        return self._safety_net(key, route)

    def _safety_net(self, key: str, route: CorrectionRoute) -> CorrectionRoute:
        if looks_latin_or_foreign(key):
            return CorrectionRoute(strategy="skip", confidence=max(route.confidence, 0.9), reason="Ключ выглядит как латинский термин или иностранный эквивалент.")
        if route.strategy == "capitalizer" and not is_clean_cyrillic_caps(key):
            return CorrectionRoute(strategy="keep" if looks_clean_russian_term(key) else "skip", confidence=0.75, reason="Маршрут capitalizer отклонён safety-net: ключ не является чистым русским капсом.")
        if route.strategy in {"corrector", "restorer"} and looks_clean_russian_term(key):
            return CorrectionRoute(strategy="keep", confidence=0.85, reason="Термин уже читаемый; активная коррекция не требуется.")
        return route


class LightOCRCorrectorAgent:
    def __init__(self) -> None:
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты исправляешь только лёгкие OCR-ошибки в ключе термина. Не восстанавливай по смыслу. Верни только JSON."),
            (
                "human",
                "Candidate JSON:\n{candidate_json}\n\nRAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n{{\"normalized_key\": \"string\", \"normalized_value\": null, \"confidence\": 0.0, \"reason\": \"причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def correct(self, state: CorrectionState) -> CorrectionSuggestion:
        data = self._invoke(state)
        return _suggestion_from_data(data, state, "corrector", "corrected")

    def _invoke(self, state: CorrectionState) -> dict[str, Any]:
        content = self.chain.invoke({
            "candidate_json": json.dumps(_candidate_payload(state), ensure_ascii=False),
            "rag_json": json.dumps(state.get("rag_matches", []), ensure_ascii=False),
        })
        return loads_json_relaxed(content)


class DefinitionRestorerAgent(LightOCRCorrectorAgent):
    def __init__(self) -> None:
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Ты осторожно восстанавливаешь сильно повреждённый OCR-ключ по определению и RAG. Если уверенности нет — верни исходный key и confidence 0. Верни только JSON."),
            (
                "human",
                "Candidate JSON:\n{candidate_json}\n\nRAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n{{\"normalized_key\": \"string\", \"normalized_value\": null, \"confidence\": 0.0, \"reason\": \"причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def restore(self, state: CorrectionState) -> CorrectionSuggestion:
        data = self._invoke(state)
        return _suggestion_from_data(data, state, "restorer", "restored")


class OCRCorrectionWorkflow:
    """Small LangGraph subgraph for one saved extracted item."""

    def __init__(self) -> None:
        self.rag = RagStore()
        self.orchestrator = CorrectionOrchestratorAgent()
        self.corrector = LightOCRCorrectorAgent()
        self.restorer = DefinitionRestorerAgent()
        self.graph = self._build_graph()

    def normalize_item(self, db: Session, item: ExtractedItem) -> CorrectionSuggestion:
        state = _state_from_item(item)
        state["rag_matches"] = self._retrieve_matches(db, item)
        result = self.graph.invoke(state, config={"recursion_limit": 20})
        suggestion = CorrectionSuggestion.model_validate(result["suggestion"])
        self._persist(db, item, suggestion, state.get("rag_matches", []))
        self._apply(db, item, suggestion)
        return suggestion

    def _build_graph(self):
        graph = StateGraph(CorrectionState)
        graph.add_node("route", self._route_node)
        graph.add_node("keep", self._keep_node)
        graph.add_node("capitalizer", self._capitalizer_node)
        graph.add_node("corrector", self._corrector_node)
        graph.add_node("restorer", self._restorer_node)
        graph.add_node("skip", self._skip_node)
        graph.add_node("post_filter", self._post_filter_node)
        graph.set_entry_point("route")
        graph.add_conditional_edges("route", _route_key, {
            "keep": "keep",
            "capitalizer": "capitalizer",
            "corrector": "corrector",
            "restorer": "restorer",
            "skip": "skip",
        })
        for node in ("keep", "capitalizer", "corrector", "restorer", "skip"):
            graph.add_edge(node, "post_filter")
        graph.add_edge("post_filter", END)
        return graph.compile()

    def _retrieve_matches(self, db: Session, item: ExtractedItem) -> list[dict[str, Any]]:
        if not settings.enable_rag_validation:
            return []
        with AgentTimer("rag.retrieve_for_correction", document_id=item.document_id, item_id=item.id, key=item.key, top_k=settings.rag_top_k) as trace:
            matches = self.rag.retrieve(db, f"{item.key}\n{item.value}\n{item.source_text or ''}", settings.rag_top_k)
            trace.set(matches_count=len(matches), best_score=matches[0].score if matches else None)
        return _matches_payload(matches)

    def _route_node(self, state: CorrectionState) -> CorrectionState:
        with AgentTimer("agent.correction_orchestrator", document_id=state["document_id"], item_id=state["item_id"], key=state["key"], model=settings.llm_model) as trace:
            route = self.orchestrator.choose(_candidate_payload(state), state.get("rag_matches", []))
            trace.set(selected_strategy=route.strategy, confidence=route.confidence, reason=route.reason[:240])
        return {**state, "route": route.model_dump()}

    def _keep_node(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        return _with_suggestion(state, CorrectionSuggestion(normalized_key=state["key"], normalized_value=None, confidence=0.0, reason=route.reason, strategy="keep", status="kept", orchestrator_reason=route.reason))

    def _skip_node(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        return _with_suggestion(state, CorrectionSuggestion(normalized_key=state["key"], normalized_value=None, confidence=0.0, reason=route.reason, strategy="skip", status="skipped", orchestrator_reason=route.reason))

    def _capitalizer_node(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        normalized = titlecase_cyrillic_caps(state["key"])
        status = "capitalized" if normalized != state["key"] else "kept"
        return _with_suggestion(state, CorrectionSuggestion(normalized_key=normalized, normalized_value=None, confidence=0.75 if normalized != state["key"] else 0.0, reason=route.reason, strategy="capitalizer", status=status, orchestrator_reason=route.reason))

    def _corrector_node(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        try:
            suggestion = self.corrector.correct(state)
            suggestion.orchestrator_reason = route.reason
        except Exception as exc:
            suggestion = CorrectionSuggestion(normalized_key=state["key"], normalized_value=None, confidence=0.0, reason=f"Ошибка corrector: {exc}", strategy="corrector", status="unrecoverable", orchestrator_reason=route.reason)
        return _with_suggestion(state, suggestion)

    def _restorer_node(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        try:
            suggestion = self.restorer.restore(state)
            suggestion.orchestrator_reason = route.reason
        except Exception as exc:
            suggestion = CorrectionSuggestion(normalized_key=state["key"], normalized_value=None, confidence=0.0, reason=f"Ошибка restorer: {exc}", strategy="restorer", status="unrecoverable", orchestrator_reason=route.reason)
        return _with_suggestion(state, suggestion)

    def _post_filter_node(self, state: CorrectionState) -> CorrectionState:
        suggestion = CorrectionSuggestion.model_validate(state["suggestion"])
        if _bad_normalized_key(state["key"], suggestion.normalized_key) or suggestion.confidence <= 0:
            suggestion.normalized_key = state["key"]
            suggestion.normalized_value = None
            if suggestion.strategy in {"corrector", "restorer"}:
                suggestion.status = "unrecoverable"
            suggestion.confidence = 0.0
        return _with_suggestion(state, suggestion)

    def _persist(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion, matches: list[dict[str, Any]]) -> None:
        db.add(ItemValidation(
            item_id=item.id,
            document_id=item.document_id,
            agent_name="langgraph_ocr_correction_workflow",
            decision=suggestion.status,
            confidence=suggestion.confidence,
            reason=suggestion.reason,
            normalized_key=suggestion.normalized_key,
            normalized_value=suggestion.normalized_value,
            rag_evidence={"matches": matches},
            payload={"key": item.key, "value": item.value, "strategy": suggestion.strategy, "orchestrator_reason": suggestion.orchestrator_reason},
        ))
        db.commit()

    def _apply(self, db: Session, item: ExtractedItem, suggestion: CorrectionSuggestion) -> None:
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


def _route_key(state: CorrectionState) -> str:
    route = state.get("route") or {}
    strategy = str(route.get("strategy") or "skip")
    return strategy if strategy in {"keep", "capitalizer", "corrector", "restorer", "skip"} else "skip"


def _state_from_item(item: ExtractedItem) -> CorrectionState:
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
    }


def _candidate_payload(state: CorrectionState) -> dict[str, Any]:
    return {k: state.get(k) for k in ("key", "value", "source_text", "page_from", "page_to", "confidence", "status")}


def _with_suggestion(state: CorrectionState, suggestion: CorrectionSuggestion) -> CorrectionState:
    return {**state, "suggestion": suggestion.model_dump()}


def _suggestion_from_data(data: dict[str, Any], state: CorrectionState, strategy: str, status: str) -> CorrectionSuggestion:
    normalized_key = str(data.get("normalized_key") or state["key"]).strip() or state["key"]
    confidence = clamp_float(data.get("confidence"), default=0.0)
    if normalized_key == state["key"]:
        confidence = 0.0
        status = "unrecoverable" if strategy == "restorer" else "unchanged"
    return CorrectionSuggestion(
        normalized_key=normalized_key,
        normalized_value=clean_optional_text(data.get("normalized_value")),
        confidence=confidence,
        reason=str(data.get("reason") or f"{strategy} suggestion"),
        strategy=strategy,
        status=status,
    )


def _bad_normalized_key(original_key: str, normalized_key: str | None) -> bool:
    if not normalized_key or not normalized_key.strip():
        return True
    nk = normalized_key.strip()
    if len(nk) > 100 or len(nk.split()) > 8:
        return True
    definition_fragments = ("представляет собой", "образуется", "содержит", "является", "под ними", "вследствие")
    if any(fragment in nk.lower() for fragment in definition_fragments):
        return True
    if looks_latin_or_foreign(original_key) and original_key.strip() != nk:
        return True
    return False


def _matches_payload(matches: list[RagMatch]) -> list[dict[str, Any]]:
    return [
        {"term": m.term, "definition": m.definition[:400], "score": m.score, "source_item_id": m.source_item_id}
        for m in matches
    ]
