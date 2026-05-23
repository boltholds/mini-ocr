from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from mini_ocr.core.config import settings
from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.observability import AgentTimer
from mini_ocr.utils.json_utils import loads_json_relaxed
from mini_ocr.services.policies.text import (
    CLEAN_CYRILLIC_CAPS_TEXT_POLICY,
    CLEAN_RUSSIAN_TERM_TEXT_POLICY,
    LATIN_OR_FOREIGN_TEXT_POLICY,
    TextPolicy,
)
from mini_ocr.utils.text import clean_optional_text, clamp_float, titlecase_cyrillic_caps


CorrectionStrategy = Literal["keep", "capitalizer", "corrector", "restorer", "skip"]


class CorrectionRoute(BaseModel):
    strategy: CorrectionStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class CorrectionSuggestion(BaseModel):
    normalized_key: str
    normalized_value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    strategy: CorrectionStrategy
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


@dataclass(frozen=True)
class CorrectionRoutingContext:
    key: str
    candidate: dict[str, Any]
    matches: list[dict[str, Any]]


class CorrectionRoutingPolicy(Protocol):
    """May choose a correction route before the LLM is called."""

    def choose(self, ctx: CorrectionRoutingContext) -> CorrectionRoute | None:
        ...


class CorrectionRouteAdjustmentPolicy(Protocol):
    """May rewrite an already selected route after LLM/pre-routing."""

    def adjust(self, ctx: CorrectionRoutingContext, route: CorrectionRoute) -> CorrectionRoute:
        ...


@dataclass(frozen=True)
class TextRoutePolicy:
    text_policy: TextPolicy
    strategy: CorrectionStrategy
    confidence: float
    reason: str | None = None

    def choose(self, ctx: CorrectionRoutingContext) -> CorrectionRoute | None:
        if not self.text_policy.matches(ctx.key):
            return None
        return CorrectionRoute(
            strategy=self.strategy,
            confidence=self.confidence,
            reason=self.reason or self.text_policy.reason,
        )


class LLMRoutingPolicy:
    def __init__(self, chain: Any) -> None:
        self.chain = chain

    def choose(self, ctx: CorrectionRoutingContext) -> CorrectionRoute | None:
        content = self.chain.invoke({
            "candidate_json": json.dumps(ctx.candidate, ensure_ascii=False),
            "rag_json": json.dumps(ctx.matches, ensure_ascii=False),
        })
        data = loads_json_relaxed(content)
        return CorrectionRoute.model_validate({
            "strategy": normalize_strategy(data.get("strategy")),
            "confidence": clamp_float(data.get("confidence"), default=0.5),
            "reason": str(data.get("reason") or "Маршрут выбран агентом."),
        })


@dataclass(frozen=True)
class ForceTextRouteAdjustmentPolicy:
    text_policy: TextPolicy
    strategy: CorrectionStrategy
    min_confidence: float
    reason: str | None = None

    def adjust(self, ctx: CorrectionRoutingContext, route: CorrectionRoute) -> CorrectionRoute:
        if not self.text_policy.matches(ctx.key):
            return route
        return CorrectionRoute(
            strategy=self.strategy,
            confidence=max(route.confidence, self.min_confidence),
            reason=self.reason or self.text_policy.reason,
        )


@dataclass(frozen=True)
class RejectBadCapitalizerPolicy:
    caps_policy: TextPolicy = CLEAN_CYRILLIC_CAPS_TEXT_POLICY
    clean_term_policy: TextPolicy = CLEAN_RUSSIAN_TERM_TEXT_POLICY

    def adjust(self, ctx: CorrectionRoutingContext, route: CorrectionRoute) -> CorrectionRoute:
        if route.strategy != "capitalizer" or self.caps_policy.matches(ctx.key):
            return route
        fallback: CorrectionStrategy = "keep" if self.clean_term_policy.matches(ctx.key) else "skip"
        return CorrectionRoute(
            strategy=fallback,
            confidence=0.75,
            reason="Маршрут capitalizer отклонён safety-net: ключ не является чистым русским капсом.",
        )


@dataclass(frozen=True)
class RejectActiveCorrectionForCleanRussianPolicy:
    clean_term_policy: TextPolicy = CLEAN_RUSSIAN_TERM_TEXT_POLICY

    def adjust(self, ctx: CorrectionRoutingContext, route: CorrectionRoute) -> CorrectionRoute:
        if route.strategy in {"corrector", "restorer"} and self.clean_term_policy.matches(ctx.key):
            return CorrectionRoute(
                strategy="keep",
                confidence=max(route.confidence, 0.85),
                reason="Термин уже читаемый; активная коррекция не требуется.",
            )
        return route


class CorrectionRoutingPipeline:
    def __init__(
        self,
        pre_policies: list[CorrectionRoutingPolicy],
        llm_policy: CorrectionRoutingPolicy,
        adjustment_policies: list[CorrectionRouteAdjustmentPolicy],
    ) -> None:
        self.pre_policies = pre_policies
        self.llm_policy = llm_policy
        self.adjustment_policies = adjustment_policies

    def choose(self, ctx: CorrectionRoutingContext) -> CorrectionRoute:
        route: CorrectionRoute | None = None
        for policy in self.pre_policies:
            route = policy.choose(ctx)
            if route is not None:
                break
        if route is None:
            route = self.llm_policy.choose(ctx)
        if route is None:
            route = CorrectionRoute(strategy="skip", confidence=0.0, reason="Не удалось выбрать маршрут коррекции.")
        for policy in self.adjustment_policies:
            route = policy.adjust(ctx, route)
        return route


def default_correction_routing_pipeline(chain: Any) -> CorrectionRoutingPipeline:
    return CorrectionRoutingPipeline(
        pre_policies=[
            TextRoutePolicy(LATIN_OR_FOREIGN_TEXT_POLICY, "skip", 0.9),
            TextRoutePolicy(CLEAN_CYRILLIC_CAPS_TEXT_POLICY, "capitalizer", 0.85),
            TextRoutePolicy(CLEAN_RUSSIAN_TERM_TEXT_POLICY, "keep", 0.9),
        ],
        llm_policy=LLMRoutingPolicy(chain),
        adjustment_policies=[
            ForceTextRouteAdjustmentPolicy(LATIN_OR_FOREIGN_TEXT_POLICY, "skip", 0.9),
            RejectBadCapitalizerPolicy(),
            RejectActiveCorrectionForCleanRussianPolicy(),
        ],
    )


class CorrectionOrchestratorAgent:
    """Routes a correction candidate to one concrete correction strategy."""

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
        self.routing = default_correction_routing_pipeline(self.chain)

    def choose(self, candidate: dict[str, Any], matches: list[dict[str, Any]]) -> CorrectionRoute:
        ctx = CorrectionRoutingContext(
            key=str(candidate.get("key") or "").strip(),
            candidate=candidate,
            matches=matches,
        )
        return self.routing.choose(ctx)


class LLMCorrectionAgent:
    """Base LLM action: prompt invocation and JSON parsing only."""

    def __init__(self, system_prompt: str) -> None:
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            (
                "human",
                "Candidate JSON:\n{candidate_json}\n\nRAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n{{\"normalized_key\": \"string\", \"normalized_value\": null, \"confidence\": 0.0, \"reason\": \"причина на русском\"}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, state: CorrectionState) -> dict[str, Any]:
        content = self.chain.invoke({
            "candidate_json": json.dumps(candidate_payload(state), ensure_ascii=False),
            "rag_json": json.dumps(state.get("rag_matches", []), ensure_ascii=False),
        })
        return loads_json_relaxed(content)


class LightOCRCorrectorAgent(LLMCorrectionAgent):
    def __init__(self) -> None:
        super().__init__("Ты исправляешь только лёгкие OCR-ошибки в ключе термина. Не восстанавливай по смыслу. Верни только JSON.")


class DefinitionRestorerAgent(LLMCorrectionAgent):
    def __init__(self) -> None:
        super().__init__("Ты осторожно восстанавливаешь сильно повреждённый OCR-ключ по определению и RAG. Если уверенности нет — верни исходный key и confidence 0. Верни только JSON.")


class CorrectionOperation(Protocol):
    strategy: CorrectionStrategy
    status: str
    timer_stage: str

    def suggest(self, state: CorrectionState, route: CorrectionRoute) -> CorrectionSuggestion:
        ...


class KeepOperation:
    strategy: CorrectionStrategy = "keep"
    status = "kept"
    timer_stage = "agent.keep_correction"

    def suggest(self, state: CorrectionState, route: CorrectionRoute) -> CorrectionSuggestion:
        return CorrectionSuggestion(
            normalized_key=state["key"],
            normalized_value=None,
            confidence=0.0,
            reason=route.reason,
            strategy=self.strategy,
            status=self.status,
            orchestrator_reason=route.reason,
        )


class SkipOperation:
    strategy: CorrectionStrategy = "skip"
    status = "skipped"
    timer_stage = "agent.skip_correction"

    def suggest(self, state: CorrectionState, route: CorrectionRoute) -> CorrectionSuggestion:
        return CorrectionSuggestion(
            normalized_key=state["key"],
            normalized_value=None,
            confidence=0.0,
            reason=route.reason,
            strategy=self.strategy,
            status=self.status,
            orchestrator_reason=route.reason,
        )


class CapitalizerOperation:
    strategy: CorrectionStrategy = "capitalizer"
    status = "capitalized"
    timer_stage = "agent.capitalizer"

    def suggest(self, state: CorrectionState, route: CorrectionRoute) -> CorrectionSuggestion:
        normalized = titlecase_cyrillic_caps(state["key"])
        changed = normalized != state["key"]
        return CorrectionSuggestion(
            normalized_key=normalized,
            normalized_value=None,
            confidence=0.75 if changed else 0.0,
            reason=route.reason,
            strategy=self.strategy,
            status=self.status if changed else "kept",
            orchestrator_reason=route.reason,
        )


class LLMCorrectionOperation:
    def __init__(self, strategy: CorrectionStrategy, status: str, timer_stage: str, agent: LLMCorrectionAgent) -> None:
        self.strategy = strategy
        self.status = status
        self.timer_stage = timer_stage
        self.agent = agent

    def suggest(self, state: CorrectionState, route: CorrectionRoute) -> CorrectionSuggestion:
        data = self.agent.invoke(state)
        suggestion = suggestion_from_data(data, state, self.strategy, self.status)
        suggestion.orchestrator_reason = route.reason
        return suggestion


class RouteNode:
    def __init__(self, orchestrator: CorrectionOrchestratorAgent) -> None:
        self.orchestrator = orchestrator

    def __call__(self, state: CorrectionState) -> CorrectionState:
        with AgentTimer("agent.correction_orchestrator", document_id=state["document_id"], item_id=state["item_id"], key=state["key"], model=settings.llm_model) as trace:
            route = self.orchestrator.choose(candidate_payload(state), state.get("rag_matches", []))
            trace.set(selected_strategy=route.strategy, confidence=route.confidence, reason=route.reason[:240])
        return {**state, "route": route.model_dump()}


class CorrectionActionNode:
    """Generic graph node for keep/skip/capitalizer/corrector/restorer.

    The node protocol is identical for all correction actions: read route from
    state, run timed operation, convert failures to a safe suggestion, write the
    suggestion back to state.
    """

    def __init__(self, operation: CorrectionOperation) -> None:
        self.operation = operation

    def __call__(self, state: CorrectionState) -> CorrectionState:
        route = CorrectionRoute.model_validate(state["route"])
        with AgentTimer(self.operation.timer_stage, document_id=state["document_id"], item_id=state["item_id"], key=state["key"]) as trace:
            try:
                suggestion = self.operation.suggest(state, route)
            except Exception as exc:
                suggestion = failed_suggestion(state, self.operation.strategy, route.reason, exc)
            trace.set(normalized_key=suggestion.normalized_key, correction_confidence=suggestion.confidence, status=suggestion.status)
        return with_suggestion(state, suggestion)


class NormalizedKeyPolicy(Protocol):
    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        ...


class EmptyNormalizedKeyPolicy:
    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        return not normalized_key or not normalized_key.strip()


class TooLongNormalizedKeyPolicy:
    def __init__(self, max_chars: int = 100, max_words: int = 8) -> None:
        self.max_chars = max_chars
        self.max_words = max_words

    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        nk = (normalized_key or "").strip()
        return len(nk) > self.max_chars or len(nk.split()) > self.max_words


class DefinitionFragmentNormalizedKeyPolicy:
    def __init__(self, fragments: tuple[str, ...] | None = None) -> None:
        self.fragments = fragments or ("представляет собой", "образуется", "содержит", "является", "под ними", "вследствие")

    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        nk = (normalized_key or "").strip().lower()
        return any(fragment in nk for fragment in self.fragments)


class ForeignOriginalChangedPolicy:
    def __init__(self, foreign_policy: TextPolicy = LATIN_OR_FOREIGN_TEXT_POLICY) -> None:
        self.foreign_policy = foreign_policy

    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        return self.foreign_policy.matches(original_key) and (original_key or "").strip() != (normalized_key or "").strip()


class NormalizedKeyGuard:
    def __init__(self, policies: list[NormalizedKeyPolicy] | None = None) -> None:
        self.policies = policies or [
            EmptyNormalizedKeyPolicy(),
            TooLongNormalizedKeyPolicy(),
            DefinitionFragmentNormalizedKeyPolicy(),
            ForeignOriginalChangedPolicy(),
        ]

    def is_bad(self, original_key: str, normalized_key: str | None) -> bool:
        return any(policy.is_bad(original_key, normalized_key) for policy in self.policies)


class PostFilterNode:
    def __init__(self, guard: NormalizedKeyGuard | None = None) -> None:
        self.guard = guard or NormalizedKeyGuard()

    def __call__(self, state: CorrectionState) -> CorrectionState:
        suggestion = CorrectionSuggestion.model_validate(state["suggestion"])
        if self.guard.is_bad(state["key"], suggestion.normalized_key) or suggestion.confidence <= 0:
            suggestion.normalized_key = state["key"]
            suggestion.normalized_value = None
            if suggestion.strategy in {"corrector", "restorer"}:
                suggestion.status = "unrecoverable"
            suggestion.confidence = 0.0
        return with_suggestion(state, suggestion)


class CorrectionGraph:
    """Pure correction graph. It does not know about SQLAlchemy, DB sessions or RAG storage."""

    def __init__(self) -> None:
        self.orchestrator = CorrectionOrchestratorAgent()
        self.operations: dict[CorrectionStrategy, CorrectionOperation] = {
            "keep": KeepOperation(),
            "skip": SkipOperation(),
            "capitalizer": CapitalizerOperation(),
            "corrector": LLMCorrectionOperation("corrector", "corrected", "agent.light_ocr_corrector", LightOCRCorrectorAgent()),
            "restorer": LLMCorrectionOperation("restorer", "restored", "agent.definition_restorer", DefinitionRestorerAgent()),
        }
        self.graph = self._build_graph()

    def run(self, state: CorrectionState) -> CorrectionSuggestion:
        result = self.graph.invoke(state, config={"recursion_limit": 20})
        return CorrectionSuggestion.model_validate(result["suggestion"])

    def _build_graph(self):
        graph = StateGraph(CorrectionState)
        graph.add_node("route", RouteNode(self.orchestrator))
        for strategy, operation in self.operations.items():
            graph.add_node(strategy, CorrectionActionNode(operation))
        graph.add_node("post_filter", PostFilterNode())
        graph.set_entry_point("route")
        graph.add_conditional_edges("route", route_key, {
            "keep": "keep",
            "capitalizer": "capitalizer",
            "corrector": "corrector",
            "restorer": "restorer",
            "skip": "skip",
        })
        for strategy in self.operations:
            graph.add_edge(strategy, "post_filter")
        graph.add_edge("post_filter", END)
        return graph.compile()





def deterministic_route(key: str) -> CorrectionRoute | None:
    """Compatibility helper for tests/old callers.

    New code should compose CorrectionRoutingPolicy objects instead of calling
    text predicate helpers directly.
    """
    ctx = CorrectionRoutingContext(key=(key or "").strip(), candidate={"key": key}, matches=[])
    for policy in [
        TextRoutePolicy(LATIN_OR_FOREIGN_TEXT_POLICY, "skip", 0.9),
        TextRoutePolicy(CLEAN_CYRILLIC_CAPS_TEXT_POLICY, "capitalizer", 0.85),
        TextRoutePolicy(CLEAN_RUSSIAN_TERM_TEXT_POLICY, "keep", 0.9),
    ]:
        route = policy.choose(ctx)
        if route is not None:
            return route
    return None


def safety_net_route(key: str, route: CorrectionRoute) -> CorrectionRoute:
    """Compatibility helper for tests/old callers.

    New code should use CorrectionRouteAdjustmentPolicy composition.
    """
    ctx = CorrectionRoutingContext(key=(key or "").strip(), candidate={"key": key}, matches=[])
    for policy in [
        ForceTextRouteAdjustmentPolicy(LATIN_OR_FOREIGN_TEXT_POLICY, "skip", 0.9),
        RejectBadCapitalizerPolicy(),
        RejectActiveCorrectionForCleanRussianPolicy(),
    ]:
        route = policy.adjust(ctx, route)
    return route


def normalize_strategy(value: Any) -> CorrectionStrategy:
    strategy = str(value or "skip").strip().lower()
    aliases = {"no_correction": "keep", "unchanged": "keep", "as_is": "keep", "restore": "restorer", "correction": "corrector"}
    strategy = aliases.get(strategy, strategy)
    return strategy if strategy in {"keep", "capitalizer", "corrector", "restorer", "skip"} else "skip"  # type: ignore[return-value]


def route_key(state: CorrectionState) -> str:
    route = state.get("route") or {}
    return normalize_strategy(route.get("strategy"))


def candidate_payload(state: CorrectionState) -> dict[str, Any]:
    return {k: state.get(k) for k in ("key", "value", "source_text", "page_from", "page_to", "confidence", "status")}


def with_suggestion(state: CorrectionState, suggestion: CorrectionSuggestion) -> CorrectionState:
    return {**state, "suggestion": suggestion.model_dump()}


def suggestion_from_data(data: dict[str, Any], state: CorrectionState, strategy: CorrectionStrategy, status: str) -> CorrectionSuggestion:
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


def failed_suggestion(state: CorrectionState, strategy: CorrectionStrategy, orchestrator_reason: str | None, exc: Exception) -> CorrectionSuggestion:
    return CorrectionSuggestion(
        normalized_key=state["key"],
        normalized_value=None,
        confidence=0.0,
        reason=f"Ошибка {strategy}: {exc}",
        strategy=strategy,
        status="unrecoverable" if strategy in {"corrector", "restorer"} else "skipped",
        orchestrator_reason=orchestrator_reason,
    )


def bad_normalized_key(original_key: str, normalized_key: str | None) -> bool:
    """Compatibility wrapper around NormalizedKeyGuard policy composition."""
    return NormalizedKeyGuard().is_bad(original_key, normalized_key)
