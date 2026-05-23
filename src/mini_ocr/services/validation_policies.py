from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence, Any

from mini_ocr.services.validation_models import RagMatchLike, ValidationCandidate, ValidationDecision


class ValidationDecisionPolicy(Protocol):
    def decide(self, ctx: "ValidationContext") -> ValidationDecision | "ValidationContext" | None:
        ...


class ValidationDecisionApplicator(Protocol):
    def apply(self, ctx: "DecisionApplicationContext") -> bool:
        ...


@dataclass(frozen=True, slots=True)
class ValidationContext:
    item: Any
    candidate: ValidationCandidate
    matches: Sequence[RagMatchLike]
    confidence: float
    reason_prefix: str = ""

    @classmethod
    def from_item(
        cls,
        item: Any,
        matches: Sequence[RagMatchLike],
        reason_prefix: str = "",
    ) -> "ValidationContext":
        return cls(
            item=item,
            candidate=ValidationCandidate.model_validate(item, from_attributes=True),
            matches=matches,
            confidence=float(getattr(item, "confidence", None) or 0.5),
            reason_prefix=reason_prefix,
        )

    def with_confidence(self, confidence: float) -> "ValidationContext":
        return replace(self, confidence=confidence)


@dataclass(frozen=True, slots=True)
class DecisionApplicationContext:
    item: Any
    decision: ValidationDecision


class EmptyCandidatePolicy:
    def decide(self, ctx: ValidationContext) -> ValidationDecision | None:
        if not ctx.candidate.key or not ctx.candidate.value:
            return ValidationDecision(decision="rejected", confidence=0.0, reason="Empty candidate")
        return None


class ShortDefinitionPolicy:
    def __init__(self, min_chars: int = 8) -> None:
        self.min_chars = min_chars

    def decide(self, ctx: ValidationContext) -> ValidationDecision | None:
        if len((ctx.candidate.value or "").strip()) < self.min_chars:
            reason = _join_reason(ctx.reason_prefix, "definition is too short")
            return ValidationDecision(decision="rejected", confidence=0.2, reason=reason)
        return None


class StrongRagMatchConfidencePolicy:
    def __init__(self, threshold: float = 0.88, max_confidence: float = 0.85) -> None:
        self.threshold = threshold
        self.max_confidence = max_confidence

    def decide(self, ctx: ValidationContext) -> ValidationContext | None:
        best = ctx.matches[0] if ctx.matches else None
        if best is None or float(best.score) <= self.threshold:
            return None
        return ctx.with_confidence(max(ctx.confidence, min(float(best.score), self.max_confidence)))


class NeedsReviewFallbackPolicy:
    def decide(self, ctx: ValidationContext) -> ValidationDecision:
        return ValidationDecision(
            decision="needs_review",
            confidence=ctx.confidence,
            reason=ctx.reason_prefix or "heuristic validation",
        )


class SequentialValidationDecisionPolicy:
    """Runs validation policies one by one."""

    def __init__(self, policies: Sequence[ValidationDecisionPolicy]) -> None:
        self.policies = list(policies)

    def decide(self, ctx: ValidationContext) -> ValidationDecision:
        current = ctx
        for policy in self.policies:
            result = policy.decide(current)
            if result is None:
                continue
            if isinstance(result, ValidationDecision):
                return result
            current = result
        return NeedsReviewFallbackPolicy().decide(current)


class RejectedDecisionApplicator:
    def apply(self, ctx: DecisionApplicationContext) -> bool:
        if ctx.decision.decision != "rejected":
            return False
        ctx.item.status = "rejected"
        ctx.item.confidence = min(float(getattr(ctx.item, "confidence", None) or 0.5), ctx.decision.confidence)
        return True


class AutoDecisionApplicator:
    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def apply(self, ctx: DecisionApplicationContext) -> bool:
        if ctx.decision.decision != "auto" or ctx.decision.confidence < self.threshold:
            return False
        ctx.item.status = "auto"
        ctx.item.confidence = min(max(float(getattr(ctx.item, "confidence", None) or 0.5), ctx.decision.confidence), 0.95)
        return True


class NeedsReviewDecisionApplicator:
    def apply(self, ctx: DecisionApplicationContext) -> bool:
        ctx.item.status = "needs_review"
        ctx.item.confidence = min(float(getattr(ctx.item, "confidence", None) or ctx.decision.confidence or 0.5), ctx.decision.confidence)
        return True


class DecisionApplicationPolicy:
    def __init__(self, applicators: Sequence[ValidationDecisionApplicator]) -> None:
        self.applicators = list(applicators)

    def apply(self, item: Any, decision: ValidationDecision) -> None:
        ctx = DecisionApplicationContext(item=item, decision=decision)
        for applicator in self.applicators:
            if applicator.apply(ctx):
                return


def default_heuristic_validation_policy() -> SequentialValidationDecisionPolicy:
    return SequentialValidationDecisionPolicy([
        EmptyCandidatePolicy(),
        ShortDefinitionPolicy(min_chars=8),
        StrongRagMatchConfidencePolicy(threshold=0.88, max_confidence=0.85),
        NeedsReviewFallbackPolicy(),
    ])


def default_decision_application_policy(auto_threshold: float = 0.75) -> DecisionApplicationPolicy:
    return DecisionApplicationPolicy([
        RejectedDecisionApplicator(),
        AutoDecisionApplicator(threshold=auto_threshold),
        NeedsReviewDecisionApplicator(),
    ])


def _join_reason(prefix: str, reason: str) -> str:
    return f"{prefix}; {reason}" if prefix else reason
