import unittest
from types import SimpleNamespace

from mini_ocr.services.validation_models import ValidationCandidate, ValidationDecision, RagEvidence
from mini_ocr.services.validation_policies import (
    AutoDecisionApplicator,
    DecisionApplicationPolicy,
    EmptyCandidatePolicy,
    NeedsReviewDecisionApplicator,
    RejectedDecisionApplicator,
    ShortDefinitionPolicy,
    StrongRagMatchConfidencePolicy,
    ValidationContext,
    default_heuristic_validation_policy,
)


class ValidationServiceRefactorTest(unittest.TestCase):
    def item(self, **overrides):
        data = dict(
            item_type="term",
            key="стандарт",
            value="нормативный документ",
            source_text="стандарт: нормативный документ",
            page_from=1,
            page_to=1,
            confidence=0.49,
            status="needs_review",
            normalized_key=None,
            normalized_value=None,
            correction_confidence=None,
        )
        data.update(overrides)
        return SimpleNamespace(**data)

    def test_validation_candidate_uses_pydantic_from_attributes(self):
        candidate = ValidationCandidate.model_validate(self.item(), from_attributes=True)
        self.assertEqual(candidate.key, "стандарт")
        self.assertEqual(candidate.value, "нормативный документ")
        self.assertEqual(candidate.confidence, 0.49)

    def test_empty_candidate_policy_rejects_without_agent_or_db(self):
        item = self.item(key="")
        ctx = ValidationContext.from_item(item, [])
        decision = EmptyCandidatePolicy().decide(ctx)
        self.assertIsInstance(decision, ValidationDecision)
        self.assertEqual(decision.decision, "rejected")

    def test_short_definition_policy_rejects(self):
        item = self.item(value="abc")
        ctx = ValidationContext.from_item(item, [])
        decision = ShortDefinitionPolicy(min_chars=8).decide(ctx)
        self.assertIsInstance(decision, ValidationDecision)
        self.assertEqual(decision.decision, "rejected")
        self.assertIn("too short", decision.reason)

    def test_strong_rag_policy_only_adjusts_confidence(self):
        item = self.item(confidence=0.49)
        match = SimpleNamespace(term="стандарт", definition="нормативный документ", score=0.93, source_item_id="1")
        ctx = ValidationContext.from_item(item, [match])
        updated = StrongRagMatchConfidencePolicy().decide(ctx)
        self.assertIsNotNone(updated)
        self.assertGreater(updated.confidence, ctx.confidence)
        self.assertLessEqual(updated.confidence, 0.85)

    def test_heuristic_policy_composes_rules(self):
        item = self.item(confidence=0.49)
        match = SimpleNamespace(term="стандарт", definition="нормативный документ", score=0.93, source_item_id="1")
        decision = default_heuristic_validation_policy().decide(ValidationContext.from_item(item, [match]))
        self.assertEqual(decision.decision, "needs_review")
        self.assertEqual(decision.confidence, 0.85)

    def test_application_policy_replaces_apply_decision_if_chain(self):
        item = self.item(confidence=0.4, status="needs_review")
        policy = DecisionApplicationPolicy([
            RejectedDecisionApplicator(),
            AutoDecisionApplicator(threshold=0.75),
            NeedsReviewDecisionApplicator(),
        ])
        policy.apply(item, ValidationDecision(decision="auto", confidence=0.82, reason="ok"))
        self.assertEqual(item.status, "auto")
        self.assertEqual(item.confidence, 0.82)

        item = self.item(confidence=0.7, status="needs_review")
        policy.apply(item, ValidationDecision(decision="rejected", confidence=0.2, reason="bad"))
        self.assertEqual(item.status, "rejected")
        self.assertEqual(item.confidence, 0.2)

    def test_rag_evidence_is_pydantic_payload(self):
        evidence = RagEvidence(term="стандарт", definition="x" * 500, score=0.9, source_item_id="1")
        payload = evidence.model_dump()
        self.assertEqual(payload["term"], "стандарт")
        self.assertEqual(payload["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
