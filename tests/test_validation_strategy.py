import unittest

from mini_ocr.schemas.extraction import ExtractedEntity
from mini_ocr.services.extraction_validator import (
    ExtractionValidator,
    SequentialValidationStrategy,
    ValidationContext,
    ValidationDecision,
)


class ValidationStrategyTest(unittest.TestCase):
    def test_unknown_item_type_is_rejected(self):
        entity = ExtractedEntity(key="стандарт", value="нормативный документ", source_text="стандарт: нормативный документ", confidence=0.9)
        decision = ExtractionValidator().validate("unknown", entity, entity.source_text, "terms")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "unknown item_type")

    def test_abbreviation_strategy_rejects_wrong_section(self):
        entity = ExtractedEntity(key="IDT", value="identical", source_text="IDT identical", confidence=0.9)
        decision = ExtractionValidator().validate("abbreviation", entity, entity.source_text, "terms")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "abbreviation outside abbreviation section")

    def test_sequential_strategy_stops_on_first_rejection(self):
        calls = []

        class RejectRule:
            def apply(self, ctx):
                calls.append("reject")
                return ValidationDecision(False, 0.0, "rejected", "stop")

        class LaterRule:
            def apply(self, ctx):
                calls.append("later")
                return None

        ctx = ValidationContext("term", "k", "value", "k value", "k value", 0.9, "terms")
        decision = SequentialValidationStrategy([RejectRule(), LaterRule()]).validate(ctx)

        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "stop")
        self.assertEqual(calls, ["reject"])

    def test_foreign_alias_is_kept_for_review_not_auto(self):
        entity = ExtractedEntity(key="D. Rohraalzhaut", value="foreign alias", source_text="D. Rohraalzhaut foreign alias", confidence=0.95)
        decision = ExtractionValidator().validate("term", entity, entity.source_text, "terms")
        self.assertTrue(decision.keep)
        self.assertLessEqual(decision.confidence, 0.49)
        self.assertEqual(decision.status, "needs_review")


if __name__ == "__main__":
    unittest.main()
