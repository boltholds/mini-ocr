from __future__ import annotations

import unittest
from types import SimpleNamespace

from mini_ocr.schemas.extraction import ExtractedEntity
try:
    from mini_ocr.services.langgraph_workflow import (
        ExtractedCandidate,
        LowConfidenceNormalizationStrategy,
        NeedsReviewNormalizationStrategy,
        NormalizationPolicy,
        OCRNoisyNormalizationStrategy,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency absent in lightweight test env
    if exc.name != "langgraph":
        raise
    ExtractedCandidate = None
    LowConfidenceNormalizationStrategy = None
    NeedsReviewNormalizationStrategy = None
    NormalizationPolicy = None
    OCRNoisyNormalizationStrategy = None
from mini_ocr.services.section_detector import SectionCandidate


class LangGraphWorkflowRefactorTest(unittest.TestCase):
    def setUp(self):
        if ExtractedCandidate is None:
            self.skipTest("langgraph is not installed in this lightweight test environment")

    def test_section_candidate_uses_pydantic_serialization(self):
        candidate = SectionCandidate(
            section_type="terms",
            text="стандарт: нормативный документ",
            page_from=1,
            page_to=2,
            score=90,
            source="header",
        )

        restored = SectionCandidate.model_validate(candidate.model_dump())

        self.assertEqual(restored.section_type, "terms")
        self.assertEqual(restored.page_to, 2)
        self.assertEqual(restored.text, candidate.text)

    def test_extracted_candidate_validates_nested_entity(self):
        payload = {
            "item_type": "term",
            "entity": {
                "key": "стандарт",
                "value": "нормативный документ",
                "source_text": "стандарт: нормативный документ",
                "confidence": 0.9,
            },
            "page_from": 1,
            "page_to": 1,
            "section_type": "terms",
            "chunk_text": "стандарт: нормативный документ",
        }

        candidate = ExtractedCandidate.model_validate(payload)

        self.assertIsInstance(candidate.entity, ExtractedEntity)
        self.assertEqual(candidate.entity.key, "стандарт")
        self.assertEqual(candidate.extractor, "langchain_llm")

    def test_normalization_policy_is_or_composition(self):
        policy = NormalizationPolicy([
            NeedsReviewNormalizationStrategy(),
            LowConfidenceNormalizationStrategy(threshold=0.75),
            OCRNoisyNormalizationStrategy(),
        ])

        clean_auto = SimpleNamespace(status="auto", confidence=0.95, key="стандарт")
        low_conf = SimpleNamespace(status="auto", confidence=0.49, key="стандарт")
        review = SimpleNamespace(status="needs_review", confidence=0.95, key="стандарт")
        noisy = SimpleNamespace(status="auto", confidence=0.95, key="ОСТATКИ OKAЛH-")

        self.assertFalse(policy.should_normalize(clean_auto))
        self.assertTrue(policy.should_normalize(low_conf))
        self.assertTrue(policy.should_normalize(review))
        self.assertTrue(policy.should_normalize(noisy))


if __name__ == "__main__":
    unittest.main()
