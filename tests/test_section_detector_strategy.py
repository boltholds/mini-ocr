from __future__ import annotations

import unittest

from mini_ocr.services.section_detector import (
    AbbreviationPageStrategy,
    HeaderSectionStrategy,
    PageText,
    SectionCandidate,
    SectionDetectionContext,
    SectionDetectionStrategy,
    SectionDetector,
    TermTablePageStrategy,
)


class SectionDetectorStrategyTest(unittest.TestCase):
    def test_header_strategy_finds_terms_section(self):
        detector = SectionDetector(strategies=[HeaderSectionStrategy("terms", ["термины и определения"])])

        candidates = detector.detect([
            (1, "1 Область применения"),
            (2, "Термины и определения\nстандарт: нормативный документ"),
        ])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].section_type, "terms")
        self.assertEqual(candidates[0].source, "header")
        self.assertEqual(candidates[0].page_from, 2)
        self.assertIn("стандарт", candidates[0].text)

    def test_term_table_strategy_detects_table_like_page(self):
        detector = SectionDetector(strategies=[TermTablePageStrategy()])

        candidates = detector.detect([
            PageText(5, "Термин\nОпределение\n1. стандарт\n2. регламент", layout_type="table_like"),
        ])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].section_type, "terms")
        self.assertEqual(candidates[0].source, "table")
        self.assertGreaterEqual(candidates[0].score, 75)

    def test_abbreviation_strategy_detects_abbreviation_page(self):
        detector = SectionDetector(strategies=[AbbreviationPageStrategy()])

        candidates = detector.detect([
            PageText(3, "Перечень сокращений и обозначений\nIDT identical\nMOD modified", layout_type="table_like"),
        ])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].section_type, "abbreviations")

    def test_detector_deduplicates_candidates_from_multiple_strategies(self):
        class DuplicateStrategy:
            def detect(self, ctx: SectionDetectionContext) -> list[SectionCandidate]:
                return [
                    SectionCandidate(section_type="terms", text="a", page_from=1, page_to=1, score=90, source="table"),
                    SectionCandidate(section_type="terms", text="b", page_from=1, page_to=1, score=80, source="table"),
                ]

        detector = SectionDetector(strategies=[DuplicateStrategy()])
        candidates = detector.detect([(1, "Термин Определение")])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "a")

    def test_custom_strategy_can_be_injected(self):
        class CustomStrategy:
            def detect(self, ctx: SectionDetectionContext) -> list[SectionCandidate]:
                return [SectionCandidate(section_type="custom", text="text", page_from=10, page_to=10, score=100, source="custom")]

        detector = SectionDetector(strategies=[CustomStrategy()])
        candidates = detector.detect([])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].section_type, "custom")


if __name__ == "__main__":
    unittest.main()
