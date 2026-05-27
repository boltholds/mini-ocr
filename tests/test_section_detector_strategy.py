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
    has_numbered_term_rows,
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

class SectionDetectorCandidatePolicyTest(unittest.TestCase):
    def test_small_table_doc_prefers_focused_table_candidates_over_broad_headers(self):
        detector = SectionDetector()
        pages = [
            PageText(1, "ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ\nТермин\nОпределение\n1. первый\n2. второй", layout_type="table_like"),
            PageText(2, "Термин\nОпределение\n1. третий\n2. четвертый", layout_type="table_like"),
            PageText(3, "обычный текст", layout_type="plain"),
        ]

        candidates = detector.detect(pages)

        self.assertTrue(candidates)
        self.assertTrue(all(c.source == "table" for c in candidates))
        self.assertEqual({(c.page_from, c.page_to) for c in candidates}, {(1, 1), (2, 2)})



class NumberedTermRowsPreservePolicyTest(unittest.TestCase):
    def test_detects_subsection_numbered_term_rows(self):
        self.assertTrue(has_numbered_term_rows("2.1. Устройство — часть объекта\n2.2. Элемент — часть объекта"))
        self.assertFalse(has_numbered_term_rows("1. первый пункт 2. второй пункт"))

    def test_table_pruning_preserves_header_window_with_numbered_term_rows(self):
        detector = SectionDetector(max_pages_after_header=2)
        pages = [
            PageText(1, "ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ\nТермин\nОпределение\n2.1. Устройство — часть объекта\n2.2. Элемент — часть объекта", layout_type="table_like"),
            PageText(2, "Термин\nОпределение\n2.3. Линия — связь между частями\n2.4. Объект — установка", layout_type="table_like"),
            PageText(3, "обычный текст", layout_type="plain"),
        ]

        candidates = detector.detect(pages)

        self.assertTrue(any(c.source == "table" for c in candidates))
        self.assertTrue(any(c.source == "header" and c.page_from == 1 and c.page_to == 2 for c in candidates))
