from __future__ import annotations

import unittest

try:
    from mini_ocr.services.agents.correction import (
        CapitalizerOperation,
        CorrectionRoute,
        CorrectionState,
        KeepOperation,
        PostFilterNode,
        deterministic_route,
        normalize_strategy,
        suggestion_from_data,
    )
except Exception as exc:  # pragma: no cover - optional runtime deps may be missing locally
    raise unittest.SkipTest(f"correction runtime dependencies are not available: {exc}")


class CorrectionRefactorTest(unittest.TestCase):
    def test_deterministic_route_skips_latin_or_foreign(self) -> None:
        route = deterministic_route("basic standard")
        self.assertIsNotNone(route)
        self.assertEqual(route.strategy, "skip")

    def test_deterministic_route_keeps_clean_russian_term(self) -> None:
        route = deterministic_route("международный стандарт")
        self.assertIsNotNone(route)
        self.assertEqual(route.strategy, "keep")

    def test_strategy_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_strategy("restore"), "restorer")
        self.assertEqual(normalize_strategy("no_correction"), "keep")
        self.assertEqual(normalize_strategy("unknown"), "skip")

    def test_keep_operation_returns_unchanged_key(self) -> None:
        state: CorrectionState = {"key": "стандарт"}
        route = CorrectionRoute(strategy="keep", confidence=0.9, reason="ok")
        suggestion = KeepOperation().suggest(state, route)
        self.assertEqual(suggestion.normalized_key, "стандарт")
        self.assertEqual(suggestion.status, "kept")

    def test_capitalizer_operation_titlecases_clean_caps(self) -> None:
        state: CorrectionState = {"key": "ТЕХНИЧЕСКИЙ РЕГЛАМЕНТ"}
        route = CorrectionRoute(strategy="capitalizer", confidence=0.9, reason="caps")
        suggestion = CapitalizerOperation().suggest(state, route)
        self.assertEqual(suggestion.normalized_key, "Технический регламент")
        self.assertEqual(suggestion.status, "capitalized")

    def test_post_filter_rolls_back_zero_confidence_changed_key(self) -> None:
        node = PostFilterNode()
        state: CorrectionState = {
            "key": "Дефект",
            "suggestion": {
                "normalized_key": "дефект продольной ориентации",
                "normalized_value": None,
                "confidence": 0.0,
                "reason": "bad",
                "strategy": "corrector",
                "status": "corrected",
                "orchestrator_reason": "route",
            },
        }
        result = node(state)
        self.assertEqual(result["suggestion"]["normalized_key"], "Дефект")
        self.assertEqual(result["suggestion"]["status"], "unrecoverable")

    def test_suggestion_from_data_marks_unchanged_corrector_as_unchanged(self) -> None:
        state: CorrectionState = {"key": "стандарт"}
        suggestion = suggestion_from_data({"normalized_key": "стандарт", "confidence": 0.9}, state, "corrector", "corrected")
        self.assertEqual(suggestion.confidence, 0.0)
        self.assertEqual(suggestion.status, "unchanged")


if __name__ == "__main__":
    unittest.main()
