from __future__ import annotations

import unittest

from mini_ocr.services.policies.text import (
    CLEAN_CYRILLIC_CAPS_TEXT_POLICY,
    CLEAN_RUSSIAN_TERM_TEXT_POLICY,
    LATIN_OR_FOREIGN_TEXT_POLICY,
    MIXED_CYRILLIC_LATIN_TEXT_POLICY,
    OCR_NOISY_TEXT_POLICY,
    TextFeatures,
)


class TextPoliciesTest(unittest.TestCase):
    def test_text_features_are_shared_between_policies(self) -> None:
        features = TextFeatures.from_text("ОСТATКИ OKAЛH-")
        self.assertGreater(features.cyrillic, 0)
        self.assertGreater(features.latin, 0)
        self.assertGreater(features.letters, 0)

    def test_mixed_cyrillic_latin_policy(self) -> None:
        self.assertTrue(MIXED_CYRILLIC_LATIN_TEXT_POLICY.matches("амendment"))
        self.assertFalse(MIXED_CYRILLIC_LATIN_TEXT_POLICY.matches("технический регламент"))

    def test_latin_or_foreign_policy_matches_english_term(self) -> None:
        self.assertTrue(LATIN_OR_FOREIGN_TEXT_POLICY.matches("basic standard"))
        self.assertTrue(LATIN_OR_FOREIGN_TEXT_POLICY.matches("IDT"))
        self.assertFalse(LATIN_OR_FOREIGN_TEXT_POLICY.matches("международный стандарт"))

    def test_clean_caps_policy_matches_only_clean_cyrillic_caps(self) -> None:
        self.assertTrue(CLEAN_CYRILLIC_CAPS_TEXT_POLICY.matches("ТЕХНИЧЕСКИЙ РЕГЛАМЕНТ"))
        self.assertFalse(CLEAN_CYRILLIC_CAPS_TEXT_POLICY.matches("ОСТATКИ OKAЛH-"))
        self.assertFalse(CLEAN_CYRILLIC_CAPS_TEXT_POLICY.matches("PAORBA-3EAD"))

    def test_clean_russian_policy_matches_normal_term(self) -> None:
        self.assertTrue(CLEAN_RUSSIAN_TERM_TEXT_POLICY.matches("международный стандарт"))
        self.assertFalse(CLEAN_RUSSIAN_TERM_TEXT_POLICY.matches("basic standard"))
        self.assertFalse(CLEAN_RUSSIAN_TERM_TEXT_POLICY.matches("ТЕХНИЧЕСКИЙ РЕГЛАМЕНТ"))

    def test_ocr_noisy_policy(self) -> None:
        self.assertTrue(OCR_NOISY_TEXT_POLICY.matches("ОСТATКИ OKAЛH-"))
        self.assertTrue(OCR_NOISY_TEXT_POLICY.matches("амendment"))
        self.assertFalse(OCR_NOISY_TEXT_POLICY.matches("международный стандарт"))


if __name__ == "__main__":
    unittest.main()

class ServiceAndAbbreviationPolicyTest(unittest.TestCase):
    def test_cyrillic_abbreviation_policy_keeps_all_caps_abbreviation(self):
        from mini_ocr.services.policies.text import CYRILLIC_ABBREVIATION_TEXT_POLICY

        self.assertTrue(CYRILLIC_ABBREVIATION_TEXT_POLICY.matches("СЭВ"))
        self.assertFalse(CYRILLIC_ABBREVIATION_TEXT_POLICY.matches("КЛАССИФИКАЦИЯ"))

    def test_service_heading_policy_detects_table_headers_and_section_headings(self):
        from mini_ocr.services.policies.text import SERVICE_HEADING_TEXT_POLICY

        self.assertTrue(SERVICE_HEADING_TEXT_POLICY.matches("Группа"))
        self.assertTrue(SERVICE_HEADING_TEXT_POLICY.matches("ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ"))
        self.assertFalse(SERVICE_HEADING_TEXT_POLICY.matches("Линия электрической связи"))

