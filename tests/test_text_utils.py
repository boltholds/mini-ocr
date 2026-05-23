import unittest

from mini_ocr.utils.text import (
    is_clean_cyrillic_caps,
    looks_clean_russian_term,
    looks_latin_or_foreign,
    looks_ocr_noisy,
    titlecase_cyrillic_caps,
)


class TextUtilsTest(unittest.TestCase):
    def test_clean_russian_term_is_kept(self):
        self.assertTrue(looks_clean_russian_term("межгосударственная стандартизация"))
        self.assertTrue(looks_clean_russian_term("технический регламент"))

    def test_latin_terms_are_foreign_not_russian(self):
        self.assertTrue(looks_latin_or_foreign("basic standard"))
        self.assertTrue(looks_latin_or_foreign("IDT"))
        self.assertFalse(looks_clean_russian_term("basic standard"))

    def test_mixed_cyrillic_latin_is_noisy(self):
        self.assertTrue(looks_ocr_noisy("ОСТATКИ OKAЛH-"))
        self.assertTrue(looks_ocr_noisy("амendment"))

    def test_capitalizer_accepts_only_clean_cyrillic_caps(self):
        self.assertTrue(is_clean_cyrillic_caps("ПРОПЛАВЛЕНИЕ"))
        self.assertEqual(titlecase_cyrillic_caps("ПРОПЛАВЛЕНИЕ"), "Проплавление")
        self.assertFalse(is_clean_cyrillic_caps("PAORBA-3EAD"))
        self.assertFalse(is_clean_cyrillic_caps("ОСТATКИ OKAЛH-"))


if __name__ == "__main__":
    unittest.main()
