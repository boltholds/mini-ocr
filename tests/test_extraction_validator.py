import unittest

from mini_ocr.schemas.extraction import ExtractedEntity
from mini_ocr.services.extraction_validator import ExtractionValidator


class ExtractionValidatorTest(unittest.TestCase):
    def test_keeps_grounded_term(self):
        entity = ExtractedEntity(
            key="стандарт",
            value="нормативный документ, разработанный на основе консенсуса",
            source_text="стандарт: нормативный документ, разработанный на основе консенсуса",
            confidence=0.95,
        )
        decision = ExtractionValidator().validate("term", entity, entity.source_text, "terms")
        self.assertTrue(decision.keep)
        self.assertEqual(decision.status, "auto")

    def test_rejects_term_in_abbreviation_section(self):
        entity = ExtractedEntity(key="стандарт", value="нормативный документ", source_text="стандарт: нормативный документ", confidence=0.8)
        decision = ExtractionValidator().validate("term", entity, entity.source_text, "abbreviations")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "term inside abbreviation section")

    def test_rejects_service_phrase(self):
        entity = ExtractedEntity(key="нормативный документ", value="см. документ", source_text="нормативный документ: см. документ", confidence=0.9)
        decision = ExtractionValidator().validate("term", entity, entity.source_text, "terms")
        self.assertFalse(decision.keep)


if __name__ == "__main__":
    unittest.main()

class ExtractionValidatorServiceNoiseTest(unittest.TestCase):
    def test_rejects_table_header_key(self):
        validator = ExtractionValidator()
        entity = ExtractedEntity(key="Группа", value="Шифр Наименование Определение", source_text="Группа Шифр Наименование Определение", confidence=0.49)
        decision = validator.validate("term", entity, "Группа Шифр Наименование Определение", "terms")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "table header is not a term")

    def test_rejects_section_heading_key(self):
        validator = ExtractionValidator()
        entity = ExtractedEntity(key="ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ", value="Термины и определения", source_text="ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ", confidence=0.9)
        decision = validator.validate("term", entity, "ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ", "terms")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "section heading is not a term")

