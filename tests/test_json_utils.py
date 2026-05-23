import unittest

from mini_ocr.utils.json_utils import loads_json_relaxed


class JsonUtilsTest(unittest.TestCase):
    def test_accepts_fenced_json(self):
        data = loads_json_relaxed('```json\n{"strategy": "keep", "confidence": 0.9}\n```')
        self.assertEqual(data["strategy"], "keep")
        self.assertEqual(data["confidence"], 0.9)

    def test_extracts_object_from_prose(self):
        data = loads_json_relaxed('answer: {"decision": "needs_review"}')
        self.assertEqual(data, {"decision": "needs_review"})

    def test_repairs_invalid_backslash_escape(self):
        data = loads_json_relaxed('{"key": "ГОСТ\\A"}')
        self.assertEqual(data["key"], "ГОСТ\\A")


if __name__ == "__main__":
    unittest.main()
