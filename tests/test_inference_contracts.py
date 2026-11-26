import unittest
from pathlib import Path

from src.model_runtime import GenerationConfig


class RuntimeTests(unittest.TestCase):
    def test_greedy_has_no_temperature(self):self.assertNotIn("temperature",GenerationConfig().kwargs())
    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):GenerationConfig(temperature=.7).kwargs()


class UiCopyTests(unittest.TestCase):
    def test_review_warnings_are_not_rendered(self):
        app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("Câu mô tả này nên được đối chiếu thêm với ảnh.", app_source)
        self.assertNotIn("Câu trả lời nên được kiểm tra lại trên ảnh.", app_source)
