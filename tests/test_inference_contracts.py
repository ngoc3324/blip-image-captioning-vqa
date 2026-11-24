import unittest
from src.model_runtime import GenerationConfig
class RuntimeTests(unittest.TestCase):
    def test_greedy_has_no_temperature(self):self.assertNotIn("temperature",GenerationConfig().kwargs())
    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):GenerationConfig(temperature=.7).kwargs()
