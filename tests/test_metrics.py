import unittest
from src.metrics import exact_match,token_f1
class MetricTests(unittest.TestCase):
    def test_normalized_match(self):self.assertEqual(exact_match("A red sofa","red sofa"),1)
    def test_partial_f1(self):self.assertGreater(token_f1("red sofa","large red sofa"),.5)
