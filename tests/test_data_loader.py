import json, tempfile, unittest
from pathlib import Path
from src.data_loader import load_annotations
ROOT=Path(__file__).parents[1]
class DataTests(unittest.TestCase):
    def test_manifest(self):
        rows=load_annotations(ROOT/"dataset/annotations_300.json",ROOT);self.assertEqual(len(rows),300);self.assertEqual(sum(len(x.qa_pairs) for x in rows),900)
    def test_unverified_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.json";p.write_text(json.dumps({"count":0,"items":[],"annotation_status":"draft"}))
            with self.assertRaises(ValueError):load_annotations(p,ROOT,require_verified=True)
