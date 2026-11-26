import json
import tempfile
import unittest
from pathlib import Path

from src.data_loader import load_annotations


ROOT = Path(__file__).parents[1]


class DataTests(unittest.TestCase):
    def test_manifest(self):
        manifest_path = ROOT / "dataset/annotations_300.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = load_annotations(manifest_path, ROOT)
        self.assertEqual(len(rows), manifest["count"])
        self.assertEqual(sum(len(row.qa_pairs) for row in rows), manifest["count"] * 3)

    def test_unverified_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.json"
            path.write_text(json.dumps({"count": 0, "items": [], "annotation_status": "draft"}))
            with self.assertRaises(ValueError):
                load_annotations(path, ROOT, require_verified=True)
