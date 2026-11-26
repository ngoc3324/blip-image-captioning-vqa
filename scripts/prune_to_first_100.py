"""Keep only dataset records for image IDs 0001 through 0100."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def keep_id(value: str) -> bool:
    try:
        return int(Path(str(value)).stem) <= 100
    except (TypeError, ValueError):
        return False


def prune_annotations():
    path = ROOT / "dataset" / "annotations_300.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"] = [item for item in payload["items"] if keep_id(item.get("image_id"))]
    payload["count"] = len(payload["items"])
    payload["annotation_notes"] = "Pruned to image IDs 0001-0100; review annotations before ground-truth use."
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prune_validation():
    path = ROOT / "dataset" / "answerability_validation.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"] = [item for item in payload.get("items", []) if keep_id(item.get("id", "").split("-")[0])]
    payload["count"] = len(payload["items"])
    payload["instructions"] = "Review the retained validation rows before calibration."
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    prune_annotations()
    prune_validation()
    print("Kept dataset records for image IDs 0001-0100")
