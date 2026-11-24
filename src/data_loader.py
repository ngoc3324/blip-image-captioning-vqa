"""Safe loading and validation for the reviewed annotation manifest."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

QA_TYPES = {"object", "scene", "visual_detail"}

@dataclass(frozen=True)
class QAPair:
    type: str
    question: str
    answer: str

@dataclass(frozen=True)
class AnnotationRecord:
    image_id: str
    image_path: Path
    source_label: str
    caption: str
    qa_pairs: tuple[QAPair, ...]
    status: str

def _image_path(root: Path, relative: str) -> Path:
    base = (root / "dataset").resolve()
    path = (base / relative).resolve()
    if base != path and base not in path.parents:
        raise ValueError(f"Image path escapes dataset directory: {relative}")
    return path

def load_annotations(path: str | Path, repo_root: str | Path | None = None,
                     require_verified: bool = False) -> list[AnnotationRecord]:
    manifest_path = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root else manifest_path.parents[1]
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("Manifest must contain items")
    if require_verified and data.get("annotation_status") != "verified_ground_truth":
        raise ValueError("Manifest is not marked verified_ground_truth")
    seen: set[str] = set(); records = []
    for item in data["items"]:
        _validate_item(item, seen, require_verified)
        image = _image_path(root, item["image_path"])
        if not image.is_file():
            raise FileNotFoundError(f"Missing image: {image}")
        pairs = tuple(QAPair(q["type"], q["question"], q["answer"]) for q in item["qa_pairs"])
        records.append(AnnotationRecord(item["image_id"], image, item["source_label"], item["caption"], pairs, item["status"]))
    if data.get("count") != len(records):
        raise ValueError("Manifest count does not match records")
    return records

def _validate_item(item: dict, seen: set[str], require_verified: bool) -> None:
    required = {"image_id", "image_path", "source_label", "caption", "qa_pairs", "status"}
    if not isinstance(item, dict) or not required.issubset(item):
        raise ValueError("Record is missing required fields")
    image_id = item["image_id"]
    if not isinstance(image_id, str) or not image_id or image_id in seen:
        raise ValueError(f"Invalid or duplicate image_id: {image_id}")
    seen.add(image_id)
    if require_verified and item["status"] != "verified":
        raise ValueError(f"Record {image_id} is not verified")
    if not all(isinstance(item[k], str) and item[k].strip() for k in ("image_path", "source_label", "caption")):
        raise ValueError(f"Record {image_id} has empty text")
    pairs = item["qa_pairs"]
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError(f"Record {image_id} must have 3 QA pairs")
    for pair in pairs:
        if (not isinstance(pair, dict) or pair.get("type") not in QA_TYPES
                or not isinstance(pair.get("question"), str)
                or not isinstance(pair.get("answer"), str)
                or not pair["question"].strip() or not pair["answer"].strip()):
            raise ValueError(f"Invalid QA in record {image_id}")

def validate_image_files(records: list[AnnotationRecord]) -> list[str]:
    from PIL import Image
    errors = []
    for record in records:
        try:
            with Image.open(record.image_path) as image:
                image.verify()
            with Image.open(record.image_path) as image:
                if image.width < 1 or image.height < 1 or image.mode != "RGB":
                    errors.append(f"{record.image_id}: invalid dimensions/mode {image.size}/{image.mode}")
        except Exception as exc:
            errors.append(f"{record.image_id}: {exc}")
    return errors
