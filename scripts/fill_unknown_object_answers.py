"""Fill placeholder object answers from the retained source labels."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "dataset" / "annotations_300.json"
payload = json.loads(PATH.read_text(encoding="utf-8"))
filled = 0

for item in payload["items"]:
    label = str(item.get("source_label", "")).strip()
    for pair in item.get("qa_pairs", []):
        if pair.get("type") == "object" and str(pair.get("answer", "")).strip().lower() == "unknown":
            if not label:
                raise ValueError(f"Missing source_label for image {item.get('image_id')}")
            pair["answer"] = label
            filled += 1

payload["count"] = len(payload["items"])
payload["annotation_status"] = "auto_filled_needs_review"
payload["annotation_notes"] = "Object answers were filled from source_label; review against each image before using as ground truth."
for item in payload["items"]:
    item["status"] = "auto_filled_needs_review"
    item["annotation_note"] = "Object answer copied from source_label; verify manually against the image."

PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Filled {filled} object placeholders")
