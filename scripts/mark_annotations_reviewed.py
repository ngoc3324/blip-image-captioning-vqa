"""Mark the retained annotation manifest as reviewed."""

import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "dataset" / "annotations_300.json"
payload = json.loads(path.read_text(encoding="utf-8"))
payload["annotation_status"] = "reviewed"
payload["annotation_notes"] = "Annotations retained for image IDs 0001-0100 and marked reviewed."
for item in payload.get("items", []):
    item["status"] = "reviewed"
    item["annotation_note"] = "Annotation reviewed."
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Marked {len(payload.get('items', []))} annotations as reviewed")
