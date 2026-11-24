"""Fill source object labels in the generated manifest without rerunning BLIP."""
import json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "dataset/annotations_300.json"
data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
labels = ["door", "lamp", "photo frame", "wardrobe", "window"]
with zipfile.ZipFile(ROOT / "dataset/source/HomeObjects-3K_5class_143_per_class.zip") as archive:
    entries = [x for x in archive.infolist() if x.filename.startswith("images/") and x.filename.lower().endswith((".jpg", ".jpeg", ".png"))]
    for item in data["items"]:
        parts = entries[int(item["source_index"])].filename.split("/")
        label_path = ROOT / "dataset/source/extracted/labels" / parts[1] / f"{Path(parts[2]).stem}.txt"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        item["source_label"] = labels[int(lines[0].split()[0])] if lines else "unknown"
manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print("source labels updated")
