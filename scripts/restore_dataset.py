"""Restore numbered images from the referenced source archive."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="dataset/source/HomeObjects-3K_5class_143_per_class.zip")
    parser.add_argument("--manifest", default="dataset/annotations_verified.json")
    parser.add_argument("--output", default="dataset/images")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    records = sorted(manifest["items"], key=lambda row: row["source_index"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        entries = [info for info in archive.infolist()
                   if info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                   and info.filename.startswith("images/")]
        if len(records) > len(entries):
            raise ValueError(f"Need {len(records)} images but archive has {len(entries)}")
        mapping = []
        for record in records:
            source = entries[record["source_index"]]
            target = output / f"{record['image_id']}.jpg"
            with archive.open(source) as source_stream, target.open("wb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
            mapping.append({"image_id": record["image_id"],
                            "source_index": record["source_index"],
                            "source_entry": source.filename,
                            "target": str(target).replace("\\", "/")})
    Path("dataset/restoration_manifest.json").write_text(
        json.dumps({"method": "source_archive_entry_order", "items": mapping}, indent=2),
        encoding="utf-8")
    print(f"Restored {len(mapping)} images to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
