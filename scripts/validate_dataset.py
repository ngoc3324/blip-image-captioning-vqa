import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_annotations, validate_image_files
if __name__ == "__main__":
    records=load_annotations("dataset/annotations_300.json", require_verified=False); errors=validate_image_files(records)
    print(f"records={len(records)}\nqa_pairs={sum(len(r.qa_pairs) for r in records)}\nverified={sum(r.status=='verified' for r in records)}\nimage_errors={len(errors)}")
    [print(f"ERROR {e}") for e in errors]; raise SystemExit(bool(errors))
