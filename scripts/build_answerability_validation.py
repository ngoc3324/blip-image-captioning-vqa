"""Build a reviewable answerability validation draft from the annotation manifest."""

import argparse
import json
from pathlib import Path


UNANSWERABLE = [
    "What happened five minutes before this photo was taken?",
    "What is behind the camera?",
    "What is the exact temperature in the room?",
    "What is the Wi-Fi password at this location?",
    "What is the full legal name of the photographer?",
]


def build(source: Path, limit: int) -> dict:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for index, item in enumerate(manifest["items"][:limit]):
        valid_pairs = [pair for pair in item.get("qa_pairs", []) if pair.get("answer") not in {"", "unknown"}]
        if valid_pairs:
            pair = valid_pairs[0]
            rows.append(
                {
                    "id": f"{item['image_id']}-answerable",
                    "image_path": f"dataset/{item['image_path']}",
                    "question": pair["question"],
                    "reference_answer": pair["answer"],
                    "answerable": True,
                    "review_status": "needs_review",
                }
            )
        rows.append(
            {
                "id": f"{item['image_id']}-unanswerable",
                "image_path": f"dataset/{item['image_path']}",
                "question": UNANSWERABLE[index % len(UNANSWERABLE)],
                "reference_answer": None,
                "answerable": False,
                "review_status": "needs_review",
            }
        )
    return {
        "status": "draft_needs_human_review",
        "instructions": "Review every image/question pair; change review_status to approved before calibration.",
        "count": len(rows),
        "items": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/annotations_300.json"))
    parser.add_argument("--output", type=Path, default=Path("dataset/answerability_validation.json"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    payload = build(args.source, args.limit)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {payload['count']} review candidates to {args.output}")


if __name__ == "__main__":
    main()
