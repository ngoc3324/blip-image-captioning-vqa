"""Normalize VQA ground-truth answers into short, reproducible references.

This script deliberately does not call BLIP VQA. Ground truth must remain
independent from the model whose predictions are being evaluated.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "annotations_300.json"

QUESTIONS = {
    "object": "What is a prominent object?",
    "scene": "What kind of scene is shown?",
    "visual_detail": "What other objects are visible?",
}

# Longer/more specific expressions come first. Answers use a controlled
# vocabulary so exact match and token-F1 are interpretable.
OBJECT_PATTERNS = [
    (r"\bcoffee table\b", "coffee table"),
    (r"\bflat screen tv\b", "television"),
    (r"\bsliding glass door\b", "sliding door"),
    (r"\bframed pictures?\b", "framed pictures"),
    (r"\bceiling fan\b", "ceiling fan"),
    (r"\bnightstand\b", "nightstand"),
    (r"\bsideboard\b", "sideboard"),
    (r"\bbookshel(?:f|ves)?\b", "bookshelf"),
    (r"\bfire\s*place\b", "fireplace"),
    (r"\btelevision\b|\btv\b", "television"),
    (r"\bcouches\b", "couches"),
    (r"\bcouch\b|\bsofa\b", "couch"),
    (r"\bchairs\b", "chairs"),
    (r"\bchair\b", "chair"),
    (r"\bdesks\b", "desks"),
    (r"\bdesk\b", "desk"),
    (r"\btables\b", "tables"),
    (r"\btable\b", "table"),
    (r"\bwindows\b", "windows"),
    (r"\bwindow\b", "window"),
    (r"\bcabinets\b", "cabinets"),
    (r"\bcabinet\b", "cabinet"),
    (r"\bstaircase\b", "staircase"),
    (r"\bpaintings?\b|\bpictures?\b|\bart\b", "pictures"),
    (r"\bplants\b", "plants"),
    (r"\bplant\b", "plant"),
    (r"\bpillows\b", "pillows"),
    (r"\bmirror\b", "mirror"),
    (r"\blaptop\b", "laptop"),
    (r"\bcomputer\b", "computer"),
    (r"\bbed\b", "bed"),
    (r"\bbench\b", "bench"),
    (r"\bshelf\b", "shelf"),
    (r"\bbooks\b", "books"),
    (r"\bcurtain\b", "curtain"),
    (r"\blamp\b", "lamp"),
    (r"\bdoor\b", "door"),
    (r"\bisland\b", "island"),
    (r"\bsuitcase\b", "suitcase"),
    (r"\bflowers\b", "flowers"),
    (r"\bbowl\b", "bowl"),
    (r"\bpumpkins\b", "pumpkins"),
    (r"\blights\b", "lights"),
    (r"\bchandel\w*\b", "chandelier"),
    (r"\bfurniture\b", "furniture"),
]


def scene_from_caption(caption: str) -> str:
    text = caption.lower()
    if "living room" in text:
        return "living room"
    if "bedroom" in text or re.search(r"\bbed\b", text):
        return "bedroom"
    if "dining room" in text:
        return "dining room"
    if "kitchen and dining" in text:
        return "kitchen and dining area"
    if "kitchen" in text:
        return "kitchen"
    if "office" in text:
        return "office"
    if "porch" in text:
        return "porch"
    if "patio" in text:
        return "patio"
    if re.search(r"\bdesk\b|\blaptop\b|\bcomputer\b", text):
        return "workspace"
    if re.search(r"\bcouch(?:es)?\b|\bsofa\b", text):
        return "living room"
    if re.search(r"\btable\b", text) and re.search(r"\bchairs?\b", text):
        return "dining room"
    return "room"


def objects_from_caption(caption: str) -> str:
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, answer in OBJECT_PATTERNS:
        for match in re.finditer(pattern, caption.lower()):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            matches.append((span[0], span[1], answer))
            occupied.append(span)

    answers: list[str] = []
    for _, _, answer in sorted(matches):
        canonical = answer.rstrip("s")
        if any(existing.rstrip("s") == canonical for existing in answers):
            continue
        answers.append(answer)
        if len(answers) == 3:
            break

    if not answers and "bedroom" in caption.lower():
        return "bed"
    if not answers:
        return "furniture"
    if len(answers) == 1:
        return answers[0]
    return ", ".join(answers[:-1]) + " and " + answers[-1]


def main() -> int:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8-sig"))
    for item in data["items"]:
        answers = {
            "object": item["source_label"].strip().lower(),
            "scene": scene_from_caption(item["caption"]),
            "visual_detail": objects_from_caption(item["caption"]),
        }
        item["qa_pairs"] = [
            {"type": kind, "question": QUESTIONS[kind], "answer": answers[kind]}
            for kind in ("object", "scene", "visual_detail")
        ]
        item["annotation_note"] = (
            "Short VQA references normalized from the source label and reviewed caption."
        )

    data["annotation_notes"] = (
        "100 images and 300 short VQA references. Object answers use source labels; "
        "scene and visual-detail answers are normalized from reviewed captions. "
        "BLIP predictions are kept separate for unbiased evaluation."
    )
    DATASET_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Normalized {len(data['items'])} images and {sum(len(x['qa_pairs']) for x in data['items'])} QA pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
