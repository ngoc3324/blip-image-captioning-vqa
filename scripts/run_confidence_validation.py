"""Run BLIP VQA on an approved answerability validation set."""

import argparse
import json
from pathlib import Path

from src.confidence import ConfidenceAssessor, ConfidenceEvidence, token_f1
from src.model_runtime import GenerationConfig
from src.vqa import BlipVQA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path, default=Path("dataset/answerability_validation.json"), nargs="?")
    parser.add_argument("--output", type=Path, default=Path("artifacts/confidence_predictions.jsonl"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    payload = json.loads(args.validation.read_text(encoding="utf-8"))
    rows = payload["items"]
    pending = [row["id"] for row in rows if row.get("review_status") != "approved"]
    if pending:
        raise SystemExit(f"Review all validation rows first; {len(pending)} rows are still pending.")

    model = BlipVQA(requested_device=args.device)
    assessor = ConfidenceAssessor()
    generation = GenerationConfig(num_beams=3, max_new_tokens=30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            result = model.answer_with_evidence(row["image_path"], row["question"], generation)
            decision = assessor.assess(
                ConfidenceEvidence(
                    mean_logprob=result.mean_logprob,
                    beam_margin=result.beam_margin,
                    primary_text=result.text,
                    alternative_text=result.alternative_text,
                )
            )
            if row["answerable"]:
                correct = token_f1(result.text, row.get("reference_answer") or "") >= 0.5
            else:
                correct = False
            record = {
                **row,
                "prediction": result.text,
                "alternative": result.alternative_text,
                "mean_logprob": result.mean_logprob,
                "beam_margin": result.beam_margin,
                "confidence_score": decision.score,
                "correct": correct,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row['id']}: {decision.label}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
