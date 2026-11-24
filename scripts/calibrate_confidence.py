"""Evaluate confidence predictions and recommend abstention/trust thresholds."""

import argparse
import json
from pathlib import Path

from src.confidence import accuracy_coverage_curve, expected_calibration_error, roc_auc


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("items", payload) if isinstance(payload, dict) else payload


def recommend(curve: list[dict], minimum_coverage: float, target_error_reduction: float) -> dict:
    baseline_risk = curve[0]["selective_risk"]
    risk_target = baseline_risk * (1 - target_error_reduction)
    candidates = [p for p in curve if p["coverage"] >= minimum_coverage and p["selective_risk"] <= risk_target]
    abstain = min(candidates, key=lambda p: p["threshold"], default=min(curve, key=lambda p: abs(p["coverage"] - minimum_coverage)))
    trust_candidates = [p for p in curve if p["accuracy"] >= 0.90]
    trust = min(trust_candidates, key=lambda p: p["threshold"], default=max(curve, key=lambda p: p["accuracy"]))
    trust_threshold = max(abstain["threshold"] + 0.01, trust["threshold"])
    return {"abstain_below": abstain["threshold"], "trust_at_or_above": min(1.0, trust_threshold), "operating_point": abstain}


def main():
    parser = argparse.ArgumentParser(description="Calibrate confidence from approved validation predictions")
    parser.add_argument("predictions", type=Path, help="JSON/JSONL rows with confidence_score and correct")
    parser.add_argument("--minimum-coverage", type=float, default=0.70)
    parser.add_argument("--target-error-reduction", type=float, default=0.30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/confidence_calibration.json"))
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    if not rows or any("confidence_score" not in row or "correct" not in row for row in rows):
        raise SystemExit("Each prediction needs confidence_score and correct fields.")
    scores = [float(row["confidence_score"]) for row in rows]
    labels = [int(bool(row["correct"])) for row in rows]
    curve = accuracy_coverage_curve(scores, labels)
    result = {
        "sample_count": len(rows),
        "auroc": roc_auc(scores, labels),
        "expected_calibration_error": expected_calibration_error(scores, labels),
        "accuracy_coverage": curve,
        "recommendation": recommend(curve, args.minimum_coverage, args.target_error_reduction),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["recommendation"], indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
