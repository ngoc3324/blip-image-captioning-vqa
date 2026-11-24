"""Evidence aggregation and abstention for generative BLIP outputs.

The raw score is useful for ranking and calibration, but is deliberately not
presented as a probability until thresholds have been fitted on validation data.
"""

from dataclasses import dataclass, field
from math import exp
import json
from pathlib import Path
import re
from typing import Mapping


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def token_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    remaining = list(b)
    matches = 0
    for token in a:
        if token in remaining:
            remaining.remove(token)
            matches += 1
    if not matches:
        return 0.0
    precision, recall = matches / len(a), matches / len(b)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class ConfidencePolicy:
    abstain_below: float = 0.34
    trust_at_or_above: float = 0.62
    calibrated: bool = False
    source: str = "heuristic-v1"

    def __post_init__(self):
        if not 0 <= self.abstain_below < self.trust_at_or_above <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= low < high <= 1")


@dataclass(frozen=True)
class ConfidenceEvidence:
    mean_logprob: float | None = None
    beam_margin: float | None = None
    primary_text: str = ""
    alternative_text: str | None = None
    clip_similarity: float | None = None


@dataclass(frozen=True)
class ConfidenceDecision:
    level: str
    score: float
    should_abstain: bool
    calibrated: bool
    signals: Mapping[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return {
            "trusted": "Tin cậy",
            "review": "Cần kiểm tra",
            "abstain": "Không đủ bằng chứng",
        }[self.level]


class ConfidenceAssessor:
    def __init__(self, policy: ConfidencePolicy | None = None):
        self.policy = policy or ConfidencePolicy()

    def assess(self, evidence: ConfidenceEvidence) -> ConfidenceDecision:
        signals: dict[str, float] = {}
        weights: dict[str, float] = {}

        if evidence.mean_logprob is not None:
            signals["token_likelihood"] = min(1.0, max(0.0, exp(evidence.mean_logprob)))
            weights["token_likelihood"] = 0.55
        if evidence.alternative_text:
            signals["generation_agreement"] = token_f1(evidence.primary_text, evidence.alternative_text)
            weights["generation_agreement"] = 0.30
        if evidence.beam_margin is not None:
            signals["beam_separation"] = 1 - exp(-4 * max(0.0, evidence.beam_margin))
            weights["beam_separation"] = 0.15
        if evidence.clip_similarity is not None:
            signals["image_text_alignment"] = min(1.0, max(0.0, evidence.clip_similarity))
            weights["image_text_alignment"] = 0.25

        total_weight = sum(weights.values())
        score = sum(signals[name] * weights[name] for name in signals) / total_weight if total_weight else 0.0

        normalized = evidence.primary_text.strip().lower()
        no_answer_markers = ("unknown", "not sure", "cannot tell", "can't tell", "không biết", "không rõ")
        if not normalized or any(marker in normalized for marker in no_answer_markers):
            score = min(score, self.policy.abstain_below - 0.01)

        if score < self.policy.abstain_below:
            level = "abstain"
        elif score >= self.policy.trust_at_or_above:
            level = "trusted"
        else:
            level = "review"
        return ConfidenceDecision(
            level=level,
            score=round(score, 6),
            should_abstain=level == "abstain",
            calibrated=self.policy.calibrated,
            signals=signals,
        )


def load_policy(path: str | Path = "artifacts/confidence_calibration.json") -> ConfidencePolicy:
    calibration = Path(path)
    if not calibration.exists():
        return ConfidencePolicy()
    try:
        payload = json.loads(calibration.read_text(encoding="utf-8"))
        recommendation = payload["recommendation"]
        low = float(recommendation["abstain_below"])
        high = float(recommendation["trust_at_or_above"])
        if high <= low:
            high = min(1.0, low + 0.01)
        return ConfidencePolicy(low, high, calibrated=True, source=str(calibration))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ConfidencePolicy()


def render_confidence(decision: ConfidenceDecision) -> str:
    notes = {
        "trusted": "Các tín hiệu sinh câu tương đối nhất quán.",
        "review": "Nên đối chiếu câu trả lời với nội dung ảnh.",
        "abstain": "Mô hình không tìm thấy đủ bằng chứng ổn định trong ảnh.",
    }
    calibration = "đã hiệu chỉnh" if decision.calibrated else "ngưỡng thử nghiệm, chưa hiệu chỉnh trên validation"
    return f"**{decision.label}** · {notes[decision.level]}  \n_{calibration}_"


def roc_auc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for i, (score_i, label_i) in enumerate(zip(scores, labels)):
        if not label_i:
            continue
        for score_j, label_j in zip(scores, labels):
            if label_j:
                continue
            wins += 1.0 if score_i > score_j else 0.5 if score_i == score_j else 0.0
    return wins / (positives * negatives)


def expected_calibration_error(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores) if low <= score < high or index == bins - 1 and score == 1]
        if not members:
            continue
        confidence = sum(scores[i] for i in members) / len(members)
        accuracy = sum(labels[i] for i in members) / len(members)
        error += len(members) / len(scores) * abs(confidence - accuracy)
    return error


def accuracy_coverage_curve(scores: list[float], labels: list[int]) -> list[dict[str, float]]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must have the same non-zero length")
    points = []
    for threshold in sorted(set(scores)):
        accepted = [i for i, score in enumerate(scores) if score >= threshold]
        accuracy = sum(labels[i] for i in accepted) / len(accepted)
        points.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(scores),
                "accuracy": accuracy,
                "selective_risk": 1 - accuracy,
            }
        )
    return points
