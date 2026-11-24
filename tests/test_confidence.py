import json
import tempfile
import unittest
from pathlib import Path

from src.confidence import (
    ConfidenceAssessor,
    ConfidenceEvidence,
    accuracy_coverage_curve,
    expected_calibration_error,
    load_policy,
    roc_auc,
)


class ConfidenceTests(unittest.TestCase):
    def test_three_decision_levels(self):
        assessor = ConfidenceAssessor()
        low = assessor.assess(ConfidenceEvidence(mean_logprob=-5, primary_text="answer", alternative_text="different"))
        high = assessor.assess(ConfidenceEvidence(mean_logprob=-0.05, beam_margin=1, primary_text="blue sofa", alternative_text="blue sofa"))
        self.assertTrue(low.should_abstain)
        self.assertEqual(high.level, "trusted")

    def test_metrics(self):
        scores, labels = [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]
        self.assertEqual(roc_auc(scores, labels), 1.0)
        self.assertLess(expected_calibration_error(scores, labels), 0.2)
        self.assertEqual(accuracy_coverage_curve(scores, labels)[-1]["coverage"], 0.25)

    def test_load_calibrated_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"recommendation": {"abstain_below": 0.4, "trust_at_or_above": 0.7}}))
            policy = load_policy(path)
            self.assertTrue(policy.calibrated)
            self.assertEqual(policy.abstain_below, 0.4)
