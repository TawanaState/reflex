"""
Unit tests for Conformal Risk Gate and calibration logic.
"""

import unittest
import numpy as np
import torch
from src.risk_gate import ConformalRiskGate, ExitAction, SlotPrediction


class TestConformalRiskGate(unittest.TestCase):
    def setUp(self):
        self.gate = ConformalRiskGate(epsilon=0.05, delta=0.05, default_threshold=0.80)

    def test_calibration_bounds(self):
        np.random.seed(42)
        n = 500
        # Simulated confidences in [0.6, 1.0]
        confidences = np.random.uniform(0.6, 1.0, n)
        # Errors happen mostly at lower confidences
        ground_truth = np.ones(n, dtype=int)
        predictions = ground_truth.copy()
        # Inject error when confidence < 0.82
        for i in range(n):
            if confidences[i] < 0.82:
                if np.random.rand() < 0.25:
                    predictions[i] = 0

        cal_thresh = self.gate.calibrate(confidences, predictions, ground_truth)
        self.assertTrue(self.gate.is_calibrated)
        self.assertGreaterEqual(cal_thresh, 0.80)
        # Verify empirical error on exited subset is within acceptable range
        exit_mask = confidences >= cal_thresh
        emp_err = np.mean(predictions[exit_mask] != ground_truth[exit_mask])
        self.assertLessEqual(emp_err, 0.05)

    def test_evaluate_logits_exit_fast_path(self):
        self.gate.confidence_threshold = 0.80
        # Create logits strongly favoring token index 1
        logits = torch.tensor([-5.0, 5.0, -5.0])
        slot_logits = {
            "route": (logits, ["click", "scroll", "wait"], [0, 1, 2])
        }
        res = self.gate.evaluate_logits(slot_logits, has_synthesis_field=False)
        self.assertEqual(res.action, ExitAction.EXIT)
        self.assertTrue(res.is_safe_to_exit)
        self.assertEqual(res.slot_predictions["route"].top_label, "scroll")
        self.assertGreaterEqual(res.overall_confidence, 0.99)
        self.assertLess(res.overall_entropy, 0.05)

    def test_evaluate_logits_expand_on_uncertainty(self):
        self.gate.confidence_threshold = 0.80
        # Create uniform logits (high uncertainty / entropy)
        logits = torch.tensor([0.1, 0.1, 0.1])
        slot_logits = {
            "route": (logits, ["click", "scroll", "wait"], [0, 1, 2])
        }
        res = self.gate.evaluate_logits(slot_logits, has_synthesis_field=False)
        self.assertEqual(res.action, ExitAction.EXPAND)
        self.assertFalse(res.is_safe_to_exit)
        self.assertIn("Confidence", res.escalation_reason)

    def test_evaluate_logits_expand_on_synthesis_request(self):
        self.gate.confidence_threshold = 0.50
        logits = torch.tensor([10.0, 0.0])
        slot_logits = {
            "route": (logits, ["click", "scroll"], [0, 1])
        }
        res = self.gate.evaluate_logits(slot_logits, has_synthesis_field=True)
        self.assertEqual(res.action, ExitAction.EXPAND)
        self.assertFalse(res.is_safe_to_exit)
        self.assertIn("synthesis", res.escalation_reason)


if __name__ == "__main__":
    unittest.main()

