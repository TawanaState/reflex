#!/usr/bin/env python3
"""
Unit and statistical tests for ConformalRiskGate.
Verifies UCB computation, Bernstein bounds, monotonicity across tolerances,
and test-set empirical coverage / selective risk guarantees.
"""

import math
import unittest
import numpy as np
from src.risk_gate import ConformalRiskGate, ExitAction


class TestConformalCalibration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_ucb_computation(self):
        gate_hoeff = ConformalRiskGate(epsilon=0.05, delta=0.05, bound_type="hoeffding")
        gate_bern = ConformalRiskGate(epsilon=0.05, delta=0.05, bound_type="empirical_bernstein")

        # Zero error scenario
        errors = np.zeros(100)
        ucb_h = gate_hoeff._compute_ucb(errors, 100)
        ucb_b = gate_bern._compute_ucb(errors, 100)

        self.assertGreater(ucb_h, 0.0)
        self.assertGreater(ucb_b, 0.0)
        # Bernstein should be tighter when variance is 0
        self.assertLessEqual(ucb_b, ucb_h)

    def test_monotonicity_across_tolerances(self):
        """Stricter tolerance epsilon should result in higher (more conservative) confidence threshold."""
        gate = ConformalRiskGate(delta=0.05, bound_type="empirical_bernstein")

        # Synthetic calibration set with realistic confidence-accuracy profile
        n = 1000
        confidences = np.random.uniform(0.5, 1.0, n)
        # Probability of correctness correlates with confidence
        p_correct = confidences**2
        ground_truth = np.random.randint(0, 2, n)
        predictions = np.where(np.random.rand(n) < p_correct, ground_truth, 1 - ground_truth)

        tolerances = [0.10, 0.05, 0.01]
        results = gate.calibrate_multi_tolerance(confidences, predictions, ground_truth, tolerances=tolerances)

        thresholds = [results[eps]["calibrated_threshold"] for eps in tolerances]
        # Verify non-decreasing threshold as epsilon gets stricter
        for i in range(len(thresholds) - 1):
            self.assertLessEqual(thresholds[i], thresholds[i + 1])

    def test_conformal_guarantee_on_test_split(self):
        """Empirical error on test split should strictly satisfy target epsilon."""
        gate = ConformalRiskGate(epsilon=0.05, delta=0.05, bound_type="empirical_bernstein")

        n = 2000
        confidences = np.random.uniform(0.6, 1.0, n)
        p_correct = confidences
        ground_truth = np.random.randint(0, 2, n)
        predictions = np.where(np.random.rand(n) < p_correct, ground_truth, 1 - ground_truth)

        # 50/50 Cal / Test split
        cal_conf, test_conf = confidences[:1000], confidences[1000:]
        cal_pred, test_pred = predictions[:1000], predictions[1000:]
        cal_gt, test_gt = ground_truth[:1000], ground_truth[1000:]

        gate.calibrate(cal_conf, cal_pred, cal_gt, target_epsilon=0.05)
        test_eval = gate.evaluate_test_split(test_conf, test_pred, test_gt)

        # Conformal bound guarantee
        self.assertTrue(test_eval["bound_satisfied"])
        self.assertLessEqual(test_eval["test_selective_error"], 0.05)

    def test_safe_withholding_under_high_noise(self):
        """Under complete noise (random guessing), gate must not release unconfident fast-path exits."""
        gate = ConformalRiskGate(epsilon=0.01, delta=0.05, bound_type="empirical_bernstein")

        n = 500
        confidences = np.random.uniform(0.5, 0.6, n)  # Low confidence
        predictions = np.random.randint(0, 2, n)
        ground_truth = np.random.randint(0, 2, n)     # 50% error rate

        thresh = gate.calibrate(confidences, predictions, ground_truth)
        test_eval = gate.evaluate_test_split(confidences, predictions, ground_truth)

        # Coverage should be 0 because 50% error rate cannot satisfy epsilon=0.01
        self.assertEqual(test_eval["test_exited"], 0)
        self.assertEqual(test_eval["test_coverage"], 0.0)
        self.assertTrue(test_eval["bound_satisfied"])


if __name__ == "__main__":
    unittest.main()

