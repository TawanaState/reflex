"""
Reflex Conformal Risk Gate: Calibrated stopping and early-exit policy for single-step reflex decisions.
Implements Conformal Risk Control (Angelopoulos et al.) with Upper Confidence Bounds (UCB)
to guarantee P(error | exit) <= eps.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class ExitAction(str, Enum):
    EXIT = "exit"       # Terminate at Step 1: Emit typed struct directly
    EXPAND = "expand"   # Escalate / Expand: Materialize generative canvas or multi-step refine


@dataclass
class SlotPrediction:
    """Prediction for a single typed slot from Step 1 denoise logits."""
    slot_name: str
    top_label: str
    top_token_id: int
    confidence: float                    # max_y f(x)_y
    entropy: float                       # Shannon entropy H(f(x))
    confidence_gap: float                # Top-1 prob - Top-2 prob (Prophet gap)
    probabilities: Dict[str, float]      # label -> normalized probability
    candidate_labels: List[str]


@dataclass
class RiskGateResult:
    """Result of the Conformal Risk Gate evaluation."""
    action: ExitAction
    overall_confidence: float
    overall_entropy: float
    is_safe_to_exit: bool
    threshold_lambda: float
    target_epsilon: float
    slot_predictions: Dict[str, SlotPrediction]
    escalation_reason: Optional[str] = None


class ConformalRiskGate:
    """
    Conformal Risk Gate for Project Reflex.
    Calibrates early-exit thresholds on a validation split to provide statistical guarantees
    on fast-path error rate: P(error | exit) <= epsilon.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        delta: float = 0.05,
        default_threshold: float = 0.85,
        bound_type: str = "empirical_bernstein",
    ):
        """
        Args:
            epsilon: Maximum allowed error rate on fast-path exits (e.g. 0.05 for 5% max error).
            delta: Confidence parameter for Upper Confidence Bound (e.g. 0.05 for 95% confidence).
            default_threshold: Default confidence threshold (1 - lambda) if uncalibrated.
            bound_type: Bound method - 'empirical_bernstein' or 'hoeffding'.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.confidence_threshold = default_threshold
        self.bound_type = bound_type
        self.is_calibrated = False
        self.calibrated_lambda = 1.0 - default_threshold
        self.calibration_stats: Dict[str, Any] = {}

    def _compute_ucb(self, errors_exited: np.ndarray, n_exit: int) -> float:
        """Computes Upper Confidence Bound on empirical risk using selected bound."""
        if n_exit == 0:
            return 1.0
        emp_risk = float(np.mean(errors_exited))

        if self.bound_type == "empirical_bernstein" and n_exit > 2:
            var_emp = float(np.var(errors_exited, ddof=1))
            log_term = math.log(2.0 / self.delta)
            term1 = math.sqrt(2.0 * var_emp * log_term / n_exit)
            term2 = 7.0 * log_term / (3.0 * (n_exit - 1))
            return min(1.0, emp_risk + term1 + term2)
        else:
            # Hoeffding inequality: emp_risk + sqrt(ln(1/delta) / (2 * n_exit))
            log_term = math.log(1.0 / self.delta)
            return min(1.0, emp_risk + math.sqrt(log_term / (2.0 * n_exit)))

    def calibrate(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        target_epsilon: Optional[float] = None,
    ) -> float:
        """
        Calibrates the exit threshold lambda* on a calibration set D_cal = {(x_i, y_i)}_{i=1}^N:
            lambda* = sup { lambda in [0, 1] : R_UCB(lambda) <= epsilon }

        Args:
            confidences: Array of predicted top-1 confidences max_y f(x_i)_y in [0, 1].
            predictions: Array of top-1 predicted label indices.
            ground_truth: Array of true label indices.
            target_epsilon: Optional override for tolerance epsilon.

        Returns:
            Calibrated confidence threshold (1 - lambda*).
        """
        eps = target_epsilon if target_epsilon is not None else self.epsilon
        n_samples = len(confidences)
        if n_samples == 0:
            return self.confidence_threshold

        errors = (predictions != ground_truth).astype(float)
        # Search candidate thresholds from fine resolution in [0.50, 0.999]
        threshold_candidates = np.linspace(0.50, 0.999, 500)
        best_thresh = 0.999  # Conservative default

        for thresh in reversed(threshold_candidates):
            exit_mask = confidences >= thresh
            n_exit = int(np.sum(exit_mask))
            if n_exit == 0:
                continue

            errors_exited = errors[exit_mask]
            ucb = self._compute_ucb(errors_exited, n_exit)

            if ucb <= eps:
                best_thresh = thresh
                break

        self.confidence_threshold = float(best_thresh)
        self.calibrated_lambda = float(1.0 - best_thresh)
        self.is_calibrated = True

        exit_mask = confidences >= best_thresh
        n_exit = int(np.sum(exit_mask))
        errors_exited = errors[exit_mask] if n_exit > 0 else np.array([])
        emp_risk = float(np.mean(errors_exited)) if n_exit > 0 else 0.0
        ucb = self._compute_ucb(errors_exited, n_exit) if n_exit > 0 else 1.0

        self.calibration_stats = {
            "cal_samples": n_samples,
            "cal_exited": n_exit,
            "cal_coverage": float(n_exit / n_samples) if n_samples > 0 else 0.0,
            "cal_emp_risk": emp_risk,
            "cal_ucb": ucb,
            "calibrated_threshold": self.confidence_threshold,
            "epsilon": eps,
            "delta": self.delta,
            "bound_type": self.bound_type,
        }
        return self.confidence_threshold

    def calibrate_multi_tolerance(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        tolerances: Optional[List[float]] = None,
    ) -> Dict[float, Dict[str, Any]]:
        """
        Calibrates thresholds across multiple risk tolerances (e.g. 0.001, 0.005, 0.01, 0.05, 0.10).
        """
        if tolerances is None:
            tolerances = [0.001, 0.005, 0.01, 0.05, 0.10]

        results = {}
        for eps in tolerances:
            thresh = self.calibrate(confidences, predictions, ground_truth, target_epsilon=eps)
            results[eps] = dict(self.calibration_stats)
        return results

    def evaluate_test_split(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates calibrated policy on held-out test split:
          - Fast-path coverage: proportion of queries exiting at Step 1
          - Empirical fast-path error rate: P(error | EXIT)
          - Conformal guarantee satisfied: empirical error <= epsilon
        """
        thresh = threshold if threshold is not None else self.confidence_threshold
        n_samples = len(confidences)
        if n_samples == 0:
            return {"test_coverage": 0.0, "test_selective_error": 0.0, "bound_satisfied": True}

        errors = (predictions != ground_truth).astype(float)
        exit_mask = confidences >= thresh
        n_exit = int(np.sum(exit_mask))

        coverage = float(n_exit / n_samples)
        if n_exit > 0:
            sel_error = float(np.sum(errors[exit_mask]) / n_exit)
        else:
            sel_error = 0.0

        satisfied = bool(sel_error <= self.epsilon)
        return {
            "test_samples": n_samples,
            "test_exited": n_exit,
            "test_coverage": coverage,
            "test_selective_error": sel_error,
            "threshold": thresh,
            "target_epsilon": self.epsilon,
            "bound_satisfied": satisfied,
        }


    def evaluate_logits(
        self,
        slot_logits: Dict[str, Tuple[torch.Tensor, List[str], List[int]]],
        has_synthesis_field: bool = False,
    ) -> RiskGateResult:
        """
        Evaluates Step 1 logits across all micro-control canvas slots.

        Args:
            slot_logits: Mapping from slot_name -> (raw_logits_at_slot, candidate_labels, candidate_token_ids)
            has_synthesis_field: True if schema explicitly requires open-ended text synthesis.

        Returns:
            RiskGateResult containing exit/expand action, confidence metrics, and predictions.
        """
        slot_preds: Dict[str, SlotPrediction] = {}
        min_confidence = 1.0
        max_entropy = 0.0

        for slot_name, (logits, labels, token_ids) in slot_logits.items():
            # Extract logits for candidate tokens only
            if len(token_ids) > 0:
                cand_logits = logits[token_ids]
            else:
                cand_logits = logits

            cand_logits_float = cand_logits.float()
            probs = torch.softmax(cand_logits_float, dim=-1)
            probs_np = probs.detach().cpu().numpy()

            # Top-1 and Top-2
            top1_idx = int(np.argmax(probs_np))
            top1_prob = float(probs_np[top1_idx])
            top1_label = labels[top1_idx] if top1_idx < len(labels) else str(top1_idx)
            top1_tok_id = token_ids[top1_idx] if top1_idx < len(token_ids) else -1

            # Prophet confidence gap (top1 - top2)
            if len(probs_np) > 1:
                sorted_probs = np.sort(probs_np)
                gap = float(sorted_probs[-1] - sorted_probs[-2])
            else:
                gap = 1.0

            # Shannon entropy H(p)
            eps = 1e-12
            entropy = float(-np.sum(probs_np * np.log(probs_np + eps)))

            prob_dict = {
                (labels[i] if i < len(labels) else str(i)): float(probs_np[i])
                for i in range(len(probs_np))
            }

            slot_preds[slot_name] = SlotPrediction(
                slot_name=slot_name,
                top_label=top1_label,
                top_token_id=top1_tok_id,
                confidence=top1_prob,
                entropy=entropy,
                confidence_gap=gap,
                probabilities=prob_dict,
                candidate_labels=labels,
            )

            if top1_prob < min_confidence:
                min_confidence = top1_prob
            if entropy > max_entropy:
                max_entropy = entropy

        # Check exit condition
        # 1. If explicit synthesis field requested -> MUST EXPAND
        if has_synthesis_field:
            return RiskGateResult(
                action=ExitAction.EXPAND,
                overall_confidence=min_confidence,
                overall_entropy=max_entropy,
                is_safe_to_exit=False,
                threshold_lambda=self.calibrated_lambda,
                target_epsilon=self.epsilon,
                slot_predictions=slot_preds,
                escalation_reason="Schema requires open-ended text synthesis.",
            )

        # 2. Conformal risk check: min_confidence >= 1 - lambda
        if min_confidence >= self.confidence_threshold:
            return RiskGateResult(
                action=ExitAction.EXIT,
                overall_confidence=min_confidence,
                overall_entropy=max_entropy,
                is_safe_to_exit=True,
                threshold_lambda=self.calibrated_lambda,
                target_epsilon=self.epsilon,
                slot_predictions=slot_preds,
                escalation_reason=None,
            )
        else:
            return RiskGateResult(
                action=ExitAction.EXPAND,
                overall_confidence=min_confidence,
                overall_entropy=max_entropy,
                is_safe_to_exit=False,
                threshold_lambda=self.calibrated_lambda,
                target_epsilon=self.epsilon,
                slot_predictions=slot_preds,
                escalation_reason=(
                    f"Confidence {min_confidence:.3f} below calibrated safety threshold "
                    f"{self.confidence_threshold:.3f} (eps={self.epsilon})."
                ),
            )

