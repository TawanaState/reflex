"""
Unit tests for Canvas Expansion Manager and drift elimination.
"""

import unittest
import torch
from src.canvas import CompiledCanvas, ControlSlot, SlotType
from src.risk_gate import ExitAction, RiskGateResult, SlotPrediction
from src.expansion import CanvasExpansionManager


class TestCanvasExpansion(unittest.TestCase):
    def setUp(self):
        self.manager = CanvasExpansionManager(default_gen_length=64, mask_token_id=4, pad_token_id=0)

    def test_freeze_and_expand(self):
        # Micro canvas: [ 101, <mask=4>, 103, <mask=4>, 102 ]
        canvas_tokens = torch.tensor([[101, 4, 103, 4, 102]])
        slots = {
            "route": ControlSlot("route", SlotType.CHOICE, ["click", "scroll"], [201, 202], canvas_position=1),
            "gate": ControlSlot("gate", SlotType.NOUL, ["safe", "review"], [301, 302], canvas_position=3),
        }
        compiled = CompiledCanvas(
            canvas_tokens=canvas_tokens,
            slots=slots,
            slot_positions=[1, 3],
            allowed_token_ids={1: [201, 202], 3: [301, 302]},
            canvas_length=5,
            mask_token_id=4,
        )

        risk_res = RiskGateResult(
            action=ExitAction.EXPAND,
            overall_confidence=0.95,
            overall_entropy=0.1,
            is_safe_to_exit=False,
            threshold_lambda=0.1,
            target_epsilon=0.05,
            slot_predictions={
                "route": SlotPrediction("route", "click", 201, 0.95, 0.1, 0.9, {"click": 0.95, "scroll": 0.05}, ["click", "scroll"]),
                "gate": SlotPrediction("gate", "safe", 301, 0.98, 0.05, 0.96, {"safe": 0.98, "review": 0.02}, ["safe", "review"]),
            },
        )

        dummy_kv = "cached_prompt_kv_representation"
        expanded = self.manager.freeze_and_expand(
            compiled_canvas=compiled,
            risk_result=risk_res,
            step1_canvas=canvas_tokens,
            prompt_kv_cache=dummy_kv,
            generative_length=64,
        )

        # 1. Verify frozen control slots are preserved and mask tokens are replaced with finalized predictions
        self.assertEqual(expanded.frozen_control_tokens[0, 1].item(), 201)  # 'click' token id
        self.assertEqual(expanded.frozen_control_tokens[0, 3].item(), 301)  # 'safe' token id

        # 2. Verify generative canvas is materialized with 64 mask tokens
        self.assertEqual(expanded.expanded_canvas_tokens.shape, (1, 64))
        self.assertTrue(torch.all(expanded.expanded_canvas_tokens == 4))

        # 3. Verify KV cache is reused without alteration
        self.assertEqual(expanded.prompt_kv_cache, dummy_kv)
        self.assertEqual(expanded.finalized_slots["route"], "click")
        self.assertEqual(expanded.finalized_slots["gate"], "safe")


if __name__ == "__main__":
    unittest.main()

