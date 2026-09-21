"""
Unit tests for Reflex Canvas Compiler.
"""

import unittest
import torch
from unittest.mock import MagicMock
from src.canvas import CanvasCompiler, Choice, Score, Noul, Schema, SlotType, SynthesisField


class MockTokenizer:
    """Mock tokenizer for unit testing canvas compilation without downloading heavy weights."""
    def __init__(self):
        self.vocab = {
            "[": 101,
            "]": 102,
            ",": 103,
            "<mask": 4,
            "<pad>": 0,
            "click": 201,
            "scroll": 202,
            "type": 203,
            "wait": 204,
            "1": 301,
            "2": 302,
            "3": 303,
            "yes": 401,
            "no": 402,
        }
        self.mask_token_id = 4
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        text_clean = text.strip()
        if text_clean in self.vocab:
            return [self.vocab[text_clean]]
        return [999]


class TestCanvasCompiler(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.compiler = CanvasCompiler(tokenizer=self.tokenizer, mask_token_id=4)

    def test_choice_slot_creation(self):
        choice = Choice(options=["click", "scroll", "type", "wait"])
        slot = choice.to_slot("route", self.tokenizer)
        self.assertEqual(slot.name, "route")
        self.assertEqual(slot.slot_type, SlotType.CHOICE)
        self.assertEqual(slot.candidate_labels, ["click", "scroll", "type", "wait"])
        self.assertEqual(slot.candidate_token_ids, [201, 202, 203, 204])

    def test_score_slot_creation(self):
        score = Score(min_val=1, max_val=3)
        slot = score.to_slot("priority", self.tokenizer)
        self.assertEqual(slot.name, "priority")
        self.assertEqual(slot.slot_type, SlotType.SCORE)
        self.assertEqual(slot.candidate_labels, ["1", "2", "3"])
        self.assertEqual(slot.candidate_token_ids, [301, 302, 303])

    def test_noul_slot_creation(self):
        noul = Noul(true_label="yes", false_label="no")
        slot = noul.to_slot("confirm", self.tokenizer)
        self.assertEqual(slot.name, "confirm")
        self.assertEqual(slot.slot_type, SlotType.NOUL)
        self.assertEqual(slot.candidate_labels, ["yes", "no"])
        self.assertEqual(slot.candidate_token_ids, [401, 402])

    def test_schema_compilation_micro_canvas(self):
        schema = Schema(fields={
            "route": Choice(options=["click", "scroll", "wait"]),
            "confirm": Noul(true_label="yes", false_label="no"),
        })
        compiled = self.compiler.compile(schema)
        self.assertIsInstance(compiled.canvas_tokens, torch.LongTensor)
        # Check that canvas length is small (e.g. 4 or 8 tokens)
        self.assertLessEqual(compiled.canvas_length, 16)
        self.assertGreaterEqual(compiled.canvas_length, 4)

        # Check mask tokens placed at slot positions
        for pos in compiled.slot_positions:
            self.assertEqual(compiled.canvas_tokens[0, pos].item(), 4)

        # Check candidate tokens mapped properly
        self.assertEqual(len(compiled.allowed_token_ids), 2)
        self.assertEqual(compiled.get_candidate_labels_for_slot("route"), ["click", "scroll", "wait"])


if __name__ == "__main__":
    unittest.main()

