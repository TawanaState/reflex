"""Gate tests: only a stable complete zero-argument native call may exit."""
import unittest

import torch

from src.engine.early_exit import AtomicToolReady, NativeDraftObserver
from src.engine.runner import ReflexEngine


class FakeTokenizer:
    def __init__(self, drafts):
        self.drafts = drafts

    def decode(self, ids, **kwargs):
        return self.drafts[int(ids[0])]


class TestNativeDraftObserver(unittest.TestCase):
    def test_exits_only_after_consecutive_valid_atomic_drafts(self):
        drafts = [
            '<|tool_call>call:mute_audio{}<tool_call|>',
            '<|tool_call>call:mute_audio{',
            '<|tool_call>call:mute_audio{}<tool_call|>',
            '<|tool_call>call:mute_audio{}<tool_call|>',
            '<|tool_call>call:mute_audio{}<tool_call|>',
        ]
        observed = []
        observer = NativeDraftObserver(FakeTokenizer(drafts), {'mute_audio'}, 3, on_draft=observed.append)
        for i in range(4):
            observer.put_draft(torch.tensor([[i]]))
        self.assertEqual(observer.stable_steps, 2)
        self.assertIsNone(observer.accepted_name)
        with self.assertRaises(AtomicToolReady):
            observer.put_draft(torch.tensor([[4]]))
        self.assertEqual(observer.accepted_name, 'mute_audio')
        self.assertEqual(observer.step, 5)
        self.assertEqual([o.stable_steps for o in observed], [1, 0, 1, 2, 3])

    def test_never_exits_on_arguments_or_unknown_tool(self):
        drafts = [
            '<|tool_call>call:set_volume{level:57}<tool_call|>',
            '<|tool_call>call:unknown{}<tool_call|>',
            '<|tool_call>call:mute_audio{}<tool_call|><|tool_call>call:mute_audio{}<tool_call|>',
        ]
        observer = NativeDraftObserver(FakeTokenizer(drafts), {'mute_audio', 'set_volume'}, 1)
        for i in range(len(drafts)):
            observer.put_draft(torch.tensor([[i]]))
        self.assertIsNone(observer.accepted_name)

    def test_observation_mode_never_exits(self):
        drafts = ['<|tool_call>call:mute_audio{}<tool_call|>']
        observer = NativeDraftObserver(FakeTokenizer(drafts), {'mute_audio'}, 1, enable_exit=False)
        observer.put_draft(torch.tensor([[0]]))
        self.assertIsNone(observer.accepted_name)
        self.assertEqual(observer.stable_steps, 1)


class TestStepTelemetry(unittest.TestCase):
    def test_uses_nonpad_token_count(self):
        class Output:
            tokens_per_forward = torch.tensor([5.0])

        generated = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19] + [0] * 246)
        self.assertEqual(ReflexEngine._estimate_forward_passes(Output(), generated, 0), 2)


if __name__ == '__main__':
    unittest.main()
