"""Gate tests: only a stable complete zero-argument native call may exit."""
import unittest
import json
import time
from types import SimpleNamespace

import torch

from src.engine.early_exit import AtomicToolReady, NativeDraftObserver
from src.engine.runner import ReflexEngine
from src.schema.inspector import inspect_tool_schema


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


class TestScalarDraftObserver(unittest.TestCase):
    @staticmethod
    def profile(properties, required):
        return inspect_tool_schema({"name": "set_volume", "parameters": {
            "type": "object", "properties": properties, "required": required,
        }})

    def test_exits_after_exact_stable_validated_call(self):
        profile = self.profile({"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"])
        drafts = [
            '<|tool_call>call:set_volume{level:57}<tool_call|>',
            '<|tool_call>call:set_volume{level:57}<tool_call|>',
        ]
        observer = NativeDraftObserver(FakeTokenizer(drafts), set(),
                                       scalar_profiles={"set_volume": profile},
                                       scalar_stable_steps_required=2)
        observer.put_draft(torch.tensor([[0]]))
        self.assertIsNone(observer.accepted_name)
        with self.assertRaises(AtomicToolReady):
            observer.put_draft(torch.tensor([[1]]))
        self.assertEqual(observer.accepted_kind, "scalar")
        self.assertEqual(observer.accepted_arguments, {"level": 57})
        self.assertEqual(observer.step, 2)

    def test_field_order_does_not_break_semantic_stability(self):
        profile = self.profile({"level": {"type": "integer"}, "quiet": {"type": "boolean"}},
                               ["level", "quiet"])
        drafts = [
            '<|tool_call>call:set_volume{level:57,quiet:true}<tool_call|>',
            '<|tool_call>call:set_volume{quiet:true,level:57}<tool_call|>',
        ]
        observer = NativeDraftObserver(FakeTokenizer(drafts), set(),
                                       scalar_profiles={"set_volume": profile},
                                       scalar_stable_steps_required=2)
        observer.put_draft(torch.tensor([[0]]))
        with self.assertRaises(AtomicToolReady):
            observer.put_draft(torch.tensor([[1]]))
        self.assertEqual(observer.accepted_arguments, {"level": 57, "quiet": True})

    def test_changed_value_resets_stability(self):
        profile = self.profile({"level": {"type": "integer"}}, ["level"])
        drafts = [f'<|tool_call>call:set_volume{{level:{value}}}<tool_call|>' for value in (56, 57, 57)]
        observer = NativeDraftObserver(FakeTokenizer(drafts), set(),
                                       scalar_profiles={"set_volume": profile},
                                       scalar_stable_steps_required=2)
        observer.put_draft(torch.tensor([[0]]))
        observer.put_draft(torch.tensor([[1]]))
        self.assertEqual(observer.stable_steps, 1)
        with self.assertRaises(AtomicToolReady):
            observer.put_draft(torch.tensor([[2]]))
        self.assertEqual(observer.accepted_arguments, {"level": 57})

    def test_missing_optional_or_invalid_value_cannot_exit(self):
        profile = self.profile({
            "level": {"type": "integer", "minimum": 0, "maximum": 100},
            "quiet": {"type": "boolean"},
        }, ["level"])
        drafts = [
            '<|tool_call>call:set_volume{level:57}<tool_call|>',
            '<|tool_call>call:set_volume{level:101,quiet:false}<tool_call|>',
            '<|tool_call>call:set_volume{level:57,quiet:false,extra:1}<tool_call|>',
            '<|tool_call>call:set_volume{level:57,quiet:false}<tool_call|>',
            '<|tool_call>call:set_volume{level:57,quiet:false}<tool_call|>',
        ]
        observer = NativeDraftObserver(FakeTokenizer(drafts), set(),
                                       scalar_profiles={"set_volume": profile},
                                       scalar_stable_steps_required=2)
        for i in range(4):
            observer.put_draft(torch.tensor([[i]]))
        self.assertEqual(observer.stable_steps, 1)
        with self.assertRaises(AtomicToolReady):
            observer.put_draft(torch.tensor([[4]]))
        self.assertEqual(observer.accepted_arguments, {"level": 57, "quiet": False})

    def test_open_string_and_extra_incomplete_call_never_exit(self):
        profile = self.profile({"message": {"type": "string"}}, ["message"])
        drafts = [
            '<|tool_call>call:set_volume{message:<|"|>hello<|"|>}<tool_call|>',
            '<|tool_call>call:set_volume{message:<|"|>hello<|"|>}<tool_call|>',
        ]
        observer = NativeDraftObserver(FakeTokenizer(drafts), set(),
                                       scalar_profiles={"set_volume": profile},
                                       scalar_stable_steps_required=1)
        for i in range(2):
            observer.put_draft(torch.tensor([[i]]))
        self.assertIsNone(observer.accepted_name)

        bounded = self.profile({"level": {"type": "integer"}}, ["level"])
        extra = ['<|tool_call>call:set_volume{level:57}<tool_call|><|tool_call>call:other{']
        observer = NativeDraftObserver(FakeTokenizer(extra), set(),
                                       scalar_profiles={"set_volume": bounded},
                                       scalar_stable_steps_required=1)
        observer.put_draft(torch.tensor([[0]]))
        self.assertIsNone(observer.accepted_name)


class TestScalarRunnerIntegration(unittest.TestCase):
    def test_runner_returns_validated_scalar_call_from_draft(self):
        drafts = ['<|tool_call>call:set_volume{level:57}<tool_call|>']
        engine = object.__new__(ReflexEngine)
        engine.tokenizer = FakeTokenizer(drafts)
        engine.settings = SimpleNamespace(
            REFLEX_ATOMIC_EARLY_EXIT=False, REFLEX_ATOMIC_STABLE_STEPS=1,
            REFLEX_SCALAR_EARLY_EXIT=True, REFLEX_SCALAR_STABLE_STEPS=2,
        )
        engine.model = SimpleNamespace(config=SimpleNamespace(canvas_length=256))
        def generate(**kwargs):
            kwargs["streamer"].put_draft(torch.tensor([[0]]))
            kwargs["streamer"].put_draft(torch.tensor([[0]]))
            raise AssertionError("The observer should have stopped generation")
        engine.model.generate = generate
        engine._build_native_chat_inputs = lambda messages, images, tools: {
            "input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]]),
        }
        tools = [{"type": "function", "function": {
            "name": "set_volume", "parameters": {"type": "object", "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            }, "required": ["level"]},
        }}]
        result = engine._run_reflex([{"role": "user", "content": "Set volume to 57"}], [],
                                    tools, 256, time.perf_counter())
        self.assertEqual(result.execution_path, "NATIVE_SCALAR_EARLY_EXIT")
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"level": 57})
        self.assertEqual(result.steps_executed, 2)
        self.assertEqual(result.confidence, 0.0)


class TestStepTelemetry(unittest.TestCase):
    def test_uses_nonpad_token_count(self):
        class Output:
            tokens_per_forward = torch.tensor([5.0])

        generated = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19] + [0] * 246)
        self.assertEqual(ReflexEngine._estimate_forward_passes(Output(), generated, 0), 2)


if __name__ == '__main__':
    unittest.main()
