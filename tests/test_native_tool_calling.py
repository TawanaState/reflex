"""Unit tests for the native-Gemma tool-call parser (src/engine/native_tool_calling.py).

Fixture strings are taken verbatim from real generated output captured during
the GB10 live probe on 2026-09-22 (experiments/native_tool_routing_*.jsonl),
not hand-invented examples.
"""
import unittest

from src.engine.native_tool_calling import (
    normalize_json_schema_types,
    parse_gemma_tool_call,
    parse_gemma_tool_call_args,
    validate_and_cast_call_args,
)
from src.schema.inspector import inspect_tool_schema


class TestParseGemmaToolCallArgs(unittest.TestCase):
    def test_flat_string_and_number_args(self):
        body = 'loc:<|"|>123 Hanoi Street, Hà Nội<|"|>,time:10,type:<|"|>plus<|"|>'
        args = parse_gemma_tool_call_args(body)
        self.assertEqual(args, {"loc": "123 Hanoi Street, Hà Nội", "time": 10, "type": "plus"})

    def test_nested_object_args(self):
        body = (
            'drink_id:<|"|>latte<|"|>,new_preferences:{milk_type:<|"|>coconut<|"|>,'
            'size:<|"|>large<|"|>,special_instructions:<|"|>boiling hot<|"|>,'
            'sweetness_level:<|"|>extra<|"|>,temperature:<|"|>hot<|"|>}'
        )
        args = parse_gemma_tool_call_args(body)
        self.assertEqual(args["drink_id"], "latte")
        self.assertEqual(args["new_preferences"], {
            "milk_type": "coconut", "size": "large",
            "special_instructions": "boiling hot",
            "sweetness_level": "extra", "temperature": "hot",
        })

    def test_string_value_containing_comma_and_colon_is_not_corrupted(self):
        body = 'query:<|"|>ratio 3:1, mix well<|"|>'
        args = parse_gemma_tool_call_args(body)
        self.assertEqual(args, {"query": "ratio 3:1, mix well"})

    def test_empty_body_via_parse_gemma_tool_call(self):
        text = "<|tool_call>call:mute_audio{}<tool_call|>"
        name, args = parse_gemma_tool_call(text)
        self.assertEqual(name, "mute_audio")
        self.assertEqual(args, {})

    def test_full_generated_text_with_thinking_preamble(self):
        text = (
            "<|channel>thought\nI'll change the drink.<channel|>"
            '<|tool_call>call:ChaDri.change_drink{drink_id:<|"|>latte<|"|>,'
            'new_preferences:{size:<|"|>large<|"|>}}<tool_call|>'
        )
        name, args = parse_gemma_tool_call(text)
        self.assertEqual(name, "ChaDri.change_drink")
        self.assertEqual(args, {"drink_id": "latte", "new_preferences": {"size": "large"}})

    def test_no_tool_call_present_returns_none(self):
        text = "<|channel>thought\nHello! How can I help you today?<channel|>"
        name, args = parse_gemma_tool_call(text)
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_malformed_body_raises_valueerror_not_silently_wrong(self):
        with self.assertRaises(ValueError):
            parse_gemma_tool_call_args('broken:<|"|>unterminated')


class TestValidateAndCastCallArgs(unittest.TestCase):
    def _complexity(self, tool_def):
        return inspect_tool_schema(tool_def)

    def test_valid_primitive_call_is_cast(self):
        tool = {"function": {"name": "set_volume", "parameters": {
            "type": "object", "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["level"],
        }}}
        complexity = self._complexity(tool)
        result = validate_and_cast_call_args({"level": "57"}, complexity)
        self.assertEqual(result, {"level": 57})

    def test_missing_required_field_fails_closed(self):
        tool = {"function": {"name": "set_volume", "parameters": {
            "type": "object", "properties": {"level": {"type": "integer"}},
            "required": ["level"],
        }}}
        complexity = self._complexity(tool)
        self.assertIsNone(validate_and_cast_call_args({}, complexity))

    def test_out_of_range_numeric_fails_closed(self):
        tool = {"function": {"name": "set_volume", "parameters": {
            "type": "object", "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["level"],
        }}}
        complexity = self._complexity(tool)
        self.assertIsNone(validate_and_cast_call_args({"level": 999}, complexity))

    def test_nested_object_required_field_passes_through(self):
        tool = {"function": {"name": "change_drink", "parameters": {
            "type": "object",
            "properties": {
                "drink_id": {"type": "string"},
                "new_preferences": {"type": "object", "properties": {}},
            },
            "required": ["drink_id", "new_preferences"],
        }}}
        complexity = self._complexity(tool)
        result = validate_and_cast_call_args(
            {"drink_id": "latte", "new_preferences": {"size": "large"}}, complexity,
        )
        self.assertEqual(result, {"drink_id": "latte", "new_preferences": {"size": "large"}})


class TestNormalizeJsonSchemaTypes(unittest.TestCase):
    def test_maps_python_style_types_to_json_schema(self):
        schema = {"type": "dict", "properties": {"n": {"type": "int"}, "s": {"type": "str"}}}
        normalized = normalize_json_schema_types(schema)
        self.assertEqual(normalized["type"], "object")
        self.assertEqual(normalized["properties"]["n"]["type"], "integer")
        self.assertEqual(normalized["properties"]["s"]["type"], "string")

    def test_leaves_already_standard_types_untouched(self):
        schema = {"type": "object", "properties": {"x": {"type": "boolean"}}}
        self.assertEqual(normalize_json_schema_types(schema), schema)


if __name__ == "__main__":
    unittest.main()
