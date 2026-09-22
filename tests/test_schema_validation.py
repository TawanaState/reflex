import unittest

from src.engine.canvas import extract_unambiguous_numeric_argument, parse_and_validate_primitives
from src.schema.inspector import Tier, inspect_tool_schema


class SchemaValidationTests(unittest.TestCase):
    def setUp(self):
        self.tool = {
            "function": {
                "name": "set_volume",
                "parameters": {
                    "type": "object",
                    "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["level"],
                },
            }
        }
        self.complexity = inspect_tool_schema(self.tool)

    def test_requires_exact_in_range_integer(self):
        self.assertEqual(self.complexity.tier, Tier.PARAMETRIC_PRIMITIVE)
        self.assertEqual(parse_and_validate_primitives('{"level":57}', self.complexity), {"level": 57})
        for raw in ('{}', '{"level":101}', '{"level":false}', '{"level":"57dB"}', '57'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_and_validate_primitives(raw, self.complexity), {})

    def test_rejects_non_finite_numeric_json(self):
        tool = {"function": {"name": "set_gain", "parameters": {
            "type": "object", "properties": {"gain": {"type": "number"}}, "required": ["gain"],
        }}}
        complexity = inspect_tool_schema(tool)
        self.assertEqual(parse_and_validate_primitives('{"gain":NaN}', complexity), {})
        self.assertEqual(parse_and_validate_primitives('{"gain":Infinity}', complexity), {})

    def test_invalid_enum_is_not_replaced_by_first_option(self):
        tool = {"function": {"name": "mode", "parameters": {
            "type": "object", "properties": {"mode": {"type": "string", "enum": ["heat", "cool"]}},
            "required": ["mode"],
        }}}
        complexity = inspect_tool_schema(tool)
        self.assertEqual(parse_and_validate_primitives('{"mode":"other"}', complexity), {})
        self.assertEqual(parse_and_validate_primitives('{"mode":"heat"}', complexity), {"mode": "heat"})

    def test_optional_property_is_not_atomic(self):
        tool = {"function": {"name": "set_mode", "parameters": {
            "type": "object", "properties": {"mode": {"type": "string", "enum": ["eco", "heat"]}},
            "required": [],
        }}}
        self.assertEqual(inspect_tool_schema(tool).tier, Tier.PARAMETRIC_PRIMITIVE)

    def test_explicit_numeric_extraction_abstains_on_ambiguity(self):
        self.assertEqual(
            extract_unambiguous_numeric_argument("Set system volume to 57 percent", self.complexity),
            {"level": 57},
        )
        for prompt in ("Set volume between 10 and 20", "Set volume to 101", "Set volume to fifty seven"):
            with self.subTest(prompt=prompt):
                self.assertIsNone(extract_unambiguous_numeric_argument(prompt, self.complexity))


if __name__ == '__main__':
    unittest.main()
