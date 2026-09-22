"""
Native DiffusionGemma tool-calling support.

The base checkpoint's own chat template (see the installed tokenizer's
chat_template.jinja) already implements a trained function-calling format:

    <|tool_call>call:tool_name{key1:<|"|>value<|"|>,key2:123}<tool_call|>

emitted by the OFFICIAL `model.generate()` (its EntropyBoundSampler-driven
block-diffusion loop), not by
Reflex's earlier custom "select a tool index into <unusedN>" micro-canvas.
That earlier scheme asked the model to perform a task (index classification)
it was never instruction-tuned for, which is the most likely reason Tier 2
(argument values) and Tier 3 (free-text arguments) previously failed outright
(see RESULTS.md and NOTES.md, 2026-09-22 entries). This module builds the
native-format prompt and parses the native-format response so the engine can
use the model's real trained capability instead of reinventing one.

The argument body is NOT valid JSON: keys are unquoted, strings are quoted
with the literal token `<|"|>` instead of `"`. `parse_gemma_tool_call_args`
below is a small tokenizer-aware converter, not a regex hack that could
corrupt string contents containing commas/colons.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..schema.inspector import ToolComplexity
from .canvas import cast_primitive_value

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:([\w\.]+)\{(.*?)\}<tool_call\|>", re.DOTALL)
STRING_QUOTE = '<|"|>'

_TYPE_MAP = {
    "dict": "object", "str": "string", "int": "integer", "float": "number",
    "bool": "boolean", "list": "array", "tuple": "array",
}


def normalize_json_schema_types(node: Any) -> Any:
    """Maps Python-style type names (as BFCL/legacy schemas use) to JSON Schema
    type names the chat template's format_parameters macro expects."""
    if isinstance(node, dict):
        return {
            k: (_TYPE_MAP.get(v.lower(), v) if k == "type" and isinstance(v, str) else normalize_json_schema_types(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [normalize_json_schema_types(x) for x in node]
    return node


def parse_gemma_tool_call(generated_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Extracts (tool_name, arguments_dict) from a generated response containing
    a <|tool_call>...<tool_call|> block. Returns (None, None) if absent or the
    argument body cannot be parsed."""
    match = TOOL_CALL_RE.search(generated_text)
    if not match:
        return None, None
    name = match.group(1)
    body = match.group(2).strip()
    if not body:
        return name, {}
    try:
        args = parse_gemma_tool_call_args(body)
    except (ValueError, json.JSONDecodeError):
        return name, None
    return name, args


def parse_gemma_tool_call_args(body: str) -> Dict[str, Any]:
    """Converts a Gemma-native argument body (unquoted keys, <|"|>-quoted
    strings) into a Python dict via standard json.loads, without regex-mangling
    string contents that happen to contain '{', ',' or ':'."""
    wrapped = "{" + body + "}"
    segments = wrapped.split(STRING_QUOTE)
    out: List[str] = []
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            # Inside a string literal: JSON-escape and re-quote.
            escaped = segment.replace("\\", "\\\\").replace('"', '\\"')
            out.append('"' + escaped + '"')
        else:
            # Outside any string literal: safe to quote bare object keys.
            quoted = re.sub(r'([{,]\s*)([A-Za-z_][\w.]*)(\s*:)', r'\1"\2"\3', segment)
            out.append(quoted)
    if len(segments) % 2 == 0:
        raise ValueError("Unbalanced <|\"|> string delimiters in tool-call arguments")
    json_text = "".join(out)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed tool-call arguments were not a JSON object")
    return parsed


def validate_and_cast_call_args(
    raw_args: Dict[str, Any], complexity: ToolComplexity
) -> Optional[Dict[str, Any]]:
    """Validates a parsed native tool-call argument dict against the declared
    schema. Required primitive (int/float/bool/enum-string) fields are cast
    and bounds-checked; required object/array fields only need to be present,
    since cast_primitive_value only handles scalar types and the model's own
    trained format already emits well-formed nested structures (verified
    against real generated output in tests/test_native_tool_calling.py).
    Returns None (fail-closed) if a required field is missing or an in-scope
    primitive fails validation. Unlisted extra keys are dropped.
    """
    if not isinstance(raw_args, dict):
        return None
    validated: Dict[str, Any] = {}
    for name, pspec in complexity.parameters.items():
        if name not in raw_args:
            continue
        value = raw_args[name]
        if pspec.type in ("object", "array") or isinstance(value, (dict, list)):
            validated[name] = value
        else:
            cast_value = cast_primitive_value(value, pspec)
            if cast_value is None:
                if name in complexity.required_parameters:
                    return None
                continue
            validated[name] = cast_value
    if any(name not in validated for name in complexity.required_parameters):
        return None
    return validated


def build_native_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Normalizes an OpenAI-style tools list for the native chat template
    (maps any Python-style parameter type names to JSON Schema names)."""
    if not tools:
        return None
    normalized = []
    for t in tools:
        fn = t.get("function", t)
        params = normalize_json_schema_types(fn.get("parameters", {"type": "object", "properties": {}}))
        normalized.append({"type": "function", "function": {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": params,
        }})
    return normalized
