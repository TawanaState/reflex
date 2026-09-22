"""
Canvas memory allocation, template seeding, and token decoding for Project Reflex engine.
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from ..schema.compiler import (
    CanvasCompiler,
    Choice,
    CompiledCanvas,
    ControlSlot,
    MicroArgumentCanvas,
    Noul,
    Schema,
    SlotType,
)
from ..schema.inspector import ParameterSpec, Tier, ToolComplexity


@dataclass
class ToolDefinition:
    """Represents a parsed OpenAI tool function."""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_atomic: bool = True


def parse_openai_tools(tools: Optional[List[Dict[str, Any]]]) -> List[ToolDefinition]:
    """Parses OpenAI tool specification dictionaries into structured ToolDefinitions."""
    if not tools:
        return []
    parsed = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        required = params.get("required", []) if isinstance(params, dict) else []
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        is_atomic = (len(required) == 0 and len(props) == 0)
        parsed.append(ToolDefinition(name=name, description=desc, parameters=params, is_atomic=is_atomic))
    return parsed


def compile_tools_to_schema(
    tools: List[Dict[str, Any]],
    tokenizer: Any,
    mask_token_id: int = 4,
    allow_direct_response: bool = False,
) -> Tuple[CompiledCanvas, Dict[str, ToolDefinition]]:
    """
    Compiles OpenAI tool definitions into a micro-control canvas with decision slots:
      - 'action': tool names mapped to candidate tokens [6, 7, ...]
    """
    parsed_tools = parse_openai_tools(tools)
    tool_dict = {t.name: t for t in parsed_tools}

    tool_options = [t.name for t in parsed_tools]
    if (allow_direct_response and len(tool_options) > 0) or not tool_options:
        tool_options.append("direct_response")

    # Tokens 6, 7, ... (<unused0>, <unused1>, ...) correspond to BFCL indexed routing classes
    cand_ids = [6 + i for i in range(len(tool_options))]

    action_slot = ControlSlot(
        name="action",
        slot_type=SlotType.CHOICE,
        candidate_labels=tool_options,
        candidate_token_ids=cand_ids,
        is_generative=False,
        description="Selected tool action or direct_response",
    )

    open_bracket = tokenizer.encode("[", add_special_tokens=False)
    open_b = open_bracket[0] if open_bracket else 101
    close_bracket = tokenizer.encode("]", add_special_tokens=False)
    close_b = close_bracket[0] if close_bracket else 102
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 4-token micro-canvas: [ [ , <mask , ] , <pad ] matching Phase 0 / LoRA training
    canvas_tokens = torch.tensor([[open_b, mask_token_id, close_b, pad_id]], dtype=torch.long)
    action_slot.canvas_position = 1

    return CompiledCanvas(
        canvas_tokens=canvas_tokens,
        slots={"action": action_slot},
        slot_positions=[1],
        allowed_token_ids={1: cand_ids},
        canvas_length=4,
        mask_token_id=mask_token_id,
        has_synthesis=False,
    ), tool_dict


def parse_and_validate_primitives(
    raw_text: str,
    complexity: ToolComplexity,
) -> Dict[str, Any]:
    """
    Extracts and type-casts primitive arguments from decoded canvas text.
    Handles partial JSON, key-value pairs, and regex extractions.
    """
    # 1. Try standard JSON parse
    text_clean = raw_text.strip()
    try:
        data = json.loads(text_clean)
        if isinstance(data, dict):
            # Validate types
            validated = {}
            for k, pspec in complexity.parameters.items():
                if k in data:
                    val = data[k]
                    val = cast_primitive_value(val, pspec)
                    if val is None:
                        return {}
                    validated[k] = val
            if all(k in validated for k in complexity.required_parameters):
                return validated
    except Exception:
        pass

    # 2. Extract key: value matches using regex
    validated = {}
    for k, pspec in complexity.parameters.items():
        # Match "key": value or key: value
        pattern = rf'["\']?{re.escape(k)}["\']?\s*[:=]\s*([^\s,\}}\]]+)'
        match = re.search(pattern, text_clean)
        if match:
            raw_val = match.group(1).strip('"\' ')
            val = cast_primitive_value(raw_val, pspec)
            if val is not None:
                validated[k] = val

    if any(k not in validated for k in complexity.required_parameters):
        return {}
    return validated



def extract_unambiguous_numeric_argument(user_text: str, complexity: ToolComplexity) -> Optional[Dict[str, Any]]:
    """Extract one explicit number for a one-number required schema; otherwise abstain.

    This is a transparent deterministic fallback, not model-generated argument synthesis.
    It deliberately rejects prompts containing multiple numeric literals.
    """
    if len(complexity.parameters) != 1 or len(complexity.required_parameters) != 1:
        return None
    name = complexity.required_parameters[0]
    spec = complexity.parameters.get(name)
    if spec is None or spec.type not in ("integer", "int", "number", "float"):
        return None
    numerals = re.findall(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])", user_text)
    if len(numerals) != 1:
        return None
    value = cast_primitive_value(numerals[0], spec)
    return {name: value} if value is not None else None


def cast_primitive_value(raw_val: Any, pspec: ParameterSpec) -> Optional[Any]:
    """Cast a primitive only when its complete value satisfies the schema."""
    if pspec.enum_values is not None:
        return next((v for v in pspec.enum_values if str(raw_val).casefold() == str(v).casefold()), None)
    kind = pspec.type
    if kind in ("integer", "int"):
        if isinstance(raw_val, bool):
            return None
        if isinstance(raw_val, int):
            value = raw_val
        elif isinstance(raw_val, str) and re.fullmatch(r"[+-]?\d+", raw_val.strip()):
            value = int(raw_val.strip())
        else:
            return None
    elif kind in ("number", "float"):
        if isinstance(raw_val, bool):
            return None
        if isinstance(raw_val, (int, float)):
            value = float(raw_val)
        elif isinstance(raw_val, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw_val.strip()):
            value = float(raw_val.strip())
        else:
            return None
    elif kind in ("boolean", "bool"):
        if isinstance(raw_val, bool):
            value = raw_val
        elif str(raw_val).casefold().strip() in ("true", "1", "yes"):
            value = True
        elif str(raw_val).casefold().strip() in ("false", "0", "no"):
            value = False
        else:
            return None
    elif kind == "string" and isinstance(raw_val, str):
        value = raw_val
    else:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        if pspec.minimum is not None and value < pspec.minimum:
            return None
        if pspec.maximum is not None and value > pspec.maximum:
            return None
    return value

