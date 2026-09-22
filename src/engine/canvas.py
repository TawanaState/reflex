"""
Canvas memory allocation, template seeding, and token decoding for Project Reflex engine.
"""

import json
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
                    if val is not None:
                        validated[k] = val
            if len(validated) > 0:
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
        else:
            # Fallback for standalone integer or float in text
            if pspec.type in ("integer", "int"):
                num_match = re.search(r'\b\d+\b', text_clean)
                if num_match:
                    try:
                        validated[k] = int(num_match.group(0))
                    except ValueError:
                        pass
            elif pspec.type in ("number", "float"):
                num_match = re.search(r'\b\d+(\.\d+)?\b', text_clean)
                if num_match:
                    try:
                        validated[k] = float(num_match.group(0))
                    except ValueError:
                        pass

    return validated


def cast_primitive_value(raw_val: Any, pspec: ParameterSpec) -> Optional[Any]:
    """Casts raw extracted token value into specified primitive type."""
    try:
        t = pspec.type
        if t in ("integer", "int"):
            if isinstance(raw_val, (int, float)):
                return int(raw_val)
            cleaned = re.sub(r'[^\d\-]', '', str(raw_val))
            return int(cleaned) if cleaned else None
        elif t in ("number", "float"):
            if isinstance(raw_val, (int, float)):
                return float(raw_val)
            cleaned = re.sub(r'[^\d\.\-]', '', str(raw_val))
            return float(cleaned) if cleaned else None
        elif t in ("boolean", "bool"):
            if isinstance(raw_val, bool):
                return raw_val
            s = str(raw_val).lower().strip()
            return s in ("true", "1", "yes")
        elif pspec.enum_values:
            val_str = str(raw_val).strip()
            for enum_opt in pspec.enum_values:
                if str(enum_opt).lower() == val_str.lower():
                    return enum_opt
            return pspec.enum_values[0]
        else:
            return str(raw_val).strip()
    except Exception:
        return None

