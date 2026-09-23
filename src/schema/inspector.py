"""
Schema Complexity Classifier for Project Reflex.
Inspects OpenAI function schemas and classifies tool calls into compute tiers:
  - Tier 1 (ATOMIC): no declared arguments
  - Tier 2 (PARAMETRIC_PRIMITIVE): primitive declared arguments
  - Tier 3 (GENERATIVE_SYNTHESIS): unbounded strings or structures
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Tier(str, Enum):
    ATOMIC = "atomic"
    PARAMETRIC_PRIMITIVE = "parametric_primitive"
    GENERATIVE_SYNTHESIS = "generative_synthesis"


@dataclass
class ParameterSpec:
    """Specification of an individual tool parameter."""
    name: str
    type: str
    description: str = ""
    required: bool = False
    enum_values: Optional[List[Any]] = None
    default_value: Optional[Any] = None
    estimated_tokens: int = 4
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @property
    def is_primitive(self) -> bool:
        """Returns True for scalar types supported by the primitive validator."""
        if self.type in ("integer", "int", "number", "float", "boolean", "bool"):
            return True
        if self.type == "string" and self.enum_values is not None and len(self.enum_values) > 0:
            return True
        return False


@dataclass
class ToolComplexity:
    """Complexity classification profile for a tool schema."""
    tool_name: str
    tier: Tier
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    estimated_canvas_length: int = 4
    recommended_denoise_steps: int = 1
    is_primitive_only: bool = True
    description: str = ""


def inspect_tool_schema(tool_def: Dict[str, Any]) -> ToolComplexity:
    """
    Traverses an OpenAI tool function schema and classifies it into compute tiers.

    Args:
        tool_def: OpenAI tool definition dict (or function dict).

    Returns:
        ToolComplexity profile with recommended canvas length and step budget.
    """
    # Extract function body if nested under {"type": "function", "function": {...}}
    fn = tool_def.get("function", tool_def)
    name = fn.get("name", "unknown_tool")
    desc = fn.get("description", "")
    params = fn.get("parameters", {})

    if not isinstance(params, dict):
        params = {}

    props = params.get("properties", {})
    if not isinstance(props, dict):
        props = {}

    required_list = params.get("required", [])
    if not isinstance(required_list, list):
        required_list = []

    # Optional properties still require an argument-capable tier.
    if len(props) == 0:
        return ToolComplexity(
            tool_name=name,
            tier=Tier.ATOMIC,
            parameters={},
            required_parameters=[],
            estimated_canvas_length=4,
            recommended_denoise_steps=1,
            is_primitive_only=True,
            description=desc,
        )

    # 2. Inspect individual parameters
    parsed_params: Dict[str, ParameterSpec] = {}
    has_generative_string = False
    total_estimated_tokens = 6  # base JSON syntax overhead `{"":}`

    for prop_name, prop_spec in props.items():
        if not isinstance(prop_spec, dict):
            prop_spec = {"type": "string"}

        raw_type = str(prop_spec.get("type", "string")).lower()
        prop_desc = prop_spec.get("description", "")
        enum_vals = prop_spec.get("enum")
        is_req = prop_name in required_list

        # Estimate parameter token footprint
        if raw_type in ("integer", "int"):
            est_tokens = 2
        elif raw_type in ("number", "float"):
            est_tokens = 3
        elif raw_type in ("boolean", "bool"):
            est_tokens = 2
        elif raw_type == "string" and enum_vals:
            est_tokens = 3
        else:
            # Unbounded string or structured object
            est_tokens = 32
            has_generative_string = True

        parsed_params[prop_name] = ParameterSpec(
            name=prop_name,
            type=raw_type,
            description=prop_desc,
            required=is_req,
            enum_values=enum_vals,
            default_value=prop_spec.get("default"),
            estimated_tokens=est_tokens,
            minimum=prop_spec.get("minimum"),
            maximum=prop_spec.get("maximum"),
        )
        total_estimated_tokens += (est_tokens + len(prop_name) // 3 + 2)

    # 3. Classify Tier 2 vs Tier 3
    if has_generative_string:
        # Tier 3: Open Generative Synthesis
        return ToolComplexity(
            tool_name=name,
            tier=Tier.GENERATIVE_SYNTHESIS,
            parameters=parsed_params,
            required_parameters=required_list,
            estimated_canvas_length=min(max(total_estimated_tokens, 64), 256),
            recommended_denoise_steps=16,
            is_primitive_only=False,
            description=desc,
        )

    # Tier 2: Parametric Primitive (All required parameters are primitive)
    # Align canvas length to power of 2 or multiple of 8 in [8, 24]
    canvas_len = max(8, min(24, ((total_estimated_tokens + 7) // 8) * 8))
    return ToolComplexity(
        tool_name=name,
        tier=Tier.PARAMETRIC_PRIMITIVE,
        parameters=parsed_params,
        required_parameters=required_list,
        estimated_canvas_length=canvas_len,
        recommended_denoise_steps=3,
        is_primitive_only=True,
        description=desc,
    )

