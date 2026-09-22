"""
Reflex Canvas Compiler (Backward Compatibility Module).
Re-exports modular components from src.schema and src.engine.
"""

from .schema.compiler import (
    SlotType,
    ControlSlot,
    Choice,
    Score,
    Noul,
    SynthesisField,
    Schema,
    CompiledCanvas,
    CanvasCompiler,
    MicroArgumentCanvas,
    compile_tier2_argument_canvas,
)
from .engine.canvas import (
    ToolDefinition,
    parse_openai_tools,
    compile_tools_to_schema,
    parse_and_validate_primitives,
)

__all__ = [
    "SlotType",
    "ControlSlot",
    "Choice",
    "Score",
    "Noul",
    "SynthesisField",
    "Schema",
    "CompiledCanvas",
    "CanvasCompiler",
    "MicroArgumentCanvas",
    "compile_tier2_argument_canvas",
    "ToolDefinition",
    "parse_openai_tools",
    "compile_tools_to_schema",
    "parse_and_validate_primitives",
]
