"""
Schema Compiler for Project Reflex.
Compiles typed schemas and tool definitions into:
  - Micro-control canvases for Step-1 action routing
  - Micro-argument canvases for Tier-2 low-step primitive parameter extraction
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from .inspector import ParameterSpec, Tier, ToolComplexity


class SlotType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"
    SYNTHESIS = "synthesis"


@dataclass
class ControlSlot:
    """Represents a single typed decision slot in a micro-control canvas."""
    name: str
    slot_type: SlotType
    candidate_labels: List[str]
    candidate_token_ids: List[int] = field(default_factory=list)
    canvas_position: int = -1
    is_generative: bool = False
    description: str = ""


@dataclass
class Choice:
    options: List[str]
    description: str = ""

    def to_slot(self, name: str, tokenizer: Any) -> ControlSlot:
        candidate_ids = []
        for opt in self.options:
            tokens = tokenizer.encode(opt, add_special_tokens=False)
            if not tokens:
                tokens = tokenizer.encode(" " + opt, add_special_tokens=False)
            candidate_ids.append(tokens[0] if tokens else 0)
        return ControlSlot(
            name=name,
            slot_type=SlotType.CHOICE,
            candidate_labels=self.options,
            candidate_token_ids=candidate_ids,
            is_generative=False,
            description=self.description,
        )


@dataclass
class Score:
    min_val: int = 1
    max_val: int = 5
    description: str = ""

    def to_slot(self, name: str, tokenizer: Any) -> ControlSlot:
        labels = [str(i) for i in range(self.min_val, self.max_val + 1)]
        candidate_ids = []
        for lbl in labels:
            tokens = tokenizer.encode(lbl, add_special_tokens=False)
            candidate_ids.append(tokens[0] if tokens else 0)
        return ControlSlot(
            name=name,
            slot_type=SlotType.SCORE,
            candidate_labels=labels,
            candidate_token_ids=candidate_ids,
            is_generative=False,
            description=self.description,
        )


@dataclass
class Noul:
    true_label: str = "yes"
    false_label: str = "no"
    description: str = ""

    def to_slot(self, name: str, tokenizer: Any) -> ControlSlot:
        labels = [self.true_label, self.false_label]
        candidate_ids = []
        for lbl in labels:
            tokens = tokenizer.encode(lbl, add_special_tokens=False)
            if not tokens:
                tokens = tokenizer.encode(" " + lbl, add_special_tokens=False)
            candidate_ids.append(tokens[0] if tokens else 0)
        return ControlSlot(
            name=name,
            slot_type=SlotType.NOUL,
            candidate_labels=labels,
            candidate_token_ids=candidate_ids,
            is_generative=False,
            description=self.description,
        )


@dataclass
class SynthesisField:
    max_tokens: int = 128
    description: str = ""

    def to_slot(self, name: str, tokenizer: Any) -> ControlSlot:
        return ControlSlot(
            name=name,
            slot_type=SlotType.SYNTHESIS,
            candidate_labels=["<SYNTH>"],
            candidate_token_ids=[],
            is_generative=True,
            description=self.description,
        )


@dataclass
class Schema:
    fields: Dict[str, Union[Choice, Score, Noul, SynthesisField]]

    def compile(self, tokenizer: Any, mask_token_id: int = 4) -> "CompiledCanvas":
        compiler = CanvasCompiler(tokenizer=tokenizer, mask_token_id=mask_token_id)
        return compiler.compile(self)


@dataclass
class CompiledCanvas:
    """Compiled micro-control canvas."""
    canvas_tokens: torch.LongTensor
    slots: Dict[str, ControlSlot]
    slot_positions: List[int]
    allowed_token_ids: Dict[int, List[int]]
    canvas_length: int
    mask_token_id: int
    has_synthesis: bool = False

    def get_candidate_ids_for_slot(self, slot_name: str) -> List[int]:
        slot = self.slots.get(slot_name)
        if slot is None:
            raise KeyError(f"Slot {slot_name} not found in compiled canvas.")
        return slot.candidate_token_ids

    def get_candidate_labels_for_slot(self, slot_name: str) -> List[str]:
        slot = self.slots.get(slot_name)
        if slot is None:
            raise KeyError(f"Slot {slot_name} not found in compiled canvas.")
        return slot.candidate_labels


class CanvasCompiler:
    """Compiles generic schemas into 4-16 token micro-control sequences."""

    def __init__(self, tokenizer: Any, mask_token_id: int = 4):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def compile(self, schema: Schema) -> CompiledCanvas:
        slots: Dict[str, ControlSlot] = {}
        tokens: List[int] = []
        slot_positions: List[int] = []
        allowed_tokens: Dict[int, List[int]] = {}
        has_synthesis = False

        open_bracket = self.tokenizer.encode("[", add_special_tokens=False)
        tokens.extend(open_bracket if open_bracket else [101])

        field_items = list(schema.fields.items())
        for idx, (field_name, field_spec) in enumerate(field_items):
            slot = field_spec.to_slot(field_name, self.tokenizer)
            if slot.is_generative:
                has_synthesis = True

            pos = len(tokens)
            tokens.append(self.mask_token_id)
            slot.canvas_position = pos
            slots[field_name] = slot
            slot_positions.append(pos)
            allowed_tokens[pos] = slot.candidate_token_ids

            if idx < len(field_items) - 1:
                comma = self.tokenizer.encode(",", add_special_tokens=False)
                tokens.extend(comma if comma else [103])

        close_bracket = self.tokenizer.encode("]", add_special_tokens=False)
        tokens.extend(close_bracket if close_bracket else [102])

        min_len = 4
        while min_len < len(tokens):
            min_len *= 2
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        while len(tokens) < min_len:
            tokens.append(pad_token_id)

        return CompiledCanvas(
            canvas_tokens=torch.tensor([tokens], dtype=torch.long),
            slots=slots,
            slot_positions=slot_positions,
            allowed_token_ids=allowed_tokens,
            canvas_length=len(tokens),
            mask_token_id=self.mask_token_id,
            has_synthesis=has_synthesis,
        )


@dataclass
class MicroArgumentCanvas:
    """Compact structured argument canvas for Tier 2 low-step parameter extraction."""
    canvas_tokens: torch.Tensor
    canvas_length: int
    slot_positions: Dict[str, List[int]]       # param_name -> token positions to decode
    param_types: Dict[str, str]                # param_name -> type
    template_str: str


def compile_tier2_argument_canvas(
    complexity: ToolComplexity,
    tokenizer: Any,
    mask_token_id: int = 4,
    pad_token_id: int = 0,
) -> MicroArgumentCanvas:
    """
    Constructs a micro-argument canvas strictly sized to [8, 24] tokens.
    Pre-seeds JSON syntax keys and places mask tokens only at argument values.
    Example: `{"level": <mask <mask }`
    """
    canvas_len = complexity.estimated_canvas_length  # e.g., 8, 16, 24

    # Seed structured prefix
    # e.g. {"param1": <mask>, ...}
    tokens = tokenizer.encode('{"', add_special_tokens=False)
    slot_positions: Dict[str, List[int]] = {}
    param_types: Dict[str, str] = {}

    param_items = list(complexity.parameters.items())
    for idx, (pname, pspec) in enumerate(param_items):
        if not pspec.required and idx > 0:
            continue

        param_types[pname] = pspec.type
        # Add key
        key_tokens = tokenizer.encode(f'{pname}":', add_special_tokens=False)
        tokens.extend(key_tokens)

        # Allocate 2-4 mask tokens for primitive value
        num_masks = min(pspec.estimated_tokens, 4)
        start_pos = len(tokens)
        mask_positions = []
        for _ in range(num_masks):
            mask_positions.append(len(tokens))
            tokens.append(mask_token_id)
        slot_positions[pname] = mask_positions

        if idx < len(param_items) - 1:
            tokens.extend(tokenizer.encode(',"', add_special_tokens=False))

    # Closing bracket
    tokens.extend(tokenizer.encode("}", add_special_tokens=False))

    # Pad or truncate to canvas_len
    target_len = max(len(tokens), canvas_len)
    # round up to multiple of 8
    target_len = min(24, max(8, ((target_len + 7) // 8) * 8))

    while len(tokens) < target_len:
        tokens.append(pad_token_id)
    if len(tokens) > target_len:
        tokens = tokens[:target_len]

    canvas_tensor = torch.tensor([tokens], dtype=torch.long)
    return MicroArgumentCanvas(
        canvas_tokens=canvas_tensor,
        canvas_length=target_len,
        slot_positions=slot_positions,
        param_types=param_types,
        template_str=f"Tier2[{complexity.tool_name}]",
    )

