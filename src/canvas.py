"""
Reflex Canvas Compiler: Compiles typed schemas (Choice, Score, Noul) into micro-control canvases.
Provides pre-seeded syntax tokens and constrained candidate token maps for discrete diffusion models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


class SlotType(str, Enum):
    CHOICE = "choice"       # Categorical selection among K options
    SCORE = "score"         # Discrete numeric score (e.g. 1-5, 0-10)
    NOUL = "noul"           # Boolean / binary decision (yes/no, true/false)
    SYNTHESIS = "synthesis" # Open-ended text field requiring generative canvas expansion


@dataclass
class ControlSlot:
    """Represents a single typed decision slot in the micro-control canvas."""
    name: str
    slot_type: SlotType
    candidate_labels: List[str]
    candidate_token_ids: List[int] = field(default_factory=list)
    canvas_position: int = -1
    is_generative: bool = False
    description: str = ""


@dataclass
class Choice:
    """Choice schema: Categorical option from a list of strings."""
    options: List[str]
    description: str = ""

    def to_slot(self, name: str, tokenizer: Any) -> ControlSlot:
        candidate_ids = []
        for opt in self.options:
            # Tokenize option (single token representation or first token)
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
    """Score schema: Discrete score from min_val to max_val."""
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
    """Noul schema: Boolean / Binary flag (e.g. True/False or Yes/No)."""
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
    """Field indicating open-ended text synthesis is required."""
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
    """Collection of typed fields representing an agent decision or tool invocation."""
    fields: Dict[str, Union[Choice, Score, Noul, SynthesisField]]

    def compile(self, tokenizer: Any, mask_token_id: int = 4) -> "CompiledCanvas":
        """Compiles this schema into a micro-control canvas layout."""
        compiler = CanvasCompiler(tokenizer=tokenizer, mask_token_id=mask_token_id)
        return compiler.compile(self)


@dataclass
class CompiledCanvas:
    """Compiled micro-control canvas ready for single-step diffusion denoising."""
    canvas_tokens: torch.LongTensor           # (1, canvas_length)
    slots: Dict[str, ControlSlot]             # name -> ControlSlot with assigned canvas_position
    slot_positions: List[int]                 # indices of masked decision slots in canvas
    allowed_token_ids: Dict[int, List[int]]   # canvas_pos -> list of valid token IDs
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
    """
    Compiles typed schemas into compact micro-control canvas token sequences (4 to 16 tokens).
    Injects pre-seeded syntax tokens and inserts mask tokens (<mask) at decision positions.
    """

    def __init__(self, tokenizer: Any, mask_token_id: int = 4):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def compile(self, schema: Schema) -> CompiledCanvas:
        """
        Builds a compact canvas representation:
        Format: `[ @field1 , @field2 , ... ]`
        Pre-seeds brackets and delimiters, leaving `@field` slots as `<mask` tokens.
        """
        slots: Dict[str, ControlSlot] = {}
        tokens: List[int] = []
        slot_positions: List[int] = []
        allowed_tokens: Dict[int, List[int]] = {}
        has_synthesis = False

        # Open bracket token
        open_bracket = self.tokenizer.encode("[", add_special_tokens=False)
        if not open_bracket:
            open_bracket = [self.tokenizer.vocab.get("[", 235284)]
        tokens.extend(open_bracket)

        field_items = list(schema.fields.items())
        for idx, (field_name, field_spec) in enumerate(field_items):
            slot = field_spec.to_slot(field_name, self.tokenizer)
            if slot.is_generative:
                has_synthesis = True

            # Record mask token position
            pos = len(tokens)
            tokens.append(self.mask_token_id)
            slot.canvas_position = pos
            slots[field_name] = slot
            slot_positions.append(pos)
            allowed_tokens[pos] = slot.candidate_token_ids

            # Separator token (comma or space) if not last
            if idx < len(field_items) - 1:
                comma = self.tokenizer.encode(",", add_special_tokens=False)
                if not comma:
                    comma = [self.tokenizer.vocab.get(",", 235269)]
                tokens.extend(comma)

        # Close bracket token
        close_bracket = self.tokenizer.encode("]", add_special_tokens=False)
        if not close_bracket:
            close_bracket = [self.tokenizer.vocab.get("]", 235285)]
        tokens.extend(close_bracket)

        # Pad to nearest power of 2 or minimum 4 tokens (e.g. 4, 8, 16)
        min_len = 4
        while min_len < len(tokens):
            min_len *= 2
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        while len(tokens) < min_len:
            tokens.append(pad_token_id)

        canvas_tensor = torch.tensor([tokens], dtype=torch.long)
        return CompiledCanvas(
            canvas_tokens=canvas_tensor,
            slots=slots,
            slot_positions=slot_positions,
            allowed_token_ids=allowed_tokens,
            canvas_length=len(tokens),
            mask_token_id=self.mask_token_id,
            has_synthesis=has_synthesis,
        )


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
        # A function is atomic if it requires no parameters or has empty properties
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



