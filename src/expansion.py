"""
Reflex Canvas Expansion Manager: Handles conditional materialization from micro-control canvas
to expanded generative canvas (64 to 256 tokens) while reusing cached prompt KV states.
Freezes Step 1 control decisions to mathematically eliminate canvas drift.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .canvas import CompiledCanvas, ControlSlot
from .risk_gate import RiskGateResult


@dataclass
class ExpandedCanvasState:
    """State for the materialized generative canvas."""
    expanded_canvas_tokens: torch.LongTensor     # (batch_size, generative_length)
    frozen_control_tokens: torch.LongTensor       # Finalized Step 1 control tokens
    generative_length: int
    prompt_kv_cache: Any                         # Reused prompt KV cache (from encoder)
    finalized_slots: Dict[str, Any]


class CanvasExpansionManager:
    """
    Manages the transition from a micro-control canvas (4-16 tokens)
    to an expanded generative canvas (64-256 tokens).
    """

    def __init__(
        self,
        default_gen_length: int = 128,
        mask_token_id: int = 4,
        pad_token_id: int = 0,
    ):
        self.default_gen_length = default_gen_length
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def freeze_and_expand(
        self,
        compiled_canvas: CompiledCanvas,
        risk_result: RiskGateResult,
        step1_canvas: torch.Tensor,
        prompt_kv_cache: Any,
        generative_length: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> ExpandedCanvasState:
        """
        Freezes finalized Step 1 control tokens and initializes an expanded generative canvas.

        Args:
            compiled_canvas: Initial compiled micro-control canvas.
            risk_result: Risk gate result from Step 1 containing finalized slot predictions.
            step1_canvas: Raw or accepted canvas tensor after Step 1.
            prompt_kv_cache: Cached KV representations from the encoder pass.
            generative_length: Number of generative tokens to allocate (default 128).
            device: Target torch device.

        Returns:
            ExpandedCanvasState configured for multi-step diffusion denoising.
        """
        gen_len = generative_length or self.default_gen_length
        if device is None:
            device = step1_canvas.device

        # 1. Freeze control tokens: replace mask positions with finalized top-1 token IDs
        frozen_tokens = step1_canvas.clone()
        finalized_slots = {}

        for slot_name, pred in risk_result.slot_predictions.items():
            slot = compiled_canvas.slots.get(slot_name)
            if slot and slot.canvas_position >= 0:
                pos = slot.canvas_position
                if pred.top_token_id > 0:
                    frozen_tokens[0, pos] = pred.top_token_id
                finalized_slots[slot_name] = pred.top_label

        # 2. Materialize generative canvas: initialized with mask tokens (<mask)
        expanded_canvas = torch.full(
            (1, gen_len),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        return ExpandedCanvasState(
            expanded_canvas_tokens=expanded_canvas,
            frozen_control_tokens=frozen_tokens,
            generative_length=gen_len,
            prompt_kv_cache=prompt_kv_cache,
            finalized_slots=finalized_slots,
        )

