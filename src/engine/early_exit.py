"""Observe native DiffusionGemma drafts and stop on a stable atomic tool call.

DiffusionGemma's official generate() calls ``streamer.put_draft`` after each
denoising forward. Raising ``AtomicToolReady`` there leaves the official
sampler before another forward; no second model invocation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .native_tool_calling import TOOL_CALL_RE, parse_gemma_tool_call


class AtomicToolReady(Exception):
    """Internal control signal for a validated early exit."""


@dataclass
class DraftObservation:
    step: int
    tool_name: Optional[str]
    has_valid_arguments: bool
    arguments: Optional[dict]
    complete_call: bool
    stable_steps: int


class NativeDraftObserver:
    """A DiffusionGemma streamer that watches one request's denoising drafts.

    ``on_draft`` receives only compact observations. It can be used for an
    offline convergence trace without retaining a full 256-token draft at
    every step. An atomic early exit requires a complete, parseable native
    call observed for the configured number of consecutive drafts. This is a
    gate, not a calibrated probability of correctness.
    """

    _takes_logits = False

    def __init__(
        self,
        tokenizer: Any,
        atomic_names: set[str],
        stable_steps_required: int = 1,
        enable_exit: bool = True,
        on_draft: Optional[Callable[[DraftObservation], None]] = None,
    ) -> None:
        if stable_steps_required < 1:
            raise ValueError("stable_steps_required must be positive")
        self.tokenizer = tokenizer
        self.atomic_names = atomic_names
        self.stable_steps_required = stable_steps_required
        self.enable_exit = enable_exit
        self.on_draft = on_draft
        self.step = 0
        self.stable_steps = 0
        self.previous_name: Optional[str] = None
        self.accepted_name: Optional[str] = None
        self.accepted_text: Optional[str] = None

    def put(self, value: Any) -> None:
        """The official generator sends the prompt and completed blocks here."""

    def end(self) -> None:
        pass

    def put_draft(self, value: Any, **kwargs: Any) -> None:
        self.step += 1
        draft_ids = value[0] if len(value.shape) == 2 else value
        text = self.tokenizer.decode(draft_ids, skip_special_tokens=False)
        name, args = parse_gemma_tool_call(text)
        complete = name is not None and len(TOOL_CALL_RE.findall(text)) == 1
        valid_atomic = complete and name in self.atomic_names and args == {}

        if valid_atomic:
            self.stable_steps = self.stable_steps + 1 if name == self.previous_name else 1
            self.previous_name = name
        else:
            self.stable_steps = 0
            self.previous_name = None

        if self.on_draft is not None:
            self.on_draft(DraftObservation(
                step=self.step,
                tool_name=name,
                has_valid_arguments=args is not None,
                arguments=args,
                complete_call=complete,
                stable_steps=self.stable_steps,
            ))

        if self.enable_exit and valid_atomic and self.stable_steps >= self.stable_steps_required:
            self.accepted_name = name
            self.accepted_text = text
            raise AtomicToolReady()
