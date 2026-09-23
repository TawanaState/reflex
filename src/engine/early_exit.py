"""Observe native DiffusionGemma drafts and stop on validated tool actions.

The official generator calls ``streamer.put_draft`` after each denoising
forward. Reflex can terminate that same generation once an offered atomic call
is complete, or (experimentally) once a fully specified typed-scalar call
remains identical for a configured number of consecutive drafts. A valid
schema is not a probability of tool relevance or correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..schema.inspector import Tier, ToolComplexity
from .native_tool_calling import TOOL_CALL_RE, parse_gemma_tool_call, validate_and_cast_call_args


class NativeToolReady(Exception):
    """Internal control signal for a validated native draft exit."""


# Retain the old internal name for callers/tests that imported it.
AtomicToolReady = NativeToolReady


@dataclass
class DraftObservation:
    step: int
    tool_name: Optional[str]
    has_valid_arguments: bool
    arguments: Optional[dict]
    complete_call: bool
    stable_steps: int
    exit_kind: Optional[str] = None
    validated_arguments: Optional[dict] = None


class NativeDraftObserver:
    """Watch one request's denoising drafts without changing the sampler.

    Atomic calls may exit after ``stable_steps_required`` complete drafts.
    Scalar calls use ``scalar_stable_steps_required`` and must include *every*
    declared field, including optional ones. That conservative rule prevents a
    draft with missing optional fields from exiting before the model adds them.
    All fields must be numeric/boolean/enum scalars according to the
    existing schema inspector; open strings, arrays, and objects never exit.
    """

    _takes_logits = False

    def __init__(
        self,
        tokenizer: Any,
        atomic_names: set[str],
        stable_steps_required: int = 1,
        enable_exit: bool = True,
        on_draft: Optional[Callable[[DraftObservation], None]] = None,
        scalar_profiles: Optional[dict[str, ToolComplexity]] = None,
        scalar_stable_steps_required: int = 2,
    ) -> None:
        if stable_steps_required < 1 or scalar_stable_steps_required < 1:
            raise ValueError("stable draft counts must be positive")
        self.tokenizer = tokenizer
        self.atomic_names = atomic_names
        self.scalar_profiles = scalar_profiles or {}
        self.stable_steps_required = stable_steps_required
        self.scalar_stable_steps_required = scalar_stable_steps_required
        self.enable_exit = enable_exit
        self.on_draft = on_draft
        self.step = 0
        self.stable_steps = 0
        self.previous_action: Optional[tuple[str, str, str]] = None
        self.accepted_name: Optional[str] = None
        self.accepted_arguments: Optional[dict] = None
        self.accepted_kind: Optional[str] = None
        self.accepted_text: Optional[str] = None

    def put(self, value: Any) -> None:
        """The official generator sends the prompt and completed blocks here."""

    def end(self) -> None:
        pass

    def _candidate(self, name: Optional[str], args: Optional[dict], complete: bool):
        if not complete or name is None or args is None:
            return None
        if name in self.atomic_names and args == {}:
            return "atomic", {}
        profile = self.scalar_profiles.get(name)
        if profile is None or profile.tier != Tier.PARAMETRIC_PRIMITIVE:
            return None
        if not profile.parameters or not all(spec.is_primitive for spec in profile.parameters.values()):
            return None
        # Optional fields may be omitted in the final call, but a partial draft
        # can add them later. Require every declared field before early exit.
        if set(args) != set(profile.parameters):
            return None
        validated = validate_and_cast_call_args(args, profile)
        if validated is None or set(validated) != set(profile.parameters):
            return None
        return "scalar", validated

    def put_draft(self, value: Any, **kwargs: Any) -> None:
        self.step += 1
        draft_ids = value[0] if len(value.shape) == 2 else value
        text = self.tokenizer.decode(draft_ids, skip_special_tokens=False)
        name, args = parse_gemma_tool_call(text)
        # Reject an additional complete or incomplete call in the same draft.
        complete = (name is not None and len(TOOL_CALL_RE.findall(text)) == 1
                    and text.count("<|tool_call>") == 1 and text.count("<tool_call|>") == 1)
        candidate = self._candidate(name, args, complete)
        kind, validated = candidate if candidate is not None else (None, None)
        if candidate is not None:
            action = (kind, name, json.dumps(validated, sort_keys=True, separators=(",", ":")))
            self.stable_steps = self.stable_steps + 1 if action == self.previous_action else 1
            self.previous_action = action
        else:
            self.stable_steps = 0
            self.previous_action = None

        if self.on_draft is not None:
            self.on_draft(DraftObservation(
                step=self.step,
                tool_name=name,
                has_valid_arguments=args is not None,
                arguments=args,
                complete_call=complete,
                stable_steps=self.stable_steps,
                exit_kind=kind,
                validated_arguments=validated,
            ))

        required = (self.stable_steps_required if kind == "atomic"
                    else self.scalar_stable_steps_required)
        if self.enable_exit and candidate is not None and self.stable_steps >= required:
            self.accepted_name = name
            self.accepted_arguments = validated
            self.accepted_kind = kind
            self.accepted_text = text
            raise NativeToolReady()
