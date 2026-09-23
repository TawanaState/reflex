"""
Execution Runner for Project Reflex.

Native tool routing uses DiffusionGemma's trained function-call format and
official sampler. Its draft streamer exposes intermediate denoising canvases.
An atomic call may exit that same generation after a complete native call is
stable; all other outputs continue through the official sampler. See
RESULTS.md for the measured baseline and the status of early-exit evidence.
"""

import base64
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, DiffusionGemmaForBlockDiffusion

from ..config import Settings, get_settings
from ..schema.inspector import Tier, inspect_tool_schema
from .early_exit import NativeToolReady, NativeDraftObserver
from .native_tool_calling import (
    build_native_tools,
    parse_gemma_tool_call,
    validate_and_cast_call_args,
)


@dataclass
class EngineOutput:
    """Structured response from ReflexEngine."""
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: str
    execution_path: str
    tier: Optional[str]
    latency_ms: float
    confidence: float
    steps_executed: int
    prompt_tokens: int
    completion_tokens: int
    candidate_action: Optional[str] = None


class ReflexEngine:
    """
    Unified Inference Engine for Project Reflex with Tiered Adaptive Compute.
    Enforces the principle: Compute Matches Entropy.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.device = torch.device(self.settings.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"[ReflexEngine] Initializing on device: {self.device}")
        print(f"[ReflexEngine] Mode: {self.settings.REFLEX_MODE.upper()}")
        print(f"[ReflexEngine] Model Path: {self.settings.DIFFUSION_GEMMA_PATH}")

        # 1. Load Tokenizer
        print("[ReflexEngine] Loading AutoTokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.DIFFUSION_GEMMA_PATH)
        self.mask_token_id = self.tokenizer.mask_token_id or 4
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        # 2. Load Multimodal Processor
        print("[ReflexEngine] Loading AutoProcessor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.settings.DIFFUSION_GEMMA_PATH)
        except Exception as e:
            print(f"[ReflexEngine] AutoProcessor warning: {e}, running text-only fallback.")
            self.processor = None

        # 3. Load Base DiffusionGemma Model
        print("[ReflexEngine] Loading DiffusionGemma 26B/A4B in bfloat16...")
        t0 = time.time()
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.settings.DIFFUSION_GEMMA_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if "cuda" in str(self.device) else "cpu",
        )
        self.model.eval()
        print(f"[ReflexEngine] Base model loaded in {time.time() - t0:.1f}s.")

        # 4. Attach LoRA Adapter (opt-in only)
        # models/reflex_lora_v1 and v2 were trained to classify a tool index
        # into <unusedN> control tokens -- a task the engine no longer
        # performs (see the module docstring). They were never trained or
        # evaluated against native <|tool_call>...<tool_call|> generation, so
        # they are NOT auto-attached here. Set REFLEX_LORA_PATH to a real
        # adapter directory only after validating it against the native
        # tool-calling path (e.g. re-running experiments/eval_native_tool_calling.py
        # with --skip-lora removed) -- attaching an unvalidated adapter can
        # silently perturb the base model's native tool-calling quality.
        self.is_lora_loaded = False
        if self.settings.REFLEX_MODE == "reflex" and os.path.exists(self.settings.REFLEX_LORA_PATH):
            print(f"[ReflexEngine] WARNING: attaching LoRA from {self.settings.REFLEX_LORA_PATH}. "
                  f"This adapter was trained for the retired index-selection scheme and has not "
                  f"been validated against native tool calling. See runner.py module docstring.")
            self.model.model.decoder = PeftModel.from_pretrained(
                self.model.model.decoder,
                self.settings.REFLEX_LORA_PATH,
            )
            self.model.eval()
            self.is_lora_loaded = True

        # Conformal risk calibration (src/risk_gate.py) and the discrete step
        # scheduler (src/engine/scheduler.py) were built around the retired
        # per-slot softmax micro-canvas and are not wired into the native
        # tool-calling path. Both remain as independently tested modules for
        # future calibration work (see PROMPT.md / RESULTS.md "Forget
        # conformal prediction for now" -- deferred, not deleted).

    def _parse_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Parses OpenAI messages into chat-template-ready role/content dicts and
        extracts any images. Tools are NOT baked into message text here -- the
        native tool-calling path passes them to apply_chat_template's own
        `tools=` argument, which renders the model's trained function-declaration
        block instead of a hand-written prompt."""
        parsed_messages = []
        images: List[Image.Image] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                parsed_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype == "image_url":
                        img_info = part.get("image_url", {})
                        url = img_info.get("url", "")
                        loaded_img = self._load_image(url)
                        if loaded_img is not None:
                            images.append(loaded_img)
                            text_parts.append("<|image|>")
                parsed_messages.append({"role": role, "content": "\n".join(text_parts)})
            else:
                parsed_messages.append({"role": role, "content": str(content)})

        return parsed_messages, images

    def _load_image(self, url: str) -> Optional[Image.Image]:
        try:
            if url.startswith("data:image"):
                header, encoded = url.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif url.startswith("http://") or url.startswith("https://"):
                resp = requests.get(url, timeout=10)
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            elif os.path.exists(url):
                return Image.open(url).convert("RGB")
        except Exception:
            pass
        return None

    def _prepare_inputs(
        self, prompt_text: str, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        if images and self.processor is not None:
            inputs = self.processor(
                text=prompt_text,
                images=images if len(images) > 1 else images[0],
                return_tensors="pt",
            )
        else:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)

        tensor_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                tensor_inputs[k] = v.to(self.device)
            else:
                tensor_inputs[k] = v
        return tensor_inputs

    def _build_native_chat_inputs(
        self,
        parsed_messages: List[Dict[str, Any]],
        images: List[Image.Image],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """Builds tokenized inputs via the model's own chat template, passing
        `tools` straight through so the template renders DiffusionGemma's
        trained <|tool>...<tool|> function-declaration block -- the mechanism
        validated in experiments/native_tool_routing_*.jsonl -- rather than a
        hand-written tool-menu prompt.

        The images branch (self.processor.apply_chat_template) is best-effort:
        the 2026-09-22 evidence run and live smoke test were text-only. Verify
        against tests/test_multimodal.py against a live server before relying
        on multimodal + native tool calling together."""
        native_tools = build_native_tools(tools)
        chat_kwargs: Dict[str, Any] = dict(
            tools=native_tools, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        )
        if images and self.processor is not None:
            inputs = self.processor.apply_chat_template(
                parsed_messages, images=images if len(images) > 1 else images[0], **chat_kwargs,
            )
        else:
            inputs = self.tokenizer.apply_chat_template(parsed_messages, **chat_kwargs)

        tensor_inputs = {}
        for k, v in inputs.items():
            tensor_inputs[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return tensor_inputs

    @staticmethod
    def _estimate_forward_passes(generation_output: Any, generated_ids: torch.Tensor, pad_token_id: int) -> int:
        """Recover the decoder forward count from official generation output.

        ``tokens_per_forward`` uses non-pad generated tokens as its numerator.
        The returned block is always canvas-sized, so using its raw width here
        would substantially over-report the denoising work.
        """
        valid_tokens = int((generated_ids != pad_token_id).sum().item())
        tokens_per_forward = getattr(generation_output, "tokens_per_forward", None)
        if tokens_per_forward is None or valid_tokens == 0:
            return max(valid_tokens, 1)
        ratio = float(tokens_per_forward.reshape(-1)[0].item())
        if ratio <= 0:
            return max(valid_tokens, 1)
        return max(1, round(valid_tokens / ratio))

    _THOUGHT_CHANNEL_RE = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)
    _ANGLE_TAG_RE = re.compile(r"<[^<>]{1,64}>")

    @classmethod
    def _clean_display_text(cls, raw_generated_text: str) -> str:
        """Strips the model's internal reasoning channel and special-token
        markers from raw (skip_special_tokens=False) decoded text, for the
        case where no tool call was emitted and the text is shown to the user."""
        text = cls._THOUGHT_CHANNEL_RE.sub("", raw_generated_text)
        text = cls._ANGLE_TAG_RE.sub("", text)
        return text.strip()

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> EngineOutput:
        """Entry point for inference generation."""
        if tool_choice == "none":
            tools = None
        elif isinstance(tool_choice, dict):
            forced_name = (tool_choice.get("function") or {}).get("name")
            matches = [t for t in (tools or []) if (t.get("function") or {}).get("name") == forced_name]
            if not forced_name or len(matches) != 1:
                raise ValueError("tool_choice names no offered function")
            tools = matches

        parsed_messages, images = self._parse_messages(messages)
        t0 = time.perf_counter()

        if self.settings.REFLEX_MODE == "vanilla":
            prompt_text = self.tokenizer.apply_chat_template(
                parsed_messages, tokenize=False, add_generation_prompt=True,
            )
            tensor_inputs = self._prepare_inputs(prompt_text, images)
            return self._run_vanilla(
                tensor_inputs=tensor_inputs,
                prompt_tokens=tensor_inputs["input_ids"].shape[1],
                max_tokens=max_tokens or self.settings.MAX_CANVAS_LENGTH,
                t0=t0,
            )
        else:
            return self._run_reflex(
                parsed_messages=parsed_messages,
                images=images,
                tools=tools,
                max_tokens=max_tokens,
                t0=t0,
            )

    def _run_vanilla(
        self,
        tensor_inputs: Dict[str, torch.Tensor],
        prompt_tokens: int,
        max_tokens: int,
        t0: float,
    ) -> EngineOutput:
        gen_len = min(max_tokens, self.settings.MAX_CANVAS_LENGTH)
        canvas = torch.full((1, gen_len), fill_value=self.mask_token_id, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            enc_out = self.model.model.encoder(**tensor_inputs)
            past_kv = enc_out.past_key_values

            steps = self.settings.GENERATIVE_STEPS
            for _ in range(steps):
                out = self.model(input_ids=None, past_key_values=past_kv, decoder_input_ids=canvas)
                canvas = torch.argmax(out.logits, dim=-1)

        total_lat = (time.perf_counter() - t0) * 1000.0
        content = self.tokenizer.decode(canvas[0], skip_special_tokens=True).strip()

        return EngineOutput(
            content=content,
            tool_calls=None,
            finish_reason="stop",
            execution_path="VANILLA_FIXED_DIFFUSION",
            tier=Tier.GENERATIVE_SYNTHESIS.value,
            latency_ms=total_lat,
            confidence=1.0,
            steps_executed=steps,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_len,
        )

    def _run_reflex(
        self,
        parsed_messages: List[Dict[str, Any]],
        images: List[Image.Image],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: Optional[int],
        t0: float,
    ) -> EngineOutput:
        has_tools = bool(tools and len(tools) > 0)
        tool_by_name: Dict[str, Dict[str, Any]] = {}
        if has_tools:
            tool_by_name = {t.get("function", t).get("name"): t for t in tools}

        # DiffusionGemma generates in fixed self.model.config.canvas_length
        # (256) blocks: max_new_tokens only controls how many WHOLE blocks
        # run (ceil(max_new_tokens / canvas_length)), not compute within one.
        # An earlier version of this method scaled max_new_tokens down per
        # tool-schema tier (e.g. 48 for atomic-only menus) intending to save
        # compute; measured completion_tokens showed this had NO effect below
        # one block (still 256 tokens generated either way), so it has been
        # removed rather than left in as a fake optimization. The full path
        # retains the official sampler's entropy-based stopping. Reflex adds
        # draft exits for complete atomic calls and, when explicitly enabled,
        # stable fully specified bounded-scalar calls.
        canvas_length = getattr(self.model.config, "canvas_length", 256)
        gen_budget = min(max_tokens, canvas_length) if max_tokens else canvas_length

        tensor_inputs = self._build_native_chat_inputs(parsed_messages, images, tools)
        prompt_len = tensor_inputs["input_ids"].shape[1]

        profiles = {name: inspect_tool_schema(tool) for name, tool in tool_by_name.items()}
        atomic_names = {
            name for name, profile in profiles.items() if profile.tier == Tier.ATOMIC
        } if self.settings.REFLEX_ATOMIC_EARLY_EXIT else set()
        scalar_profiles = {
            name: profile for name, profile in profiles.items()
            if profile.tier == Tier.PARAMETRIC_PRIMITIVE
            and profile.parameters
            and all(spec.is_primitive for spec in profile.parameters.values())
        } if self.settings.REFLEX_SCALAR_EARLY_EXIT else {}
        observer = NativeDraftObserver(
            tokenizer=self.tokenizer,
            atomic_names=atomic_names,
            stable_steps_required=self.settings.REFLEX_ATOMIC_STABLE_STEPS,
            scalar_profiles=scalar_profiles,
            scalar_stable_steps_required=self.settings.REFLEX_SCALAR_STABLE_STEPS,
        ) if atomic_names or scalar_profiles else None

        with torch.inference_mode():
            generation_kwargs = {k: v for k, v in tensor_inputs.items() if k not in ("input_ids", "attention_mask")}
            try:
                gen_out = self.model.generate(
                    input_ids=tensor_inputs["input_ids"],
                    attention_mask=tensor_inputs.get("attention_mask"),
                    max_new_tokens=gen_budget,
                    streamer=observer,
                    **generation_kwargs,
                )
            except NativeToolReady:
                if observer is None or observer.accepted_name not in tool_by_name:
                    raise
                name = observer.accepted_name
                if observer.accepted_kind == "atomic":
                    if name not in atomic_names or observer.accepted_arguments != {}:
                        raise
                    accepted_args = {}
                    path = "NATIVE_ATOMIC_EARLY_EXIT"
                    tier = Tier.ATOMIC.value
                elif observer.accepted_kind == "scalar":
                    profile = scalar_profiles.get(name)
                    if profile is None or observer.accepted_arguments is None:
                        raise
                    accepted_args = validate_and_cast_call_args(observer.accepted_arguments, profile)
                    if accepted_args is None or set(accepted_args) != set(profile.parameters):
                        raise
                    path = "NATIVE_SCALAR_EARLY_EXIT"
                    tier = Tier.PARAMETRIC_PRIMITIVE.value
                else:
                    raise
                return EngineOutput(
                    content=None,
                    tool_calls=[{
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(accepted_args)},
                    }],
                    finish_reason="tool_calls",
                    execution_path=path,
                    tier=tier,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    # Draft stability is not a calibrated probability.
                    confidence=0.0,
                    steps_executed=observer.step,
                    prompt_tokens=prompt_len,
                    completion_tokens=canvas_length,
                    candidate_action=name,
                )

        sequences = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        gen_ids = sequences[0, prompt_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        completion_tokens = int(gen_ids.shape[0])
        forward_passes = self._estimate_forward_passes(gen_out, gen_ids, self.pad_token_id)
        total_lat = (time.perf_counter() - t0) * 1000.0

        tool_name, raw_args = parse_gemma_tool_call(raw_text)

        if tool_name is None:
            # The model answered directly instead of calling a tool -- either
            # no tools were offered, the request didn't need one, or (for
            # ambiguous/context-dependent requests) it is asking for
            # clarification rather than guessing. See RESULTS.md for examples
            # of this from the 2026-09-22 native-calling probe.
            return EngineOutput(
                content=self._clean_display_text(raw_text),
                tool_calls=None,
                finish_reason="stop",
                execution_path="NATIVE_DIRECT_RESPONSE",
                tier=None,
                latency_ms=total_lat,
                confidence=0.0,
                steps_executed=forward_passes,
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
            )

        if tool_name not in tool_by_name:
            return EngineOutput(
                content=f"Model referenced a tool not in the offered menu: {tool_name}",
                tool_calls=None,
                finish_reason="stop",
                execution_path="NATIVE_UNKNOWN_TOOL_REFERENCE",
                tier=None,
                latency_ms=total_lat,
                confidence=0.0,
                steps_executed=forward_passes,
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                candidate_action=tool_name,
            )

        complexity = inspect_tool_schema(tool_by_name[tool_name])
        validated_args = validate_and_cast_call_args(raw_args, complexity) if raw_args is not None else None

        if validated_args is None:
            # raw_args is None either because the body failed to parse, or
            # validate_and_cast_call_args rejected a missing/out-of-range
            # required field. Fail closed rather than emit a call the caller
            # cannot trust, matching the project's established policy.
            return EngineOutput(
                content="Unable to produce valid required tool arguments.",
                tool_calls=None,
                finish_reason="stop",
                execution_path="ARGUMENT_VALIDATION_FAILED",
                tier=complexity.tier.value,
                latency_ms=total_lat,
                confidence=0.3,
                steps_executed=forward_passes,
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                candidate_action=tool_name,
            )

        tool_call = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(validated_args)},
        }
        return EngineOutput(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls",
            execution_path="NATIVE_TOOL_CALL",
            tier=complexity.tier.value,
            latency_ms=total_lat,
            confidence=1.0,
            steps_executed=forward_passes,
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            candidate_action=tool_name,
        )

