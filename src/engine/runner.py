"""
KV-Warmed Tiered Execution Runner for Project Reflex.
Orchestrates:
  - Phase 1: Step-1 discrete tool routing on micro-control canvas
  - Tier 1: Zero-argument atomic tool call after one decoder pass
  - Tier 2: One explicit numeric argument extracted and validated from user text
  - Tier 3: Multi-pass generative canvas, with required-field validation
  - Seamless in-memory KV-cache retention across tiers
"""

import base64
import io
import json
import os
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
from ..schema.compiler import (
    CanvasCompiler,
    Choice,
    CompiledCanvas,
    Noul,
    Schema,
    compile_tier2_argument_canvas,
)
from ..schema.inspector import Tier, ToolComplexity, inspect_tool_schema
from .canvas import (
    ToolDefinition,
    compile_tools_to_schema,
    parse_and_validate_primitives,
    extract_unambiguous_numeric_argument,
    parse_openai_tools,
)
from .scheduler import DynamicStepScheduler, ExecutionPlan
from ..risk_gate import ConformalRiskGate, ExitAction, RiskGateResult


@dataclass
class EngineOutput:
    """Structured response from ReflexEngine."""
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: str
    execution_path: str
    tier: str
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

        # 4. Attach LoRA Adapter
        self.is_lora_loaded = False
        if self.settings.REFLEX_MODE == "reflex" and os.path.exists(self.settings.REFLEX_LORA_PATH):
            print(f"[ReflexEngine] Injecting trained Reflex LoRA from {self.settings.REFLEX_LORA_PATH}...")
            self.model.model.decoder = PeftModel.from_pretrained(
                self.model.model.decoder,
                self.settings.REFLEX_LORA_PATH,
            )
            self.model.eval()
            self.is_lora_loaded = True
            print("[ReflexEngine] Reflex LoRA adapter attached successfully.")

        # 5. Initialize Conformal Risk Gate & Step Scheduler
        self.risk_gate = ConformalRiskGate(
            epsilon=self.settings.CONFORMAL_EPS,
            delta=0.05,
            default_threshold=0.90,
            bound_type="clopper_pearson",
        )
        # This is an explicit heuristic until a matching calibration artifact exists.
        # Never mark a fixed threshold as statistically calibrated.

        self.scheduler = DynamicStepScheduler(
            default_generative_steps=self.settings.GENERATIVE_STEPS,
            default_generative_length=self.settings.MAX_CANVAS_LENGTH,
        )

    def _parse_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Image.Image]]:
        """Parses OpenAI messages and formats tool menus and image tokens."""
        parsed_messages = []
        images: List[Image.Image] = []

        tools_prefix = ""
        tools_suffix = ""
        if tools and len(tools) > 0:
            parsed_tools = parse_openai_tools(tools)
            tools_doc = "\n".join([f"[{i}] {t.name}: {t.description}" for i, t in enumerate(parsed_tools)])
            tools_prefix = f"Available Tools:\n{tools_doc}\n\nUser Request: "
            tools_suffix = f"\n\nSelect the most appropriate tool index [0 to {len(parsed_tools)-1}]:"

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content
                if tools_prefix and idx == len(messages) - 1:
                    text = f"{tools_prefix}{text}{tools_suffix}"
                parsed_messages.append({"role": role, "content": text})
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
                text = "\n".join(text_parts)
                if tools_prefix and idx == len(messages) - 1:
                    text = f"{tools_prefix}{text}{tools_suffix}"
                parsed_messages.append({"role": role, "content": text})
            else:
                text = str(content)
                if tools_prefix and idx == len(messages) - 1:
                    text = f"{tools_prefix}{text}{tools_suffix}"
                parsed_messages.append({"role": role, "content": text})

        prompt_text = self.tokenizer.apply_chat_template(
            parsed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text, images

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
        latest_user_content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        user_text = latest_user_content if isinstance(latest_user_content, str) else ""
        prompt_text, images = self._parse_messages(messages, tools=tools)
        tensor_inputs = self._prepare_inputs(prompt_text, images)
        prompt_len = tensor_inputs["input_ids"].shape[1]

        t0 = time.perf_counter()

        if self.settings.REFLEX_MODE == "vanilla":
            return self._run_vanilla(
                tensor_inputs=tensor_inputs,
                prompt_tokens=prompt_len,
                max_tokens=max_tokens or self.settings.MAX_CANVAS_LENGTH,
                t0=t0,
            )
        else:
            return self._run_reflex(
                tensor_inputs=tensor_inputs,
                prompt_tokens=prompt_len,
                tools=tools,
                tool_choice=tool_choice,
                user_text=user_text,
                max_tokens=max_tokens or self.settings.MAX_CANVAS_LENGTH,
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
        tensor_inputs: Dict[str, torch.Tensor],
        prompt_tokens: int,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        user_text: str,
        max_tokens: int,
        t0: float,
    ) -> EngineOutput:
        has_tools = bool(tools and len(tools) > 0)
        parsed_tools = parse_openai_tools(tools) if has_tools else []
        tool_map = {t.name: t for t in parsed_tools}

        with torch.inference_mode():
            # 1. Prompt Prefill (computes and retains in-memory prompt KV representations)
            enc_out = self.model.model.encoder(**tensor_inputs)
            past_kv = enc_out.past_key_values

            if has_tools:
                compiled_canvas, _ = compile_tools_to_schema(
                    tools=tools,
                    tokenizer=self.tokenizer,
                    mask_token_id=self.mask_token_id,
                    allow_direct_response=False,
                )
            else:
                schema = Schema(fields={
                    "action": Choice(options=["direct_response", "search", "calculate"]),
                    "needs_args": Noul(true_label="yes", false_label="no"),
                })
                compiler = CanvasCompiler(tokenizer=self.tokenizer, mask_token_id=self.mask_token_id)
                compiled_canvas = compiler.compile(schema)

            micro_canvas = compiled_canvas.canvas_tokens.to(self.device)

            # 2. Phase 1: Step-1 Forward Pass on Micro-Control Canvas (reusing past_kv)
            step1_out = self.model(
                input_ids=None,
                past_key_values=past_kv,
                decoder_input_ids=micro_canvas,
            )
            step1_logits = step1_out.logits

            # 3. Extract candidate logits & evaluate Conformal Risk Gate
            slot_logits = {}
            for slot_name, slot in compiled_canvas.slots.items():
                pos = slot.canvas_position
                raw_slot_logits = step1_logits[0, pos]
                slot_logits[slot_name] = (
                    raw_slot_logits,
                    slot.candidate_labels,
                    slot.candidate_token_ids,
                )

            risk_result = self.risk_gate.evaluate_logits(
                slot_logits=slot_logits,
                has_synthesis_field=compiled_canvas.has_synthesis,
            )

            action_pred = risk_result.slot_predictions.get("action")
            selected_action = action_pred.top_label if action_pred else "direct_response"
            overall_conf = risk_result.overall_confidence

            # If no tools or model chose direct response, branch to generative expansion
            if not has_tools or selected_action not in tool_map:
                return self._unroll_generative_synthesis(
                    past_kv=past_kv,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    overall_conf=overall_conf,
                    t0=t0,
                    candidate_action=selected_action,
                )

            # A low-confidence route must not become an executable tool call.
            # Current threshold is heuristic until separately calibrated.
            if not risk_result.is_safe_to_exit:
                return self._unroll_generative_synthesis(
                    past_kv=past_kv,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    overall_conf=overall_conf,
                    t0=t0,
                    candidate_action=selected_action,
                )

            # Tool is identified! Inspect its schema complexity
            raw_tool_def = next(t for t in tools if t.get("function", t).get("name") == selected_action)
            complexity = inspect_tool_schema(raw_tool_def)
            plan = self.scheduler.schedule(complexity, requested_max_tokens=max_tokens)

            # =========================================================
            # TIER 1: ATOMIC ACTION (0 Args Required) -> 1 Forward Pass
            # =========================================================
            if plan.tier == Tier.ATOMIC:
                step1_lat = (time.perf_counter() - t0) * 1000.0
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": selected_action,
                        "arguments": "{}",
                    },
                }
                return EngineOutput(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    execution_path="FAST_PATH_STEP_1",
                    tier=Tier.ATOMIC.value,
                    latency_ms=step1_lat,
                    confidence=overall_conf,
                    steps_executed=1,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                )

            # =========================================================
            # TIER 2: PARAMETRIC PRIMITIVE -> EXPLICIT NUMERIC EXTRACTION
            # =========================================================
            if plan.tier == Tier.PARAMETRIC_PRIMITIVE:
                # The v2 adapter was trained for action routing only. Do not turn
                # an untrained value canvas into an executable numeric argument.
                # Support the narrow case whose value is explicit and unambiguous.
                parsed_args = extract_unambiguous_numeric_argument(user_text, complexity)
                total_lat = (time.perf_counter() - t0) * 1000.0
                if parsed_args is None:
                    return EngineOutput(
                        content="Unable to determine a unique valid numeric argument.",
                        tool_calls=None,
                        finish_reason="stop",
                        execution_path="ARGUMENT_VALIDATION_FAILED",
                        tier=Tier.PARAMETRIC_PRIMITIVE.value,
                        latency_ms=total_lat,
                        confidence=overall_conf,
                        steps_executed=1,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        candidate_action=selected_action,
                    )
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": selected_action, "arguments": json.dumps(parsed_args)},
                }
                return EngineOutput(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    execution_path="TIER_2_EXPLICIT_NUMERIC_EXTRACTION",
                    tier=Tier.PARAMETRIC_PRIMITIVE.value,
                    latency_ms=total_lat,
                    confidence=overall_conf,
                    steps_executed=1,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    candidate_action=selected_action,
                )

            # =========================================================
            # TIER 3: OPEN GENERATIVE SYNTHESIS -> 12-20 Denoising Steps
            # =========================================================
            gen_len = plan.canvas_length
            gen_canvas = torch.full((1, gen_len), fill_value=self.mask_token_id, dtype=torch.long, device=self.device)

            steps_done = 0
            for _ in range(plan.max_denoise_steps):
                out = self.model(input_ids=None, past_key_values=past_kv, decoder_input_ids=gen_canvas)
                gen_canvas = torch.argmax(out.logits, dim=-1)
                steps_done += 1

            total_lat = (time.perf_counter() - t0) * 1000.0
            generated_text = self.tokenizer.decode(gen_canvas[0], skip_special_tokens=True).strip()

            # A generated payload is an executable tool call only when it parses
            # and all declared required fields pass validation.
            try:
                raw_arguments = json.loads(generated_text)
            except json.JSONDecodeError:
                raw_arguments = None
            parsed_args = parse_and_validate_primitives(generated_text, complexity) if isinstance(raw_arguments, dict) else {}
            if not isinstance(raw_arguments, dict) or any(name not in parsed_args for name in complexity.required_parameters):
                return EngineOutput(
                    content="Unable to produce valid required tool arguments.",
                    tool_calls=None,
                    finish_reason="stop",
                    execution_path="ARGUMENT_VALIDATION_FAILED",
                    tier=Tier.GENERATIVE_SYNTHESIS.value,
                    latency_ms=total_lat,
                    confidence=overall_conf,
                    steps_executed=1 + steps_done,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=gen_len,
                )
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            arg_str = json.dumps(parsed_args)
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": selected_action,
                    "arguments": arg_str,
                },
            }
            return EngineOutput(
                content=None,
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                execution_path="TIER_3_GENERATIVE_SYNTHESIS",
                tier=Tier.GENERATIVE_SYNTHESIS.value,
                latency_ms=total_lat,
                confidence=overall_conf,
                steps_executed=1 + steps_done,
                prompt_tokens=prompt_tokens,
                completion_tokens=gen_len,
            )

    def _unroll_generative_synthesis(
        self,
        past_kv: Any,
        prompt_tokens: int,
        max_tokens: int,
        overall_conf: float,
        t0: float,
        candidate_action: Optional[str] = None,
    ) -> EngineOutput:
        """Handles pure text generation when no tool is invoked."""
        gen_len = min(max_tokens, self.settings.MAX_CANVAS_LENGTH)
        gen_canvas = torch.full((1, gen_len), fill_value=self.mask_token_id, dtype=torch.long, device=self.device)

        steps = self.settings.GENERATIVE_STEPS
        for _ in range(steps):
            out = self.model(input_ids=None, past_key_values=past_kv, decoder_input_ids=gen_canvas)
            gen_canvas = torch.argmax(out.logits, dim=-1)

        total_lat = (time.perf_counter() - t0) * 1000.0
        generated_text = self.tokenizer.decode(gen_canvas[0], skip_special_tokens=True).strip()

        return EngineOutput(
            content=generated_text,
            tool_calls=None,
            finish_reason="stop",
            execution_path="EXPANDED_GENERATIVE_PATH",
            tier=Tier.GENERATIVE_SYNTHESIS.value,
            latency_ms=total_lat,
            confidence=overall_conf,
            steps_executed=1 + steps,
            prompt_tokens=prompt_tokens,
            completion_tokens=gen_len,
            candidate_action=candidate_action,
        )

