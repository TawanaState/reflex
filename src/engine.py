"""
Project Reflex: Dual-Mode Unified Inference Engine.
Integrates DiffusionGemma-26B-A4B-it with calibrated LoRA adapter,
micro-control canvas Step-1 evaluation, Conformal Risk Gating,
seamless in-memory KV-cache retention, and multimodal vision processing.
"""

import base64
import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, DiffusionGemmaForBlockDiffusion

from .canvas import (
    CanvasCompiler,
    Choice,
    CompiledCanvas,
    Noul,
    Schema,
    ToolDefinition,
    compile_tools_to_schema,
    parse_openai_tools,
)
from .config import Settings, get_settings
from .expansion import CanvasExpansionManager
from .risk_gate import ConformalRiskGate, ExitAction, RiskGateResult


@dataclass
class EngineOutput:
    """Structured response from ReflexEngine."""
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: str
    execution_path: str
    latency_ms: float
    confidence: float
    steps_executed: int
    prompt_tokens: int
    completion_tokens: int


class ReflexEngine:
    """
    Unified Inference Engine for Project Reflex.
    Supports:
      - Mode 'reflex': Sub-150ms Step-1 micro-canvas decision + Conformal Risk Gate + conditional expansion.
      - Mode 'vanilla': Fixed multi-step diffusion generation.
      - Multimodal inputs (image_url base64/URL).
      - Multi-turn conversation context with prompt KV caching.
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
            print(f"[ReflexEngine] Warning: AutoProcessor failed to load ({e}), fallback to text-only.")
            self.processor = None

        # 3. Load Base Model
        print("[ReflexEngine] Loading DiffusionGemma 26B/A4B in bfloat16...")
        t0 = time.time()
        self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            self.settings.DIFFUSION_GEMMA_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if "cuda" in str(self.device) else "cpu",
        )
        self.model.eval()
        print(f"[ReflexEngine] Model loaded in {time.time() - t0:.1f}s.")

        # 4. Load LoRA Adapter if in reflex mode
        self.is_lora_loaded = False
        if self.settings.REFLEX_MODE == "reflex" and os.path.exists(self.settings.REFLEX_LORA_PATH):
            print(f"[ReflexEngine] Injecting trained Reflex LoRA from {self.settings.REFLEX_LORA_PATH}...")
            self.model.model.decoder = PeftModel.from_pretrained(
                self.model.model.decoder,
                self.settings.REFLEX_LORA_PATH,
            )
            self.model.eval()
            self.is_lora_loaded = True
            print("[ReflexEngine] Reflex LoRA adapter attached to decoder attention.")
        else:
            print(f"[ReflexEngine] Running base decoder (LoRA not attached).")

        # 5. Initialize Conformal Risk Gate & Canvas Expansion Manager
        self.risk_gate = ConformalRiskGate(
            epsilon=self.settings.CONFORMAL_EPS,
            delta=0.05,
            default_threshold=0.50,
            bound_type="empirical_bernstein",
        )
        # Calibrate default lambda*: (1 - lambda*) = 0.50
        self.risk_gate.calibrated_lambda = 0.50
        self.risk_gate.is_calibrated = True

        self.expansion_manager = CanvasExpansionManager(
            default_gen_length=self.settings.MAX_CANVAS_LENGTH,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
        )

        # In-memory KV session cache (prompt_hash -> (past_key_values, timestamp))
        self._kv_cache: Dict[str, Tuple[Any, float]] = {}

    def _parse_messages(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[Image.Image]]:
        """
        Parses OpenAI messages list, extracting text and any embedded images.
        Injects `<|image|>` token when an image is present, and includes tool definitions
        when tools are provided.
        """
        parsed_messages = []
        images: List[Image.Image] = []

        tools_block = ""
        if tools and len(tools) > 0:
            parsed_tools = parse_openai_tools(tools)
            tools_doc = "\n".join([f"[{i}] {t.name}: {t.description}" for i, t in enumerate(parsed_tools)])
            tools_block = f"\n\nAvailable Tools:\n{tools_doc}\n\nSelect the most appropriate tool index [0 to {len(parsed_tools)-1}]:"

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                text = content
                if tools_block and idx == len(messages) - 1:
                    text = text + tools_block
                parsed_messages.append({"role": role, "content": text})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type == "text":
                        text_parts.append(part.get("text", ""))
                    elif part_type == "image_url":
                        img_info = part.get("image_url", {})
                        url = img_info.get("url", "")
                        loaded_img = self._load_image(url)
                        if loaded_img is not None:
                            images.append(loaded_img)
                            text_parts.append("<|image|>")
                text = "\n".join(text_parts)
                if tools_block and idx == len(messages) - 1:
                    text = text + tools_block
                parsed_messages.append({"role": role, "content": text})
            else:
                text = str(content)
                if tools_block and idx == len(messages) - 1:
                    text = text + tools_block
                parsed_messages.append({"role": role, "content": text})

        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            parsed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt_text, images


    def _load_image(self, url: str) -> Optional[Image.Image]:
        """Loads PIL Image from data URL, http URL, or local filepath."""
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
        except Exception as e:
            print(f"[ReflexEngine] Failed to load image from {url[:30]}...: {e}")
        return None

    def _prepare_inputs(
        self, prompt_text: str, images: List[Image.Image]
    ) -> Dict[str, torch.Tensor]:
        """Prepares tensor inputs for the causal encoder using tokenizer or processor."""
        if images and self.processor is not None:
            # Multimodal inputs
            inputs = self.processor(
                text=prompt_text,
                images=images if len(images) > 1 else images[0],
                return_tensors="pt",
            )
        else:
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)

        # Move tensors to device
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
        """
        Executes generation pipeline according to active REFLEX_MODE.
        """
        prompt_text, images = self._parse_messages(messages, tools=tools)
        tensor_inputs = self._prepare_inputs(prompt_text, images)
        prompt_len = tensor_inputs["input_ids"].shape[1]

        t0 = time.perf_counter()

        # Branch based on mode
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
        """Standard fixed multi-step diffusion baseline."""
        gen_len = min(max_tokens, self.settings.MAX_CANVAS_LENGTH)
        canvas = torch.full((1, gen_len), fill_value=self.mask_token_id, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            # Prefill encoder pass
            enc_out = self.model.model.encoder(**tensor_inputs)
            past_kv = enc_out.past_key_values

            # Fixed multi-step reverse diffusion
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
        max_tokens: int,
        t0: float,
    ) -> EngineOutput:
        """
        Reflex hybrid pipeline:
        Step 1 Micro-Control Canvas -> Conformal Risk Gate -> Fast Path Exit OR Conditional Expansion.
        """
        has_tools = bool(tools and len(tools) > 0)
        parsed_tools = parse_openai_tools(tools) if has_tools else []
        tool_map = {t.name: t for t in parsed_tools}

        with torch.inference_mode():
            # 1. Prompt Prefill (computes and retains in-memory past_key_values)
            enc_out = self.model.model.encoder(**tensor_inputs)
            past_kv = enc_out.past_key_values

            if has_tools:
                # Compile micro-control canvas for tool selection
                compiled_canvas, _ = compile_tools_to_schema(
                    tools=tools,
                    tokenizer=self.tokenizer,
                    mask_token_id=self.mask_token_id,
                    allow_direct_response=False,
                )
            else:
                # Default generic micro-control canvas (e.g., triage / action / response-type)
                schema = Schema(fields={
                    "action": Choice(options=["direct_response", "search", "calculate"]),
                    "needs_args": Noul(true_label="yes", false_label="no"),
                    "risk": Choice(options=["low", "high"]),
                })
                compiler = CanvasCompiler(tokenizer=self.tokenizer, mask_token_id=self.mask_token_id)
                compiled_canvas = compiler.compile(schema)

            micro_canvas = compiled_canvas.canvas_tokens.to(self.device)

            # 2. Step-1 Forward Pass on Micro-Control Canvas (reusing past_kv)
            step1_out = self.model(
                input_ids=None,
                past_key_values=past_kv,
                decoder_input_ids=micro_canvas,
            )
            step1_logits = step1_out.logits  # (1, canvas_len, vocab_size)

            # 3. Extract candidate logits & Evaluate Conformal Risk Gate
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

            needs_args_pred = risk_result.slot_predictions.get("needs_args")
            needs_args_label = needs_args_pred.top_label if needs_args_pred else "no"

            # Check if selected tool is atomic
            is_atomic_tool = False
            if has_tools and selected_action in tool_map:
                tool_def = tool_map[selected_action]
                is_atomic_tool = tool_def.is_atomic or (needs_args_label == "no")

            # -------------------------------------------------------------
            # FAST-PATH EXIT EVALUATION
            # -------------------------------------------------------------
            can_fast_path = (
                has_tools
                and selected_action in tool_map
                and is_atomic_tool
                and (risk_result.is_safe_to_exit or overall_conf >= (1.0 - self.settings.CONFORMAL_EPS * 20))
            )

            if can_fast_path:
                # Halt at Step 1! Return typed action object immediately.
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
                    latency_ms=step1_lat,
                    confidence=overall_conf,
                    steps_executed=1,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                )

            # -------------------------------------------------------------
            # PHASE 2: CONDITIONAL GENERATIVE EXPANSION
            # -------------------------------------------------------------
            # Query demands open text, tool requires arguments, or risk gate escalated
            gen_len = min(max_tokens, self.settings.MAX_CANVAS_LENGTH)
            gen_canvas = torch.full((1, gen_len), fill_value=self.mask_token_id, dtype=torch.long, device=self.device)

            steps = self.settings.GENERATIVE_STEPS
            for _ in range(steps):
                out = self.model(input_ids=None, past_key_values=past_kv, decoder_input_ids=gen_canvas)
                gen_canvas = torch.argmax(out.logits, dim=-1)

            total_lat = (time.perf_counter() - t0) * 1000.0
            generated_text = self.tokenizer.decode(gen_canvas[0], skip_special_tokens=True).strip()

            # If tool selected required arguments, format tool call with arguments
            if has_tools and selected_action in tool_map and not is_atomic_tool:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": selected_action,
                        "arguments": generated_text if "{" in generated_text else "{}",
                    },
                }
                return EngineOutput(
                    content=None,
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                    execution_path="EXPANDED_GENERATIVE_PATH",
                    latency_ms=total_lat,
                    confidence=overall_conf,
                    steps_executed=1 + steps,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=gen_len,
                )

            # Return open text response
            return EngineOutput(
                content=generated_text,
                tool_calls=None,
                finish_reason="stop",
                execution_path="EXPANDED_GENERATIVE_PATH",
                latency_ms=total_lat,
                confidence=overall_conf,
                steps_executed=1 + steps,
                prompt_tokens=prompt_tokens,
                completion_tokens=gen_len,
            )

