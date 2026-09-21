## Project Reflex: Open-Source Serving & Public Release Packaging

You are the **Lead ML Systems Engineer** tasked with preparing **Project Reflex** for open-source public release. The core research, Phase 0 probes, Step-1 logit extraction, LoRA weights (`models/reflex_lora_v1`), and conformal risk calibration are already validated and working in the repository.

Your objective is to turn this research codebase into a clean, reproducible, high-performance public repository that anyone with access to an NVIDIA GPU workstation (DGX, A100, H100, or modern 80GB+ VRAM machine) can clone, configure via `.env`, and launch as an **OpenAI-compatible inference server**.

---

### NON-NEGOTIABLE OPERATIONAL DIRECTIVES

1. **Zero Mocking / Zero Fakes:**
   * Never inject simulated sleep times, synthetic mock predictions, or fake metrics.
   * All server endpoints, token reads, and tool dispatches must run through the actual PyTorch / Transformers model graph.
2. **Preserve Validated Artifacts:**
   * Do **NOT** delete, break, or overwrite existing working experiments (`experiments/`), dataset splits (`data/`), model weights (`models/reflex_lora_v1/`), or result JSONs (`results/` and `experiments/*.json`).
3. **Patience, Research & Verification:**
   * Before writing code, inspect existing files (`experiments/01_step1_real_spike.py`, `experiments/test_kv_retention_timing.py`, `train_reflex_lora.py`, and `PROPOSAL.md`).
   * When integrating tools or OpenAI API schemas, follow official OpenAI API specs and PyTorch best practices.
   * Run live end-to-end integration tests using the official `openai` Python SDK before declaring any task complete.

---

### ARCHITECTURAL REQUIREMENTS & DELIVERABLES

#### 1. Environment Configuration (`.env` & `src/config.py`)
Create a standardized `.env.example` and a strict settings loader (`src/config.py` using `pydantic-settings` or `python-dotenv`):
* `DIFFUSION_GEMMA_PATH`: Path to local model directory or Hugging Face repo ID (e.g., `google/diffusiongemma-26b-it` or NVIDIA FP4/BF16 weights).
* `REFLEX_LORA_PATH`: Path to the trained LoRA adapter (default: `models/reflex_lora_v1`).
* `REFLEX_MODE`: Toggle runtime mode:
  * `reflex`: The full hybrid runtime (sub-150ms Step-1 micro-canvas decision + conformal risk gate + conditional generative expansion).
  * `vanilla`: Standard DiffusionGemma baseline (fixed multi-step canvas denoising, no early exit).
* `HOST`: Server bind address (default: `0.0.0.0`).
* `PORT`: Server port (default: `8000`).
* `CONFORMAL_EPS`: Operational error tolerance for risk gating (default: `0.02`).
* `DEVICE`: Target CUDA device (default: `cuda:0`).
* `MAX_CANVAS_LENGTH`: Maximum generative expansion buffer length (default: `256`).

#### 2. Dual-Mode Inference Engine (`src/engine.py`)
Implement a unified runtime engine supporting both modes cleanly:
* **Mode A (`vanilla`):**
  * Standard diffusion pipeline: Ingests user prompt, causal encoder caches KV state, runs full 15–25 denoising steps on the generation canvas, and returns text.
* **Mode B (`reflex`):**
  * **Phase 1 (The Micro-Control Canvas):**
    * Allocates a minimal 4-to-16 token canvas (`action: @ \n needs_args: @ \n risk: @`).
    * Extracts candidate logits exclusively on the masked slots using `logprob_token_ids` in a single forward pass (~110ms).
    * Evaluates the conformal risk gate ($p_{\text{target}} \ge 1 - \lambda^*$).
  * **Fast-Path Exit:** If the selected action is atomic/discrete and passes the risk gate, halts at Step 1 and immediately returns the typed action object.
  * **Phase 2 (Conditional Expansion):** If the selected action requires arguments or the query demands free-form text/reasoning, materializes an expanded generative canvas (64–256 tokens) and unrolls 10–20 denoising steps, reusing the cached prompt KV without re-encoding.

#### 3. Full Modality & Conversation Support
* **Multi-Turn Context:** The causal encoder processes conversation history (`messages: [{"role": "user", ...}, {"role": "assistant", ...}]`) and retains prompt KV states across turns.
* **Multimodal Vision:** Support base64 or URL image payloads inside the standard OpenAI message format:
  ```json
  {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]}

    ```

Pass image tensors directly into DiffusionGemma’s vision encoder tower.

* **Tool Calling / Function Routing:**
* Accept standard OpenAI `tools: [{"type": "function", "function": {...}}]`.
* The schema compiler (`src/canvas.py`) automatically maps candidate tool names to slot-constrained candidate tokens on the micro-control canvas.



#### 4. OpenAI-Compatible API Server (`src/server.py`)

Implement a FastAPI application exposing:

* `GET /health` (System status, active GPU memory, loaded mode).
* `GET /v1/models` (Returns model ID and active mode metadata).
* `POST /v1/chat/completions`:
* Full support for OpenAI request and response specifications.
* When `reflex` mode executes an atomic tool reflex on Step 1, format the output as:
```json
{
  "id": "chatcmpl-reflex-xyz",
  "object": "chat.completion",
  "created": 1789999999,
  "model": "reflex-diffusiongemma",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {"name": "target_tool_name", "arguments": "{}"}
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 128, "completion_tokens": 0, "total_tokens": 128},
  "reflex_metadata": {
    "execution_path": "FAST_PATH_STEP_1",
    "latency_ms": 112.4,
    "confidence": 0.982
  }
}

```


* When generating open text, return standard `message: {"role": "assistant", "content": "..."}`.



#### 5. Verification & Test Suite (`tests/`)

Write standalone validation scripts that execute against the live server:

* `tests/test_fast_path_reflex.py`: Sends single-turn discrete routing queries and asserts response time is $<150\text{ ms}$ with valid `tool_calls`.
* `tests/test_generative_expansion.py`: Sends open-ended text and coding prompts, verifying valid multi-step text generation.
* `tests/test_multimodal.py`: Sends an image + question payload, verifying the vision encoder functions without runtime crashes.
* `tests/test_openai_client.py`: Verifies direct compatibility using the official `from openai import OpenAI` Python SDK.

#### 6. Public Documentation & Presentation (`README.md`)

Write an executive, publication-grade `README.md` containing:

1. **Hero Overview:** Crisp explanation of Reflex—unifying sub-150ms discrete decision reflexes (System 1) with generative diffusion synthesis (System 2) in a single model.
2. **Architectural Comparison Table:** Reflex vs. Monolithic AR LLMs vs. TypeSafe Jev vs. Two-Model Cascades.
3. **Hardware Requirements & Prerequisites:** GPU memory guidelines (80GB+ VRAM recommendations for 26B/A4B checkpoint), CUDA drivers, Python version.
4. **Quickstart Guide (5 Minutes to Run):**
* Environment setup & dependency installation (`requirements.txt`).
* `.env` file configuration.
* One-line server launch command: `python -m src.server`.


5. **Client Usage Examples:**
* Python snippet using standard `openai` library for fast tool dispatch.
* Python snippet using standard `openai` library for conversational generation.


6. **Benchmark & Research Artifacts:** References to paper findings, latency scaling benchmarks, and Pareto plots.

---

### EXECUTION STAGES

1. **Audit & Scaffolding:** Inspect existing files, create clean directory structure (`src/`, `tests/`, `configs/`).
2. **Engine & Configuration:** Build `src/config.py`, `src/canvas.py`, and `src/engine.py` connecting the verified model pipeline to the dual-mode switch.
3. **API Implementation:** Build `src/server.py` with FastAPI, handling JSON tool extraction, micro-canvas seeding, and OpenAI response formatting.
4. **Testing & Validation:** Spin up the server locally on the GPU, execute the test suite, and record real stdout traces in `NOTES.md`.
5. **Documentation & Polish:** Finalize `README.md`, `.env.example`, and `requirements.txt`.
