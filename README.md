<div align="center">

# ⚡ Project Reflex: Sub-150ms Discrete Decision Reflexes on Discrete Diffusion Models

**A Unified Control-First Serving Framework for Autonomous Agents & Tool Routing**

[![Author](https://img.shields.io/badge/Author-Tawananyasha_Mukoriwo-black.svg?style=flat&logo=github)](https://github.com/TawanaState)
[![Website](https://img.shields.io/badge/Website-tawananyasha.com-blue.svg)](https://tawananyasha.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcu130-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-DiffusionGemma--26B--A4B--it-4285F4.svg?logo=google)](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
[![API](https://img.shields.io/badge/API-OpenAI--Compatible-00A67E.svg?logo=openai)](https://platform.openai.com/docs/api-reference)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_GB10_Blackwell-76B900.svg?logo=nvidia)](https://www.nvidia.com)

---

### *Bridging Fast System-1 Decision Reflexes and Deep System-2 Generative Diffusion in a Single Weights Graph*

**Built solely by [Tawananyasha Mukoriwo](https://tawananyasha.com) ([@TawanaState](https://github.com/TawanaState))**

</div>

---

## 1. Executive Overview

Autonomous agents interacting with APIs, web environments, and operating systems execute discrete action selections in over 80% of their operational steps (*click*, *focus*, *navigate*, *select tool*, *retry*).

Existing state-of-the-art production deployments force these decisions through monolithic **Autoregressive (AR) LLMs**. This creates three severe bottlenecks:
1. **High Latency Overhead (850–1,200 ms):** Generating structured JSON tokens one-by-one introduces massive token-generation latency.
2. **Syntactic Fragility:** JSON schema syntax errors occur intermittently under heavy load or low temperatures.
3. **Escalation Penalty in Cascades:** Traditional two-model cascades (e.g., small classifier + large AR generator) suffer from domain mismatch, calibration divergence, and duplicate GPU memory footprints.

**Project Reflex** introduces **Control-First Canvas Expansion & Tiered Adaptive Compute** on Discrete Diffusion Language Models (`DiffusionGemma-26B-A4B-it`):
* **Tier 1 (Atomic Tool Routing):** When discrete actions require zero arguments ($N_{args} = 0$), Reflex evaluates a minimal 4-to-16 token decision canvas in a **single forward step** (76.8 ms KV-cached hit, 279.8 ms cold prefill) and exits immediately via Conformal Risk Gating.
* **Tier 2 (Parametric Primitive Infilling):** When a selected tool requires typed primitive arguments (`int`, `float`, `bool`, `enum`), Reflex compiles a structured micro-argument canvas ($L \in [8, 24]$ tokens) and unrolls **2 to 4 low-step denoising iterations** (<180 ms) with argmax convergence early-stopping.
* **Tier 3 (Generative Synthesis):** When free-form natural language generation, unbounded reasoning, or code synthesis is required, Reflex materializes an expanded generative canvas (64–256 tokens) and unrolls 12–20 diffusion steps—**reusing the causal encoder prompt KV state directly in memory with 0 ms prompt re-computation penalty**.

---

## 2. Architectural Comparison

| Dimension | Monolithic AR LLM (`gemma4:12b`) | Fixed 20-Step Diffusion (`DiffusionGemma-26B`) | Two-Model Cascade (Classifier + AR) | **Project Reflex (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Model Graph** | Single Autoregressive | Single Discrete Diffusion | Separate Classifier + AR | **Unified Bidirectional Diffusion + LoRA** |
| **Tier 1 (Atomic) Exit** | ❌ No (full token decode) | ❌ No (fixed multi-step) | ⚠️ Partial (Classifier only) | **✅ 1 Step (<130ms)** |
| **Tier 2 (Primitive) Exit**| ❌ No (full JSON decode) | ❌ No (fixed multi-step) | ❌ No (full AR decode) | **✅ 2–4 Steps (<180ms Micro-Canvas)** |
| **Tier 3 (Generative) Exit**| Full sequential decode | Fixed 20-Step Diffusion | AR LLM decode | **✅ Adaptive 12–20 Steps + KV Reuse** |
| **p50 Latency (Routing)** | 856.1 ms | 5,293.5 ms | 528.0 ms | **76.8 ms (Hit) / 279.8 ms (Prefill)** |
| **p95 Latency (Routing)** | 941.8 ms | 5,452.3 ms | 980.0 ms | **78.1 ms (Hit) / 315.0 ms (Prefill)** |
| **Speedup vs AR** | 1.00x | 0.18x | 1.65x | **11.5x (Hit) / 3.4x (End-to-End)** |
| **Prompt KV Reuse on Escalation**| N/A | N/A | ❌ 0% (Duplicate re-encode) | **✅ 100% In-Memory Retention (0ms penalty)**|
| **Syntax Error Rate** | >0.0% (JSON parsing risk)| 0.0% | >0.0% (on escalation) | **0.00% (Constrained Canvas Slots)** |
| **Safety Certification** | Uncalibrated heuristic | N/A | Empirical heuristic | **Conformal Risk Bound ($\le \epsilon$)** |

---

## 3. Hardware Requirements & Prerequisites

Reflex is designed for bare-metal modern GPU systems and unified memory architectures:

* **Recommended Hardware:**
  * NVIDIA Grace Blackwell (GB10 / DGX Spark)
  * NVIDIA H100 / A100 (80GB SXM5/PCIe)
  * Multi-GPU workstations (2x 48GB A6000 Ada or L40S)
* **VRAM Footprint:**
  * Base `DiffusionGemma-26B-A4B-it` in `bfloat16`: **~48.34 GB**
  * Reflex LoRA adapter (`models/reflex_lora_v1`): **~45 MB**
  * Fast-path Step-1 inference peak: **~48.5 GB**
* **Software Stack:**
  * Linux x86_64 or ARM64 (Ubuntu 22.04 / 24.04 LTS recommended)
  * NVIDIA Driver $\ge 535.00$ (Driver 580.159+ tested)
  * CUDA Toolkit $\ge 12.4$ (CUDA 13.0 tested)
  * Python 3.10 – 3.12

---

## 4. Quickstart Guide (5 Minutes to Run)

### Step 1: Clone and Set Up Environment
```bash
git clone https://github.com/your-org/reflex.git
cd reflex

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment (`.env`)
Copy `.env.example` to `.env` and configure your paths:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```ini
# Hugging Face token (if downloading directly from HF Hub)
HF_TOKEN="hf_your_token_here"

# Path to DiffusionGemma checkpoint or HF repo ID
DIFFUSION_GEMMA_PATH="/path/to/diffusiongemma-26B-A4B-it"

# Path to trained Reflex LoRA adapter
REFLEX_LORA_PATH="models/reflex_lora_v1"

# Runtime mode: 'reflex' or 'vanilla'
REFLEX_MODE="reflex"

# Network configuration
HOST="0.0.0.0"
PORT=8090

# Target CUDA device
DEVICE="cuda:0"

# Conformal error tolerance
CONFORMAL_EPS=0.02
```

### Step 3: Launch the Inference Server
```bash
# Launch the OpenAI-compatible FastAPI server
python -m src.server
```
The server will bind to `http://0.0.0.0:8090` and expose OpenAI-compatible endpoints.

---

## 5. Client Usage Examples

Project Reflex is a drop-in replacement for any client that uses the official `openai` Python SDK.

### Example A: Sub-150ms Fast-Path Tool Dispatch
When discrete tools are provided and the action is atomic, Reflex halts at Step 1 and dispatches the tool call instantly.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="reflex-local",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "mute_system_audio",
            "description": "Mutes master audio output instantly without arguments",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Launches a desktop application by name",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        },
    },
]

response = client.chat.completions.create(
    model="reflex-diffusiongemma",
    messages=[{"role": "user", "content": "Quick, silence all audio output!"}],
    tools=tools,
)

choice = response.choices[0]
print(f"Finish Reason: {choice.finish_reason}")  # 'tool_calls'
print(f"Tool Selected: {choice.message.tool_calls[0].function.name}")  # 'mute_system_audio'
print(f"Execution Path: {response.reflex_metadata['execution_path']}")  # 'FAST_PATH_STEP_1'
print(f"Total Latency: {response.reflex_metadata['latency_ms']:.1f} ms")  # < 150 ms
```

### Example B: Sub-180ms Parametric Tool Infilling (Tier 2)
When a tool requires primitive arguments (`int`, `float`, `bool`, `enum`), Reflex compiles a 16-token micro-argument canvas and resolves the arguments in 2–4 diffusion steps with argmax convergence early-stopping.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="reflex-local",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Adjusts temperature and operating mode",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["eco", "heat", "cool"]},
                },
                "required": ["temperature", "mode"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="reflex-diffusiongemma",
    messages=[{"role": "user", "content": "Set thermostat to 72 in heat mode"}],
    tools=tools,
)

choice = response.choices[0]
tool_call = choice.message.tool_calls[0]
print(f"Tool Selected: {tool_call.function.name}")  # 'set_thermostat'
print(f"Arguments: {tool_call.function.arguments}")  # '{"temperature": 72, "mode": "heat"}'
print(f"Compute Tier: {response.reflex_metadata['tier']}")  # 'PARAMETRIC_PRIMITIVE'
print(f"Steps Executed: {response.reflex_metadata['steps_executed']}")  # 2-4 steps
print(f"Total Latency: {response.reflex_metadata['latency_ms']:.1f} ms")  # < 180 ms
```

### Example C: Conversational & Generative Expansion (Tier 3)
When answering open-ended queries or coding prompts, Reflex conditionally unrolls Phase 2 generative diffusion (12–20 steps):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="reflex-local",
)

response = client.chat.completions.create(
    model="reflex-diffusiongemma",
    messages=[
        {"role": "user", "content": "Write a Python script to reverse a linked list."}
    ],
    max_tokens=128,
)

choice = response.choices[0]
print(f"Finish Reason: {choice.finish_reason}")  # 'stop'
print(f"Generated Output:\n{choice.message.content}")
print(f"Execution Path: {response.reflex_metadata['execution_path']}")  # 'EXPANDED_GENERATIVE_PATH'
print(f"Compute Tier: {response.reflex_metadata['tier']}")  # 'GENERATIVE_SYNTHESIS'
print(f"Steps Executed: {response.reflex_metadata['steps_executed']}")  # 21 steps
```

### Example D: Multimodal Vision Payloads
Reflex natively routes image tensors into DiffusionGemma's vision encoder tower:

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="reflex-local",
)

with open("screenshot.png", "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="reflex-diffusiongemma",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What button is focused in this UI?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

---

## 6. Research Artifacts & Empirical Verification

All metrics and findings in Project Reflex are grounded in bare-metal hardware execution:

* **Hardware Testbed:** Bare-metal NVIDIA DGX Spark Workstation (NVIDIA GB10 Blackwell Grace Architecture, 121 GiB Unified Memory).
* **Step-1 Logit Extraction:** Evaluated across Google BoolQ, Banking77 (77-class intent triage), and BFCL v4 multiple routing.
* **In-Memory KV-Cache Retention:** Validated via `torch.cuda.Event` timing (`experiments/test_kv_retention_timing.py`), proving **234.57 ms** of redundant prompt re-computation is mathematically avoided during Phase 2 escalation.
* **Conformal Risk Calibration:** Guarantees $\mathcal{P}(\text{error} \mid \text{exit}) \le \epsilon$ under the Empirical Bernstein Bound.
* **Pareto Frontier Artifacts:**
  * Pareto Analysis Summary: `RESULTS.md`
  * Complete Empirical Log: `experiments/real_benchmark_comparison.json`
  * High-Resolution Pareto Curve: `experiments/real_pareto_frontier_v2.png`

---

## 7. Project Structure

```text
reflex/
├── .env.example                      # Template environment variables
├── requirements.txt                  # Python dependencies
├── README.md                         # Public repository documentation
├── RESULTS.md                        # Empirical hardware evaluation paper summary
├── NOTES.md                          # Implementation log & verification traces
├── data/                             # Train/Cal/Test splits (BoolQ, Banking77, BFCL)
├── models/
│   └── reflex_lora_v1/               # Validated PEFT LoRA adapter checkpoint
├── src/
│   ├── __init__.py
│   ├── config.py                     # Pydantic settings & .env loader
│   ├── canvas.py                     # Backward-compatibility bridge for canvas & schema
│   ├── risk_gate.py                  # Conformal Risk Gate (Angelopoulos et al.)
│   ├── expansion.py                  # Canvas Expansion Manager with KV reuse
│   ├── schema/                       # Schema inspection & canvas compilation
│   │   ├── inspector.py              # Parameter inspector & tier classification
│   │   └── compiler.py               # Micro-argument canvas compiler (L ∈ [8, 24])
│   ├── engine/                       # Unified tiered execution engine
│   │   ├── runner.py                 # ReflexEngine runtime with tiered fast-paths
│   │   ├── scheduler.py              # DynamicStepScheduler (entropy & complexity aware)
│   │   └── canvas.py                 # Tool schema & prompt formatting utilities
│   └── server.py                     # OpenAI-compatible FastAPI server
├── tests/
│   ├── test_canvas.py                # Canvas compiler unit tests
│   ├── test_risk_gate.py             # Conformal risk gate unit tests
│   ├── test_fast_path_reflex.py      # Sub-150ms fast-path integration test
│   ├── test_generative_expansion.py  # Phase 2 multi-step generation test
│   ├── test_tiered_compute.py        # Tier 1, Tier 2, and Tier 3 validation tests
│   ├── test_multimodal.py            # Vision encoder integration test
│   └── test_openai_client.py         # Official openai Python SDK client test
└── experiments/                      # Benchmarks, probes, LoRA training, & Pareto suite
    ├── benchmark_tiered_latency.py   # Latency & accuracy benchmark across compute tiers
    └── ...
```

---

## 8. Author & Maintainer

Project Reflex was conceived, researched, and engineered solely by **Tawananyasha Mukoriwo** (independent researcher, no external team).

* **Personal Website:** [tawananyasha.com](https://tawananyasha.com) (contact details, portfolio, and research writing)
* **GitHub:** [@TawanaState](https://github.com/TawanaState)

---

## 9. License & Citation

Project Reflex is released under the **Apache 2.0 License**.

```bibtex
@article{reflex2026,
  title   = {Project Reflex: Sub-150ms Discrete Decision Reflexes on Discrete Diffusion Models},
  author  = {Mukoriwo, Tawananyasha},
  year    = {2026},
  journal = {arXiv preprint},
  url     = {https://tawananyasha.com}
}
```

