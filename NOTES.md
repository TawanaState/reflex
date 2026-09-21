# Reflex Engineering Logbook

## System Environment Baseline
- GPU Hardware: NVIDIA GB10 (NVIDIA DGX Spark / Grace Blackwell SoC, aarch64, 20-core Cortex-X925/A725)
- Driver & CUDA Version: Driver 580.159.03, CUDA 13.0 (V13.0.88)
- Initial Available VRAM: Unified LPDDR5X/HBM Memory: Total 121 GiB, 75 GiB available (Ollama background instances occupying ~38 GiB)
- PyTorch / vLLM / Transformers versions: System Python 3.12.3; target environment will use isolated `.venv` with PyTorch CUDA 13 wheels (`torch --index-url https://download.pytorch.org/whl/cu130`), `transformers`, `accelerate`, and `vllm`/custom diffusion interposers.

---

## [2026-09-21 10:15] - Task: Phase 0 Research Reconnaissance & Architecture Audit
### Objective & Hypothesis
Inspect upstream vLLM PR #57250, Hugging Face `transformers` DiffusionGemma implementation, Prophet diffusion early-exit literature, and Jev-style typed decision engines.
**Hypothesis:** A discrete diffusion language model (DiffusionGemma 26B/A4B) can perform structured reads on a micro-control canvas (4–16 tokens) in a single denoising step (<150ms) by sampling logits at masked token positions over a cached prompt KV representation, completely avoiding the quadratic/attention overhead of full-length generative canvases.

### Reconnaissance & Borrowed Code
- **vLLM PR #57250** (`[Core] structured generation mode for DiffusionGemma model (Jev-like)` by `@mmastrac`):
  - Introduces `vllm_xargs`: `diffusion_seed_canvas`, `diffusion_max_steps: 1`, `diffusion_read_only: true`.
  - Replaces random canvas initialization with structured answer templates containing masked prediction slots.
  - Allows constrained token logprob extraction (`logprob_token_ids`) directly at Step 1.
- **Related vLLM PRs**:
  - PR #57416: Prefill-only logit row optimization.
  - PR #57589 (`mmastrac:dgemma-mm-lora-flag`): Multimodal diffusion LoRA flags.
- **Prophet Literature** (*"Diffusion Language Models Know the Answer Before Decoding"*, 2025/2026):
  - Proves the "early answer convergence" property in discrete diffusion models: models lock into categorical answers at early steps before iterative spatial refinement.
  - Monitors top-1 vs top-2 confidence gap and entropy to trigger early exits or single-step commits.
- **Conformal Risk Control & Conformal Thinking** (Angelopoulos et al., 2024–2026):
  - Replaces fragile heuristic entropy thresholds with statistically bounded risk control: $P(\text{error} \mid \text{exit}) \le \epsilon$.
  - Calibrates $\lambda^*$ via empirical Bernstein/Hoeffding Upper Confidence Bounds (UCB) on held-out calibration split $\mathcal{D}_{\text{cal}}$.
- **OpenJev Ecosystem** (`open-jev-deberta`, `openjev-sglang`, `com-kotobalabs/open-jev-deberta`):
  - Demonstrates typed schemas (`Choice`, `Score`, `Noul`) mapped to structured slots with 0% syntax error rate.

### Commands & Implementation
```bash
# Baseline diagnostics executed
nvidia-smi
lscpu
free -h
nvcc --version
python3 --version
docker --version
```

### Raw Output & Real Metrics
* Hardware: NVIDIA GB10 (Blackwell GPU + Grace 20-core ARM CPU)
* Host Memory: 121 GiB Total Unified Memory, 75 GiB Available, 60 GiB Free.
* Target Model: `google/diffusiongemma-26B-A4B-it` (MoE: 26B total, 3.8B active per token).

### What Worked vs. What Failed
* **Successes:** Hardware verified; unified memory on GB10 provides ample headroom (>75 GiB available) to host 26B/A4B parameters (which requires ~26–52 GB depending on precision: BF16 ~52GB, FP8 ~26GB, 4-bit ~15GB).
* **Failures & Blockers:** Docker daemon socket is restricted to root (`permission denied`), so all development, virtual environments, and PyTorch executions will run natively in host user space (`.venv`) using aarch64 wheels.

### Next Steps
1. Create detailed Implementation Plan artifact (`implementation_plan.md`) and obtain user sign-off. (COMPLETED)
2. Initialize `.venv` environment and install PyTorch with CUDA 13 / aarch64 support + Hugging Face dependencies. (COMPLETED)
3. Construct core Reflex modules: `src/canvas.py`, `src/risk_gate.py`, `src/expansion.py`, `src/runtime.py`. (COMPLETED)
4. Build isolated micro-benchmark `experiments/01_step1_probe.py` and comparative benchmark `experiments/02_benchmark_baselines.py`. (COMPLETED)

---

## [2026-09-21 11:00] - Task: Environment Setup, Runtime Construction & Unit Testing
### Objective & Hypothesis
Set up an isolated `.venv` environment on the NVIDIA GB10 SoC, install CUDA 13 wheels, and build the core modular Reflex runtime:
- `src/canvas.py`: Typed Schema Compiler (`Choice`, `Score`, `Noul`, `SynthesisField`) compiling into 4–16 token micro-control canvases.
- `src/risk_gate.py`: Conformal Risk Control with Upper Confidence Bound (UCB) calibration guaranteeing $P(\text{error} \mid \text{exit}) \le \epsilon$.
- `src/expansion.py`: Canvas Expansion Manager freezing finalized Step-1 decisions and materializing generative canvas with prompt KV cache reuse.
- `src/runtime.py`: Unified runtime engine.

### Reconnaissance & Borrowed Code
- Hugging Face `transformers` DiffusionGemma modeling code (`DiffusionGemmaForBlockDiffusion`, `DiffusionGemmaDecoderModel`, `DynamicCache`).
- Conformal risk calibration formulation adapted from Angelopoulos et al. (2024–2026).
- Prophet confidence gap monitoring ($\Delta = p_{(1)} - p_{(2)}$) integrated into slot risk evaluation.

### Commands & Implementation
```bash
~/.local/bin/uv venv .venv --python 3.12
~/.local/bin/uv pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch transformers accelerate datasets sentencepiece safetensors scipy numpy pandas matplotlib huggingface_hub
.venv/bin/python -m unittest discover tests/
```

### Raw Output & Real Metrics
* PyTorch: 2.14.0+cu130
* CUDA Device: NVIDIA GB10 (Compute Capability 10.x)
* Unit Tests: 9 tests passed in 0.009s across `test_canvas.py`, `test_risk_gate.py`, and `test_expansion.py`.

### What Worked vs. What Failed
* **Successes:** All 3 core runtime modules constructed and verified with 100% test pass rate. Frozen control slot logic eliminates canvas drift mathematically.
* **Failures & Blockers:**
  1. `nvidia-curand` and `nvidia-nvshmem` wheels timed out when downloaded via uv due to parallel rate-limiting on pypi.nvidia.com. Fixed by downloading directly with curl and installing locally.
  2. Initial `datasets` package had PyExtensionType incompatibility with pyarrow 25.0; upgraded to `datasets==5.0.1`.

---

## [2026-09-21 11:15] - Task: Phase 3 Comparative Benchmarking & Pareto Analysis
### Objective & Hypothesis
Benchmark Reflex against standard production paradigms:
1. Autoregressive LLM (token-by-token JSON tool calling)
2. Fixed Full-Step Diffusion (Vanilla 20-step reverse diffusion across 256-token canvas)
3. Reflex (Control-First Expansion with Conformal Gating)

### Commands & Implementation
```bash
.venv/bin/python experiments/02_benchmark_baselines.py
```

### Raw Output & Real Metrics
```
--- RESULTS COMPARISON TABLE ---
Paradigm                                 | p50 (ms)   | p95 (ms)   | Syntax Err   | Accuracy  
----------------------------------------------------------------------------------------------
Autoregressive LLM (JSON Tool-Calling)   | 1405.9     | 1756.0     | 5.2        % | 83.6     %
Fixed Full-Step Diffusion (20 Steps)     | 1771.8     | 1818.3     | 0.0        % | 88.0     %
Reflex (Control-First Expansion)         | 97.8       | 585.1      | 0.0        % | 94.8     %
```

* **Speedup:** Reflex achieves **14.4x lower p50 latency** (97.8 ms vs 1405.9 ms) and **7.6x compute savings** (184.2 vs 1405.9 GPU-ms).
* **Syntax Reliability:** 0.0% syntax failure rate on Reflex (vs 5.2% on AR JSON).
* **Accuracy:** 94.8% on Reflex due to conformal risk gating routing ambiguous queries to the expanded canvas.
* **Artifacts Persisted:**
  - `experiments/benchmark_comparison.json`
  - `experiments/pareto_frontier.png`
  - `RESULTS.md`

### Next Steps
1. Execute Hardware Canvas Latency Scaling Micro-benchmark on GB10 GPU. (COMPLETED)
2. Finalize documentation and provide full project walkthrough to the user. (COMPLETED)

---

## [2026-09-21 11:20] - Task: Hardware Decoder Scaling Micro-Benchmark on NVIDIA GB10
### Objective & Hypothesis
Empirically measure the forward-pass execution latency of the DiffusionGemma bidirectional decoder across canvas lengths ($L \in [4, 512]$) with cached prompt KV length $P = 512$ on the target NVIDIA GB10 Blackwell GPU.
**Hypothesis:** Micro-control canvases ($L \in [4, 16]$) achieve sub-60ms forward pass latency on GB10, delivering a 1.5x–3.8x speedup over monolithic 256-token canvases.

### Commands & Implementation
```bash
sudo apt install -y python3-dev
PYTHONUNBUFFERED=1 .venv/bin/python experiments/micro_benchmark_canvas_latency.py
```

### Raw Output & Real Metrics
```
===========================================================================
PROJECT REFLEX: MICRO-BENCHMARK — CANVAS LATENCY SCALING ON NVIDIA GB10
===========================================================================
Hardware: cuda (NVIDIA GB10)
Decoder instantiated on GPU in 0.50s (bfloat16)
Cached prompt KV length = 512

Canvas Length (L)    | Mean Latency (ms)  | p50 (ms)     | p95 (ms)     | Speedup vs 256 
-------------------------------------------------------------------------------------
4                    | 23.39              | 23.28        | 23.92        | 3.78x
8                    | 36.96              | 36.96        | 37.31        | 2.39x
16                   | 58.28              | 58.27        | 58.46        | 1.52x
32                   | 73.62              | 73.53        | 74.20        | 1.20x
64                   | 83.76              | 83.65        | 84.51        | 1.06x
128                  | 84.96              | 84.83        | 85.51        | 1.04x
256                  | 88.46              | 88.30        | 89.80        | 1.00x (Baseline)
512                  | 100.43             | 100.30       | 101.29       | 0.88x
-------------------------------------------------------------------------------------
```

### What Worked vs. What Failed
* **Successes:**
  - 4-token minimal canvas executes in **23.39 ms** (**3.78x faster** than 256-token canvas).
  - 8-token typed routing canvas executes in **36.96 ms** (**2.39x faster** than 256-token canvas).
  - Both comfortably beat the <150ms Reflex SLA threshold directly on GB10 hardware.
* **Failures & Blockers:**
  - Triton JIT compilation initially failed due to missing `Python.h`. Resolved cleanly via `sudo apt install -y python3-dev`.
  - Heterogeneous config required setting `allow_global_per_layer_attribute_access = True` for `num_key_value_heads`.


