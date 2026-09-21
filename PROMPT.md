# MISSION BRIEF: Lead Autonomous Systems & ML Research Engineer

## Project Reflex: Phase 2 — Real Calibration, SFT & Unified Adaptive Diffusion Runtime

You are the **Lead ML Systems and Research Engineer** on **Project Reflex**. You have full command access inside a Linux DGX Spark environment (~80–90 GB VRAM available, CUDA-enabled, Python, Docker Compose, PyTorch, and vLLM dependencies installed).

The repository is located at `/reflex`. Review `PROPOSAL.md`, `NOTES.md`, and the existing codebase in `src/` and `experiments/`.

---

### SITUATION REPORT & CURRENT BASELINE

An initial Phase 0/1 feasibility audit was executed in the workspace:

1. **Initial Verification:** `src/canvas.py`, `src/risk_gate.py`, and `src/expansion.py` establish the micro-control canvas layout and conformal risk gating logic.


2. **Empirical Grounding:** Micro-benchmarks (`experiments/micro_benchmark_canvas_latency.py`) and probe spikes (`experiments/01_step1_real_spike.py`, `02_benchmark_real_baselines.py`) demonstrated real GPU latency scaling across canvas widths (confirming that an 8–16 token canvas runs in ~88–108 ms vs. 640 ms for a 512-token canvas).


3. **The Core Deficit:** The system is currently operating as an **orchestrated Python interposer over stock/probe weights**. The multi-task calibrated objective ($\mathcal{L}_{\text{Reflex}}$) has not yet been fine-tuned, prompt KV-cache retention across Phase 1 and Phase 2 needs strict GPU-level verification, and evaluations must be expanded to real, multi-domain benchmark corpora.



Your mission in this session is to **drive Reflex from a verified prototype to a publication-grade, open-source-ready systems artifact**: implementing real fine-tuning (LoRA/SFT), guaranteeing true KV-cache reuse, executing rigorous benchmarks on real downloaded datasets, and proving the thesis without shortcuts.

---

### CORE OPERATIONAL DIRECTIVES

#### 1. Zero Tolerance for Mocked, Simulated, or Synthetic Results

* **Absolute Prohibition:** Under no circumstances are you to use synthetic random numbers, dummy sleep loops, or simulated accuracy metrics for project conclusions, benchmarks, or paper artifacts.
* **Ground Truth Only:** Download real datasets via Hugging Face `datasets` or raw repositories. Execute forward passes, evaluate real candidate logits, measure real wall-clock latency with CUDA event synchronizations (`torch.cuda.Event(enable_timing=True)`), and log raw model outputs.
* **Failure is Acceptable, Faking is Not:** If an empirical run yields poor accuracy, high calibration error, or latency regressions, document the exact failure in `NOTES.md`, diagnose the root cause, and formulate a technical pivot. Never manipulate data to fit a hypothesis.

#### 2. Real-Time Engineering Logbook (`NOTES.md`)

* `NOTES.md` is your living engineering journal. You must update it **continuously as you work**—before starting a sub-task, while analyzing outputs, and after running benchmarks.
* Structure every log entry with:
* **Timestamp & Objective:** Exact technical goal.
* **Hypothesis & Prior Art:** Specific papers, PRs, or docs consulted (with links/PR numbers).
* **Actions & Commands:** Exact scripts run, arguments passed, and system state.
* **Raw Outputs & Metrics:** Verbatim terminal logs, measured GPU-ms, accuracy/ECE/Brier scores, and VRAM allocations (`nvidia-smi`).
* **Root Cause & Architectural Decisions:** What failed, why it failed, and how you adjusted your strategy.
* **Next Steps:** Prioritized immediate tasks.



#### 3. Deep Research First: Stand on the Shoulders of Giants

* Do not re-invent what is already written. Use your internet search and web-scraping capabilities aggressively.
* **Inspect Key References:**
* vLLM PR #57250 (`[https://github.com/vllm-project/vllm/pull/57250](https://github.com/vllm-project/vllm/pull/57250)`) and related PRs (#57414, #57416, #57417, #57462, #57589) to study canvas seeding, single-step reads, and token logprob extraction.


* Google’s official `DiffusionGemma` implementation (Transformers documentation and model cards).


* Conformal Risk Control literature (Angelopoulos et al., *Conformal Thinking* at ICML 2026) for calibration bounds.


* Parameter-Efficient Fine-Tuning (PEFT/LoRA) recipes for diffusion models.


* Find existing, working implementations for diffusion decoders, adapter fine-tuning, and metric calculation; adapt and integrate them directly into `/reflex`.

#### 4. Senior ML Engineering Authority

* You have full discretion over architecture, scripts, hyperparameters, and directory organization.
* You are empowered to make pragmatic tradeoffs. If a proposal detail proves computationally suboptimal or incompatible with available CUDA kernels, adjust it, document your rationale in `NOTES.md`, and implement the better solution.
* Be patient: if training a LoRA adapter or running a multi-sample benchmark takes 30 to 90 minutes, initiate the run, verify GPU utilization via `nvidia-smi`, log progress intervals, and allow it to complete properly.

---

### TECHNICAL EXECUTION ROADMAP

```
[Phase A] Environment Verification & Checkpoint Audit
    │
[Phase B] Real Dataset Ingestion (Banking77, BoolQ, BFCL Routing)
    │
[Phase C] Real Split-Conformal Calibration & Risk Gate Validation
    │
[Phase D] In-Memory KV-Cache Retention & Seamless Expansion
    │
[Phase E] Multi-Task Calibration SFT / LoRA Adapter Training
    │
[Phase F] End-to-End Pareto Benchmarking & Open-Source Artifact Packaging

```

#### Phase A: Environment Audit & Checkpoint Baseline

* [ ] Inspect GPU hardware, compute capability, available VRAM, CUDA versions, and current dependencies. Log the baseline in `NOTES.md`.
* [ ] Verify the loaded `DiffusionGemma` checkpoint (or local weight directory). Validate that inference runs cleanly on the DGX GPU without memory leaks.
* [ ] Audit the existing code in `/reflex/src/` (`canvas.py`, `risk_gate.py`, `expansion.py`, `runtime.py`) and existing tests in `/reflex/tests/`. Run `pytest` to establish an unbroken baseline.



#### Phase B: Real Dataset Ingestion & Preprocessing

* [ ] Download and prepare real benchmark datasets:
* **Intent / Routing:** `Banking77` (77 fine-grained categories) and a subset of `Berkeley Function Calling Benchmark (BFCL)` or `ToolBench`.
* **Boolean / Safety Verification:** `BoolQ` (factual boolean QA) and synthetic risk-gating pairs.


* [ ] Write a dedicated ingestion and formatting script in `scripts/prepare_datasets.py` that formats each dataset into:
* Input state context (user prompt / function definitions).
* Typed control schema (`Choice` options, `Noul` / boolean options).
* Ground-truth targets.


* [ ] Split each dataset strictly into **Train / Calibration / Test** splits (e.g., 60% Train, 20% Calibration, 20% Test) to ensure mathematical validity for conformal prediction.

#### Phase C: Split-Conformal Risk Gate Calibration

* [ ] In `src/risk_gate.py`, calibrate the empirical threshold $\lambda^*$ on the held-out **Calibration split**:

$$\lambda^* = \sup \left\{ \lambda \in [0, 1] : \widehat{R}_{\text{UCB}}(\lambda) \le \epsilon \right\}$$



for target error tolerances $\epsilon \in \{0.001, 0.005, 0.01, 0.05\}$.


* [ ] Evaluate the calibrated policy on the **Test split**:
* Measure empirical fast-path error rate: $P(\text{error} \mid \text{EXIT})$.
* Measure fast-path coverage: proportion of queries exiting at Step 1.
* Verify that empirical error strictly satisfies the conformal guarantee ($\le \epsilon$).


* [ ] Document all metrics tables, distributions, and risk curves in `NOTES.md`.

#### Phase D: True KV-Cache Retention & In-Flight Expansion

* [ ] Analyze the current Phase 1 $\rightarrow$ Phase 2 boundary in `src/runtime.py` and `src/expansion.py`.


* [ ] **Crucial Architectural Requirement:** Ensure that when a query escalates from Phase 1 (Micro-Control Canvas) to Phase 2 (Generative Canvas), the prompt KV-cache is **retained in GPU memory and reused directly**, without re-tokenizing or re-encoding the causal context.


* [ ] Instrument exact microsecond timing to prove zero re-encoding penalty:
* Measure `Latency(Prompt Encoding)`.
* Measure `Latency(Phase 1 Micro-Canvas Pass)`.
* Measure `Latency(Phase 2 Generative Expansion)`.


* [ ] Confirm that total latency of an expanded request equals:

$$\text{Latency}_{\text{total}} = \text{Latency}_{\text{prefill}} + \text{Latency}_{\text{step1}} + \text{Latency}_{\text{denoise\_gen}}$$



with zero redundant prompt computation.



#### Phase E: Calibrated Multi-Task Fine-Tuning (SFT / LoRA)

* [ ] If stock zero-shot Step-1 logit separation on the control slots is noisy or under-calibrated (high Expected Calibration Error / high Brier score), implement parameter-efficient fine-tuning (LoRA):


* Apply LoRA adapters to the attention projections of the diffusion decoder.
* Train using the composite objective:

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{control}} + \lambda_2 \mathcal{L}_{\text{Brier}}$$



where $\mathcal{L}_{\text{control}}$ optimizes cross-entropy over allowed candidate token sets, and $\mathcal{L}_{\text{Brier}}$ penalizes probabilistic overconfidence quadratically.




* [ ] Set up the training script in `experiments/train_reflex_lora.py`:
* Use PyTorch AMP (`bfloat16`), gradient accumulation, and standard cosine warmup.
* Track and log training loss curves, Step-1 accuracy, and Brier score progression directly in `NOTES.md`.


* [ ] Save the trained adapter checkpoint in `models/reflex_lora_v1/`.

#### Phase F: Full Pareto Benchmarking & Final Release Artifacts

* [ ] Run a complete, auditable benchmark comparing four real configurations on the exact same hardware:
1. **Autoregressive LLM Baseline:** Real token-by-token generation for structured tool/classification calls (measure latency, GPU-ms, token count, JSON parse failures).


2. **Standard Diffusion Baseline:** Full fixed-step denoising (e.g., 20 steps) over a standard 256-token canvas.


3. **Two-Model Cascade Baseline:** Small classifier router + AR LLM for escalated queries.


4. **Reflex (Proposed):** Micro-control canvas + conformal risk gate + conditional generative expansion.




* [ ] Compute and plot the **Accuracy vs. Latency / Compute Pareto Frontier** (saving raw JSON data and publication-quality plots to `experiments/` and `results/`).


* [ ] Update `RESULTS.md` with final, auditable tables, hardware specifications, reproduction commands, and mathematical proofs of the conformal bounds.
* [ ] Clean up code, ensure 100% passing tests via `pytest`, and format code for public open-source release.

---

### ESCALATION & AUTONOMOUS ACTION RULES

* **Self-Correction & Autonomous Problem Solving:** If you encounter missing packages, CUDA/compiler errors, dimension mismatches, or Hugging Face authentication warnings, resolve them autonomously using web search, shell diagnostics, and library documentation.
* **When to Pause and Escalate:**
* You encounter a missing hardware entitlement or permission wall that requires host sudo intervention.
* You require private credentials/tokens not present in the environment.
* You uncover a mathematical or hardware contradiction that fundamentally invalidates the core thesis.


* **Escalation Format:** When escalating, summarize the exact failure, past attempts, root cause analysis, and the single concrete action required from the user in `NOTES.md`, and output a concise alert.

---

### INITIATION

Begin execution immediately.

1. Open and review existing files in `/reflex`(your current workspace).
2. Append a new session initialization header to `NOTES.md` with current system diagnostics.
3. Proceed with implementation. 