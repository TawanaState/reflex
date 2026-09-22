# Reflex: A Unified Decision-and-Generation Runtime for Diffusion Language Models

> **Historical proposal, not an implementation report (updated 2026-09-22).** This document records design hypotheses. The repository currently uses a Transformers/PyTorch server, not the proposed vLLM patches; it does not implement newly trained dedicated control tokens, mixed frozen fields, GPU-millisecond compute regularization, or OSWorld/Mind2Web evaluation. Its old accuracy and safety targets have not been established. Critically, the "Micro-Control Canvas" / dedicated `<unusedN>` control-vocabulary mechanism described in Sections 2.2-2.3 below (evaluating a tool index in a single masked-token read) is **no longer the serving path**: it was superseded on 2026-09-22 after root-cause investigation found the base checkpoint already has its own trained, native tool-calling format that the custom canvas was bypassing. Using that native format through the official generation algorithm reached 93%+ tool-routing accuracy and ~77% fully-correct tool calls (including free-text and nested-object arguments) on a 1,051-example independent held-out set, with zero training -- outperforming the custom-canvas approach even after LoRA fine-tuning. See RESULTS.md for the measured evidence and README.md / `src/engine/runner.py` for the current architecture. The Conformal Risk Gating mechanism in Section 4.2 is also not currently wired into serving (see RESULTS.md "Calibration status"); it remains future work, not an implemented guarantee.


**Document Type:** Systems & Research Proposal

**Target Tracks:** MLSys / ICLR / NeurIPS (Systems & Architectures)

**Status:** Architecture Proposal & Experimental Blueprint

---

## Executive Summary

Autonomous software agents (browser operators, coding assistants, workflow orchestrators) are bottlenecked by the inference economics of autoregressive (AR) language models. Standard AR models spend uniform, multi-second compute generating structured JSON tool calls token-by-token, introducing non-zero parsing failures, high latency (3–8 seconds), and prohibitive execution costs.

In September 2026, TypeSafe introduced **Jev**, demonstrating that an AI model can abandon autoregression to evaluate typed schemas (`Choice`, `Score`, `Noul`) in a single forward pass (<100ms) with zero syntax errors. However, Jev is a proprietary, text-only classifier that cannot generate unstructured text, synthesize code, or populate dynamic tool arguments.

**Reflex** is a unified runtime built on open discrete diffusion language models (specifically **DiffusionGemma 26B/A4B**) that reconciles this divide. Rather than using two separate models or running a monolithic dual-zone canvas that wastes GPU cycles on unused tokens, Reflex introduces **Control-First Canvas Expansion**:

1. **Micro-Control Canvas:** The model evaluates a compact 4–16 token typed control canvas in a single denoise step over a cached prompt KV representation.
2. **Conformal Risk Gating:** Instead of brittle entropy thresholds, the runtime applies conformal risk control ($P(\text{error} \mid \text{exit}) \le \epsilon$) to guarantee a user-defined safety bound before terminating.
3. **Conditional Materialization:** Only when the control state explicitly indicates synthesis or the risk gate demands escalation does the runtime allocate and denoise an expanded generative canvas—reusing the initial prompt KV state without redundant re-encoding.

Reflex transforms adaptive compute from simple "early stopping" into an **Adaptive Output Topology**: dynamically collapsing the output space to a typed categorical manifold $\mathcal{Y}_{\text{typed}}$ for reflex actions, or expanding to open sequence space $\mathcal{Y}_{\text{open}}$ for deliberate generation, operating from the same model weights.

---

## 1. Problem Statement & Theoretical Motivation

### 1.1 The Agent Latency Dilemma

In closed-loop agentic workflows (e.g., OS navigation, web automation, tool dispatch), over 80% of model invocations are discrete routing decisions:

* *Should I click button $A$, scroll down, or wait?*
* *Which deterministic API tool out of 30 should handle this event?*
* *Does this log line represent a critical security alert or noise?*

Current systems handle these decisions via two imperfect paradigms:

```
PARADIGM 1: Monolithic Autoregressive LLM
[Prompt / State] ──> [Causal Attention] ──> Emits JSON String Token-by-Token ──> Parse Output
Drawbacks: 3-8s latency, non-zero syntax errors, overconfident hallucination, high cost ($/tok).

PARADIGM 2: Heterogeneous Cascade (Classifier Router + Separate LLM)
[Prompt / State] ──> [Small Classifier] ──(Lossy Text Label)──> [Large LLM for Generation]
Drawbacks: Context tokenized twice; latency multiplied when escalating; complex multi-model serving.

```

### 1.2 The Jev Model Paradigm: Value & Limitations

TypeSafe’s Jev introduced an architecture that maps unstructured state directly to typed decision slots.

| Characteristic | TypeSafe Jev (System 1) | Standard AR LLMs (System 2) | Reflex Target |
| --- | --- | --- | --- |
| **Output Type** | Typed enums / floats | Generated string tokens | **Adaptive:** Typed enum OR Token canvas |
| **Latency** | 70–500ms | 3,000–15,000ms | **70–150ms (Reflex) / 300–800ms (Gen)** |
| **Syntax Errors** | 0.0% (by construction) | 0.5% – 15.0% | **0.0% on control slots** |
| **Generative Ability** | None (cannot write text) | Full generation | **Full generation via canvas expansion** |
| **Weight Space** | Proprietary Black Box | Open / Proprietary | **Open-weight discrete diffusion** |
| **Modality** | Text only | Multimodal | **Native Multimodal (Vision + Text)** |

### 1.3 The Core Research Question

*Can a single multimodal diffusion language model act as a calibrated, typed decision engine on a minimal control canvas, while conditionally materializing an expanded generative canvas only when synthesis is strictly required?*

---

## 2. Technical Architecture: Control-First Expansion

The foundational insight of Reflex is that **discarded canvas tokens are not free**. In discrete diffusion models, the decoder operates bidirectionally across the entire canvas. Initializing a naive 512-token canvas to read 4 control slots forces the attention matrix to compute interactions over 508 useless masked tokens on the first step.

Reflex solves this by physically separating the execution into two sequential stages linked by a persistent KV-cache.

```
                                  INPUT (Text / Image)
                                           │
                                           ▼
                                 [ Causal Encoder ]
                                           │
                                  Caches Prompt KV
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  PHASE 1: CONTROL CANVAS│  (Canvas Length = 4 to 16 tokens)
                              │  <ROUTE>      [ @route ]│
                              │  <TARGET>     [ @elem  ]│  <── Single Denoising Pass (~100ms)
                              │  <SAFETY>     [ @gate  ]│
                              │  <SYNTHESIZE> [ @bool  ]│
                              └────────────┬────────────┘
                                           │
                        Extract Candidate Logits & Normalize
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │   Conformal Risk Gate       │
                            │   P(error | exit) <= eps    │
                            └──────────────┬──────────────┘
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  ▼                                                 ▼
          [ FAST REFLEX EXIT ]                              [ ESCALATE / EXPAND ]
      • Emit typed struct directly                       • Materialize Generative Canvas
      • Terminate execution                              • Length = 64 to 512 tokens
      • Latency: ~100ms | Cost: 1 step                   • Reuses cached prompt KV + control state
                                                         • Iterative diffusion unrolls payload/code

```

### 2.1 The Diffusion Backbone

Reflex is developed on **DiffusionGemma 26B/A4B** (a mixture-of-experts model with ~4B active parameters per token). Its architectural separation of a **causal prompt encoder** and a **bidirectional canvas decoder** provides the required foundation:

* **The Context Pass:** User prompt, system tools, and multimodal state (e.g., UI screenshots) are ingested by the causal encoder once and held in GPU memory as persistent key-value (KV) states.
* **The Canvas Pass:** The bidirectional decoder attends to the cached KV-state while performing denoising over the active canvas.

### 2.2 Phase 1: The Micro-Control Canvas

Instead of a broad text generation canvas, Reflex allocates an initial canvas of only **4 to 16 token slots**.

Each slot is constrained to a predefined set of candidate token IDs:

* **Route Slot (`@route`):** Bound to dedicated control tokens (`<REFLEX_CLICK>`, `<REFLEX_SCROLL>`, `<REFLEX_TOOL>`, `<REFLEX_SYNTHESIS>`, `<REFLEX_ESCALATE>`).
* **Target Slot (`@elem`):** Bound to indexed UI element identifiers (`<ELEM_0>`, ..., `<ELEM_255>`).
* **Safety Slot (`@gate`):** Bound to risk categories (`<SAFE>`, `<REVIEW>`, `<DESTRUCTIVE>`).

The runtime executes **exactly one denoising step**. Because the canvas is only 16 tokens wide, execution is dominated by attention over the cached prompt, bypassing the quadratic memory overhead of wide diffusion canvases and producing candidate logits in **sub-150ms**.

### 2.3 Dedicated Control Vocabulary vs. Natural Language Verbalizers

Traditional prompt-classification systems map output probabilities onto natural language strings (e.g., comparing the logprob of `"yes"` vs `"no"` or `"click"` vs `"scroll"`), which introduces tokenization bias and vocabulary collision.

Reflex modifies the tokenizer and model embeddings to include dedicated control tokens:


$$\mathcal{V}_{\text{control}} = \{\langle\text{REFLEX\_ACT}_i\rangle, \langle\text{RISK}_j\rangle, \langle\text{SLOT}_k\rangle\}$$


Because these tokens are dedicated identifiers, their representations are optimized solely for discrete state transitions, avoiding semantic ambiguity.

### 2.4 Phase 2: Conditional Canvas Expansion (Materialization)

If the Phase 1 control slot resolves to `<REFLEX_SYNTHESIS>` (e.g., the action requires typing free text or writing complex parameters), the runtime triggers **Canvas Expansion**:

1. Phase 1 control decisions are finalized and appended to the context.
2. An expanded generative canvas of length $L_{\text{gen}} \in [64, 512]$ is allocated.
3. The model executes $T_{\text{gen}}$ denoising steps exclusively on the new generative slots, reusing the initial prompt KV cache.

**Elimination of Canvas Drift:** In naive dual-zone architectures, noisy, unmasked generative tokens bidirectionally corrupt the discrete control slots during Step 1. Under Reflex's Control-First Expansion, control slots are finalized *before* the generative canvas is materialized. Canvas drift is mathematically precluded.

---

## 3. Schema-Conditioned Heterogeneous Compute

A major inefficiency in traditional agent frameworks is the all-or-nothing approach to tool calls: an LLM must generate every field of a JSON object through identical autoregressive compute.

Reflex introduces **Schema-Conditioned Compute Allocation**, allowing structured tools to mix typed slots (evaluated in 1 step) and generative buffers (iterated over multiple steps) within the same payload canvas:

```text
Tool: create_calendar_event(title: str, priority: enum, notify: bool)

[ CANVAS LAYOUT ]
priority:  [@HIGH, @MED, @LOW]       <── Resolved on Step 1 (Reflex)
notify:    [@TRUE, @FALSE]           <── Resolved on Step 1 (Reflex)
title:     [@ @ @ @ @ @ @ @ @ @ @]   <── Denoised across Steps 2..10 (Synthesis)

```

The runtime freezes `priority` and `notify` at Step 1, allocating iterative denoising compute *solely* to the tokens representing `title`. Compute is proportional to the entropy of each individual field in the schema.

---

## 4. Mathematical Formulation & Risk-Controlled Stopping

### 4.1 The Failure of Naive Entropy Thresholds

Prior adaptive diffusion literature relies on raw Shannon entropy:


$$H(p) = -\sum_{k} p_k \log p_k < \tau$$


This heuristic is brittle: deep neural networks are frequently overconfident on out-of-distribution (OOD) data, exhibiting near-zero entropy while outputting incorrect decisions.

### 4.2 Conformal Risk Control for Early Exits

Reflex frames the early-exit termination problem through **Conformal Risk Control** (Angelopoulos et al., 2024).

Let $x \in \mathcal{X}$ be the input state, and let $f(x)$ be the Phase 1 softmax distribution over candidate actions $\mathcal{Y}$. We define a non-conformity score $s(x, y) = 1 - f(x)_y$.

The runtime defines an exit policy $\pi_\lambda(x) \in \{\text{EXIT}, \text{EXPAND}\}$ governed by a threshold $\lambda$:


$$\pi_\lambda(x) = \begin{cases} \text{EXIT} & \text{if } \max_y f(x)_y \ge 1 - \lambda \\ \text{EXPAND} & \text{otherwise} \end{cases}$$

Given a user-specified operational error tolerance $\epsilon$ (e.g., $\epsilon = 0.005$, corresponding to a maximum allowable fast-path error of 0.5%), we calibrate $\lambda$ on a held-out calibration set $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^N$ using the Upper Confidence Bound on empirical risk:


$$\widehat{R}(\lambda) = \frac{\sum_{i=1}^N \ell(\hat{y}_i, y_i) \cdot \mathbb{I}(\pi_\lambda(x_i) = \text{EXIT})}{\sum_{i=1}^N \mathbb{I}(\pi_\lambda(x_i) = \text{EXIT}) + \gamma}$$

The runtime solves:


$$\lambda^* = \sup \left\{ \lambda \in [0, 1] : \widehat{R}_{\text{UCB}}(\lambda) \le \epsilon \right\}$$

**System Guarantee:** The developer sets `reflex_error_tolerance = 0.001`, and the runtime mathematically bounds the fast-path error rate, delegating ambiguous edge cases to expanded generation.

### 4.3 Training Objective

Reflex trains the underlying diffusion model using a joint multi-task objective:


$$\mathcal{L}_{\text{Reflex}} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{control}} + \lambda_2 \mathcal{L}_{\text{calibration}} + \lambda_3 \mathbb{E}[C(\pi)]$$

1. **$\mathcal{L}_{\text{diffusion}}$:** Standard discrete denoising cross-entropy over masked sequence tokens:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} \left[ -\sum_{i \in \text{Masked}} \log p_\theta(x_{0, i} \mid x_t) \right]$$


2. **$\mathcal{L}_{\text{control}}$:** Cross-entropy over the candidate set of dedicated control tokens at Step 1:

$$\mathcal{L}_{\text{control}} = -\sum_{k \in \mathcal{K}} y_k \log \left( \frac{\exp(z_k / T)}{\sum_{j \in \mathcal{K}} \exp(z_j / T)} \right)$$


3. **$\mathcal{L}_{\text{calibration}}$:** Proper scoring calibration penalty (Multi-Class Brier Score on control slots):

$$\mathcal{L}_{\text{calibration}} = \frac{1}{\vert{}\mathcal{K}\vert{}} \sum_{k \in \mathcal{K}} (p_k - y_k)^2$$


4. **Compute Regularization ($\mathbb{E}[C(\pi)]$):** Directly optimizes measured GPU runtime (in milliseconds) rather than an abstract token count:

$$\mathbb{E}[C(\pi)] = P(\text{EXIT}) \cdot \text{Cost}_{\text{step1}} + P(\text{EXPAND}) \cdot \left(\text{Cost}_{\text{step1}} + \text{Cost}_{\text{expand}}\right)$$



---

## 5. System Implementation: vLLM Runtime Extensions

Reflex is designed as a set of modular patches to the `vllm-project/vllm` execution engine for DiffusionGemma:

```
vLLM Core Architecture
  ├── Model Runner (DiffusionGemma)
  └── Scheduler
        └── [Reflex Extension Layer]
              ├── Control Canvas Allocator (Allocates 4-16 slots)
              ├── Logprob Extraction Engine (Evaluates exact logprob_token_ids)
              ├── Conformal Risk Gate (Evaluates fast-path safety)
              └── Canvas Expansion Manager (Conditionally updates sequence metadata)

```

### Proposed vLLM Engine Modifiers

* `control_canvas_tokens`: Injects pre-seeded syntax tokens into the canvas without running full prefill steps.
* `fixed_canvas_positions`: Marks specific slot indices as frozen, shielding them from noise injection during reverse diffusion.
* `allowed_token_sets`: Constrains logit normalization to valid candidate subsets per slot position, eliminating out-of-schema logits.
* `conditional_canvas_expand`: Extends the internal sequence length metadata in-flight if the risk gate triggers `<REFLEX_EXPAND>`, avoiding a full scheduler re-queue.

---

## 6. Experimental Evaluation Plan

The primary thesis is that Reflex establishes a new **Pareto Frontier** across Accuracy, Latency, and Compute Cost.

```
Accuracy / Error Rate
  ▲
  │                     AR LLM (Slow, Expensive, High Accuracy)
  │                      ▲
  │                     ╱
  │   Reflex Frontier  * ─── * (Adaptive Compute Path)
  │                  *
  │                 *
  │   Classifier+LLM Cascade (Brittle Transfer)
  │
  └──────────────────────────────────────────────────────────► GPU Latency (ms)

```

### 6.1 Baseline Architectures for Comparison

1. **Autoregressive LLM (Baseline A):** Llama-3-8B / Gemma-2-9B outputting JSON tool calls.
2. **Two-Model Cascade (Baseline B):** DeBERTa-v3 classifier (routing/intent) cascading to an AR LLM on low confidence.
3. **Standard Diffusion LM (Baseline C):** Vanilla DiffusionGemma executing fixed 20-step denoising across a standard 256-token canvas for all inputs.
4. **Reflex (Proposed System):** DiffusionGemma 26B/A4B with Control-First Canvas Expansion and Conformal Gating.

### 6.2 Benchmarks & Workloads

* **Agent Navigation:** **OSWorld** & **Mind2Web** (measuring UI action selection, click accuracy, and step latency).
* **Tool Dispatch:** **Berkeley Function Calling Leaderboard (BFCL)** (measuring routing precision and argument synthesis correctness).
* **Calibrated Triage:** **Banking77** & **BoolQ** (measuring classification accuracy, ECE, and selective risk coverage).

### 6.3 Evaluation Metrics

* **Reflex Latency (p50 / p95 / p99):** Time to return a decision when terminating on Phase 1.
* **Effective Cost (GPU-ms per task):** Total GPU occupancy time across multi-step execution traces.
* **Fast-Path Coverage:** Percentage of benchmark tasks resolved in Phase 1 without triggering expansion.
* **Selective Risk:** Error rate on the subset of requests resolved via Phase 1 at varying risk thresholds ($\epsilon$).
* **Syntactic Reliability:** Frequency of malformed JSON or unparseable tool invocations (Target: 0.0% on Reflex).

---

## 7. Minimal Viable Prototype (MVP) Execution Plan

To de-risk the scientific hypothesis before full fine-tuning, the project executes an immediate zero-training validation phase:

```
[Phase 0: 7 Days]  Zero-Training Feasibility Study on Stock DiffusionGemma
        │
[Phase 1: 3 Weeks] vLLM Inference Engine Extensions (Control Canvas + Expansion)
        │
[Phase 2: 4 Weeks] Synthetic Dataset Generation & Multi-Task Calibration SFT
        │
[Phase 3: 3 Weeks] Benchmarking vs Baselines & Systems Paper Preparation

```

### Phase 0: The 7-Day Feasibility Study

* **Hypothesis:** Off-the-shelf DiffusionGemma exposed to a micro-control canvas on Step 1 already possesses enough discriminative signal to beat random choice and approach standard zero-shot classifier baselines.
* **Method:**
1. Load stock `diffusiongemma-26B-A4B-it` in PyTorch.
2. Setup a 4-token control canvas: `[Category: @, Urgent: @]`.
3. Run Step 1 forward pass on 1,000 samples from **Banking77** and **BoolQ**.
4. Measure accuracy and Shannon entropy from the Step 1 candidate logits.
5. *Success Metric:* Step 1 top-1 accuracy exceeding 65% with a distinct correlation between low entropy and correctness validates the core premise immediately.



---

## 8. Strategic Commercial Roadmap

While academic dissemination targets systems venues (MLSys/ICLR), Reflex provides immediate commercial value as **open infrastructure for autonomous agents**:

1. **High-Frequency OS & Browser Agents:** Dropping single-step action latency from 4,000ms to ~100ms enables interactive desktop automation at continuous 5–10Hz control frequencies, turning sluggish script execution into responsive real-time software.
2. **Deterministic Enterprise Tool Gateways:** Enterprise deployments cannot tolerate LLM JSON hallucination. Reflex provides mathematical zero-error guarantees on discrete tool selection while preserving generative parameter generation.
3. **Air-Gapped / Edge Appliance Serving:** Because Reflex operates within a single quantized model footprint, it can be deployed on a single workstation GPU or localized edge server—eliminating reliance on third-party API dependencies and external bandwidth costs.

Reflex unifies what was previously split: fast deterministic software logic and open-ended generative AI, operating as a single, cohesive runtime.