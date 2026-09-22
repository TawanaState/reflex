# TASK BRIEF: Tiered Adaptive Compute for Parametric Tool Calls & Schema-Aware Step Scheduling

You are the **Lead ML Systems Engineer** on **Project Reflex**. The server and base hybrid runtime are operational. 

Your objective is to implement **Schema-Conditioned Low-Step Denoising**: an adaptive step scheduler that allocates compute dynamically based on parameter schema complexity. When a tool requires primitive arguments (integers, floats, enums, bounded choices), the engine must allocate a compact micro-canvas and execute only **1 to 4 denoising steps**, rather than burning 15–20 steps on predictable, low-entropy data. Full 15–20 step denoising must be strictly reserved for open-ended string generation and free-form code/text.

You must build this with **clean modularization** so the engine is decoupled, readable, and trivial to unit test and debug.

---

### CORE ARCHITECTURAL PRINCIPLE: COMPUTE MATCHES ENTROPY

1. **Tier 1: Atomic Action (No Args)**
   * *Schema:* Tool with 0 required parameters.
   * *Canvas:* Micro-control canvas ($L \le 8$).
   * *Budget:* **1 forward pass** (~110ms) via Step-1 logit extraction.
2. **Tier 2: Parametric Primitive (Ints, Floats, Enums, Bounded Identifiers)**
   * *Schema:* Tool parameters are typed as `int`, `float`, `bool`, or `enum` (e.g., `set_volume(level: int)`, `transfer(amount: float, currency: enum)`).
   * *Canvas:* Sized strictly to expected parameter footprint ($L \in [8, 24]$ tokens).
   * *Budget:* **2 to 4 denoising steps** (~130–160ms total). Low-entropy primitives stabilize within 2–3 iterations over the warm prompt KV cache.
3. **Tier 3: Open Generative Synthesis (Unbounded Strings / Code)**
   * *Schema:* Tool parameters contain free-form text/strings (e.g., `send_email(body: str)`, `generate_sql(query: str)`).
   * *Canvas:* Generative buffer ($L \in [64, 256]$ tokens).
   * *Budget:* **12 to 20 denoising steps** (~350–500ms).

---

### REFACTORING & MODULARIZATION PLAN

Refactor the execution pipeline into clean, single-responsibility modules under `src/`:

```text
src/
├── config.py             # Environment & model runtime configs
├── schema/
│   ├── inspector.py      # Inspects tool JSON schemas and classifies into Tier 1, 2, or 3
│   └── compiler.py       # Compiles schemas into canvas slot indices & candidate token sets
├── engine/
│   ├── scheduler.py      # Dynamic step allocator: maps Tier & entropy to optimal denoise steps
│   ├── canvas.py         # Canvas memory allocation, mask indexing, and seeded templates
│   └── runner.py         # Causal encoder pass, prompt KV retention, and bidirectional diffusion steps
└── server.py             # FastAPI OpenAI-compatible routing (/v1/chat/completions)

```

#### Detailed Deliverables

##### 1. `src/schema/inspector.py` (Schema Complexity Classifier)

Implement `inspect_tool_schema(tool_def: dict) -> ToolComplexity`:

* Traverses the tool's JSON schema `parameters.properties`.
* If no properties $\rightarrow$ `Tier.ATOMIC` (Budget: 1 step).
* If all required properties are `integer`, `number`, `boolean`, or `enum` $\rightarrow$ `Tier.PARAMETRIC_PRIMITIVE` (Budget: 2–4 steps, calculate max token length $L$).
* If any property is an unconstrained `string` $\rightarrow$ `Tier.GENERATIVE_SYNTHESIS` (Budget: default full steps).

##### 2. `src/engine/scheduler.py` (Adaptive Step Allocator)

Implement `DynamicStepScheduler`:

* Calculates execution plan: `(canvas_length, denoise_steps, candidate_mask)`.
* For Tier 2 calls, sets `max_steps = 3` (or adaptive early stopping if argmax tokens stabilize across 2 consecutive steps).
* Exposes clean debug logs: `[SCHEDULER] Tool: transfer | Tier: PRIMITIVE | Canvas: 16 | Steps: 3`.

##### 3. `src/engine/runner.py` (KV-Warmed Micro-Expansion)

Ensure the micro-argument canvas directly reuses the initial prompt KV cache:

* Phase 1 resolves tool choice on Step 1.
* If Tier 2, inject argument template (e.g., `amt: [ @ @ @ ] \n curr: [ @ ]`), allocate only the required slice, and run 2–3 reverse diffusion iterations.
* Decode and validate extracted primitives against expected types (`int(val)`, `float(val)`).

---

### VERIFICATION & BENCHMARKING

1. **Zero Mocking:** All execution must run on actual GPU weights through the diffusion graph. Do not simulate latency or fake step outputs.
2. **Unit & Integration Tests (`tests/test_tiered_compute.py`):**
* Test Tier 1 call (atomic routing): Verify completion in 1 step ($<130\text{ ms}$).
* Test Tier 2 call (integer/float parameter tool): Verify completion in $\le 4$ steps ($<180\text{ ms}$) and verify argument parsing.
* Test Tier 3 call (open string parameter tool): Verify full generative expansion and output validity.


3. **Benchmark Script (`experiments/benchmark_tiered_latency.py`):**
* Measure and compare:
* Tier 1 Latency (p50/p95)
* Tier 2 Latency (p50/p95) vs. Fixed 20-step baseline
* Argument accuracy (% valid parsed primitives)


* Save outputs to `experiments/tiered_latency_results.json`.



---

### OPERATIONAL DIRECTIVE

1. Maintain `NOTES.md` continuously with implementation decisions, commands executed, and measured latency traces.
2. Do not break existing server endpoints (`/v1/chat/completions`) or overwrite validated model weights (`models/reflex_lora_v1/`).
3. Begin by creating `src/schema/inspector.py`, verify with a standalone test, and integrate into `src/engine/runner.py`.