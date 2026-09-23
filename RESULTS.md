# Project Reflex: results and evidence status

**Updated 2026-09-23 (completed matched scalar early-exit benchmark).** The current evidence is the paired same-checkpoint experiments below: the atomic-only matched benchmark, and the newer scalar-argument early-exit extension. Historical v1/v2 and earlier unmatched probes remain documented afterward; their old latency comparisons are superseded by the paired results.

## Matched DiffusionGemma vs Reflex benchmark (2026-09-23)

The base policy uses the official DiffusionGemma `model.generate()` sampler with no draft observer. Reflex uses that *same loaded BF16 checkpoint*, native chat template, 256-token budget, and parser, with a one-draft schema-aware atomic exit added. There is no LoRA. Policy order was seeded and randomized within each request; CUDA was synchronized around end-to-end wall timing, and two warmup calls were excluded. The run used the local NVIDIA GB10. The frozen 500-row workload, hashes, scoring rules, and reproduction commands are in [BENCHMARK.md](BENCHMARK.md) and [provenance](data/bfcl/paired_eval_500.provenance.json). Raw [trace](experiments/paired_bfcl_20260923T080412Z.jsonl), [manifest](experiments/paired_bfcl_20260923T080412Z.manifest.json), [summary](experiments/paired_bfcl_20260923T080412Z.summary.json), and measured [figure](experiments/paired_bfcl_20260923T080412Z.summary.svg) are retained.

Positive-call outcome below means the selected name and typed argument fields match the published BFCL possible-answer alternatives under the documented strict-field scorer. No-tool outcome means the API returned no executable call; text quality is ungraded. This is **not the official BFCL AST score**. Intervals are 95% bootstraps over normalized exact-query clusters (484 distinct query strings), for the paired *mean milliseconds saved*; positive means Reflex is faster.

| Workload | n | Official mean / p50 / p95 (ms) | Reflex mean / p50 / p95 (ms) | Paired mean saved, 95% interval (ms) | Outcome, both policies | Early exits, correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Selected mix** | 500 | 1966 / 1404 / 5710 | 1957 / 1404 / 5677 | **+8.9 [-12.5, +23.0]** | 396/500 | 28, 24 correct |
| Empty-argument calls | 25 | 979 / 1002 / 1085 | 702 / 708 / 826 | **+277.4 [+247.6, +294.5]** | 25/25 | 24, 24 correct |
| Primitive arguments | 303 | 1391 / 1349 / 2096 | 1387 / 1354 / 2081 | +3.7 [+0.1, +8.6] | 237/303 | 0 |
| Longer-text arguments | 52 | 1585 / 1408 / 2987 | 1592 / 1408 / 2990 | -6.3 [-13.9, -0.4] | 29/52 | 0 |
| No applicable tool | 120 | 3791 / 3390 / 7121 | 3818 / 3400 / 7193 | -27.3 [-107.1, +16.5] | 105/120 | 4, **0 correct** |

The returned tool name, arguments, and assistant text matched in **all 500 pairs**, so this run observed no accuracy change; that observation is not an unseen-error guarantee. On the 25 atomic positives, Reflex saved **28.3% of mean latency** (979.3 to 701.9 ms) and reduced estimated decoder forwards from 2.00 to 1.04. The earlier historical ~29% signal survived a same-model, interleaved comparison. The selected mixed workload saved only **0.45% of mean latency**, with an interval crossing zero and essentially unchanged p50. It does not establish an overall speedup for a general agent workload.

The 43 BFCL live-irrelevance requests with an empty-schema distractor expose the gate's central limit. Reflex exited early on **4/43 wrong tool choices**; the official sampler returned the same four wrong tools, so they did not cause paired accuracy regressions. The gate validates native syntax and the selected schema, **not relevance to the user's intent**. Four faster wrong actions cannot count as successful fast-path coverage. Across all 120 no-tool requests, 105 returned no executable call in both policies. One 13-forward no-tool pair had a 4.26-second negative latency difference; the no-tool subgroup's *paired median* was +1.2 ms saved, so its negative mean should not be read as a stable observer cost without replication.

### Stability ablation on the 25 atomic positives

| Policy | Correct | Early exits | Mean wall latency | Mean decoder forwards |
| --- | ---: | ---: | ---: | ---: |
| Reflex, 1 stable draft | 25/25 | 24 | 702 ms | 1.04 |
| Reflex, 2 stable drafts | 25/25 | 24 | 976 ms | 2.00 |
| Reflex, 3 stable drafts | 25/25 | 0 | 977 ms | 2.00 |
| Official sampler | 25/25 | 0 | 979 ms | 2.00 |

The two- and three-draft arms ran after each randomized base/one-draft pair, so this ablation is exploratory. Waiting two drafts erases almost all measured speedup; three drafts cannot exit before the sampler finishes these examples. This positive-only ablation does not estimate the false-exit risk on no-tool prompts.

### Current research verdict

Reflex has a reproducible, narrow fast path for empty-argument tool calls on this hardware. The mechanism did not change the returned tool name, arguments, or assistant text in 500 paired requests, but it also did **not** deliver a convincing speedup on the selected mixed workload. The 25 positive atomic examples are too few for a safety or broad generalization claim, and four incorrect exits on no-tool requests show that schema validity alone is not semantic correctness. This is a credible experimental systems result and a useful boundary finding, **not yet a submission-ready paper claim**. Next evidence priorities are a larger independent atomic-positive set with hard negatives, official BFCL scoring, measured generative quality, and GPU-time/energy profiling. No multimodal quality claim is made from this text-only run.

## Matched scalar early-exit extension (2026-09-23)

This experiment extends the atomic (empty-argument) draft gate to fully specified typed-scalar calls (every declared field, including optional ones, present and schema-valid, identical for N consecutive drafts) and asks whether it delivers a further, safe speedup on the *same* loaded checkpoint. The rule, frozen workload, and reproduction commands are in [SCALAR_EXPERIMENT.md](SCALAR_EXPERIMENT.md) and [provenance](data/bfcl/scalar_eval.provenance.json). Four policies were interleaved per request under one seed, one loaded BF16 checkpoint, no LoRA: `official` (no gate), `atomic_only` (current shipped gate), `scalar_2` (atomic gate plus a two-stable-draft scalar gate — the shipped `REFLEX_SCALAR_STABLE_STEPS` default), and `scalar_1` (a one-stable-draft ablation, run only on the scalar-positive and no-tool-distractor subsets). Raw [trace](experiments/scalar_exit_20260923T092825Z.jsonl), [manifest](experiments/scalar_exit_20260923T092825Z.manifest.json), [summary](experiments/scalar_exit_20260923T092825Z.summary.json), and [figure](experiments/scalar_exit_20260923T092825Z.summary.svg) are retained. All 214 rows completed the three-arm comparison; 158 of those (scalar positives + no-tool scalar-distractor negatives) also completed the `scalar_1` arm, matching the frozen 800-row trace exactly.

| Workload | n | Official mean/p50 (ms) | Reflex atomic-only mean/p50 (ms) | Reflex scalar-2 mean/p50 (ms) | scalar-2 saved vs atomic-only, 95% CI (ms) | Correct, all arms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Selected mix (overall)** | 214 | 2347 / 1627 | 2318 / 1584 | 2307 / 1552 | **+10.8 [-3.5, +31.0]** | 173/214 |
| Scalar-schema positives | 58 | 1360 / 1177 | 1369 / 1170 | 1342 / 1190 | **+27.6 [-1.4, +85.5]** | 46/58 |
| Atomic controls (replication) | 25 | 988 / 1009 | 706 / 710 | 707 / 724 | -1.3 [-6.1, +3.7] | 25/25 |
| Open-text controls | 31 | 1810 / 1506 | 1835 / 1510 | 1817 / 1526 | +17.2 [-17.2, +75.4] | 17/31 |
| No-tool + scalar distractor | 100 | 3424 / 2993 | 3421 / 2996 | 3419 / 2988 | +2.1 [-8.9, +18.2] | 85/100 |

At the shipped `scalar_2` (two-stable-draft) setting, the scalar gate fired 37 times across the 214-row mix (33 on scalar positives, 4 on no-tool distractors) but **did not produce a statistically reliable speedup anywhere it fired**: every relevant interval above includes zero or near-zero. The atomic-controls row replicates the earlier matched atomic result almost exactly (28.6% mean reduction here vs. 28.3% previously), confirming the harness and hardware are behaving consistently between runs. The open-text-control row confirms the scalar gate correctly never fires outside eligible schemas (0 exits, 0 latency effect beyond noise).

A one-stable-draft ablation (`scalar_1`) tells a different story on the scalar-positive subset: mean latency fell from 1369 ms (atomic-only) to 1172 ms, a **197.2 ms / 14.4% mean reduction, 95% CI [136.1, 276.7] ms** — a real, CI-clear effect, unlike `scalar_2`. This reproduces the same pattern already seen in the atomic-gate stability ablation (`RESULTS.md`, "Stability ablation on the 25 atomic positives"): waiting for a second stable draft erases almost all of the measured benefit, because most eligible scalar calls are already fully formed by the first draft that could exit atomically-equivalent calls.

**Correctness and regression check (the important part):** across every one of the 214 three-arm rows and all 158 four-arm rows, the scalar gate's returned tool name, arguments, and assistant content were **byte-identical to what the same request produced under `atomic_only` and under `official`** — `matching_content_and_call` is 214/214 and 158/158 in the summary contrasts, at both N=1 and N=2. Five scalar exits (1 of the 33 on positives, 4 of the 4 on no-tool distractors) did not match the strict-field score against the published BFCL gold answer. Inspecting those five directly (identical requests, decoded independently, not taken from the summary) confirms that in **every case** the official, un-gated sampler and the atomic-only policy independently produced the exact same wrong tool call or wrong argument value — the scalar gate reproduced a pre-existing base-model error faster, it did not introduce a new one. This mirrors the earlier atomic-gate finding (4/43 wrong atomic exits on no-tool requests, same base-model errors) almost exactly, including landing on a very similar absolute rate (4/100 here vs. 4/43 there).

**Verdict:** the scalar gate is schema-valid-and-stable, not relevance- or correctness-aware, exactly like the atomic gate before it. At its shipped default (`REFLEX_SCALAR_STABLE_STEPS=2`), it adds negligible or no measured latency benefit anywhere in this workload, so there is no case for defaulting it on. The one-stable-draft variant shows a genuine, non-regressive speedup on true scalar positives, but with only 58 positive and 100 negative examples this is not enough to bound its false-exit risk, and its absolute wrong-exit rate on no-tool distractors (4/100) is identical to the two-draft setting — an extra stable draft bought no additional safety here. **`REFLEX_SCALAR_EARLY_EXIT` remains `false` by default**, per the pre-registered decision rule in `SCALAR_EXPERIMENT.md` (only enable on measured, non-erased speedup plus a credible error analysis). This is a negative-leaning, honestly reported result, not a launch.

## Earlier evidence and research history

### Earlier native-path transition (before the matched benchmark)

Earlier in this repair sprint, Tier 2 argument synthesis was a deterministic regex extractor (not learned), and Tier 3 free-text arguments were 0/5. A senior review of that state recommended one decisive experiment before deciding whether to keep pursuing full tool-calling: a large (500+), genuinely independent, order-randomized routing test, plus replacing the hand-rolled generation loop with DiffusionGemma's *official* sampler.

Running that experiment surfaced the actual root cause: the installed checkpoint's own chat template (`chat_template.jinja`) already implements a trained, native function-calling protocol (`<|tool_call>call:name{key:<|"|>value<|"|>,...}<tool_call|>`, driven by a standard `tools=` argument to `apply_chat_template`), completely independent of Reflex's custom "select a tool index into `<unusedN>`" micro-canvas. The whole Tier 1/2/3 architecture had been asking the model to do a task (index classification into unused-vocabulary slots) it was never instruction-tuned for, instead of the task it was tuned for. That misalignment, not a capability ceiling, is the most likely explanation for the old Tier 2/3 failures.

Using the native format with the *official* `model.generate()` (no custom canvas, no manual argmax loop, no LoRA) on a frozen, **1,051-row held-out routing set** (§ "Native tool-calling evidence" below) reached **93.3% tool-name accuracy** and **~77% calls passing a loose gold-value containment check** (not an exact or official BFCL argument score), similar under one deterministic candidate-order permutation. `src/engine/runner.py` has been rewritten to use this mechanism as the live serving path; a smoke test of the rewritten engine against the real model confirmed both previously-broken cases now work: `set_volume(level=57)` (exact, was `level=1` under the old scheme) and a full coherent apology email for the exact prompt that scored 0/5 before.

On 2026-09-23, the serving path gained the missing Reflex-controlled transition. The official DiffusionGemma sampler exposes an argmax draft after each decoder forward. `NativeDraftObserver` exits when that draft contains one complete, parseable call to an offered argument-free schema; every other case continues through the same official generation call. The mechanism allocates fewer denoising steps to simple actions while preserving native argument and free-text generation.

**Status at that earlier point:** the atomic gate was syntactic and schema based, the sample contained only 25 atomic-bucket items, and the latency comparison below used an earlier full-generation run. The matched run above now replaces that latency comparison. The old loose argument score remains non-official, and some free-text tool calls still fail or become direct responses.

## Native early-exit evidence (2026-09-23)

| Observation | Source | Interpretation |
| --- | --- | --- |
| **25/25 correct tool names and calls on all held-out atomic-bucket examples; 24/25 exited after one decoder step** | `experiments/atomic_early_exit_20260923T070911Z.jsonl` / `.summary.json` | The remaining item had optional parameters, so schema inspection correctly sent it through the argument-capable path; it remained correct and completed in two measured decoder steps. Base model, no LoRA. |
| **697 ms mean, 707 ms p50, 798 ms p95; mean 1.04 decoder steps** | Same trace | The prior full-generation result on these 25 items was 978 ms mean / 1,006 ms p50 with the same 25/25 accuracy. This historical, non-interleaved comparison is about 29% lower mean and p50 latency (1.40x throughput-equivalent speedup), not a matched randomized A/B. |
| Deterministic cross-tier probe: primitive 50/52 names, 40/52 loose args, mean **2.96 steps / 1,490 ms**; free-text/nested 45/52 names, 33/52 loose args, mean **3.65 steps / 1,627 ms** | `experiments/tiered_adaptive_compute_20260923T071541Z.jsonl` / `.summary.json` | No atomic exit occurred on any of these 104 non-atomic requests. Together with the atomic result (1.04 steps / 697 ms), this shows measured compute ordering. The complete free-text bucket exactly reproduced the earlier full-generation 45/52 name and 33/52 loose-argument counts. |
| Corrected full-path step telemetry | `src/engine/runner.py::_estimate_forward_passes`; regression test in `tests/test_native_early_exit.py` | The official statistic divides non-pad generated tokens by decoder forwards. The old code incorrectly used the padded 256-slot block width and therefore inflated reported 26-39 counts. Current traces report 2 steps for the primitive smoke call and 9 for one failed generative email call. |

## Native tool-calling evidence (2026-09-22)

| Observation | Source | Interpretation |
| --- | --- | --- |
| Kill-criterion probe (n=15, zero-shot, no LoRA, base model): atomic 5/5, primitive 3/5, free-text 3/5 tool-name correct; one item's arguments matched a nested multi-field BFCL gold answer exactly | `/tmp` scratch probe (not committed; see NOTES.md 2026-09-22 entry for the transcript) | Misses were a 128-token budget cutting off the model's own unprompted reasoning preamble before it reached the tool call, or the model correctly declining an ambiguous/context-dependent request -- not argument-quality failures. Motivated raising the token budget before scaling up. |
| **1,051-row held-out routing set, original candidate order: 981/1051 = 93.3% tool-name accuracy; 813/1051 = 77.4% loose argument-match calls** | `experiments/native_tool_routing_20260922T115413Z.jsonl` / `.summary.json` | Zero-shot, base model, no LoRA, official `generate()`. Source is `BFCL_v4_live_multiple` (real user queries), a source file disjoint from the `BFCL_v4_multiple` category used for `data/bfcl/{train,cal,test}`; provenance and the one dropped exact-query overlap are in `data/bfcl/live_eval_provenance.json`. |
| **Same 1,051 examples, deterministic per-item candidate-order permutation: 979/1051 = 93.1% tool-name accuracy; 812/1051 = 77.3% loose argument-match calls** | Same trace file, `condition=base_permuted` | Close to the original-order result; this check does not establish complete positional robustness. Candidate-count cells are small at some menu sizes (for example, 7 and 10). |
| By argument-complexity bucket (original order): atomic 25/25 name and args; primitive (numeric/enum/bool-typed) 911/974 name, 755/974 full args; free-text/nested-object 45/52 name, 33/52 full args | Same trace file | The free-text/nested bucket includes genuinely complex cases, e.g. a drink-order call correctly producing a nested `new_preferences` object with five sub-fields matching the official BFCL gold answer. Argument scoring is a loose containment check against gold values (see `args_roughly_match` in `experiments/eval_native_tool_calling.py`), not the official BFCL AST scorer -- may both overcount coincidental text matches and undercount correct paraphrases. |
| Mean latency 1,470-1,473 ms per call across both conditions (no KV-cache reuse implemented in the eval script) | Same trace file | Real, substantially higher than the old scheme's fabricated ~280 ms (which was fast because it was answering the wrong question). This is a different mechanism from the old fixed-20-step Tier-3 loop; no matched AR timing claim is made. |
| Live smoke test of the *rewritten* `src/engine/runner.py` (not the standalone eval script) against the real GB10 model, 4 cases | `/tmp` scratch smoke test (see NOTES.md 2026-09-22 entry for the full JSON) | `mute_audio {}` (atomic, correct); `set_volume {"level": 57}` (exact -- this is the case that emitted `level: 1` under the old scheme); a distractor-menu flight search correctly called `search_flights` with a plausible date; and the free-text apology-email prompt, offered the same 3-tool menu as the other cases, produced a complete, coherent apology email as direct assistant content rather than invoking the offered `write_email` tool. That last case is a genuine, documented limitation (see below), not a hidden failure. |

## What has been measured (v1 / v2 history, superseded as the serving path)

| Observation | Source | Interpretation |
| --- | --- | --- |
| DiffusionGemma and the v1 LoRA loaded on the local NVIDIA GB10 server | Live `/health` check and three local inference calls during the audit | Real model execution is operational. HTTP success does not establish correct tool choice. |
| v1 Step-1 held-out correctness: **43/100** | `results/final_pareto_benchmark_results.json`, `reflex_proposed.results` | **43/75 BoolQ; 0/25 Banking77.** The 62.5% previously shown in this document was a training-window metric. |
| AR reference correctness: **64/100** | Same artifact, `ar_baseline.results` | **63/75 BoolQ; 1/25 Banking77.** This is a different model, quantization, and inference stack; it is an engineering reference, not a matched baseline. |
| v1 risk-policy calibration exits: **0/150**; test exits: **0/100** at all reported tolerances | Same artifact, `conformal_calibration` and `conformal_evaluation` | Selective error on exited cases is **undefined**, because there were no exits. The old file encodes it as 0%; that value must not be reported as successful safety evidence. |
| Tier-2 old artifact: **20%** exact numeric argument accuracy; **367 ms** mean and **606 ms** p95 wall latency | `experiments/tiered_latency_results.json` | Historical output from a benchmark source that was syntactically broken at audit time. Reproducibility and route correctness were not established. It does not support a sub-180 ms accuracy claim. |
| Prompt prefill **236.72 ms**, four-token decoder pass **76.76 ms**, 64-token decoder pass with reused prompt KV **123.86 ms** | `experiments/kv_retention_timing_results.json` and its script | Real GPU microbenchmark observations, pending independent rerun. They time components, not a complete generated response. Superseded as a design pattern: the native serving path calls `model.generate()` once per request and no longer manages a manual Phase-1/Phase-2 KV handoff. |
| Fixed 20-pass, 256-token diffusion latency **5293.53 ms** mean over five runs on one prompt | `results/final_pareto_benchmark_results.json` and `experiments/benchmark_pareto_suite.py` | A costly fixed operating point. Its reported 88% accuracy was inserted in code, not evaluated here. Its published p50/p95 were derived from the mean, not measured percentiles. |

The live v1 server selected `mute_audio {}` for both "Set volume to 57" and "Write an email to Alice apologizing for the delay." Those are concrete wrong-tool failures with the v1 adapter, regardless of syntactic validity. The current unit tests cover small code paths; they do not test held-out model quality by themselves -- the native tool-calling evidence above does.

## Why the previous Pareto chart is not empirical evidence

The old `experiments/benchmark_pareto_suite.py` inserts fixed diffusion accuracy, a two-model cascade point, and an expanded-path accuracy. Its expanded latency adds one 64-token decoder pass to prefill and Step 1; it does not execute a complete 12-20-step generation. `experiments/generate_real_pareto_analysis.py` also samples latency values. Figures derived from these sources are **illustrative historical artifacts**. They should not appear in a paper as a measured accuracy/latency frontier. Both scripts' `main()` now stop immediately with an explicit error; they are retained only as inspectable historical source.

## Supervision correction and version boundary (v1 -> v2)

The original BoolQ data used candidate IDs 2374 and 9484, which decode to unrelated tokens in the installed tokenizer. Correct `no` and `yes` IDs are 1904 and 4443. The original BFCL records assigned every target to option zero; the pinned official answer file places 127 of 200 targets at other positions. `scripts/prepare_datasets.py --repair-existing-supervision` corrected the committed records without changing split membership. `data/supervision_provenance.json` records the source answer URL and hash. **The v1 adapter and all v1 result artifacts predate this correction and are not corrected by editing the data files.**

## Corrected v2: measured local API results (custom index-selection scheme, superseded)

The v2 LoRA was trained on repaired BoolQ and BFCL labels with chat-wrapped prompts, targeting the custom index-selection micro-canvas described above -- **not** the native tool-calling format now used for serving. It is kept on disk (`models/reflex_lora_v2/`) as an archived, documented artifact and is **not auto-attached** by the current server (see `src/engine/runner.py` module docstring and `.env.example`): it has never been evaluated against native tool calling, and attaching an unvalidated adapter to a different serving mechanism could silently change output quality in untested ways.

| API probe | Exact calls | Mean wall latency | Evidence and limit |
| --- | ---: | ---: | --- |
| Local BFCL single-tool routing, original order | **40/40** | See per-item trace | `experiments/bfcl_routing_v2_20260922T084010Z.jsonl`; all used the one-step atomic path. This measures tool name only, not official BFCL argument scoring. Two of the 40 exactly overlapped BFCL train queries; excluding them gives **38/38**. |
| Same 40 queries, deterministically permuted menu | **40/40** (38/38 excluding overlap) | See per-item trace | Same trace; each tool's index changed according to a fixed per-item shuffle. |
| Atomic `mute_audio`, five trials | **5/5** | **281.1 ms** | `experiments/tiered_latency_summary_20260922T082929Z.json`; one prompt repeated five times. |
| Explicit numeric `set_volume(level=57)`, five trials | **5/5** | **285.2 ms** | Same summary. A deterministic single-number extractor supplies the value after learned tool selection; this is **not learned argument synthesis**. Compare to the native path's exact match via real generation, above. |
| Free-text email call, five trials | **0/5** | **2625.7 ms** | Same summary. Required fields were not generated and the server returned no executable tool call. Compare to the native path's coherent output on the same prompt, above. |

The BFCL train and test IDs are disjoint at the ID level; this 40/38-item probe is far smaller than the 1,051-item native-path evaluation above, and was superseded by it as the decisive routing evidence.

The deployed exit threshold for this scheme was an **uncalibrated 0.90 heuristic** and is no longer used. The current native gate instead requires a complete parseable call whose selected schema declares no parameters; it also remains uncalibrated.

## Calibration status: explicitly deferred, not solved

The v1 risk-policy calibration exited on 0/100 test items ("0% error on exits" was vacuous). The v2 scheme's deployed gate was an uncalibrated 0.90 heuristic. Neither `ConformalRiskGate` (`src/risk_gate.py`) nor `DynamicStepScheduler` (`src/engine/scheduler.py`) is wired into the native serving path; both target the retired micro-canvas. The new atomic gate is deliberately narrow: the model must emit one complete native call, its name must be in the offered menu, parsing must yield `{}`, and the chosen schema must declare zero properties. It preserved 25/25 correctness in the held-out atomic-positive subset, while the matched no-tool stress set found four incorrect early exits. Neither sample establishes a risk bound. A separate calibration split, larger atomic set, and threshold/coverage curve are required before calling it calibrated.

## Reproducible next evaluation

1. Find a larger independent, naturally occurring atomic-positive set and a matching no-tool set with atomic distractors; report false-exit risk separately from positive-call speed. The current 25/43 cells are too small for calibration.
2. Run the official BFCL AST evaluator or publish a validated equivalence check for full argument scoring. The strict-field metric here is transparent but not the leaderboard score.
3. Evaluate open-ended generative quality and latency, and measure GPU kernel time/energy. The present longer-text bucket contains tool arguments, not a full generative benchmark.
4. Investigate observer overhead on long no-tool generations and test any optimization in a new paired run before changing the serving default.

## Current conclusion

The old custom index-selection path failed to provide learned argument synthesis. Native DiffusionGemma tool calling made the model usable again; Reflex's draft gate then produced a real but narrow early-exit speedup. The completed paired benchmark supports **28.3% lower mean latency on 25 empty-argument calls at identical observed outputs**, while the selected 500-request mixture shows **no reliable mean speedup**. This is a measured limitation, not a reason to invent a broader performance claim. The raw trace, benchmark protocol, and explicit failure cases above are the basis for further research.
