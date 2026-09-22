# Project Reflex: results and evidence status

**Updated 2026-09-22.** This document separates measured observations from claims that the current artifacts cannot support. The v1 adapter and historical JSON files remain available for audit. The corrected v2 adapter has completed training and local API probes; the measurements and their narrow scope are below.

## What has been measured

| Observation | Source | Interpretation |
| --- | --- | --- |
| DiffusionGemma and the v1 LoRA loaded on the local NVIDIA GB10 server | Live `/health` check and three local inference calls during the audit | Real model execution is operational. HTTP success does not establish correct tool choice. |
| v1 Step-1 held-out correctness: **43/100** | `results/final_pareto_benchmark_results.json`, `reflex_proposed.results` | **43/75 BoolQ; 0/25 Banking77.** The 62.5% previously shown in this document was a training-window metric. |
| AR reference correctness: **64/100** | Same artifact, `ar_baseline.results` | **63/75 BoolQ; 1/25 Banking77.** This is a different model, quantization, and inference stack; it is an engineering reference, not a matched baseline. |
| v1 risk-policy calibration exits: **0/150**; test exits: **0/100** at all reported tolerances | Same artifact, `conformal_calibration` and `conformal_evaluation` | Selective error on exited cases is **undefined**, because there were no exits. The old file encodes it as 0%; that value must not be reported as successful safety evidence. |
| Tier-2 old artifact: **20%** exact numeric argument accuracy; **367 ms** mean and **606 ms** p95 wall latency | `experiments/tiered_latency_results.json` | Historical output from a benchmark source that was syntactically broken at audit time. Reproducibility and route correctness were not established. It does not support a sub-180 ms accuracy claim. |
| Prompt prefill **236.72 ms**, four-token decoder pass **76.76 ms**, 64-token decoder pass with reused prompt KV **123.86 ms** | `experiments/kv_retention_timing_results.json` and its script | Real GPU microbenchmark observations, pending independent rerun. They time components, not a complete generated response. |
| Fixed 20-pass, 256-token diffusion latency **5293.53 ms** mean over five runs on one prompt | `results/final_pareto_benchmark_results.json` and `experiments/benchmark_pareto_suite.py` | A costly fixed operating point. Its reported 88% accuracy was inserted in code, not evaluated here. Its published p50/p95 were derived from the mean, not measured percentiles. |

The live server selected `mute_audio {}` for both “Set volume to 57” and “Write an email to Alice apologizing for the delay.” Those are concrete wrong-tool failures with the v1 adapter, regardless of syntactic validity. The current 18 discoverable unit tests cover small code paths; they do not test held-out model quality.

## Why the previous Pareto chart is not empirical evidence

The old `experiments/benchmark_pareto_suite.py` inserts fixed diffusion accuracy, a two-model cascade point, and an expanded-path accuracy. Its expanded latency adds one 64-token decoder pass to prefill and Step 1; it does not execute a complete 12–20-step generation. `experiments/generate_real_pareto_analysis.py` also samples latency values. Figures derived from these sources are **illustrative historical artifacts**. They should not appear in a paper as a measured accuracy/latency frontier.

The 76–83 ms cached micro-canvas timing is a **decoder-only** measurement after prompt prefill. It cannot be compared directly with end-to-end AR latency. A quality-matched end-to-end speedup has not yet been established. The runtime's repeated whole-canvas argmax is also different from the official DiffusionGemma sampling algorithm, so generative quality requires separate validation.

## Supervision correction and version boundary

The original BoolQ data used candidate IDs 2374 and 9484, which decode to unrelated tokens in the installed tokenizer. Correct `no` and `yes` IDs are 1904 and 4443. The original BFCL records assigned every target to option zero; the pinned official answer file places 127 of 200 targets at other positions. `scripts/prepare_datasets.py --repair-existing-supervision` corrected the committed records without changing split membership. `data/supervision_provenance.json` records the source answer URL and hash. **The v1 adapter and all v1 result artifacts predate this correction and are not corrected by editing the data files.**

## Corrected v2: measured local API results

The v2 LoRA was trained on repaired BoolQ and BFCL labels with chat-wrapped prompts. Its final 90-example mixed-task **calibration/validation** check was 61.11% Step-1 accuracy (not held-out test accuracy). The adapter and provenance manifest are in `models/reflex_lora_v2/`.

| API probe | Exact calls | Mean wall latency | Evidence and limit |
| --- | ---: | ---: | --- |
| Local BFCL single-tool routing, original order | **40/40** | See per-item trace | `experiments/bfcl_routing_v2_20260922T084010Z.jsonl`; all used the one-step atomic path. This measures tool name only, not official BFCL argument scoring. |
| Same 40 queries, deterministically permuted menu | **40/40** | See per-item trace | Same trace; each tool's index changed according to a fixed per-item shuffle. |
| Atomic `mute_audio`, five trials | **5/5** | **281.1 ms** | `experiments/tiered_latency_summary_20260922T082929Z.json`; one prompt repeated five times. |
| Explicit numeric `set_volume(level=57)`, five trials | **5/5** | **285.2 ms** | Same summary. A deterministic single-number extractor supplies the value after learned tool selection; this is **not learned argument synthesis**. |
| Free-text email call, five trials | **0/5** | **2625.7 ms** | Same summary. Required fields were not generated and the server returned no executable tool call. |

The BFCL train and test IDs are disjoint, but two exact query strings recur across them with different candidate menus. Removing those two gives 38/38 correct in the original and permuted exploratory traces; this does not substitute for a newly frozen independent test set. The 40 examples are a local BFCL-derived single-call routing subset, not the official BFCL benchmark. The repeated three-case tier probe is a functionality smoke test, not a population estimate. Its first v2 run before the explicit-number repair had 0/5 exact numeric calls because the model emitted `level=1` for 57; that trace is preserved as `experiments/tiered_latency_traces_20260922T082436Z.jsonl`.

The deployed exit threshold remains an **uncalibrated 0.90 heuristic**. High confidence in the local routing probe is not evidence of a distribution-free error guarantee, and no quality-matched end-to-end speedup against a matched baseline has been measured. The server's large-canvas output does not yet yield valid free-text tool arguments.

## Reproducible next evaluation

1. Preserve v1 weights and JSON as historical evidence. Record the v2 checkpoint, base model revision, package versions, and dataset hashes.
2. Run `python -m unittest discover tests/` for code-level checks. Run GPU integration scripts separately; the unit command does not execute them.
3. Repeat the routing probe on a newly frozen independent set without train/test query overlap, and add base-model and matched-baseline runs. The local 40-item probe above is a routing subset, **not** the official full BFCL score.
4. Extend the tiered probe to varied prompts and schema types. The current numeric path handles one explicit number only; train and validate a genuine argument model before claiming learned Tier-2 synthesis. The free-text path needs a valid generation method and quality evaluation.
5. Calibrate an exit policy on a disjoint set and report calibration size, exits/errors, an upper bound, independent test coverage/error, and the assumptions of the method. Report selective error as N/A at zero exits. The server currently uses an uncalibrated 0.90 heuristic and must not be described as offering a conformal guarantee.
6. For a paper, add matched baselines, official DiffusionGemma sampling, full-path generation quality, repeated latency trials, uncertainty intervals, and ablations. Build figures only from per-item measured traces.

## Current conclusion

The prototype now demonstrates useful **local tool-name routing** on a small BFCL-derived set, real model loading, a short decoder read, KV reuse, and a working API shell. It does **not** yet establish broad tool-call correctness, learned argument synthesis, a useful calibrated fast path, or a quality-matched speedup. Public release is reasonable as an explicitly experimental repository with these limits. A research paper needs an independent larger test set, credible matched baselines, measured latency and uncertainty, and a contribution beyond prior short-canvas structured reads.
