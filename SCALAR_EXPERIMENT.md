# Experimental scalar draft exit

## Question

Can Reflex stop the *same* DiffusionGemma generation earlier when a complete typed-scalar tool call has stabilized, while preserving the action that the current atomic-only Reflex policy would return?

This is a test of a new Reflex-controlled stopping rule. The existing official entropy-adaptive sampler remains the generator. Atomic exits are already supported; open strings, arrays, objects, and general free-text synthesis stay on that sampler. The scalar rule is **disabled by default** until its latency and error effects are measured.

## Rule under test

A scalar draft can exit only when it has exactly one complete native tool call, names an offered function whose *entire schema* consists of integer/number/boolean/enum-string fields, contains every declared field (including optional fields), has no extra fields, passes the same primitive casting and bounds checks as the final API path, and the canonical tool name plus validated arguments remain identical for N consecutive drafts. The experiment tests N=1 and N=2; the default experimental setting is N=2. The gate has no relevance classifier or calibrated correctness probability.

## Frozen workload

`scripts/prepare_scalar_eval.py` selects real rows from pinned BFCL v4 sources. The exact source revision, SHA-256 hashes, overlap filter, and selection seed are in `data/bfcl/scalar_eval.provenance.json`; the frozen 214 rows are in `data/bfcl/scalar_eval.jsonl`.

| Diagnostic category | Requests |
| --- | ---: |
| Scalar-schema gold calls from held-out live multiple | 58 |
| Live irrelevance with an offered scalar distractor | 100 |
| Empty-argument controls | 25 |
| Open-text tool-call controls | 31 |

Of the 58 scalar positives, 30 schemas include optional fields; the gate can exit on those only if every optional field is present in the draft. Of the 100 no-tool cases, 28 also offer an atomic distractor. There are 196 distinct normalized query strings. This selected workload is neither a production traffic mix nor BFCL's official aggregate. The scalar-positive label means the **gold tool's schema** is eligible; it does not imply that a draft will contain all fields or exit early.

## Matched run

`experiments/benchmark_scalar_early_exit.py` loads one base BF16 checkpoint with no LoRA. It interleaves these policies per request with the same RNG seed, native prompt, tool menu, 256-token budget, parser, and CUDA-synchronized end-to-end timing:

- `official`: no draft exit.
- `atomic_only`: current Reflex atomic gate, scalar gate off.
- `scalar_2`: atomic gate plus two-stable-draft scalar gate.
- `scalar_1`: one-stable-draft scalar ablation on scalar positives and scalar-distractor negatives only.

The primary contrast is `scalar_2` versus `atomic_only`. The run also measures wrong scalar exits on no-tool requests, exact returned-call/text changes, strict field agreement with BFCL alternatives, p50/p95 latency, and estimated decoder forwards. The strict-field score is documented in `BENCHMARK.md` and is **not** BFCL's official AST score. `experiments/summarize_scalar_early_exit.py` reports only complete paired requests and bootstraps latency by normalized query text.

## Reproduce

```bash
PYTHONPATH=. python scripts/prepare_scalar_eval.py
PYTHONPATH=. python experiments/benchmark_scalar_early_exit.py
PYTHONPATH=. python experiments/summarize_scalar_early_exit.py experiments/scalar_exit_<run-id>.jsonl
```

The default remains `REFLEX_SCALAR_EARLY_EXIT=false` in `.env.example`. A positive result would require measured useful speedup **and** a credible error analysis; a matching answer count on this selected set alone would not certify unseen-request safety.

## Result (2026-09-23)

The matched run completed all 800 rows (`experiments/scalar_exit_20260923T092825Z.jsonl`, manifest and summary beside it). Full numbers, tables, and figure are in [RESULTS.md](RESULTS.md#matched-scalar-early-exit-extension-2026-09-23). Short version:

- **`scalar_2` (the shipped stable-step default) does not clear the pre-registered bar.** Its mean latency saving versus `atomic_only` was not statistically distinguishable from zero on the overall mix (+10.8 ms, 95% CI [-3.5, +31.0]) or on scalar positives specifically (+27.6 ms, CI [-1.4, +85.5]).
- **`scalar_1` (one stable draft) did clear it on scalar positives**: +197.2 ms mean saving (14.4% relative), CI [136.1, 276.7] ms, with zero output regressions — every scalar-gated response, correct or not, exactly matched what `atomic_only`/`official` independently produced for the same request.
- **Five scalar exits (1/33 positives, 4/4 no-tool-distractor exits) disagreed with the strict-field gold score.** All five were traced back to the un-gated sampler making the identical error on the identical request — the gate reproduced existing base-model mistakes faster, it did not cause new ones, but it also cannot tell a stable-and-wrong draft from a stable-and-right one.

**Decision:** keep `REFLEX_SCALAR_EARLY_EXIT=false`. The shipped two-draft setting has no measured benefit to justify the added risk surface, and the promising one-draft variant needs a larger calibration/negative set before it could be defaulted on responsibly. This is reported as a negative-leaning result, not softened into a launch.
