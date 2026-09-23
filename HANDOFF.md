# Reflex handoff — 2026-09-23

## Status: complete (2026-09-23, later same day)

The GPU run finished cleanly (800/800 rows). Summary, plot, and documentation updates (`RESULTS.md`, `README.md`, `SCALAR_EXPERIMENT.md`, `NOTES.md`) are done. Verdict: the shipped `REFLEX_SCALAR_STABLE_STEPS=2` default shows no statistically reliable speedup anywhere in the workload; a one-draft ablation does show a real, non-regressive ~14% speedup on true scalar positives but isn't enough evidence to default it on. `REFLEX_SCALAR_EARLY_EXIT` stays `false`. Focused tests (20/20 in `tests/test_native_early_exit.py`, 15 total across the paired-benchmark suite), `py_compile`, and `git diff --check` all pass. See `RESULTS.md#matched-scalar-early-exit-extension-2026-09-23` for the full writeup. Remaining original objective/context preserved below for reference.

## Objective

Extend Reflex's native-draft early exit beyond zero-argument tools to typed scalar calls, then measure latency and correctness against the same loaded DiffusionGemma checkpoint. Report negative results honestly. Keep the experimental scalar gate disabled by default until evidence supports enabling it.

## Implemented

- `src/engine/early_exit.py` now accepts complete, offered scalar tool calls only when every declared field is present, there are no extra fields, schema validation succeeds, and the exact canonical call is stable for N consecutive drafts. Open strings, objects, arrays, incomplete calls, and unknown tools do not exit. The old atomic path remains.
- `src/engine/runner.py` wires this observer into native generation and returns `NATIVE_SCALAR_EARLY_EXIT` when it fires. `.env.example` and `src/config.py` add `REFLEX_SCALAR_EARLY_EXIT=false` and `REFLEX_SCALAR_STABLE_STEPS=2`.
- Focused tests in `tests/test_native_early_exit.py` passed (10 tests). Earlier combined focused suite passed 27 tests before the last added test. No full final test run yet.
- `scripts/prepare_scalar_eval.py` froze 214 real BFCL-derived requests in `data/bfcl/scalar_eval.jsonl` with source hashes in its provenance file: 58 scalar positives, 100 no-tool prompts with scalar distractors, 25 atomic controls, 31 open-text controls. `SCALAR_EXPERIMENT.md` states the protocol. New benchmark, summary, and plot scripts are in `experiments/`.

## GPU benchmark status

The matched run was launched with `PYTHONPATH=. .venv/bin/python experiments/benchmark_scalar_early_exit.py`. Trace: `experiments/scalar_exit_20260923T092825Z.jsonl`; matching manifest beside it. At last check it had completed **80/214** requests. The previous tool poll was interrupted, so check whether the process is still running before restarting anything (`pgrep -af benchmark_scalar_early_exit` and `wc -l` on the trace). Its old exec session ID was `14921` and may still be pollable. The run interleaves official sampler, atomic-only Reflex, and two-stable-draft scalar Reflex for every row; one-draft scalar ablation also runs on 58 positives and 100 hard negatives. A complete trace should have **800 rows**.

Interim at 67 complete requests: scalar positives (n=22) had 16/22 strict-call matches in each arm; mean latency atomic-only 1332 ms, scalar-1 1189 ms, scalar-2 1329 ms. Eleven scalar exits occurred in each scalar arm and were correct. No-tool scalar-distractor cases (n=25) had 20/25 correct no-call outcomes in each arm; scalar-1 and scalar-2 each exited twice on wrong calls. These are preliminary observations, not final claims.

## Next actions

1. Finish or recover the GPU run. Do not mix a resumed/new run into the existing trace; if the process died, assess complete pairs and launch a fresh run if needed.
2. Run `PYTHONPATH=. .venv/bin/python experiments/summarize_scalar_early_exit.py experiments/scalar_exit_20260923T092825Z.jsonl` and then `PYTHONPATH=. .venv/bin/python experiments/plot_scalar_early_exit.py experiments/scalar_exit_20260923T092825Z.summary.json`. Check paired correctness changes and wrong exits, especially no-tool negatives. The primary contrast is scalar-2 versus atomic-only.
3. Update `RESULTS.md`, `README.md`, and `SCALAR_EXPERIMENT.md` with measured results and limitations. Keep default scalar exit off if speed gain is erased or risk is unresolved. Do not claim official BFCL AST accuracy; the scorer is a documented strict-field proxy.
4. Run focused tests, syntax/compile checks, `git diff --check`, and inspect `git status`. The work is currently uncommitted; do not discard existing changes.

The prior 500-request matched benchmark remains the established result: atomic positives 979→702 ms (25/25 correct), but only 0.45% mean saving across its selected mixed workload and four wrong early exits on no-tool requests where baseline made the same errors. See `RESULTS.md` and `BENCHMARK.md`.
