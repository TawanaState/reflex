# Reflex paired benchmark protocol

## Question

Does stopping DiffusionGemma's *existing* sampler when a native, schema-valid, empty-argument tool call appears reduce request latency without changing the action returned? This tests a narrow semantic completion rule. It does not claim a new sampler, model training method, or general speedup for generative answers.

## Relation to prior work

[DiffusionGemma's official sampler](https://huggingface.co/docs/transformers/en/model_doc/diffusion_gemma) already uses entropy-based token acceptance. [Prophet](https://arxiv.org/abs/2508.19982) demonstrated early answer convergence and training-free early commit on other diffusion language models. Reflex's testable addition is a narrower stopping predicate: a native agent action is complete when one offered empty-schema call parses cleanly. Any novelty claim is conditional on the measured speed and error tradeoff against the official sampler, not on early stopping in general.

## Data and provenance

`data/bfcl/paired_eval_500.jsonl` is a frozen, stratified diagnostic workload, prepared by `scripts/prepare_paired_eval.py`. Its source revision and SHA-256 hashes are in `data/bfcl/paired_eval_500.provenance.json`.

| Category | Count | Source | Label |
| --- | ---: | --- | --- |
| Empty-argument calls | 25 | BFCL v4 live multiple | Published single-call gold answer; one of the 25 offers optional arguments and is not eligible for Reflex's atomic exit. |
| Primitive-argument calls | 303 | BFCL v4 live multiple | Published single-call gold answer. |
| Longer-text arguments | 52 | BFCL v4 live multiple | Published single-call gold answer. This bucket is based on a gold string longer than 25 characters; it is not a complete taxonomy of generative tasks. |
| No-tool requests | 43 | BFCL v4 live irrelevance | Category label says no offered tool applies; each case offers at least one empty-schema distractor. |
| No-tool requests | 77 | BFCL v4 irrelevance | Category label says no offered tool applies. |

The live multiple source was already filtered to 2–10 candidate functions, one gold call, and no exact query overlap with the repo's old train/cal/test splits. Preparation also excludes exact query overlaps for the no-tool rows. This is a **selected 500-request workload**, not BFCL's official aggregate or a measured production mix. The 25 atomic positives are the available held-out examples in this source; their sample size limits any safety claim. No examples, gold calls, or latency values are fabricated.

## Paired run

`experiments/benchmark_paired_bfcl.py` loads the base BF16 checkpoint once with no LoRA. Both arms use the same native chat template, tool menu, 256-token generation budget, official `model.generate()`, parsing, and schema validation. `base` disables the draft observer; `reflex_1` enables a one-stable-draft atomic exit. A seeded random swap chooses their order within each request. Both arms receive the same per-request RNG seed. The first request warms each arm and is excluded. CUDA is synchronized before and after each timed request. Reported wall time includes prompt construction, model generation, and output parsing.

The atomic rows also receive `reflex_2` and `reflex_3` for the two- and three-draft stability ablation. These extra arms run after the randomized base/one-draft pair, so their timing is exploratory. Every trace row records the execution path, output, latency, and estimated decoder forward count. The manifest records the model, GPU, library version, input hash, and git revision.

## Scoring and analysis

Tool-name accuracy compares the returned name to the published gold name. A stricter call metric also compares typed argument fields to the BFCL possible-answer alternatives, allows omission only when an empty-string alternative is listed, and rejects extra fields. This scorer is transparent and unit-tested, but **it is not BFCL's official AST evaluator**. The no-tool metric is the absence of an executable API tool call; it does not evaluate the quality of the assistant's text or distinguish a deliberate refusal from failed schema validation.

`experiments/summarize_paired_bfcl.py` uses only completed request pairs. It reports each arm's mean, p50, and p95 wall latency; paired mean and median milliseconds saved (plus an interquartile range) with a 95% mean interval bootstrapped by normalized query text (484 distinct queries among 500 rows); accuracy changes and paired bootstrap intervals when observed differences exist (an all-zero resample is not presented as a risk bound); early-exit counts split by correctness; latency savings on cases both policies answer correctly; and decoder-forward estimates. Forward counts on full generations are recovered from the official generation statistic rather than counted by an observer. GPU kernel time and energy are **not measured**. The interval describes uncertainty from the sampled query clusters, not a guarantee about unseen tasks.

## Reproduce

```bash
python scripts/prepare_paired_eval.py
PYTHONPATH=. python experiments/benchmark_paired_bfcl.py --ablate-atomic
python experiments/summarize_paired_bfcl.py experiments/paired_bfcl_<run-id>.jsonl
python experiments/plot_paired_bfcl.py experiments/paired_bfcl_<run-id>.summary.json
```

A CUDA GPU with enough memory for the 26B-A4B BF16 checkpoint is required. The raw trace and summary in `experiments/` are the evidence; `RESULTS.md` explains the interpretation and limits.
