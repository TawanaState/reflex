#!/usr/bin/env python3
"""Interleaved, same-model DiffusionGemma vs Reflex benchmark.

Both arms use ReflexEngine's native chat template and the official
model.generate() sampler. The only policy difference is the schema-aware
draft exit. The model is loaded once, calls are randomized within each pair,
and every request/arm gets the same seeded RNG state. Wall times include
prompt formatting and parsing, with CUDA synchronized at both boundaries.
"""

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import get_settings
from src.engine.native_tool_calling import build_native_tools
from src.engine.runner import ReflexEngine

ROOT = Path(__file__).resolve().parents[1]


def equal_value(actual, expected):
    if isinstance(expected, (dict, list)):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def strict_fields(predicted, gold):
    """Exact fields against BFCL's listed alternatives; not its AST scorer.

    A gold alternative of empty string permits omission. Unexpected predicted
    keys fail. No string containment or numeric coercion is used.
    """
    if not isinstance(predicted, dict) or not isinstance(gold, dict):
        return False
    if set(predicted) - set(gold):
        return False
    for key, alternatives in gold.items():
        alternatives = alternatives if isinstance(alternatives, list) else [alternatives]
        if key not in predicted:
            if "" not in alternatives:
                return False
        elif not any(equal_value(predicted[key], value) for value in alternatives):
            return False
    return True


def score(output, row):
    calls = output.tool_calls or []
    if row["arg_bucket"] == "no_tool":
        return {"name_ok": None, "strict_call_ok": None, "no_tool_ok": not calls}
    if len(calls) != 1:
        return {"name_ok": False, "strict_call_ok": False, "no_tool_ok": None}
    fn = calls[0]["function"]
    name_ok = fn["name"] == row["ground_truth_label"]
    try:
        args = json.loads(fn["arguments"])
    except (ValueError, TypeError):
        args = None
    return {
        "name_ok": name_ok,
        "strict_call_ok": name_ok and strict_fields(args, row["gold_args"]),
        "no_tool_ok": None,
    }


def run_one(engine, row, policy, seed):
    engine.settings.REFLEX_ATOMIC_EARLY_EXIT = policy != "base"
    engine.settings.REFLEX_ATOMIC_STABLE_STEPS = 1 if policy == "base" else int(policy.split("_")[-1])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    messages = [{"role": "user", "content": row["user_query"]}]
    tools = build_native_tools(row["candidate_functions"])
    start = time.perf_counter()
    output = engine.generate(messages=messages, tools=tools, max_tokens=256)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    calls = output.tool_calls or []
    parsed = json.loads(calls[0]["function"]["arguments"]) if calls else None
    result = {
        "id": row["id"], "policy": policy, "bucket": row["arg_bucket"],
        "source": row["source"], "num_candidates": row["num_candidates"],
        "gold_tool": row["ground_truth_label"],
        "pred_tool": calls[0]["function"]["name"] if calls else None,
        "pred_args": parsed, "content": output.content,
        "execution_path": output.execution_path,
        "early_exit": output.execution_path == "NATIVE_ATOMIC_EARLY_EXIT",
        "latency_ms": wall_ms, "engine_latency_ms": output.latency_ms,
        "forward_passes_estimate": output.steps_executed,
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
    }
    result.update(score(output, row))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/bfcl/paired_eval_500.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Debug run only; never use for primary claims")
    parser.add_argument("--ablate-atomic", action="store_true", help="Also run 2/3 stable steps on atomic rows")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or ROOT / f"experiments/paired_bfcl_{run_id}.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    settings = get_settings().model_copy(update={
        "REFLEX_MODE": "reflex", "REFLEX_LORA_PATH": "/tmp/reflex-no-adapter-for-paired-benchmark",
    })
    engine = ReflexEngine(settings)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {
        "run_id": run_id, "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(rows), "git_commit": commit,
        "gpu": gpu, "torch": torch.__version__,
        "model_path": settings.DIFFUSION_GEMMA_PATH,
        "canvas_length": engine.model.config.canvas_length,
        "lora_loaded": engine.is_lora_loaded,
        "max_new_tokens": 256, "seed": 20260923,
        "warmup": "first row once per base/reflex policy, excluded",
        "timing": "synchronized end-to-end wall time, includes template and output parsing",
        "base": "official model.generate() with no draft observer",
        "reflex": "same generator with NativeDraftObserver and schema-aware atomic exit",
        "ablate_atomic": args.ablate_atomic,
        "score": "strict field match with BFCL listed alternatives; not official BFCL AST score",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), **metadata}), flush=True)
    for policy in ("base", "reflex_1"):
        run_one(engine, rows[0], policy, 20260923)
    rng = random.Random(20260923)
    with output_path.open("w", buffering=1) as trace:
        for index, row in enumerate(rows):
            policies = ["base", "reflex_1"]
            rng.shuffle(policies)
            if args.ablate_atomic and row["arg_bucket"] == "atomic":
                policies += ["reflex_2", "reflex_3"]
            for order, policy in enumerate(policies):
                result = run_one(engine, row, policy, 20260923 + index)
                result["pair_index"] = index
                result["order_in_pair"] = order
                trace.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0:
                print(f"completed {index + 1}/{len(rows)} pairs", flush=True)
    print(f"Trace: {output_path}", flush=True)


if __name__ == "__main__":
    main()
