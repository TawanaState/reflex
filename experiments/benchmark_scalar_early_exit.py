#!/usr/bin/env python3
"""Matched in-process BFCL test of experimental typed-scalar draft exits."""

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from experiments.benchmark_paired_bfcl import score
from src.config import get_settings
from src.engine.native_tool_calling import build_native_tools
from src.engine.runner import ReflexEngine

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "official": (False, False, 2),
    "atomic_only": (True, False, 2),
    "scalar_1": (True, True, 1),
    "scalar_2": (True, True, 2),
}


def run_one(engine, row, policy, seed):
    atomic, scalar, stable = POLICIES[policy]
    engine.settings.REFLEX_ATOMIC_EARLY_EXIT = atomic
    engine.settings.REFLEX_SCALAR_EARLY_EXIT = scalar
    engine.settings.REFLEX_SCALAR_STABLE_STEPS = stable
    engine.settings.REFLEX_ATOMIC_STABLE_STEPS = 1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
    tools = build_native_tools(row["candidate_functions"])
    start = time.perf_counter()
    output = engine.generate(
        messages=[{"role": "user", "content": row["user_query"]}],
        tools=tools,
        max_tokens=256,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    calls = output.tool_calls or []
    arguments = json.loads(calls[0]["function"]["arguments"]) if calls else None
    result = {
        "id": row["id"], "policy": policy, "bucket": row["eval_bucket"],
        "arg_bucket": row["arg_bucket"], "source": row["source"],
        "gold_tool": row["ground_truth_label"],
        "pred_tool": calls[0]["function"]["name"] if calls else None,
        "pred_args": arguments, "content": output.content,
        "execution_path": output.execution_path,
        "early_kind": ("atomic" if output.execution_path == "NATIVE_ATOMIC_EARLY_EXIT"
                       else "scalar" if output.execution_path == "NATIVE_SCALAR_EARLY_EXIT"
                       else None),
        "latency_ms": wall_ms,
        "forward_passes_estimate": output.steps_executed,
        "prompt_tokens": output.prompt_tokens,
    }
    result.update(score(output, row))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/bfcl/scalar_eval.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Debug only; not for reported claims")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or ROOT / f"experiments/scalar_exit_{run_id}.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    settings = get_settings().model_copy(update={
        "REFLEX_MODE": "reflex", "REFLEX_LORA_PATH": "/tmp/reflex-no-adapter-for-scalar-benchmark",
    })
    engine = ReflexEngine(settings)
    manifest = {
        "run_id": run_id, "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(rows),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observer_sha256": hashlib.sha256((ROOT / "src/engine/early_exit.py").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256((ROOT / "src/engine/runner.py").read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "model_path": settings.DIFFUSION_GEMMA_PATH,
        "lora_loaded": engine.is_lora_loaded,
        "canvas_length": engine.model.config.canvas_length,
        "max_new_tokens": 256,
        "seed": 20260924,
        "policies": POLICIES,
        "scalar_1_scope": "scalar positives and scalar-distractor no-tool negatives only",
        "warmup": "first input under each policy, excluded",
        "timing": "CUDA-synchronized end-to-end wall time, includes prompt and output processing",
        "score": "strict field alternatives / no executable call, not BFCL official AST score",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), **manifest}), flush=True)
    for policy in POLICIES:
        run_one(engine, rows[0], policy, 20260924)
    rng = random.Random(20260924)
    with output_path.open("w", buffering=1) as trace:
        for index, row in enumerate(rows):
            policies = ["official", "atomic_only", "scalar_2"]
            if row["eval_bucket"] in ("scalar_positive", "no_tool_scalar_distractor"):
                policies.append("scalar_1")
            rng.shuffle(policies)
            for order, policy in enumerate(policies):
                result = run_one(engine, row, policy, 20260924 + index)
                result["pair_index"] = index
                result["order_in_request"] = order
                trace.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (index + 1) % 20 == 0:
                print(f"completed {index + 1}/{len(rows)} requests", flush=True)
    print(f"Trace: {output_path}", flush=True)


if __name__ == "__main__":
    main()
