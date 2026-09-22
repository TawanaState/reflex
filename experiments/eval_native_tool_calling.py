#!/usr/bin/env python3
"""
Experiment A (native tool-calling variant): measures whether DiffusionGemma's
own trained function-calling chat template -- <|tool_call>call:name{args}<tool_call|>
via the OFFICIAL model.generate() (its EntropyBoundSampler-driven block-diffusion loop) -- generalizes to
a frozen, independent, larger BFCL-derived routing set (data/bfcl/live_eval.jsonl,
sourced from the disjoint `live_multiple` category; see
scripts/prepare_bfcl_live_eval.py and data/bfcl/live_eval_provenance.json).

This directly answers the senior-review "kill criterion": run 500+ genuinely
unseen routing examples, randomize candidate order per item, and report
accuracy at varied candidate counts (2..10) -- but using the model's real
tool-calling mechanism instead of Reflex's custom index-selection canvas, and
scores BOTH tool-name selection and argument correctness (not just routing).

Runs three conditions against the SAME loaded model (one GPU load):
  1. base model, original candidate order
  2. base model, deterministic per-item permuted candidate order
  3. Reflex v2 LoRA attached (adapter was tuned for the OLD index-selection
     scheme, so this checks whether that tuning transfers/hurts/helps native
     tool calling, or should simply be dropped in favor of prompting alone)

Writes a timestamped per-item JSONL trace plus a summary JSON. No numbers are
inserted; every row is a real forward-pass measurement.
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "/home/tawana/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
LORA_V2_PATH = str(ROOT / "models" / "reflex_lora_v2")

TYPE_MAP = {"dict": "object", "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "list": "array", "tuple": "array"}

TOOL_CALL_RE = re.compile(r"<\|tool_call>call:([\w\.]+)\{(.*?)\}<tool_call\|>", re.DOTALL)


def normalize_schema(node):
    if isinstance(node, dict):
        return {k: (TYPE_MAP.get(v.lower(), v) if k == "type" and isinstance(v, str) else normalize_schema(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [normalize_schema(x) for x in node]
    return node


def to_openai_tools(functions):
    tools = []
    for f in functions:
        params = normalize_schema(f.get("parameters", {"type": "object", "properties": {}}))
        tools.append({"type": "function", "function": {
            "name": f["name"], "description": f.get("description", ""), "parameters": params,
        }})
    return tools


def parse_tool_call(text):
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def args_roughly_match(pred_body, gold_args):
    """Loose containment check: every gold scalar/first-listed value appears in the
    raw predicted argument text. This is intentionally conservative scoring
    (not the official BFCL AST checker) -- used only as a directional signal."""
    if not gold_args:
        return True
    if pred_body is None:
        return False
    pred_norm = pred_body.casefold()
    hits = 0
    total = 0
    for v in gold_args.values():
        candidates = v if isinstance(v, list) else [v]
        total += 1
        for c in candidates:
            if isinstance(c, str) and c.casefold() in pred_norm:
                hits += 1
                break
            if isinstance(c, (int, float, bool)) and str(c).casefold() in pred_norm:
                hits += 1
                break
    return total > 0 and hits == total


def load_eval_set(max_items):
    rows = []
    with open(ROOT / "data" / "bfcl" / "live_eval.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if max_items:
        rows = rows[:max_items]
    return rows


def run_condition(model, tokenizer, rows, permute, max_new_tokens, condition_name, trace_f):
    results = []
    for row in rows:
        names = list(row["candidate_names"])
        functions = list(row["candidate_functions"])
        if permute:
            rnd = random.Random(42 + sum(ord(c) for c in row["id"]))
            order = list(range(len(names)))
            rnd.shuffle(order)
            names = [names[i] for i in order]
            functions = [functions[i] for i in order]

        tools = to_openai_tools(functions)
        messages = [{"role": "user", "content": row["user_query"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = inputs["input_ids"].to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        seq = out.sequences if hasattr(out, "sequences") else out
        gen_ids = seq[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        pred_name, pred_body = parse_tool_call(gen_text)

        correct_name = pred_name == row["ground_truth_label"]
        correct_args = args_roughly_match(pred_body, row.get("gold_args")) if correct_name else False

        result = {
            "condition": condition_name, "id": row["id"], "permuted": permute,
            "num_candidates": row["num_candidates"], "arg_bucket": row["arg_bucket"],
            "gold_tool": row["ground_truth_label"], "pred_tool": pred_name,
            "correct_name": correct_name, "correct_args_loose": correct_args,
            "latency_ms": lat_ms, "pred_body": pred_body,
        }
        results.append(result)
        trace_f.write(json.dumps(result) + "\n")
        trace_f.flush()
        print(f"[{condition_name}] {row['id']} n_cand={row['num_candidates']} bucket={row['arg_bucket']} "
              f"gold={row['ground_truth_label']} pred={pred_name} name_ok={correct_name} args_ok={correct_args} ({lat_ms:.0f}ms)")
    return results


def summarize(results):
    summary = {"total": len(results), "correct_name": sum(r["correct_name"] for r in results)}
    summary["accuracy_name"] = summary["correct_name"] / summary["total"] if results else None
    summary["correct_args_loose"] = sum(r["correct_args_loose"] for r in results)
    summary["accuracy_args_loose"] = summary["correct_args_loose"] / summary["total"] if results else None
    summary["mean_latency_ms"] = sum(r["latency_ms"] for r in results) / len(results) if results else None
    by_bucket = defaultdict(list)
    for r in results:
        by_bucket[r["num_candidates"]].append(r["correct_name"])
    summary["accuracy_by_candidate_count"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_bucket.items())}
    by_arg_bucket = defaultdict(list)
    for r in results:
        by_arg_bucket[r["arg_bucket"]].append((r["correct_name"], r["correct_args_loose"]))
    summary["accuracy_by_arg_bucket"] = {
        k: {"name": f"{sum(n for n, _ in v)}/{len(v)}", "args_loose": f"{sum(a for _, a in v)}/{len(v)}"}
        for k, v in by_arg_bucket.items()
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Loading base DiffusionGemma 26B (bf16)...")
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    rows = load_eval_set(args.max_items)
    print(f"Loaded {len(rows)} independent eval rows")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = ROOT / "experiments" / f"native_tool_routing_{run_id}.jsonl"
    all_results = {}
    with open(trace_path, "w", encoding="utf-8") as trace_f:
        all_results["base_original"] = run_condition(model, tokenizer, rows, False, args.max_new_tokens, "base_original", trace_f)
        all_results["base_permuted"] = run_condition(model, tokenizer, rows, True, args.max_new_tokens, "base_permuted", trace_f)

        if not args.skip_lora:
            print(f"Attaching Reflex v2 LoRA from {LORA_V2_PATH}...")
            model.model.decoder = PeftModel.from_pretrained(model.model.decoder, LORA_V2_PATH)
            model.eval()
            all_results["lora_v2_original"] = run_condition(model, tokenizer, rows, False, args.max_new_tokens, "lora_v2_original", trace_f)

    summary = {"run_id": run_id, "trace_path": str(trace_path), "num_items": len(rows),
               "max_new_tokens": args.max_new_tokens,
               "conditions": {k: summarize(v) for k, v in all_results.items()}}
    summary_path = trace_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTrace: {trace_path}\nSummary: {summary_path}")
    print(json.dumps(summary["conditions"], indent=2))


if __name__ == "__main__":
    main()
