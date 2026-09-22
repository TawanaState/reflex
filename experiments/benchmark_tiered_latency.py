#!/usr/bin/env python3
"""End-to-end tier benchmark with per-request correctness and latency traces."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_settings


def tool(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


TOOLS = [
    tool("mute_audio", "Mute system audio"),
    tool("set_volume", "Set system volume percentage", {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"]),
    tool("write_email", "Draft an email", {"recipient": {"type": "string"}, "body": {"type": "string"}}, ["recipient", "body"]),
]
CASES = [
    {"id": "atomic", "prompt": "Mute system audio now", "gold_tool": "mute_audio", "gold_tier": "atomic", "gold_args": {}},
    {"id": "primitive", "prompt": "Set system volume to 57 percent", "gold_tool": "set_volume", "gold_tier": "parametric_primitive", "gold_args": {"level": 57}},
    {"id": "generative", "prompt": "Draft an email to alice@example.com apologizing for the delay", "gold_tool": "write_email", "gold_tier": "generative_synthesis", "gold_args": None},
]


def summarize_latency(rows, field):
    vals = [row[field] for row in rows if row.get(field) is not None]
    if not vals:
        return None
    return {"count": len(vals), "mean": float(np.mean(vals)), "p50": float(np.percentile(vals, 50)), "p95": float(np.percentile(vals, 95))}


def run_tiered_benchmark(num_trials=10, base_url=None):
    settings = get_settings()
    host = "127.0.0.1" if settings.HOST in ("0.0.0.0", "::") else settings.HOST
    base_url = base_url or f"http://{host}:{settings.PORT}"
    endpoint = f"{base_url}/v1/chat/completions"
    health = requests.get(f"{base_url}/health", timeout=10).json()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = Path(__file__).with_name(f"tiered_latency_traces_{run_id}.jsonl")
    cases = CASES
    all_rows = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for case in cases:
            for trial in range(num_trials):
                request = {"model": settings.MODEL_ID, "messages": [{"role": "user", "content": case["prompt"]}], "tools": TOOLS, "max_tokens": 64}
                row = {"run_id": run_id, "case_id": case["id"], "trial": trial, "gold_tool": case["gold_tool"], "gold_tier": case["gold_tier"], "gold_args": case["gold_args"]}
                start = time.perf_counter()
                try:
                    response = requests.post(endpoint, json=request, timeout=60)
                    row["wall_latency_ms"] = (time.perf_counter() - start) * 1000
                    row["http_status"] = response.status_code
                    response.raise_for_status()
                    data = response.json()
                    meta = data.get("reflex_metadata") or {}
                    row["engine_latency_ms"] = meta.get("latency_ms")
                    row["execution_path"] = meta.get("execution_path")
                    row["actual_tier"] = meta.get("tier")
                    row["steps_executed"] = meta.get("steps_executed")
                    calls = (data.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
                    row["actual_tool"] = calls[0]["function"]["name"] if calls else None
                    raw_args = calls[0]["function"]["arguments"] if calls else None
                    row["raw_arguments"] = raw_args
                    try:
                        args = json.loads(raw_args) if raw_args is not None else None
                    except (TypeError, json.JSONDecodeError):
                        args = None
                    row["json_object"] = isinstance(args, dict)
                    row["route_correct"] = row["actual_tool"] == case["gold_tool"]
                    row["tier_correct"] = row["actual_tier"] == case["gold_tier"]
                    if case["id"] == "atomic":
                        row["argument_correct"] = args == {}
                    elif case["id"] == "primitive":
                        row["argument_correct"] = args == case["gold_args"]
                    else:
                        row["argument_correct"] = isinstance(args, dict) and args.get("recipient") == "alice@example.com" and isinstance(args.get("body"), str) and bool(args["body"].strip())
                    row["call_correct"] = bool(row["route_correct"] and row["tier_correct"] and row["argument_correct"])
                except Exception as exc:
                    row["wall_latency_ms"] = (time.perf_counter() - start) * 1000
                    row["error"] = str(exc)
                    row["call_correct"] = False
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                all_rows.append(row)
                print(f"{case['id']} trial={trial} correct={row['call_correct']} route={row.get('actual_tool')} tier={row.get('actual_tier')} wall_ms={row['wall_latency_ms']:.1f}")
    summary = {"run_id": run_id, "server_health": health, "num_trials_per_case": num_trials, "trace_file": str(trace_path), "cases": {}}
    for case in cases:
        rows = [row for row in all_rows if row["case_id"] == case["id"]]
        correct = [row for row in rows if row["call_correct"]]
        summary["cases"][case["id"]] = {
            "gold_tool": case["gold_tool"], "gold_tier": case["gold_tier"],
            "call_correct": len(correct), "total": len(rows), "call_accuracy": len(correct) / len(rows),
            "all_wall_latency_ms": summarize_latency(rows, "wall_latency_ms"),
            "correct_call_wall_latency_ms": summarize_latency(correct, "wall_latency_ms"),
            "all_engine_latency_ms": summarize_latency(rows, "engine_latency_ms"),
        }
    summary_path = Path(__file__).with_name(f"tiered_latency_summary_{run_id}.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Traces: {trace_path}\nSummary: {summary_path}")
    return summary


if __name__ == "__main__":
    run_tiered_benchmark(num_trials=int(os.environ.get("REFLEX_BENCH_TRIALS", "10")))
