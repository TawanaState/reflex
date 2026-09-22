#!/usr/bin/env python3
"""Held-out BFCL single-call routing probe through the real Reflex API."""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


def tool_menu(row, permute=False):
    names = list(row["candidate_names"])
    description_by_name = {}
    for line in row["prompt"].splitlines():
        match = re.match(r"\[\d+\] ([^:]+): (.*)", line)
        if match:
            description_by_name[match.group(1)] = match.group(2)
    if permute:
        random.Random(42 + sum(ord(c) for c in row["id"])).shuffle(names)
    return [{"type": "function", "function": {
        "name": name,
        "description": description_by_name.get(name, ""),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }} for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=40)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    settings = get_settings()
    host = "127.0.0.1" if settings.HOST in ("0.0.0.0", "::") else settings.HOST
    base = args.base_url or f"http://{host}:{settings.PORT}"
    health = requests.get(f"{base}/health", timeout=10).json()
    rows = [json.loads(line) for line in (ROOT / "data/bfcl/test.jsonl").open() if line.strip()][:args.max_items]
    train_queries = {json.loads(line)["user_query"].strip().casefold() for line in (ROOT / "data/bfcl/train.jsonl").open() if line.strip()}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trace_path = ROOT / "experiments" / f"bfcl_routing_v2_{run_id}.jsonl"
    outcomes = []
    with trace_path.open("w", encoding="utf-8") as trace:
        for permuted in (False, True):
            for row in rows:
                tools = tool_menu(row, permuted)
                payload = {"model": settings.MODEL_ID, "messages": [{"role": "user", "content": row["user_query"]}], "tools": tools}
                result = {"run_id": run_id, "id": row["id"], "permuted": permuted, "candidate_names": [t["function"]["name"] for t in tools], "gold_tool": row["ground_truth_label"], "query_seen_in_train": row["user_query"].strip().casefold() in train_queries}
                start = time.perf_counter()
                try:
                    response = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=60)
                    result["wall_latency_ms"] = (time.perf_counter() - start) * 1000
                    result["http_status"] = response.status_code
                    response.raise_for_status()
                    body = response.json()
                    calls = (body.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
                    result["predicted_tool"] = calls[0]["function"]["name"] if calls else None
                    result["arguments"] = calls[0]["function"]["arguments"] if calls else None
                    result["metadata"] = body.get("reflex_metadata")
                    result["correct"] = result["predicted_tool"] == result["gold_tool"]
                except Exception as exc:
                    result["wall_latency_ms"] = (time.perf_counter() - start) * 1000
                    result["error"] = str(exc)
                    result["correct"] = False
                outcomes.append(result)
                trace.write(json.dumps(result) + "\n")
                trace.flush()
                print(f"{row['id']} permuted={permuted} correct={result['correct']} predicted={result.get('predicted_tool')}")
    summary = {"run_id": run_id, "server_health": health, "trace_path": str(trace_path), "counts": {}}
    for permuted in (False, True):
        subset = [x for x in outcomes if x["permuted"] == permuted]
        clean = [x for x in subset if not x["query_seen_in_train"]]
        summary["counts"]["permuted" if permuted else "original"] = {
            "correct": sum(x["correct"] for x in subset), "total": len(subset),
            "query_overlap_ids": [x["id"] for x in subset if x["query_seen_in_train"]],
            "clean_correct": sum(x["correct"] for x in clean), "clean_total": len(clean),
            "selected_tools": dict(Counter(x.get("predicted_tool") for x in subset)),
        }
    summary_path = trace_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Traces: {trace_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
