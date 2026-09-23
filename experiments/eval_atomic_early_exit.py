#!/usr/bin/env python3
"""Evaluate the live draft gate on every held-out BFCL atomic example."""
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine.native_tool_calling import build_native_tools


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main():
    rows = [json.loads(line) for line in (ROOT / 'data/bfcl/live_eval.jsonl').open()]
    rows = [row for row in rows if row['arg_bucket'] == 'atomic']
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    trace_path = ROOT / 'experiments' / f'atomic_early_exit_{run_id}.jsonl'
    results = []
    with trace_path.open('w', encoding='utf-8') as trace:
        for index, row in enumerate(rows):
            request = {
                'model': 'reflex-diffusiongemma',
                'messages': [{'role': 'user', 'content': row['user_query']}],
                'tools': build_native_tools(row['candidate_functions']),
                'max_tokens': 256,
            }
            start = time.perf_counter()
            response = requests.post('http://127.0.0.1:8090/v1/chat/completions', json=request, timeout=120)
            latency = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            body = response.json()
            calls = body['choices'][0]['message'].get('tool_calls') or []
            predicted = calls[0]['function']['name'] if calls else None
            metadata = body['reflex_metadata']
            result = {
                'id': row['id'], 'gold_tool': row['ground_truth_label'], 'predicted_tool': predicted,
                'correct': predicted == row['ground_truth_label'], 'wall_latency_ms': latency,
                **metadata,
            }
            results.append(result)
            trace.write(json.dumps(result) + '\n')
            trace.flush()
            print(f"{index + 1}/{len(rows)} correct={result['correct']} steps={result['steps_executed']} "
                  f"path={result['execution_path']} latency={latency:.1f}ms", flush=True)
    latencies = [row['wall_latency_ms'] for row in results]
    summary = {
        'run_id': run_id, 'trace_path': str(trace_path), 'total': len(results),
        'correct': sum(row['correct'] for row in results),
        'accuracy': sum(row['correct'] for row in results) / len(results),
        'early_exits': sum(row['execution_path'] == 'NATIVE_ATOMIC_EARLY_EXIT' for row in results),
        'mean_latency_ms': statistics.mean(latencies),
        'p50_latency_ms': statistics.median(latencies),
        'p95_latency_ms': percentile(latencies, .95),
        'mean_warm_latency_ms_excluding_first': statistics.mean(latencies[1:]),
        'mean_steps': statistics.mean(row['steps_executed'] for row in results),
    }
    summary_path = trace_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
