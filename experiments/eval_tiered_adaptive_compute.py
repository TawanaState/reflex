#!/usr/bin/env python3
"""Held-out cross-tier probe of correctness, latency, and decoder steps."""
import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from eval_native_tool_calling import args_roughly_match

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-bucket', type=int, default=52)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / 'data/bfcl/live_eval.jsonl').open()]
    selected = []
    for bucket in ('primitive', 'freetext'):
        candidates = [row for row in rows if row['arg_bucket'] == bucket]
        random.Random(args.seed + len(bucket)).shuffle(candidates)
        selected.extend(candidates[:args.per_bucket])

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    trace_path = ROOT / 'experiments' / f'tiered_adaptive_compute_{run_id}.jsonl'
    results = []
    with trace_path.open('w', encoding='utf-8') as trace:
        for index, row in enumerate(selected):
            tools = [{'type': 'function', 'function': fn} for fn in row['candidate_functions']]
            start = time.perf_counter()
            response = requests.post('http://127.0.0.1:8090/v1/chat/completions', json={
                'model': 'reflex-diffusiongemma',
                'messages': [{'role': 'user', 'content': row['user_query']}],
                'tools': tools, 'max_tokens': 256,
            }, timeout=120)
            wall_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            body = response.json()
            calls = body['choices'][0]['message'].get('tool_calls') or []
            name = calls[0]['function']['name'] if calls else None
            raw_args = calls[0]['function']['arguments'] if calls else None
            name_correct = name == row['ground_truth_label']
            args_correct = name_correct and args_roughly_match(raw_args, row.get('gold_args'))
            result = {
                'id': row['id'], 'bucket': row['arg_bucket'], 'gold_tool': row['ground_truth_label'],
                'predicted_tool': name, 'name_correct': name_correct, 'args_correct_loose': args_correct,
                'wall_latency_ms': round(wall_ms, 3), **body['reflex_metadata'],
            }
            results.append(result)
            trace.write(json.dumps(result) + '\n')
            trace.flush()
            print(f"{index+1}/{len(selected)} {row['arg_bucket']} name={name_correct} args={args_correct} "
                  f"steps={result['steps_executed']} latency={wall_ms:.0f}ms", flush=True)

    summary = {'run_id': run_id, 'trace_path': str(trace_path), 'seed': args.seed, 'per_bucket': {}}
    for bucket in ('primitive', 'freetext'):
        bucket_rows = [row for row in results if row['bucket'] == bucket]
        summary['per_bucket'][bucket] = {
            'total': len(bucket_rows),
            'name_correct': sum(row['name_correct'] for row in bucket_rows),
            'args_correct_loose': sum(row['args_correct_loose'] for row in bucket_rows),
            'early_exits': sum(row['execution_path'] == 'NATIVE_ATOMIC_EARLY_EXIT' for row in bucket_rows),
            'wrong_early_exits': sum(
                row['execution_path'] == 'NATIVE_ATOMIC_EARLY_EXIT' and not row['name_correct']
                for row in bucket_rows
            ),
            'mean_steps': statistics.mean(row['steps_executed'] for row in bucket_rows),
            'median_steps': statistics.median(row['steps_executed'] for row in bucket_rows),
            'mean_latency_ms': statistics.mean(row['wall_latency_ms'] for row in bucket_rows),
            'median_latency_ms': statistics.median(row['wall_latency_ms'] for row in bucket_rows),
        }
    summary_path = trace_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
