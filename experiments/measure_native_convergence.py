#!/usr/bin/env python3
"""Measure when native tool identity and arguments emerge during diffusion.

This is observational: it lets the official sampler finish and records each
argmax draft through its public streamer hook. A gate can then be evaluated
without assuming that a plausible early tool name is correct. Run the full
held-out split before making paper claims; small --max-items runs are probes.
"""
import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.engine.early_exit import NativeDraftObserver
from src.engine.native_tool_calling import build_native_tools, parse_gemma_tool_call
from src.schema.inspector import Tier, inspect_tool_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-items', type=int, default=None)
    parser.add_argument('--bucket', choices=['atomic', 'primitive', 'freetext'])
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--stable-steps', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = get_settings()
    rows = [json.loads(line) for line in (ROOT / 'data/bfcl/live_eval.jsonl').open()]
    if args.bucket:
        rows = [row for row in rows if row['arg_bucket'] == args.bucket]
    random.Random(args.seed).shuffle(rows)
    if args.max_items is not None:
        rows = rows[:args.max_items]

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        settings.DIFFUSION_GEMMA_PATH, torch_dtype=torch.bfloat16, device_map='cuda'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(settings.DIFFUSION_GEMMA_PATH)
    output = args.output or ROOT / 'experiments' / f'native_convergence_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as trace:
        for index, row in enumerate(rows):
            tools = build_native_tools(row['candidate_functions'])
            atomic_names = {
                tool['function']['name'] for tool in tools
                if inspect_tool_schema(tool).tier == Tier.ATOMIC
            }
            inputs = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['user_query']}],
                tools=tools, add_generation_prompt=True, return_tensors='pt', return_dict=True,
            )
            input_ids = inputs['input_ids'].to(model.device)
            observations = []
            start = time.perf_counter()

            def record(observation):
                data = asdict(observation)
                data['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 2)
                observations.append(data)

            observer = NativeDraftObserver(
                tokenizer, atomic_names, stable_steps_required=args.stable_steps,
                enable_exit=False, on_draft=record,
            )
            with torch.inference_mode():
                output_ids = model.generate(input_ids=input_ids, max_new_tokens=model.config.canvas_length, streamer=observer)
            elapsed_ms = (time.perf_counter() - start) * 1000
            sequence = output_ids.sequences if hasattr(output_ids, 'sequences') else output_ids
            generated = tokenizer.decode(sequence[0, input_ids.shape[1]:], skip_special_tokens=False)
            final_name, final_args = parse_gemma_tool_call(generated)
            gate = next((o for o in observations if o['stable_steps'] >= args.stable_steps), None)
            result = {
                'id': row['id'], 'bucket': row['arg_bucket'],
                'gold_tool': row['ground_truth_label'],
                'final_tool': final_name, 'final_correct': final_name == row['ground_truth_label'],
                'final_arguments': final_args,
                'full_latency_ms': round(elapsed_ms, 2),
                'steps': len(observations),
                'first_gold_tool_step': next((o['step'] for o in observations if o['tool_name'] == row['ground_truth_label']), None),
                'first_final_tool_step': next((o['step'] for o in observations if o['tool_name'] == final_name and final_name is not None), None),
                'atomic_gate_step': gate['step'] if gate else None,
                'atomic_gate_tool': gate['tool_name'] if gate else None,
                'atomic_gate_elapsed_ms': gate['elapsed_ms'] if gate else None,
                'observations': observations,
            }
            trace.write(json.dumps(result, ensure_ascii=False) + '\n')
            trace.flush()
            print(f"{index + 1}/{len(rows)} {row['id']} bucket={row['arg_bucket']} "
                  f"full={final_name} gold={row['ground_truth_label']} steps={len(observations)} "
                  f"gate={result['atomic_gate_step']} time={elapsed_ms:.0f}ms", flush=True)
    print(f'Trace: {output}')


if __name__ == '__main__':
    main()
