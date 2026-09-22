#!/usr/bin/env python3
"""
Benchmark: Empirical Latency & Scaling across Adaptive Compute Tiers.
Measures:
  - Tier 1: Atomic (1 step)
  - Tier 2: Parametric Primitive (2-4 steps) vs. Fixed 20-step baseline
  - Tier 3: Generative Synthesis (12-20 steps)
Saves results to experiments/tiered_latency_results.json.
"""

import json
import os
import sys
import time
import numpy as np
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_settings


def run_tiered_benchmark(num_trials: int = 10):
    settings = get_settings()
    endpoint = f"http://{settings.HOST}:{settings.PORT}/v1/chat/completions"

    print("=" * 80)
    print("PROJECT REFLEX: EMPIRICAL TIERED ADAPTIVE COMPUTE BENCHMARK")
    print(f"Hardware: NVIDIA GB10 Blackwell SoC (Bare-Metal GPU Execution)")
    print(f"Endpoint: {endpoint} (Trials per tier: {num_trials})")
    print("=" * 80)

    tools = [
        # Tier 1
    # Tool definitions per Tier
    t1_tools = [
        {
            "type": "function",
            "function": {
                "name": "mute_audio",
                "description": "Mutes system audio output",
                "description": "Mutes audio",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    t2_tools = [
        {
            "type": "function",
            "function": {
                "name": "mute_audio",
                "description": "Mutes audio",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        # Tier 2
        {
            "type": "function",
            "function": {
                "name": "set_volume",
                "description": "Sets the master volume level",
                "description": "Sets volume level",
                "parameters": {
                    "type": "object",
                    "properties": {"level": {"type": "integer", "description": "Volume 0-100"}},
                    "required": ["level"],
                },
            },
        },
        # Tier 3
    ]
    t3_tools = [
        {
            "type": "function",
            "function": {
                "name": "compose_summary",
                "description": "Writes a detailed summary of a document",
                "name": "mute_audio",
                "description": "Mutes audio",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_email",
                "description": "Drafts and sends an email message with custom text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "properties": {
                        "recipient": {"type": "string", "description": "Email address"},
                        "body": {"type": "string", "description": "Free-form email content"},
                    },
                    "required": ["recipient", "body"],
                },
            },
        },
    ]

    # Warmup
    print("\nWarming up engine kernels...")
    requests.post(endpoint, json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Mute"}], "tools": tools}, timeout=60)
    requests.post(endpoint, json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Mute"}], "tools": t1_tools}, timeout=60)

    # 1. Benchmark Tier 1 (Atomic)
    print("\n[1/3] Benchmarking Tier 1: Atomic (1-Step Fast Path)...")
    t1_wall_latencies = []
    t1_engine_latencies = []
    for i in range(num_trials):
        t0 = time.perf_counter()
        resp = requests.post(
            endpoint,
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Mute audio now"}], "tools": tools},
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Mute audio now"}], "tools": t1_tools},
            timeout=30,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        t1_wall_latencies.append(lat)
        data = resp.json()
        t1_engine_latencies.append(data.get("reflex_metadata", {}).get("latency_ms", lat))

    # 2. Benchmark Tier 2 (Parametric Primitive)
    print("\n[2/3] Benchmarking Tier 2: Parametric Primitive (2-4 Steps Micro-Canvas)...")
    t2_wall_latencies = []
    t2_engine_latencies = []
    t2_valid_parses = 0
    for i in range(num_trials):
        t0 = time.perf_counter()
        resp = requests.post(
            endpoint,
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": f"Set volume to {50 + i}"}], "tools": tools},
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": f"Set volume to {50 + i}"}], "tools": t2_tools},
            timeout=30,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        t2_wall_latencies.append(lat)
        data = resp.json()
        meta = data.get("reflex_metadata", {})
        t2_engine_latencies.append(meta.get("latency_ms", lat))
        tc = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        if tc:
            try:
                args = json.loads(tc[0]["function"]["arguments"])
                if isinstance(args, dict):
                if isinstance(args, dict) and args.get("level") == (50 + i):
                    t2_valid_parses += 1
            except Exception:
                pass

    # 3. Benchmark Tier 3 (Generative Synthesis)
    print("\n[3/3] Benchmarking Tier 3: Generative Synthesis (12-20 Steps Full Canvas)...")
    t3_wall_latencies = []
    t3_engine_latencies = []
    for i in range(num_trials):
        t0 = time.perf_counter()
        resp = requests.post(
            endpoint,
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Summarize the history of discrete diffusion models in computing."}], "max_tokens": 64},
            json={"model": settings.MODEL_ID, "messages": [{"role": "user", "content": "Write an email to Alice apologizing for the delay"}], "tools": t3_tools, "max_tokens": 64},
            timeout=60,
        )
        lat = (time.perf_counter() - t0) * 1000.0
        t3_wall_latencies.append(lat)
        data = resp.json()
        t3_engine_latencies.append(data.get("reflex_metadata", {}).get("latency_ms", lat))

    # Aggregate Statistics
    summary = {
        "hardware": "NVIDIA GB10 Blackwell SoC (Bare-Metal 121GB Unified)",
        "num_trials": num_trials,
        "tier1_atomic": {
            "steps": 1,
            "wall_latency_ms": {
                "mean": float(np.mean(t1_wall_latencies)),
                "p50": float(np.percentile(t1_wall_latencies, 50)),
                "p95": float(np.percentile(t1_wall_latencies, 95)),
            },
            "engine_latency_ms": {
                "mean": float(np.mean(t1_engine_latencies)),
                "p50": float(np.percentile(t1_engine_latencies, 50)),
                "p95": float(np.percentile(t1_engine_latencies, 95)),
            },
        },
        "tier2_parametric_primitive": {
            "steps": "2-4",
            "wall_latency_ms": {
                "mean": float(np.mean(t2_wall_latencies)),
                "p50": float(np.percentile(t2_wall_latencies, 50)),
                "p95": float(np.percentile(t2_wall_latencies, 95)),
            },
            "engine_latency_ms": {
                "mean": float(np.mean(t2_engine_latencies)),
                "p50": float(np.percentile(t2_engine_latencies, 50)),
                "p95": float(np.percentile(t2_engine_latencies, 95)),
            },
            "argument_parse_accuracy_pct": float((t2_valid_parses / num_trials) * 100.0),
        },
        "tier3_generative_synthesis": {
            "steps": "12-20",
            "wall_latency_ms": {
                "mean": float(np.mean(t3_wall_latencies)),
                "p50": float(np.percentile(t3_wall_latencies, 50)),
                "p95": float(np.percentile(t3_wall_latencies, 95)),
            },
            "engine_latency_ms": {
                "mean": float(np.mean(t3_engine_latencies)),
                "p50": float(np.percentile(t3_engine_latencies, 50)),
                "p95": float(np.percentile(t3_engine_latencies, 95)),
            },
        },
        "comparisons": {
            "tier1_speedup_vs_tier3": float(np.mean(t3_wall_latencies) / np.mean(t1_wall_latencies)),
            "tier2_speedup_vs_tier3": float(np.mean(t3_wall_latencies) / np.mean(t2_wall_latencies)),
        },
    }

    out_path = os.path.join(os.path.dirname(__file__), "tiered_latency_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Tier 1 (Atomic, 1 Step):        Mean = {summary['tier1_atomic']['wall_latency_ms']['mean']:.2f} ms | p50 = {summary['tier1_atomic']['wall_latency_ms']['p50']:.2f} ms")
    print(f"Tier 2 (Primitive, 2-4 Steps):  Mean = {summary['tier2_parametric_primitive']['wall_latency_ms']['mean']:.2f} ms | p50 = {summary['tier2_parametric_primitive']['wall_latency_ms']['p50']:.2f} ms (Parse Acc: {summary['tier2_parametric_primitive']['argument_parse_accuracy_pct']:.1f}%)")
    print(f"Tier 3 (Generative, 16-20 Steps):Mean = {summary['tier3_generative_synthesis']['wall_latency_ms']['mean']:.2f} ms | p50 = {summary['tier3_generative_synthesis']['wall_latency_ms']['p50']:.2f} ms")
    print("-" * 80)
    print(f"Speedup Tier 2 vs Full Generation: {summary['comparisons']['tier2_speedup_vs_tier3']:.2f}x faster compute allocation!")
    print(f"Results saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_tiered_benchmark(num_trials=10)

