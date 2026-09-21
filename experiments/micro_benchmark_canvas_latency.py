#!/usr/bin/env python3
"""
Micro-Benchmark: Decoder Canvas Length Latency Scaling on NVIDIA GB10
Empirically measures the latency (ms) and throughput of the DiffusionGemma bidirectional
decoder as a function of canvas length:
  - L = 4 (Reflex Minimal Control Slot)
  - L = 8 (Reflex Typed Route)
  - L = 16 (Reflex Multi-Slot Control Canvas)
  - L = 32 (Extended Control Canvas)
  - L = 64 (Expanded Generative Sub-block)
  - L = 128 (Expanded Generative Canvas)
  - L = 256 (Standard Monolithic Diffusion Canvas)
  - L = 512 (Wide Diffusion Canvas)
Proves the core systems thesis of Reflex: discarded canvas tokens have severe latency penalties,
and a 4-16 token micro-canvas achieves sub-150ms execution.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoConfig
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderModel,
    DiffusionGemmaConfig,
)
from transformers.cache_utils import DynamicCache


def benchmark_canvas_scaling():
    print("=" * 75, flush=True)
    print("PROJECT REFLEX: MICRO-BENCHMARK — CANVAS LATENCY SCALING ON NVIDIA GB10", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Load official DiffusionGemma configuration
    config = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")
    text_config = config.text_config

    # To isolate attention scaling across canvas lengths with zero CPU overhead,
    # we benchmark the native DiffusionGemma decoder architecture configured for 4 representative layers
    bench_config = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")
    bench_config.text_config.allow_global_per_layer_attribute_access = True
    bench_config.text_config.num_hidden_layers = 4

    print(f"Instantiating DiffusionGemmaDecoderModel directly on CUDA in bfloat16...", flush=True)
    t0 = time.time()
    
    # Initialize directly on device to prevent CPU memory overhead
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    decoder = DiffusionGemmaDecoderModel(bench_config)
    decoder.eval()
    print(f"Decoder instantiated on GPU in {time.time() - t0:.2f}s", flush=True)

    canvas_lengths = [4, 8, 16, 32, 64, 128, 256, 512]
    prompt_len = 512  # Standard agent prompt length

    # Create dummy prompt KV cache representing cached prompt state
    kv_cache = DynamicCache()
    num_heads = bench_config.text_config.num_key_value_heads
    head_dim = bench_config.text_config.head_dim
    for layer_idx in range(bench_config.text_config.num_hidden_layers):
        dummy_k = torch.zeros((1, num_heads, prompt_len, head_dim), dtype=torch.bfloat16, device=device)
        dummy_v = torch.zeros((1, num_heads, prompt_len, head_dim), dtype=torch.bfloat16, device=device)
        kv_cache.update(dummy_k, dummy_v, layer_idx=layer_idx)

    results = []
    num_warmup = 10
    num_trials = 30

    print(f"\nBenchmarking forward pass with cached prompt KV length = {prompt_len}...", flush=True)
    print(f"{'Canvas Length (L)':<20} | {'Mean Latency (ms)':<18} | {'p50 (ms)':<12} | {'p95 (ms)':<12} | {'Speedup vs 256':<15}", flush=True)
    print("-" * 85, flush=True)

    base_256_lat = None

    for L in reversed(canvas_lengths):
        canvas_ids = torch.randint(0, bench_config.text_config.vocab_size, (1, L), device=device)
        pos_ids = torch.arange(prompt_len, prompt_len + L, dtype=torch.long, device=device).unsqueeze(0)

        # Warmup
        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = decoder(
                    decoder_input_ids=canvas_ids,
                    past_key_values=kv_cache,
                    decoder_position_ids=pos_ids,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Timed trials
        latencies = []
        with torch.inference_mode():
            for _ in range(num_trials):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                _ = decoder(
                    decoder_input_ids=canvas_ids,
                    past_key_values=kv_cache,
                    decoder_position_ids=pos_ids,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lat = (time.perf_counter() - t_start) * 1000.0
                latencies.append(lat)

        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))

        if L == 256:
            base_256_lat = mean_lat

        results.append({
            "canvas_length": L,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "latencies": latencies,
        })

    # Sort results by canvas length ascending
    results = sorted(results, key=lambda r: r["canvas_length"])

    for r in results:
        L = r["canvas_length"]
        speedup = (base_256_lat / r["mean_latency_ms"]) if base_256_lat else 1.0
        print(f"{L:<20} | {r['mean_latency_ms']:<18.2f} | {r['p50_latency_ms']:<12.2f} | {r['p95_latency_ms']:<12.2f} | {speedup:<15.2f}x", flush=True)

    print("-" * 85, flush=True)
    r8 = next(r for r in results if r["canvas_length"] == 8)
    speedup_8 = base_256_lat / r8["mean_latency_ms"]
    print(f"\nEmpirical Finding on GB10:", flush=True)
    print(f" - Standard 256-token canvas latency: {base_256_lat:.2f} ms", flush=True)
    print(f" - Reflex 8-token micro-canvas latency: {r8['mean_latency_ms']:.2f} ms", flush=True)
    print(f" - Speedup factor: {speedup_8:.2f}x reduction in single-step decoder latency!", flush=True)

    # Save results to JSON
    out_dir = os.path.dirname(__file__)
    json_path = os.path.join(out_dir, "canvas_latency_scaling.json")
    with open(json_path, "w") as f:
        json.dump({
            "prompt_kv_length": prompt_len,
            "results": results,
            "base_256_lat_ms": base_256_lat,
            "reflex_8_lat_ms": r8["mean_latency_ms"],
            "speedup_8_vs_256": speedup_8,
        }, f, indent=2)
    print(f"Results saved to {json_path}", flush=True)


if __name__ == "__main__":
    benchmark_canvas_scaling()
