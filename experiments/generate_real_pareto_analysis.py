#!/usr/bin/env python3
"""Archived illustrative plotter; not a source of empirical Pareto points.

It samples synthetic latencies and assumes exit error/coverage. Source is retained
for audit, but direct execution is disabled. Plot only per-request measured traces.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    raise RuntimeError("Retired: synthetic latencies and assumed risk make this plot non-empirical.")
    # 1. Load real data
    ar_path = os.path.join(EXPERIMENTS_DIR, "real_ar_baseline_results.json")
    scaling_path = os.path.join(EXPERIMENTS_DIR, "canvas_latency_scaling.json")
    probe_path = os.path.join(EXPERIMENTS_DIR, "real_step1_probe_results.json")

    with open(ar_path) as f:
        ar_data = json.load(f)
    with open(scaling_path) as f:
        scaling_data = json.load(f)
    with open(probe_path) as f:
        probe_data = json.load(f)

    # 2. Extract metrics
    # Real AR baseline
    ar_latencies = np.array(ar_data["latencies"])
    # Clip startup outliers (>5000ms) for fair steady-state distribution plotting
    ar_steady_latencies = ar_latencies[ar_latencies < 5000.0]
    ar_mean = float(np.mean(ar_steady_latencies))
    ar_p50 = float(np.median(ar_steady_latencies))
    ar_p95 = float(np.percentile(ar_steady_latencies, 95))
    ar_acc = ar_data["accuracy"]
    ar_syntax_err = ar_data["syntax_error_rate"]

    # Real Canvas scaling
    canvas_lens = [r["canvas_length"] for r in scaling_data["results"]]
    canvas_means = [r["mean_latency_ms"] for r in scaling_data["results"]]
    canvas_p50s = [r["p50_latency_ms"] for r in scaling_data["results"]]
    canvas_p95s = [r["p95_latency_ms"] for r in scaling_data["results"]]

    step1_canvas_4tok = canvas_means[0]  # 23.39 ms
    full_step_256tok = canvas_means[3]   # 88.46 ms

    # Fixed 20-step diffusion baseline compute: 20 * full_step_256tok
    fixed_diff_latencies = np.random.normal(loc=20 * full_step_256tok, scale=15, size=100)
    fixed_diff_mean = float(np.mean(fixed_diff_latencies))
    fixed_diff_p50 = float(np.median(fixed_diff_latencies))
    fixed_diff_p95 = float(np.percentile(fixed_diff_latencies, 95))

    # Real Probe metrics
    probe_samples = probe_data["samples"]
    probe_latencies = np.array([s["latency_ms"] for s in probe_samples])
    probe_steady_latencies = probe_latencies[probe_latencies < 1000.0] # exclude sample 0 warm-up
    probe_acc = probe_data["metrics"]["top1_accuracy"]
    probe_mean_entropy = probe_data["metrics"]["mean_entropy"]
    probe_mean_gap = probe_data["metrics"]["mean_confidence_gap"]

    # Hybrid Reflex system (Step-1 Fast Path when calibrated + AR / Expansion Fallback)
    # With Conformal Gate, when confident exit in step1 (23.39ms canvas), else escalate to AR (ar_mean)
    # Fast exit rate scenario analysis
    reflex_fast_lat = step1_canvas_4tok
    reflex_full_lat = float(np.mean(probe_steady_latencies))

    # Compile unified JSON comparison
    comparison = {
        "hardware": {
            "device": "NVIDIA GB10 (Grace Blackwell Architecture)",
            "memory_unified_gib": 121,
            "cuda_version": "13.0",
            "driver_version": "580.159.03",
            "diffusion_model": "google/diffusiongemma-26B-A4B-it (bfloat16, 26B parameters)",
            "ar_baseline_model": "gemma4:12b-it-qat (Ollama local JSON engine)"
        },
        "canvas_latency_scaling": {
            "4_tokens_ms": canvas_means[0],
            "8_tokens_ms": canvas_means[1],
            "16_tokens_ms": canvas_means[2],
            "256_tokens_ms": canvas_means[3],
            "512_tokens_ms": canvas_means[4],
            "speedup_4_vs_256": canvas_means[3] / canvas_means[0]
        },
        "paradigms": {
            "Autoregressive_LLM": {
                "model": "gemma4:12b-it-qat",
                "mean_latency_ms": ar_mean,
                "p50_latency_ms": ar_p50,
                "p95_latency_ms": ar_p95,
                "accuracy_pct": ar_acc,
                "syntax_error_rate_pct": ar_syntax_err,
                "mechanism": "Token-by-token sequential decode + JSON schema prompt"
            },
            "Fixed_20Step_Diffusion": {
                "model": "DiffusionGemma-26B (256-token canvas)",
                "mean_latency_ms": fixed_diff_mean,
                "p50_latency_ms": fixed_diff_p50,
                "p95_latency_ms": fixed_diff_p95,
                "accuracy_pct": 88.0,
                "syntax_error_rate_pct": 0.0,
                "mechanism": "Uniform 20 reverse diffusion steps over 256 tokens"
            },
            "Reflex_Step1_Canvas_Decoder_Only": {
                "model": "DiffusionGemma-26B (4-token micro-canvas)",
                "mean_latency_ms": step1_canvas_4tok,
                "p50_latency_ms": canvas_p50s[0],
                "p95_latency_ms": canvas_p95s[0],
                "speedup_vs_ar": ar_mean / step1_canvas_4tok,
                "syntax_error_rate_pct": 0.0,
                "mechanism": "1-step discrete denoise on 4 tokens with pre-cached prompt KV"
            },
            "Reflex_Step1_EndToEnd": {
                "model": "DiffusionGemma-26B (Encoder + 4-token Decoder)",
                "mean_latency_ms": float(np.mean(probe_steady_latencies)),
                "p50_latency_ms": float(np.median(probe_steady_latencies)),
                "p95_latency_ms": float(np.percentile(probe_steady_latencies, 95)),
                "accuracy_pct": probe_acc,
                "mean_entropy": probe_mean_entropy,
                "mean_confidence_gap": probe_mean_gap,
                "conformal_selective_risk": 0.0,
                "conformal_test_coverage": 0.0,
                "syntax_error_rate_pct": 0.0,
                "mechanism": "Full prompt encoding + 1-step canvas decode + conformal calibration"
            }
        }
    }

    out_comp = os.path.join(EXPERIMENTS_DIR, "real_benchmark_comparison.json")
    with open(out_comp, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved real comparison metrics to {out_comp}")

    # 3. Generate Publication Figure (4 Panels)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # Panel A: Latency Comparison Bar & Violin/Boxplot
    ax1 = axes[0, 0]
    paradigms = ["Reflex Step-1\n(KV-Cached Canvas)", "Reflex Step-1\n(Full End-to-End)", "AR Baseline\n(gemma4:12b)", "Fixed Diffusion\n(20-step / 256tok)"]
    p50_vals = [step1_canvas_4tok, float(np.median(probe_steady_latencies)), ar_p50, fixed_diff_p50]
    p95_vals = [canvas_p95s[0], float(np.percentile(probe_steady_latencies, 95)), ar_p95, fixed_diff_p95]
    colors = ["#1b9e77", "#2ca02c", "#d95f02", "#7570b3"]

    x = np.arange(len(paradigms))
    width = 0.35
    b1 = ax1.bar(x - width/2, p50_vals, width, label="p50 Latency (ms)", color=colors, alpha=0.85)
    b2 = ax1.bar(x + width/2, p95_vals, width, label="p95 Latency (ms)", color=colors, alpha=0.45, hatch="//")

    ax1.set_ylabel("Latency (ms, log-scale)", fontsize=12, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_title("A. Latency Distribution across Paradigms (Real Hardware)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(paradigms, fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    for bar, val in zip(b1, p50_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val * 1.15, f"{val:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: Canvas Length Latency Scaling (Real Hardware GB10)
    ax2 = axes[0, 1]
    ax2.plot(canvas_lens, canvas_means, marker="o", linewidth=2.5, color="#1f77b4", label="1-Step Forward Pass (BF16)")
    ax2.fill_between(canvas_lens, canvas_p50s, canvas_p95s, alpha=0.2, color="#1f77b4", label="p50–p95 Range")

    # Annotate speedup
    ax2.annotate(f"4-Token Micro-Canvas:\n{canvas_means[0]:.1f} ms\n({canvas_means[3]/canvas_means[0]:.2f}x faster than 256)",
                 xy=(4, canvas_means[0]), xytext=(30, 35),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="#1f77b4", lw=1.5),
                 fontweight="bold")

    ax2.annotate(f"256-Token Generative Canvas:\n{canvas_means[3]:.1f} ms",
                 xy=(256, canvas_means[3]), xytext=(150, 70),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff2e6", ec="#d95f02", lw=1.5))

    ax2.set_xlabel("Canvas Length $K$ (Tokens)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Step-1 Latency (ms)", fontsize=12, fontweight="bold")
    ax2.set_title("B. Physical Canvas Latency Scaling on NVIDIA GB10", fontsize=13, fontweight="bold")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(canvas_lens)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(loc="lower right")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Panel C: Step-1 Real Logits Calibration (Entropy vs Confidence Gap)
    ax3 = axes[1, 0]
    correct_samples = [s for s in probe_samples if s["is_correct"]]
    incorrect_samples = [s for s in probe_samples if not s["is_correct"]]

    ax3.scatter([s["confidence_gap"] for s in correct_samples], [s["entropy"] for s in correct_samples],
                color="#2ca02c", alpha=0.75, s=60, label=f"Correct Step-1 (n={len(correct_samples)})", edgecolors="k", lw=0.5)
    ax3.scatter([s["confidence_gap"] for s in incorrect_samples], [s["entropy"] for s in incorrect_samples],
                color="#d62728", alpha=0.75, s=60, marker="x", label=f"Incorrect Step-1 (n={len(incorrect_samples)})", lw=1.5)

    ax3.axhline(0.35, color="gray", linestyle=":", label="Low Entropy Threshold")
    ax3.axvline(0.70, color="purple", linestyle="--", label="Conformal Exit Boundary")

    ax3.set_xlabel("Prophet Confidence Gap ($|P(\\text{yes}) - P(\\text{no})|$)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Shannon Entropy $H_1$", fontsize=12, fontweight="bold")
    ax3.set_title("C. Real Step-1 Uncertainty Landscape on DiffusionGemma 26B", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right")
    ax3.grid(True, ls="--", alpha=0.5)

    # Panel D: Empirical Pareto Frontier (Latency vs Error Rate)
    ax4 = axes[1, 1]
    
    # Points on Pareto frontier
    labels = [
        "Reflex Fast-Path (4-tok Canvas)",
        "Reflex Full Probe (Prompt+Canvas)",
        "AR Baseline (gemma4:12b)",
        "Fixed 20-Step Diffusion"
    ]
    lat_points = [step1_canvas_4tok, float(np.mean(probe_steady_latencies)), ar_mean, fixed_diff_mean]
    # Error rates
    err_points = [46.0, 46.0, 100.0 - ar_acc, 12.0]
    syntax_errs = [0.0, 0.0, ar_syntax_err, 0.0]

    ax4.set_ylim(5, 52)
    custom_offsets = [
        (lat_points[0] * 1.3, 44),
        (lat_points[1] * 0.4, 40),
        (lat_points[2] * 0.4, 14),
        (lat_points[3] * 0.5, 8)
    ]
    for i in range(len(labels)):
        ax4.scatter(lat_points[i], err_points[i], s=200, color=colors[i], edgecolors="black", zorder=5, label=labels[i])
        ax4.annotate(f"{labels[i]}\n({lat_points[i]:.1f}ms, {err_points[i]:.1f}% err)",
                     xy=(lat_points[i], err_points[i]),
                     xytext=custom_offsets[i],
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], lw=1.5))

    # Highlight theoretical Reflex Conformal Gated Envelope
    # As conformal exit threshold varies from 0% exit (100% AR fallback) to high exit
    exit_fractions = np.linspace(0.0, 0.8, 50)
    # Blended latency = exit_frac * reflex_fast_lat + (1 - exit_frac) * ar_mean
    # Blended error = selective error under gate
    blended_lats = exit_fractions * step1_canvas_4tok + (1.0 - exit_fractions) * ar_mean
    blended_errs = exit_fractions * 10.0 + (1.0 - exit_fractions) * (100.0 - ar_acc) # with 10% risk bound
    ax4.plot(blended_lats, blended_errs, color="#1b9e77", linestyle="-.", linewidth=2, label=r"Reflex Operating Curve ($\epsilon \leq 10\%$)")

    ax4.set_xlabel("Mean Latency (ms, log-scale)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Task Error Rate (%)", fontsize=12, fontweight="bold")
    ax4.set_xscale("log")
    ax4.set_title("D. Empirical Pareto Frontier: Latency vs. Error Rate", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper left")
    ax4.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    out_fig = os.path.join(EXPERIMENTS_DIR, "real_pareto_frontier.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved publication Pareto plot to {out_fig}")


if __name__ == "__main__":
    main()
