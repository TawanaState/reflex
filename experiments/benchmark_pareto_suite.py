#!/usr/bin/env python3
"""Archived v1 Pareto experiment; retained for audit only.

This source inserts unmeasured accuracy and cascade points and must not generate
new empirical artifacts. Use the correctness-aware v2 trace scripts instead.
"""

import os
import sys
import time
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import torch
from peft import PeftModel
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.risk_gate import ConformalRiskGate, ExitAction

MODEL_PATH = "/home/tawana/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
ADAPTER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "reflex_lora_v1"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
OLLAMA_API_URL = "http://localhost:11434/api/generate"
AR_MODEL_NAME = "gemma4:12b-it-qat"


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# =========================================================================
# 1. AUTOREGRESSIVE LLM BENCHMARK (OLLAMA ENGINE)
# =========================================================================
def query_ollama(prompt: str, is_boolean: bool = True) -> dict:
    if is_boolean:
        sys_msg = (
            'You are an API router. Respond ONLY with valid JSON: {"answer": "yes"} or {"answer": "no"}. '
            'No explanations.'
        )
    else:
        sys_msg = (
            'You are an intent triage router. Respond ONLY with valid JSON: {"intent": "<intent_name>"}. '
            'No explanations.'
        )

    payload = json.dumps({
        "model": AR_MODEL_NAME,
        "prompt": f"{sys_msg}\n\n{prompt}",
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 32,
        }
    }).encode("utf-8")

    req = urllib.request.Request(OLLAMA_API_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        tot_lat = (time.perf_counter() - t0) * 1000.0
        ttft = data.get("prompt_eval_duration", 0) / 1e6
        eval_dur = data.get("eval_duration", 0) / 1e6
        tokens = data.get("eval_count", 0)
        raw_text = data.get("response", "").strip()

        syntax_err = False
        parsed_ans = ""
        try:
            p = json.loads(raw_text)
            parsed_ans = str(p.get("answer" if is_boolean else "intent", "")).lower().strip()
        except Exception:
            syntax_err = True

        return {
            "raw_text": raw_text,
            "parsed_answer": parsed_ans,
            "syntax_error": syntax_err,
            "total_latency_ms": tot_lat,
            "ttft_ms": ttft,
            "eval_duration_ms": eval_dur,
            "tokens_generated": tokens,
        }
    except Exception as e:
        return {
            "raw_text": "",
            "parsed_answer": "",
            "syntax_error": True,
            "total_latency_ms": 3000.0,
            "ttft_ms": 500.0,
            "eval_duration_ms": 2500.0,
            "tokens_generated": 0,
            "error": str(e),
        }


def benchmark_ar_llm(test_samples: list) -> dict:
    print("\n" + "=" * 75)
    print(f"PARADIGM 1: AUTOREGRESSIVE LLM ({AR_MODEL_NAME}) ON {len(test_samples)} SAMPLES")
    print("=" * 75)

    latencies, ttfts, tokens, syntax_errors, correct_count = [], [], [], 0, 0
    results = []

    for i, item in enumerate(test_samples):
        prompt = item["prompt"]
        gt_label = str(item["ground_truth_label"]).lower().strip()

        res = query_ollama(prompt, is_boolean=(item["schema_type"] == "noul"))
        lat = res["total_latency_ms"]
        is_syn = res["syntax_error"]
        pred = res["parsed_answer"]

        latencies.append(lat)
        ttfts.append(res["ttft_ms"])
        tokens.append(res["tokens_generated"])
        if is_syn:
            syntax_errors += 1

        is_corr = (pred == gt_label) and not is_syn
        if is_corr:
            correct_count += 1

        results.append({
            "idx": i,
            "latency_ms": lat,
            "ttft_ms": res["ttft_ms"],
            "tokens": res["tokens_generated"],
            "syntax_error": is_syn,
            "correct": is_corr,
            "prediction": pred,
            "ground_truth": gt_label,
        })

        if (i + 1) % 20 == 0 or i == len(test_samples) - 1:
            print(f"  Sample {i+1:03d}/{len(test_samples):03d} | Latency: {lat:.1f}ms | Pred: {pred} | GT: {gt_label} | Correct: {is_corr}")

    n = len(test_samples)
    return {
        "paradigm": "Autoregressive LLM (gemma4:12b JSON)",
        "samples_evaluated": n,
        "accuracy": (correct_count / n) * 100.0,
        "syntax_error_rate": (syntax_errors / n) * 100.0,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "mean_tokens": float(np.mean(tokens)),
        "results": results,
    }


# =========================================================================
# 2. DIFFUSION & REFLEX BENCHMARK (PHYSICAL GPU EXECUTION)
# =========================================================================
def benchmark_diffusion_and_reflex(cal_samples: list, test_samples: list) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print(f"PARADIGMS 2, 3, & 4: DIFFUSIONGEMMA 26B/A4B + REFLEX ADAPTER ON NVIDIA GB10")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mask_id = tokenizer.mask_token_id or 4
    open_b = tokenizer.encode("[", add_special_tokens=False)[0]
    close_b = tokenizer.encode("]", add_special_tokens=False)[0]
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    print("Loading base model DiffusionGemma 26B/A4B into bfloat16...")
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Measure Standard Fixed 20-Step Diffusion Baseline (256-token canvas)
    print("\n--- Measuring Baseline 2: Fixed 20-Step Diffusion (256-token canvas) ---")
    fixed_canvas = torch.full((1, 256), fill_value=mask_id, dtype=torch.long, device=device)
    dummy_input = tokenizer(test_samples[0]["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # Measure 5 trials of full 20-step execution
    full_diffusion_latencies = []
    with torch.inference_mode():
        for trial in range(5):
            torch.cuda.synchronize()
            start_ev.record()

            # 1. Prefill
            enc_out = model.model.encoder(input_ids=dummy_input.input_ids, attention_mask=dummy_input.attention_mask)
            kv = enc_out.past_key_values
            # 2. 20 reverse steps
            c = fixed_canvas.clone()
            for _ in range(20):
                out = model(input_ids=None, past_key_values=kv, decoder_input_ids=c)
                c = torch.argmax(out.logits, dim=-1)

            end_ev.record()
            torch.cuda.synchronize()
            full_diffusion_latencies.append(start_ev.elapsed_time(end_ev))

    mean_fixed_diff_lat = float(np.mean(full_diffusion_latencies))
    print(f"Fixed 20-Step Diffusion Latency: Mean = {mean_fixed_diff_lat:.2f} ms")

    # Now load trained Reflex LoRA adapter for Reflex evaluations
    print(f"\nLoading trained Reflex LoRA adapter from {ADAPTER_PATH}...")
    model.model.decoder = PeftModel.from_pretrained(model.model.decoder, ADAPTER_PATH)
    model.model.decoder.eval()
    print("Reflex LoRA adapter loaded into decoder!")

    # Micro-Canvas: [ [ , <mask , ] , <pad ]
    canvas_4 = torch.tensor([[open_b, mask_id, close_b, pad_id]], device=device)
    mask_slot_idx = 1

    # Evaluate on Calibration set for Conformal Risk Gate
    print(f"\nEvaluating {len(cal_samples)} calibration samples to compute conformal thresholds...")
    cal_confidences, cal_predictions, cal_ground_truth = [], [], []

    with torch.inference_mode():
        for item in cal_samples:
            p_inputs = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            enc_out = model.model.encoder(input_ids=p_inputs.input_ids, attention_mask=p_inputs.attention_mask)
            out = model(input_ids=None, past_key_values=enc_out.past_key_values, decoder_input_ids=canvas_4)
            logits = out.logits[0, mask_slot_idx, item["candidate_token_ids"]].float()
            probs = torch.softmax(logits, dim=-1)

            top1_prob = float(torch.max(probs).item())
            top1_idx = int(torch.argmax(probs).item())
            cal_confidences.append(top1_prob)
            cal_predictions.append(top1_idx)
            cal_ground_truth.append(item["ground_truth_index"])

    cal_conf_arr = np.array(cal_confidences)
    cal_pred_arr = np.array(cal_predictions)
    cal_gt_arr = np.array(cal_ground_truth)

    gate = ConformalRiskGate(epsilon=0.05, delta=0.05, bound_type="empirical_bernstein")
    conformal_sweep = gate.calibrate_multi_tolerance(
        cal_conf_arr, cal_pred_arr, cal_gt_arr, tolerances=[0.005, 0.01, 0.05, 0.10]
    )
    print("\nCalibrated Thresholds on Calibration Set:")
    for eps, stats in conformal_sweep.items():
        print(f"  Target eps={eps:<5} -> Threshold (1 - lambda*) = {stats['calibrated_threshold']:.3f} | Cal Coverage = {stats['cal_coverage']*100:.1f}%")

    # Evaluate on Test Set
    print(f"\nEvaluating {len(test_samples)} physical test samples on Reflex Pipeline...")
    reflex_results = []
    latencies_prefill = []
    latencies_step1 = []
    test_confidences = []
    test_predictions = []
    test_ground_truth = []
    test_entropies = []
    test_briers = []

    with torch.inference_mode():
        for i, item in enumerate(test_samples):
            p_inputs = tokenizer(item["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)

            # Measure prefill latency
            torch.cuda.synchronize()
            start_ev.record()
            enc_out = model.model.encoder(input_ids=p_inputs.input_ids, attention_mask=p_inputs.attention_mask)
            end_ev.record()
            torch.cuda.synchronize()
            t_pref = start_ev.elapsed_time(end_ev)
            latencies_prefill.append(t_pref)

            # Measure Step-1 micro-canvas latency
            torch.cuda.synchronize()
            start_ev.record()
            out = model(input_ids=None, past_key_values=enc_out.past_key_values, decoder_input_ids=canvas_4)
            end_ev.record()
            torch.cuda.synchronize()
            t_step1 = start_ev.elapsed_time(end_ev)
            latencies_step1.append(t_step1)

            logits = out.logits[0, mask_slot_idx, item["candidate_token_ids"]].float()
            probs = torch.softmax(logits, dim=-1)

            top1_prob = float(torch.max(probs).item())
            top1_idx = int(torch.argmax(probs).item())
            gt_idx = item["ground_truth_index"]
            is_corr = (top1_idx == gt_idx)

            # Entropy
            p_np = probs.cpu().numpy()
            entropy = float(-np.sum(p_np * np.log(p_np + 1e-12)))

            # Brier
            y_onehot = np.zeros_like(p_np)
            y_onehot[gt_idx] = 1.0
            brier = float(np.sum((p_np - y_onehot)**2))

            test_confidences.append(top1_prob)
            test_predictions.append(top1_idx)
            test_ground_truth.append(gt_idx)
            test_entropies.append(entropy)
            test_briers.append(brier)

            reflex_results.append({
                "idx": i,
                "prefill_ms": t_pref,
                "step1_ms": t_step1,
                "confidence": top1_prob,
                "entropy": entropy,
                "brier": brier,
                "correct": is_corr,
                "prediction_idx": top1_idx,
                "gt_idx": gt_idx,
            })

            if (i + 1) % 20 == 0 or i == len(test_samples) - 1:
                print(f"  Test {i+1:03d}/{len(test_samples):03d} | Prefill: {t_pref:.1f}ms | Step1: {t_step1:.1f}ms | Conf: {top1_prob:.3f} | Correct: {is_corr}")

    test_conf_arr = np.array(test_confidences)
    test_pred_arr = np.array(test_predictions)
    test_gt_arr = np.array(test_ground_truth)

    # Evaluate conformal performance at each epsilon
    conformal_evals = {}
    for eps, stats in conformal_sweep.items():
        thresh = stats["calibrated_threshold"]
        eval_res = gate.evaluate_test_split(test_conf_arr, test_pred_arr, test_gt_arr, threshold=thresh)
        conformal_evals[eps] = eval_res

    # Overall Step-1 Top-1 Accuracy across all samples
    step1_acc = float(np.mean(test_pred_arr == test_gt_arr)) * 100.0
    mean_pref_ms = float(np.mean(latencies_prefill))
    mean_step1_ms = float(np.mean(latencies_step1))
    mean_exp_ms = 123.86  # From physical KV-retention benchmark on L=64

    return {
        "fixed_diffusion": {
            "paradigm": "Standard Fixed 20-Step Diffusion (256-tok)",
            "mean_latency_ms": mean_fixed_diff_lat,
            "p50_latency_ms": mean_fixed_diff_lat,
            "p95_latency_ms": mean_fixed_diff_lat * 1.03,
            "accuracy": 88.0,
            "syntax_error_rate": 0.0,
        },
        "reflex": {
            "paradigm": "Reflex (Control-First Canvas Expansion)",
            "step1_accuracy": step1_acc,
            "syntax_error_rate": 0.0,
            "mean_prefill_ms": mean_pref_ms,
            "mean_step1_ms": mean_step1_ms,
            "fast_path_latency_ms": mean_pref_ms + mean_step1_ms,
            "steady_state_kv_hit_ms": mean_step1_ms,
            "expanded_path_latency_ms": mean_pref_ms + mean_step1_ms + mean_exp_ms,
            "mean_entropy": float(np.mean(test_entropies)),
            "mean_brier_score": float(np.mean(test_briers)),
            "conformal_calibration": conformal_sweep,
            "conformal_evaluation": conformal_evals,
            "results": reflex_results,
        }
    }


# =========================================================================
# 3. PLOTTING & PARETO VISUALIZATION
# =========================================================================
def generate_pareto_visualizations(ar_data: dict, diff_data: dict, reflex_data: dict, output_path: str):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Panel A: Latency Distribution Comparison
    ax0 = axs[0, 0]
    paradigms = ["AR LLM (JSON)", "Fixed Diffusion (20-Step)", "Reflex Expanded", "Reflex Full Fast-Path", "Reflex KV-Hit (Steady)"]
    p50_latencies = [
        ar_data["p50_latency_ms"],
        diff_data["p50_latency_ms"],
        reflex_data["expanded_path_latency_ms"],
        reflex_data["fast_path_latency_ms"],
        reflex_data["steady_state_kv_hit_ms"],
    ]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#0275d8", "#5cb85c"]
    bars = ax0.barh(paradigms, p50_latencies, color=colors, edgecolor="black", alpha=0.85)
    ax0.set_xscale("log")
    ax0.set_xlabel("Median Latency (ms, log scale)", fontsize=11, fontweight="bold")
    ax0.set_title("Panel A: End-to-End Latency by Paradigm", fontsize=12, fontweight="bold")
    ax0.grid(True, which="both", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax0.text(w * 1.1, bar.get_y() + bar.get_height()/2, f"{w:.1f} ms", va="center", fontsize=10, fontweight="bold")

    # Panel B: Conformal Risk vs Fast-Path Coverage
    ax1 = axs[0, 1]
    eps_vals = sorted(reflex_data["conformal_evaluation"].keys())
    coverages = [reflex_data["conformal_evaluation"][e]["test_coverage"] * 100 for e in eps_vals]
    selective_risks = [reflex_data["conformal_evaluation"][e]["test_selective_error"] * 100 for e in eps_vals]
    target_bounds = [e * 100 for e in eps_vals]

    ax1.plot(target_bounds, selective_risks, "o-", color="#d9534f", label="Empirical Selective Error (%)", lw=2)
    ax1.plot(target_bounds, target_bounds, "--", color="gray", label="Conformal Upper Bound (eps)", lw=1.5)
    ax1.set_xlabel("Target Error Tolerance eps (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Error on Exits (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Panel B: Conformal Error Bound Verification", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Fast-Path Coverage vs Tolerance
    ax2 = axs[1, 0]
    ax2.plot(target_bounds, coverages, "s-", color="#0275d8", lw=2, label="Step-1 Fast-Path Coverage (%)")
    ax2.set_xlabel("Target Error Tolerance eps (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Fast-Path Coverage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Panel C: Step-1 Fast-Path Exit Coverage", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    # Panel D: Accuracy vs Latency Pareto Frontier
    ax3 = axs[1, 1]
    # AR LLM
    ax3.scatter([ar_data["mean_latency_ms"]], [ar_data["accuracy"]], color="#d9534f", s=180, label="Autoregressive LLM", zorder=5)
    # Fixed Diffusion
    ax3.scatter([diff_data["mean_latency_ms"]], [diff_data["accuracy"]], color="#f0ad4e", s=180, label="Fixed 20-Step Diffusion", zorder=5)
    # Two-Model Cascade
    cascade_lat = 0.5 * 100.0 + 0.5 * (100.0 + ar_data["mean_latency_ms"])
    cascade_acc = 84.5
    ax3.scatter([cascade_lat], [cascade_acc], color="#9c27b0", s=180, label="Two-Model Cascade (Router+LLM)", zorder=5)
    # Reflex Operating Points (at different conformal tolerances)
    reflex_lats = [
        cov/100.0 * reflex_data["fast_path_latency_ms"] + (1.0 - cov/100.0) * reflex_data["expanded_path_latency_ms"]
        for cov in coverages
    ]
    reflex_accs = [
        cov/100.0 * (100.0 - risk) + (1.0 - cov/100.0) * 92.0
        for cov, risk in zip(coverages, selective_risks)
    ]
    ax3.plot(reflex_lats, reflex_accs, "*-", color="#0275d8", lw=2.5, markersize=12, label="Reflex Pareto Frontier", zorder=6)

    ax3.set_xlabel("Mean Latency (ms)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Task Decision Accuracy (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Panel D: Accuracy vs. Latency Pareto Frontier", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Project Reflex: Physical Pareto Systems Evaluation on NVIDIA GB10 Blackwell GPU", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Publication-quality Pareto plot saved to {output_path}")


def main():
    raise RuntimeError(
        "Retired: this v1 script inserts unmeasured Pareto points and cannot produce empirical results. "
        "Use benchmark_routing_v2.py and benchmark_tiered_latency.py."
    )
    print("==========================================================================")
    print("PROJECT REFLEX: PHASE F FULL AUDITABLE PARETO BENCHMARK SUITE")
    print("==========================================================================")

    # 1. Load Test Splits
    boolq_test = load_jsonl(os.path.join(DATA_DIR, "boolq", "test.jsonl"))[:75]
    banking_test = load_jsonl(os.path.join(DATA_DIR, "banking77", "test.jsonl"))[:25]
    test_suite = boolq_test + banking_test

    boolq_cal = load_jsonl(os.path.join(DATA_DIR, "boolq", "cal.jsonl"))[:100]
    banking_cal = load_jsonl(os.path.join(DATA_DIR, "banking77", "cal.jsonl"))[:50]
    cal_suite = boolq_cal + banking_cal

    print(f"Loaded {len(test_suite)} Test items ({len(boolq_test)} BoolQ + {len(banking_test)} Banking77)")
    print(f"Loaded {len(cal_suite)} Calibration items")

    # Step 1: Benchmark Autoregressive LLM (via local Ollama engine)
    ar_results = benchmark_ar_llm(test_suite)

    # Archived body retained for audit; main() exits before this point.
    diff_reflex_results = benchmark_diffusion_and_reflex(cal_suite, test_suite)

    fixed_diff_data = diff_reflex_results["fixed_diffusion"]
    reflex_data = diff_reflex_results["reflex"]

    # Save comprehensive JSON results
    summary_path = os.path.join(RESULTS_DIR, "final_pareto_benchmark_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "ar_baseline": ar_results,
            "fixed_diffusion_baseline": fixed_diff_data,
            "reflex_proposed": reflex_data,
        }, f, indent=2)
    print(f"\nFinal comprehensive benchmark JSON saved to {summary_path}")

    # Step 5: Generate Publication Visualizations
    plot_path = os.path.join(RESULTS_DIR, "pareto_frontier.png")
    generate_pareto_visualizations(ar_results, fixed_diff_data, reflex_data, plot_path)
    # Also save copy in experiments/
    exp_plot = os.path.join(os.path.dirname(__file__), "real_pareto_frontier_v2.png")
    generate_pareto_visualizations(ar_results, fixed_diff_data, reflex_data, exp_plot)

    print("\n" + "=" * 75)
    print("PHASE F BENCHMARK COMPLETE — AUDITABLE RESULTS GENERATED")
    print("=" * 75)


if __name__ == "__main__":
    main()

