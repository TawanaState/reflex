#!/usr/bin/env python3
"""
Experiment 01 (Real): Zero-Training Feasibility Spike on Real Model Weights
Executes real tensor forward passes of DiffusionGemma 26B/A4B on 100 real samples from BoolQ.
Measures real accuracy, real Shannon entropy, real Prophet confidence gap,
real Brier score, and real CUDA event wall-clock latencies.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "/home/tawana/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"

def run_real_spike(num_samples: int = 100):
    print("=" * 75)
    print(f"PROJECT REFLEX: REAL ZERO-TRAINING FEASIBILITY SPIKE ({num_samples} SAMPLES)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0)})")

    from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

    print(f"\n1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mask_token_id = tokenizer.mask_token_id or 4
    print(f"Tokenizer loaded. Mask Token ID: {mask_token_id}")

    # Token IDs for 'yes' and 'no'
    # Test both with and without leading space to get exact token IDs
    yes_tokens = tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("no", add_special_tokens=False)
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    print(f"Candidate Token IDs: 'yes' -> {yes_token_id}, 'no' -> {no_token_id}")

    print(f"\n2. Loading Real Weights for DiffusionGemma 26B/A4B into bfloat16...")
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s!")
    print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    print(f"\n3. Loading {num_samples} Real Samples from Google BoolQ Validation Split...")
    dataset = load_dataset("google/boolq", split=f"validation[:{num_samples}]")

    open_bracket_id = tokenizer.encode("[", add_special_tokens=False)[0]
    close_bracket_id = tokenizer.encode("]", add_special_tokens=False)[0]
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # 4-token micro-canvas: [ [ , <mask , ] , <pad ]
    canvas_tokens = torch.tensor([[open_bracket_id, mask_token_id, close_bracket_id, pad_id]], device=device)
    mask_slot_idx = 1

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = []
    latencies = []
    correct_count = 0
    brier_scores = []
    entropies = []
    confidence_gaps = []

    print("\nExecuting Real Step-1 Forward Passes on GPU...")
    print(f"{'Idx':<4} | {'Prediction':<10} | {'GT':<6} | {'P(yes)':<8} | {'P(no)':<8} | {'Entropy':<8} | {'Gap':<8} | {'Latency':<9} | {'Correct':<7}")
    print("-" * 85)

    for i, item in enumerate(dataset):
        passage = item["passage"]
        question = item["question"]
        gt_bool = item["answer"]
        gt_label = "yes" if gt_bool else "no"

        # Construct prompt
        prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer (yes or no):"
        prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.inference_mode():
            torch.cuda.synchronize()
            start_event.record()

            outputs = model(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
                decoder_input_ids=canvas_tokens,
            )

            end_event.record()
            torch.cuda.synchronize()

        lat_ms = start_event.elapsed_time(end_event)
        latencies.append(lat_ms)

        # Extract logits at [MASK] position (slot 1)
        logits = outputs.logits[0, mask_slot_idx]
        yes_logit = logits[yes_token_id].item()
        no_logit = logits[no_token_id].item()

        cand_logits = torch.tensor([yes_logit, no_logit], dtype=torch.float32)
        probs = torch.softmax(cand_logits, dim=-1)
        p_yes = probs[0].item()
        p_no = probs[1].item()

        pred_label = "yes" if p_yes >= p_no else "no"
        conf = max(p_yes, p_no)
        gap = abs(p_yes - p_no)

        eps = 1e-12
        entropy = - (p_yes * np.log(p_yes + eps) + p_no * np.log(p_no + eps))

        is_correct = (pred_label == gt_label)
        if is_correct:
            correct_count += 1

        y_true = 1.0 if gt_label == "yes" else 0.0
        brier = (p_yes - y_true)**2 + (p_no - (1.0 - y_true))**2
        brier_scores.append(brier)
        entropies.append(entropy)
        confidence_gaps.append(gap)

        results.append({
            "idx": i,
            "question": question,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "p_yes": p_yes,
            "p_no": p_no,
            "confidence": conf,
            "entropy": float(entropy),
            "confidence_gap": float(gap),
            "brier_score": float(brier),
            "latency_ms": float(lat_ms),
            "is_correct": bool(is_correct),
        })

        if (i + 1) % 10 == 0 or i == num_samples - 1:
            print(f"{i+1:<4} | {pred_label:<10} | {gt_label:<6} | {p_yes:<8.3f} | {p_no:<8.3f} | {entropy:<8.3f} | {gap:<8.3f} | {lat_ms:<7.1f}ms | {str(is_correct):<7}")

    # Summary Statistics
    n = len(results)
    acc = (correct_count / n) * 100.0
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    mean_entropy = float(np.mean(entropies))
    mean_brier = float(np.mean(brier_scores))
    mean_gap = float(np.mean(confidence_gaps))

    print("\n" + "=" * 75)
    print("REAL STEP-1 EMPIRICAL RESULTS SUMMARY ON DIFFUSIONGEMMA 26B/A4B")
    print("=" * 75)
    print(f"Total Samples Evaluated:      {n}")
    print(f"Top-1 Accuracy:               {acc:.2f}% (Random Baseline: 50.0%)")
    print(f"Mean Wall-Clock Latency:      {mean_lat:.2f} ms (Target: <150ms)")
    print(f"p50 Latency:                  {p50_lat:.2f} ms")
    print(f"p95 Latency:                  {p95_lat:.2f} ms")
    print(f"Mean Shannon Entropy (H1):    {mean_entropy:.4f}")
    print(f"Mean Prophet Confidence Gap:  {mean_gap:.4f}")
    print(f"Mean Multi-Class Brier Score: {mean_brier:.4f}")
    print("=" * 75)

    # Conformal Risk Calibration on Real Model Outputs
    print("\nEvaluating Conformal Risk Calibration on Real Logits...")
    conf_array = np.array([r["confidence"] for r in results])
    pred_arr = np.array([1 if r["prediction"] == "yes" else 0 for r in results])
    gt_arr = np.array([1 if r["ground_truth"] == "yes" else 0 for r in results])

    half = n // 2
    cal_conf, test_conf = conf_array[:half], conf_array[half:]
    cal_pred, test_pred = pred_arr[:half], pred_arr[half:]
    cal_gt, test_gt = gt_arr[:half], gt_arr[half:]

    from src.risk_gate import ConformalRiskGate
    gate = ConformalRiskGate(epsilon=0.10, delta=0.05)
    cal_thresh = gate.calibrate(cal_conf, cal_pred, cal_gt)

    test_exit_mask = test_conf >= cal_thresh
    n_test_exit = int(np.sum(test_exit_mask))
    test_coverage = (n_test_exit / len(test_conf)) * 100.0
    test_errors = (test_pred != test_gt).astype(float)
    test_risk = float(np.sum(test_errors[test_exit_mask]) / max(1, n_test_exit)) * 100.0

    print(f"Calibrated Threshold (1 - lambda*): {cal_thresh:.3f}")
    print(f"Test Fast-Path Coverage:            {test_coverage:.1f}%")
    print(f"Test Selective Error Rate:          {test_risk:.1f}% (Guaranteed <= 10.0%)")

    # Save to disk
    out_path = os.path.join(os.path.dirname(__file__), "real_step1_probe_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": {
                "total_samples": n,
                "top1_accuracy": acc,
                "mean_latency_ms": mean_lat,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "mean_entropy": mean_entropy,
                "mean_brier_score": mean_brier,
                "mean_confidence_gap": mean_gap,
                "conformal_threshold": cal_thresh,
                "conformal_test_coverage": test_coverage,
                "conformal_test_risk": test_risk,
            },
            "samples": results,
        }, f, indent=2)
    print(f"\nReal experimental results saved to {out_path}")


if __name__ == "__main__":
    run_real_spike(num_samples=100)

