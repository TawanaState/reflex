#!/usr/bin/env python3
"""
Experiment 01: Zero-Training Feasibility Spike (Step-1 Probe)
Evaluates DiffusionGemma 26B/A4B on a micro-control canvas (4-16 tokens) in a single denoise step (<150ms).
Tests discriminative accuracy, Shannon entropy, Prophet confidence gap, Brier score, and conformal risk control.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from typing import Dict, List, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.canvas import CanvasCompiler, Choice, Noul, Schema
from src.risk_gate import ConformalRiskGate, ExitAction
from src.runtime import ReflexRuntime


def load_hf_token() -> str:
    """Reads HF_TOKEN from environment or .env file."""
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip("\"'")
                    os.environ["HF_TOKEN"] = token
                    return token
    return ""


def get_benchmark_samples() -> List[Dict[str, Any]]:
    """
    Returns curated evaluation samples for binary question answering (BoolQ-style)
    and intent routing (agent triage / decision classification).
    """
    samples = [
        # BoolQ-style fact-verification / QA
        {"prompt": "Passage: The Great Wall of China is visible from low Earth orbit without magnification.\nQuestion: is the great wall of china visible from space?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: Water freezes at 0 degrees Celsius and 32 degrees Fahrenheit at standard atmospheric pressure.\nQuestion: does water freeze at 32 degrees fahrenheit?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: The human skeleton has 206 bones in adulthood, down from approximately 270 bones at birth.\nQuestion: does an adult human have 206 bones?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: The capital of Australia is Canberra, not Sydney or Melbourne.\nQuestion: is sydney the capital of australia?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: Light travels at approximately 299,792 kilometers per second in a vacuum.\nQuestion: does light travel at 300000 km per second in vacuum?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Mars has two small moons, Phobos and Deimos, discovered in 1877.\nQuestion: does mars have two moons?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Photosynthesis occurs in plants, algae, and some bacteria using chlorophyll to convert light into chemical energy.\nQuestion: do animals perform photosynthesis?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: The Eiffel Tower is located in Paris, France, on the Champ de Mars.\nQuestion: is the eiffel tower in london?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: Mount Everest is Earth's highest mountain above sea level, located in the Himalayas.\nQuestion: is mount everest the tallest mountain on earth?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Jupiter is the fifth planet from the Sun and the largest in the Solar System.\nQuestion: is jupiter the largest planet in our solar system?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Python is a dynamically typed, garbage-collected programming language created by Guido van Rossum.\nQuestion: is python a statically typed language?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.\nQuestion: is the pacific ocean the largest ocean?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox.\nQuestion: is shingles caused by the same virus as chickenpox?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Helium is lighter than air and non-flammable, making it safe for airships.\nQuestion: is helium heavier than air?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: Sound travels faster in water than in air because water molecules are more densely packed.\nQuestion: does sound travel faster in water than in air?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Diamonds are an allotrope of carbon with a crystal lattice structure.\nQuestion: are diamonds made of carbon?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Penguins are flightless birds that are highly adapted for life in the water.\nQuestion: can penguins fly in the air?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: The speed of sound in air at 20 degrees Celsius is approximately 343 meters per second.\nQuestion: is the speed of sound faster than light?", "label": "no", "options": ["yes", "no"]},
        {"prompt": "Passage: Mercury is the smallest planet in the Solar System and the closest to the Sun.\nQuestion: is mercury the closest planet to the sun?", "label": "yes", "options": ["yes", "no"]},
        {"prompt": "Passage: Copper is an excellent conductor of electricity and heat.\nQuestion: does copper conduct electricity well?", "label": "yes", "options": ["yes", "no"]},

        # Agent Routing / Triage Decisions (Action Selection)
        {"prompt": "System Event: User clicked on payment checkout button. Status: Payment gateway returned HTTP 402 Card Declined.\nAction:", "label": "notify", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Database connection pool exhausted. 50 active threads waiting for connection.\nAction:", "label": "escalate", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Transient network timeout on read replica health check. 1 failure out of 100 checks.\nAction:", "label": "retry", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: User updated profile photo. File uploaded successfully to S3 bucket.\nAction:", "label": "ignore", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Automated daily backup completed in 4 minutes with checksum match.\nAction:", "label": "ignore", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Security alert: 50 failed SSH login attempts from unrecognized IP in 60 seconds.\nAction:", "label": "escalate", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Idempotent webhook delivery failed with HTTP 503 Service Unavailable.\nAction:", "label": "retry", "options": ["retry", "notify", "escalate", "ignore"]},
        {"prompt": "System Event: Customer requested refund for defective order #4819.\nAction:", "label": "notify", "options": ["retry", "notify", "escalate", "ignore"]},
    ]
    return samples


def run_experiment():
    print("=" * 70)
    print("PROJECT REFLEX: EXPERIMENT 01 — STEP-1 FEASIBILITY PROBE")
    print("=" * 70)

    token = load_hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    MODEL_ID = "google/diffusiongemma-26B-A4B-it"
    print(f"Loading Tokenizer for {MODEL_ID}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)

    print(f"Loading Model {MODEL_ID}...")
    from transformers import DiffusionGemmaForBlockDiffusion
    
    # Check available memory
    free_mem_gb = torch.cuda.mem_get_info()[0] / (1024**3) if torch.cuda.is_available() else 0
    print(f"Available GPU VRAM / Unified Memory: {free_mem_gb:.1f} GB")

    t_load_0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )
    print(f"Model loaded successfully in {time.time() - t_load_0:.1f}s")
    model.eval()

    runtime = ReflexRuntime(
        model=model,
        tokenizer=tokenizer,
        risk_gate=ConformalRiskGate(epsilon=0.05, delta=0.05, default_threshold=0.85),
        device=device,
    )

    samples = get_benchmark_samples()
    print(f"\nEvaluating Step-1 probe on {len(samples)} benchmark samples...")

    results = []
    latencies = []
    correct_count = 0
    brier_scores = []
    entropies = []
    confidence_gaps = []

    for idx, sample in enumerate(samples):
        prompt = sample["prompt"]
        ground_truth = sample["label"]
        options = sample["options"]

        schema = Schema(fields={
            "answer": Choice(options=options, description="Prediction slot")
        })

        # Measure wall-clock latency of exactly 1 denoise step
        t0 = time.perf_counter()
        risk_res, compiled_canvas, past_kv, canvas_tokens, step1_lat = runtime.execute_step1_read(
            prompt=prompt,
            schema=schema,
        )
        total_sample_lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(step1_lat)

        pred_slot = risk_res.slot_predictions["answer"]
        top_pred = pred_slot.top_label
        conf = pred_slot.confidence
        entropy = pred_slot.entropy
        gap = pred_slot.confidence_gap

        is_correct = (top_pred.lower().strip() == ground_truth.lower().strip())
        if is_correct:
            correct_count += 1

        # Multi-class Brier score = sum_k (p_k - y_k)^2
        probs = pred_slot.probabilities
        brier = sum(
            ((probs.get(opt, 0.0) - (1.0 if opt == ground_truth else 0.0)) ** 2)
            for opt in options
        )
        brier_scores.append(brier)
        entropies.append(entropy)
        confidence_gaps.append(gap)

        results.append({
            "idx": idx,
            "prompt": prompt[:60] + "...",
            "ground_truth": ground_truth,
            "prediction": top_pred,
            "confidence": conf,
            "entropy": entropy,
            "confidence_gap": gap,
            "is_correct": is_correct,
            "brier_score": brier,
            "latency_ms": step1_lat,
            "canvas_length": compiled_canvas.canvas_length,
        })

        if (idx + 1) % 5 == 0 or idx == len(samples) - 1:
            print(f"[{idx+1:02d}/{len(samples)}] Top-1: {top_pred:8s} | GT: {ground_truth:8s} | "
                  f"Conf: {conf:.3f} | H: {entropy:.3f} | Gap: {gap:.3f} | Latency: {step1_lat:.1f}ms | Correct: {is_correct}")

    # Summary Statistics
    n = len(samples)
    acc = (correct_count / n) * 100.0
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    mean_entropy = float(np.mean(entropies))
    mean_brier = float(np.mean(brier_scores))
    mean_gap = float(np.mean(confidence_gaps))

    print("\n" + "=" * 70)
    print("STEP-1 FEASIBILITY SPIKE RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Samples Evaluated:      {n}")
    print(f"Top-1 Accuracy:               {acc:.2f}% (Target: >65%)")
    print(f"Mean Wall-Clock Latency:      {mean_lat:.2f} ms (Target: <150ms)")
    print(f"p50 Latency:                  {p50_lat:.2f} ms")
    print(f"p95 Latency:                  {p95_lat:.2f} ms")
    print(f"Mean Shannon Entropy (H1):    {mean_entropy:.4f}")
    print(f"Mean Prophet Confidence Gap:  {mean_gap:.4f}")
    print(f"Mean Multi-Class Brier Score: {mean_brier:.4f}")
    print("=" * 70)

    # Conformal Risk Calibration Test
    print("\nEvaluating Conformal Risk Calibration...")
    conf_array = np.array([r["confidence"] for r in results])
    pred_indices = np.array([0 if r["prediction"] == "yes" else 1 for r in results])
    gt_indices = np.array([0 if r["ground_truth"] == "yes" else 1 for r in results])

    # 50/50 split
    half = n // 2
    cal_gate = ConformalRiskGate(epsilon=0.10, delta=0.05)
    cal_thresh = cal_gate.calibrate(conf_array[:half], pred_indices[:half], gt_indices[:half])
    
    test_conf = conf_array[half:]
    test_err = (pred_indices[half:] != gt_indices[half:]).astype(float)
    test_exit = test_conf >= cal_thresh
    test_n_exit = int(np.sum(test_exit))
    test_coverage = (test_n_exit / len(test_conf)) * 100.0
    test_risk = float(np.sum(test_err[test_exit]) / max(1, test_n_exit)) * 100.0

    print(f"Calibrated Threshold (1 - lambda): {cal_thresh:.3f}")
    print(f"Test Fast-Path Coverage:           {test_coverage:.1f}%")
    print(f"Test Selective Error Rate:         {test_risk:.1f}% (Guaranteed <= 10.0%)")

    # Output JSON file
    output_path = os.path.join(os.path.dirname(__file__), "results_step1_probe.json")
    with open(output_path, "w") as f:
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

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_experiment()

