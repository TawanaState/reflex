#!/usr/bin/env python3
"""
Experiment: Multi-Task Calibrated Fine-Tuning (SFT / LoRA) for Project Reflex
Trains a lightweight parameter-efficient LoRA adapter on the bidirectional diffusion decoder
using the composite multi-task objective:
    L_Reflex = L_control + lambda_1 * L_diffusion + lambda_2 * L_Brier

Optimizes Step-1 micro-control slot accuracy across multi-domain datasets (BoolQ, Banking77, BFCL)
while penalizing probabilistic overconfidence via the Multi-Class Brier proper scoring rule.
"""

import os
import sys
import time
import math
import json
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion, get_cosine_schedule_with_warmup

# Ensure deterministic execution
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_PATH = "/home/tawana/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "reflex_lora_v1"))


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def train_reflex_lora(
    total_steps: int = 250,
    lr: float = 2e-4,
    grad_accum_steps: int = 4,
    lambda_brier: float = 1.0,
    lambda_diff: float = 0.5,
    eval_interval: int = 50,
):
    print("=" * 80)
    print("PROJECT REFLEX: MULTI-TASK CALIBRATED SFT / LoRA ADAPTER TRAINING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0)})")

    # 1. Load Tokenizer & Model
    print("\n[1/5] Loading Base DiffusionGemma 26B/A4B in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mask_token_id = tokenizer.mask_token_id or 4
    open_bracket_id = tokenizer.encode("[", add_special_tokens=False)[0]
    close_bracket_id = tokenizer.encode("]", add_special_tokens=False)[0]
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # 2. Freeze Base Weights & Ingest LoRA on Decoder
    print("\n[2/5] Injecting LoRA Adapters into Decoder Attention Projections...")
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model.model.decoder = get_peft_model(model.model.decoder, lora_config)
    model.model.decoder.print_trainable_parameters()
    print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 3. Load Datasets
    print("\n[3/5] Ingesting Multi-Task Training Corpora...")
    boolq_train = load_jsonl(os.path.join(DATA_DIR, "boolq", "train.jsonl"))
    banking_train = load_jsonl(os.path.join(DATA_DIR, "banking77", "train.jsonl"))
    bfcl_train = load_jsonl(os.path.join(DATA_DIR, "bfcl", "train.jsonl"))

    boolq_cal = load_jsonl(os.path.join(DATA_DIR, "boolq", "cal.jsonl"))[:100]
    banking_cal = load_jsonl(os.path.join(DATA_DIR, "banking77", "cal.jsonl"))[:100]

    # Create balanced multi-task training pool
    train_pool = []
    # Oversample smaller tasks for balanced representation
    train_pool.extend(boolq_train[:2000])
    train_pool.extend(banking_train[:2000])
    for _ in range(15):  # BFCL oversample to ~1800 instances
        train_pool.extend(bfcl_train)

    random.shuffle(train_pool)
    print(f"Total Combined Training Pool: {len(train_pool)} examples.")

    # Micro-Canvas: [ [ , <mask , ] , <pad ]
    canvas_tokens = torch.tensor([[open_bracket_id, mask_token_id, close_bracket_id, pad_id]], device=device)
    mask_pos = 1

    # 4. Setup Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5. Training Loop
    print("\n[4/5] Executing Calibrated Multi-Task Optimization...")
    print(f"{'Step':<6} | {'Loss':<8} | {'L_control':<9} | {'L_Brier':<8} | {'Accuracy':<9} | {'Brier':<8} | {'LR':<9} | {'GPU Mem':<8}")
    print("-" * 80)

    model.train()
    history = []
    accum_loss = 0.0
    accum_l_ce = 0.0
    accum_l_brier = 0.0
    running_correct = 0
    running_samples = 0
    running_brier_scores = []

    t_train_start = time.time()
    step = 0
    pool_idx = 0

    while step < total_steps:
        sample = train_pool[pool_idx % len(train_pool)]
        pool_idx += 1

        prompt_text = sample["prompt"]
        cand_token_ids = sample["candidate_token_ids"]
        target_token_id = sample["target_token_id"]
        gt_index = sample["ground_truth_index"]

        # Tokenize prompt
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 1. Forward pass: encoder prefill + step 1 decoder
        with torch.no_grad():
            enc_out = model.model.encoder(
                input_ids=prompt_inputs.input_ids,
                attention_mask=prompt_inputs.attention_mask,
            )
            prompt_kv = enc_out.past_key_values

        # Decoder pass computes gradients for LoRA
        out = model(
            input_ids=None,
            past_key_values=prompt_kv,
            decoder_input_ids=canvas_tokens,
        )

        logits = out.logits[0, mask_pos]  # (vocab_size,)
        cand_logits = logits[cand_token_ids].float()
        probs = torch.softmax(cand_logits, dim=-1)

        # 2. Objective Computation
        # L_control: Cross-Entropy over candidate tokens
        p_target = probs[gt_index]
        l_control = -torch.log(p_target + 1e-12)

        # L_Brier: Multi-Class Brier Proper Scoring Penalty
        # (p_k - y_k)^2 averaged over classes
        y_onehot = torch.zeros_like(probs)
        y_onehot[gt_index] = 1.0
        l_brier = torch.mean((probs - y_onehot)**2)

        # L_diffusion: Syntax loss on unmasked tokens [0, 2, 3]
        syntax_loss = 0.0
        if lambda_diff > 0:
            syntax_pos = [0, 2, 3]
            syntax_targets = torch.tensor([open_bracket_id, close_bracket_id, pad_id], device=device)
            syntax_logits = out.logits[0, syntax_pos].float()
            syntax_loss = nn.functional.cross_entropy(syntax_logits, syntax_targets)

        total_loss = l_control + lambda_brier * l_brier + lambda_diff * syntax_loss
        norm_loss = total_loss / grad_accum_steps
        norm_loss.backward()

        # Track stats
        accum_loss += total_loss.item()
        accum_l_ce += l_control.item()
        accum_l_brier += l_brier.item()

        pred_idx = torch.argmax(probs).item()
        is_corr = (pred_idx == gt_index)
        running_correct += int(is_corr)
        running_samples += 1

        brier_val = torch.sum((probs - y_onehot)**2).item()
        running_brier_scores.append(brier_val)

        if (pool_idx % grad_accum_steps) == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0 or step == total_steps:
                cur_loss = accum_loss / (10 * grad_accum_steps)
                cur_ce = accum_l_ce / (10 * grad_accum_steps)
                cur_br = accum_l_brier / (10 * grad_accum_steps)
                cur_acc = (running_correct / max(1, running_samples)) * 100.0
                cur_brier_score = float(np.mean(running_brier_scores))
                cur_lr = scheduler.get_last_lr()[0]
                cur_mem = torch.cuda.memory_allocated() / (1024**3)

                print(f"{step:<6} | {cur_loss:<8.4f} | {cur_ce:<9.4f} | {cur_br:<8.4f} | {cur_acc:<8.1f}% | {cur_brier_score:<8.4f} | {cur_lr:<9.2e} | {cur_mem:<6.2f}GB")

                history.append({
                    "step": step,
                    "loss": cur_loss,
                    "loss_control": cur_ce,
                    "loss_brier": cur_br,
                    "step1_accuracy": cur_acc,
                    "brier_score": cur_brier_score,
                    "learning_rate": cur_lr,
                })

                accum_loss = 0.0
                accum_l_ce = 0.0
                accum_l_brier = 0.0
                running_correct = 0
                running_samples = 0
                running_brier_scores = []

            # Validation evaluation
            if step % eval_interval == 0 or step == total_steps:
                model.eval()
                val_accs = []
                val_briers = []
                with torch.no_grad():
                    for v_item in boolq_cal[:30] + banking_cal[:30]:
                        v_p = tokenizer(v_item["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
                        enc = model.model.encoder(input_ids=v_p.input_ids, attention_mask=v_p.attention_mask)
                        v_out = model(input_ids=None, past_key_values=enc.past_key_values, decoder_input_ids=canvas_tokens)
                        v_logits = v_out.logits[0, mask_pos, v_item["candidate_token_ids"]].float()
                        v_probs = torch.softmax(v_logits, dim=-1)
                        v_pred = torch.argmax(v_probs).item()
                        val_accs.append(float(v_pred == v_item["ground_truth_index"]))
                        v_y = torch.zeros_like(v_probs)
                        v_y[v_item["ground_truth_index"]] = 1.0
                        val_briers.append(torch.sum((v_probs - v_y)**2).item())

                mean_v_acc = float(np.mean(val_accs)) * 100.0
                mean_v_br = float(np.mean(val_briers))
                print(f"  >>> [Validation @ Step {step}] Step-1 Accuracy: {mean_v_acc:.2f}% | Brier Score: {mean_v_br:.4f}")
                model.train()

    elapsed = time.time() - t_train_start
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE IN {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print("=" * 80)

    # 6. Save LoRA Adapter Checkpoint
    print(f"\n[5/5] Saving Trained Reflex LoRA Adapter Checkpoint to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.model.decoder.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training log and metadata
    log_file = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(log_file, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "training_time_seconds": elapsed,
            "final_step1_accuracy": history[-1]["step1_accuracy"] if history else 0.0,
            "final_brier_score": history[-1]["brier_score"] if history else 0.0,
            "history": history,
        }, f, indent=2)

    # Save also to experiments/
    exp_log = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "reflex_lora_training_log.json"))
    with open(exp_log, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Adapter saved successfully to {OUTPUT_DIR}")
    print(f"Training history saved to {exp_log}")


if __name__ == "__main__":
    train_reflex_lora(
        total_steps=200,
        lr=2e-4,
        grad_accum_steps=4,
        lambda_brier=1.0,
        lambda_diff=0.5,
        eval_interval=50,
    )

