#!/usr/bin/env python3
"""
Test loading real DiffusionGemma weights on NVIDIA GB10 Blackwell GPU.
Executes a real forward pass and measures latency with torch.cuda.Event.
"""

import os
import sys
import time
import torch

MODEL_PATH = "/home/tawana/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b"

print("=" * 70)
print("TESTING REAL DIFFUSIONGEMMA WEIGHT LOADING ON DGX SPARK (GB10)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Target device: {device} ({torch.cuda.get_device_name(0)})")

# Check memory before loading
free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
print(f"CUDA memory before loading: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

print(f"\n1. Loading Tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"Tokenizer loaded. Mask token ID: {tokenizer.mask_token_id}, Vocab size: {tokenizer.vocab_size}")

print(f"\n2. Loading Model weights into bfloat16 onto {device}...")
t0 = time.time()
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()
t_load = time.time() - t0
print(f"Model loaded successfully in {t_load:.2f}s!")

allocated = torch.cuda.memory_allocated() / (1024**3)
reserved = torch.cuda.memory_reserved() / (1024**3)
print(f"VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

print("\n3. Testing Single-Step Forward Pass on Micro-Control Canvas...")
prompt = "Passage: Water freezes at 0 degrees Celsius and 32 degrees Fahrenheit.\nQuestion: does water freeze at 32 degrees fahrenheit?\nAnswer:"
prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)

mask_id = tokenizer.mask_token_id or 4
# Micro-canvas: [ <mask , ]
canvas_tokens = torch.tensor([[tokenizer.encode("[", add_special_tokens=False)[0], mask_id, tokenizer.encode("]", add_special_tokens=False)[0]]], device=device)

# Measure real latency with CUDA Events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.inference_mode():
    # Warmup pass
    _ = model(
        input_ids=prompt_tokens.input_ids,
        attention_mask=prompt_tokens.attention_mask,
        decoder_input_ids=canvas_tokens,
    )
    torch.cuda.synchronize()

    # Timed pass
    start_event.record()
    outputs = model(
        input_ids=prompt_tokens.input_ids,
        attention_mask=prompt_tokens.attention_mask,
        decoder_input_ids=canvas_tokens,
    )
    end_event.record()
    torch.cuda.synchronize()

latency_ms = start_event.elapsed_time(end_event)
print(f"Step-1 Forward Pass Wall-Clock Latency: {latency_ms:.2f} ms")

logits = outputs.logits  # shape: (1, canvas_len, vocab_size)
mask_logits = logits[0, 1]  # at slot index 1 (the mask position)

yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("no", add_special_tokens=False)[0]

cand_logits = torch.tensor([mask_logits[yes_id].item(), mask_logits[no_id].item()])
probs = torch.softmax(cand_logits, dim=-1)

print(f"\nCandidate readout at [MASK] slot:")
print(f"  P('yes' | step=1): {probs[0].item():.4f} (logit: {mask_logits[yes_id].item():.2f})")
print(f"  P('no'  | step=1): {probs[1].item():.4f} (logit: {mask_logits[no_id].item():.2f})")
pred = "yes" if probs[0] > probs[1] else "no"
print(f"  Predicted Label: {pred} (Ground Truth: yes)")
print("=" * 70)

