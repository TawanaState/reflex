#!/usr/bin/env python3
"""
Model download script for google/diffusiongemma-26B-A4B-it.
Downloads model safetensors weights using huggingface_hub snapshot_download with multi-threading.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# Read HF_TOKEN from .env if not in os.environ
if "HF_TOKEN" not in os.environ and os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1].strip("\"'")

token = os.environ.get("HF_TOKEN")
print(f"Starting snapshot download for {MODEL_ID}...")
t0 = time.time()

try:
    model_dir = snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        max_workers=8,
        resume_download=True,
    )
    elapsed = time.time() - t0
    print(f"Successfully downloaded {MODEL_ID} to {model_dir} in {elapsed:.1f}s")
except Exception as e:
    print(f"Download failed with error: {e}", file=sys.stderr)
    sys.exit(1)

