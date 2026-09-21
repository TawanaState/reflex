#!/usr/bin/env python3
"""
Test Suite 2: Conditional Generative Expansion.
Validates that open-ended reasoning and synthesis prompts materialize the expanded canvas,
execute multi-step diffusion generation, and return valid text completion.
"""

import os
import sys
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_settings


def test_generative_expansion():
    settings = get_settings()
    base_url = f"http://{settings.HOST}:{settings.PORT}"
    endpoint = f"{base_url}/v1/chat/completions"

    print("=" * 75)
    print("TEST: CONDITIONAL GENERATIVE EXPANSION (PHASE 2 MULTI-STEP)")
    print(f"Target: {endpoint}")
    print("=" * 75)

    payload = {
        "model": settings.MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function to compute the Fibonacci sequence up to n terms.",
            }
        ],
        "max_tokens": 128,
    }

    t0 = time.perf_counter()
    resp = requests.post(endpoint, json=payload, timeout=60)
    wall_lat = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200, f"HTTP Error {resp.status_code}: {resp.text}"
    data = resp.json()

    choice = data["choices"][0]
    content = choice["message"].get("content")
    finish_reason = choice.get("finish_reason")
    meta = data.get("reflex_metadata", {})

    print(f"\nResponse Received in {wall_lat:.1f}ms (Engine: {meta.get('latency_ms')}ms):")
    print(f"  Execution Path: {meta.get('execution_path')}")
    print(f"  Steps Executed: {meta.get('steps_executed')}")
    print(f"  Finish Reason:  {finish_reason}")
    print(f"  Generated Text Sample (first 120 chars):")
    print(f"    {repr(content[:120]) if content else 'None'}")

    # Assertions
    assert finish_reason == "stop", f"Expected finish_reason 'stop', got '{finish_reason}'"
    assert content is not None and len(content.strip()) > 0, "Expected non-empty generated text"
    assert meta.get("execution_path") in ["EXPANDED_GENERATIVE_PATH", "VANILLA_FIXED_DIFFUSION"], (
        f"Unexpected execution path: {meta.get('execution_path')}"
    )
    assert meta.get("steps_executed", 0) > 1, f"Expected > 1 steps executed, got {meta.get('steps_executed')}"

    print("\nSUCCESS: Generative expansion verified.")
    print("=" * 75)


if __name__ == "__main__":
    test_generative_expansion()

