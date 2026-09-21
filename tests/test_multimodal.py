#!/usr/bin/env python3
"""
Test Suite 3: Multimodal Vision Input Processing.
Validates that base64 image payloads in OpenAI format are routed
into DiffusionGemma's vision tower and processed without runtime crashes.
"""

import base64
import io
import os
import sys
import time
import requests
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_settings


def create_test_image_base64() -> str:
    """Creates a small 64x64 PNG image and returns data URL."""
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def test_multimodal_vision():
    settings = get_settings()
    base_url = f"http://{settings.HOST}:{settings.PORT}"
    endpoint = f"{base_url}/v1/chat/completions"

    print("=" * 75)
    print("TEST: MULTIMODAL VISION PAYLOAD INGESTION")
    print(f"Target: {endpoint}")
    print("=" * 75)

    data_url = create_test_image_base64()

    payload = {
        "model": settings.MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the primary color shown in this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 64,
    }

    t0 = time.perf_counter()
    resp = requests.post(endpoint, json=payload, timeout=60)
    wall_lat = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200, f"HTTP Error {resp.status_code}: {resp.text}"
    data = resp.json()

    choice = data["choices"][0]
    content = choice["message"].get("content")
    meta = data.get("reflex_metadata", {})

    print(f"\nMultimodal Response Received in {wall_lat:.1f}ms (Engine: {meta.get('latency_ms')}ms):")
    print(f"  Execution Path: {meta.get('execution_path')}")
    print(f"  Generated Sample: {repr(content[:100]) if content else 'None'}")

    assert choice.get("finish_reason") in ["stop", "tool_calls"]
    print("\nSUCCESS: Vision encoder tower processed image tensor cleanly.")
    print("=" * 75)


if __name__ == "__main__":
    test_multimodal_vision()

