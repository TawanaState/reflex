#!/usr/bin/env python3
"""
Test Suite 1: Native Tool-Calling Routing.
Validates that single-turn atomic tool invocations return valid OpenAI
tool_calls through the native tool-calling path. Latency is real generation
through the official sampler (~1-2s typical for a warm request), not a
single masked-token read -- see RESULTS.md for measured figures and why the
originally proposed sub-150ms figure no longer applies beyond a pure
argument-free reflex.
"""

import os
import sys
import time
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_settings


def test_fast_path_reflex():
    settings = get_settings()
    base_url = f"http://{settings.HOST}:{settings.PORT}"
    endpoint = f"{base_url}/v1/chat/completions"

    print("=" * 75)
    print("TEST: FAST-PATH STEP 1 DISCRETE TOOL REFLEX")
    print(f"Target: {endpoint}")
    print("=" * 75)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "open_inbox",
                "description": "Opens the user email inbox directly without parameters",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mute_audio",
                "description": "Toggles master audio mute state",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]

    payload = {
        "model": settings.MODEL_ID,
        "messages": [
            {"role": "user", "content": "Please open my inbox now."}
        ],
        "tools": tools,
    }

    # Warmup request
    print("\nWarming up endpoint...")
    resp_warmup = requests.post(endpoint, json=payload, timeout=60)
    assert resp_warmup.status_code == 200, f"Warmup failed with code {resp_warmup.status_code}: {resp_warmup.text}"
    print(f"Warmup response code: {resp_warmup.status_code}")

    # Timed fast-path requests
    latencies = []
    print("\nExecuting timed trials...")
    for trial in range(3):
        t0 = time.perf_counter()
        resp = requests.post(endpoint, json=payload, timeout=30)
        wall_lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(wall_lat)

        assert resp.status_code == 200, f"HTTP Error {resp.status_code}: {resp.text}"
        data = resp.json()

        choice = data["choices"][0]
        tool_calls = choice["message"].get("tool_calls")
        finish_reason = choice.get("finish_reason")
        meta = data.get("reflex_metadata", {})

        print(f"  Trial {trial+1}: Wall Latency = {wall_lat:.1f}ms | Engine Latency = {meta.get('latency_ms')}ms | Path = {meta.get('execution_path')}")
        print(f"    Tool: {tool_calls[0]['function']['name'] if tool_calls else 'None'} | Finish: {finish_reason} | Conf: {meta.get('confidence')}")

        # Assertions
        assert finish_reason == "tool_calls", f"Expected finish_reason 'tool_calls', got '{finish_reason}'"
        assert tool_calls is not None and len(tool_calls) > 0, "No tool_calls in message"
        assert meta.get("execution_path") == "NATIVE_TOOL_CALL", f"Expected NATIVE_TOOL_CALL, got {meta.get('execution_path')}"
        assert meta.get("steps_executed", 0) >= 1, f"Expected at least 1 decoder forward pass, got {meta.get('steps_executed')}"

    mean_wall = sum(latencies) / len(latencies)
    print("\n" + "-" * 75)
    print(f"Fast-Path Mean Wall-Clock Latency: {mean_wall:.2f} ms")
    print("SUCCESS: Fast-Path Reflex Tool Routing verified.")
    print("=" * 75)


if __name__ == "__main__":
    test_fast_path_reflex()

