#!/usr/bin/env python3
"""
Test Suite 4: Official OpenAI SDK Python Client Integration.
Validates zero-friction compatibility with `from openai import OpenAI`.
"""

import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_settings


def test_openai_sdk_integration():
    settings = get_settings()
    base_url = f"http://{settings.HOST}:{settings.PORT}/v1"

    print("=" * 75)
    print("TEST: OFFICIAL OPENAI PYTHON SDK COMPATIBILITY")
    print(f"Base URL: {base_url}")
    print("=" * 75)

    client = OpenAI(base_url=base_url, api_key="reflex-local-dummy-key")

    # 1. Models list
    print("\n1. Querying client.models.list()...")
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    print(f"Available Models: {model_ids}")
    assert settings.MODEL_ID in model_ids, f"Model {settings.MODEL_ID} not found in {model_ids}"

    # 2. Tool calling (Fast-path reflex)
    print("\n2. Querying client.chat.completions.create() with tools...")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "check_weather",
                "description": "Checks the weather forecast",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    resp_tool = client.chat.completions.create(
        model=settings.MODEL_ID,
        messages=[{"role": "user", "content": "Check the weather forecast for today."}],
        tools=tools,
    )
    choice_tool = resp_tool.choices[0]
    print(f"Tool Choice Finish Reason: {choice_tool.finish_reason}")
    assert choice_tool.finish_reason == "tool_calls", f"Expected 'tool_calls', got {choice_tool.finish_reason}"
    assert choice_tool.message.tool_calls is not None
    print(f"Tool Invoked: {choice_tool.message.tool_calls[0].function.name}")

    # 3. Conversational text generation
    print("\n3. Querying client.chat.completions.create() for text completion...")
    resp_text = client.chat.completions.create(
        model=settings.MODEL_ID,
        messages=[{"role": "user", "content": "What is 10 + 25? Answer with only the number."}],
        max_tokens=32,
    )
    choice_text = resp_text.choices[0]
    print(f"Text Choice Finish Reason: {choice_text.finish_reason}")
    print(f"Generated Content: {choice_text.message.content}")
    assert choice_text.finish_reason == "stop", f"Expected 'stop', got {choice_text.finish_reason}"
    assert choice_text.message.content is not None

    print("\n" + "-" * 75)
    print("SUCCESS: Full compatibility with official openai Python SDK verified.")
    print("=" * 75)


if __name__ == "__main__":
    test_openai_sdk_integration()

