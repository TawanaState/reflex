"""
FastAPI OpenAI-Compatible Inference Server for Project Reflex.
Exposes /health, /v1/models, and /v1/chat/completions.
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .engine import EngineOutput, ReflexEngine

# Global singleton engine instance
engine: Optional[ReflexEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes ReflexEngine on startup and frees resources on shutdown."""
    global engine
    settings = get_settings()
    print(f"[Server] Initializing ReflexEngine (port: {settings.PORT})...")
    engine = ReflexEngine(settings=settings)
    print(f"[Server] ReflexEngine initialized successfully. Serving on {settings.HOST}:{settings.PORT}")
    yield
    print("[Server] Shutting down ReflexEngine...")
    engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Project Reflex Inference Server",
    description="OpenAI-compatible inference server for Project Reflex with discrete diffusion reflexes & conditional expansion.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Request and Response Schemas
# =========================================================================

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "reflex-diffusiongemma"
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ReflexMetadata(BaseModel):
    execution_path: str
    tier: Optional[str] = None
    latency_ms: float
    confidence: float
    steps_executed: int
    candidate_action: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    reflex_metadata: ReflexMetadata


# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/health")
async def health():
    """Healthcheck endpoint returning system and GPU allocation metrics."""
    global engine
    settings = get_settings()
    gpu_allocated = 0.0
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_allocated = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "status": "healthy",
        "mode": settings.REFLEX_MODE,
        "model": settings.DIFFUSION_GEMMA_PATH,
        "adapter_loaded": getattr(engine, "is_lora_loaded", False) if engine else False,
        "device": settings.DEVICE,
        "gpu_memory_allocated_gb": gpu_allocated,
        "gpu_device_name": gpu_name,
        "port": settings.PORT,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models catalog endpoint."""
    settings = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": settings.MODEL_ID,
                "object": "model",
                "created": 1789999999,
                "owned_by": "reflex-ai",
                "root": settings.DIFFUSION_GEMMA_PATH,
                "parent": None,
                "permission": [],
                "reflex_mode": settings.REFLEX_MODE,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Executes sub-150ms Step 1 reflexes for discrete tool actions or conditionally expands.
    """
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Reflex engine is not yet initialized.")

    settings = get_settings()
    created_ts = int(time.time())
    completion_id = f"chatcmpl-reflex-{uuid.uuid4().hex[:12]}"

    # Convert Pydantic messages to dict
    raw_messages = [msg.model_dump() for msg in request.messages]

    # Run inference engine
    out: EngineOutput = engine.generate(
        messages=raw_messages,
        tools=request.tools,
        tool_choice=request.tool_choice,
        max_tokens=request.max_tokens,
        temperature=request.temperature or 0.0,
    )

    # Format ToolCalls if present
    tool_calls_models = None
    if out.tool_calls:
        tool_calls_models = [
            ToolCall(
                id=tc["id"],
                type="function",
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in out.tool_calls
        ]

    assistant_msg = AssistantMessage(
        role="assistant",
        content=out.content,
        tool_calls=tool_calls_models,
    )

    choice = Choice(
        index=0,
        message=assistant_msg,
        finish_reason=out.finish_reason,
    )

    usage = Usage(
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
        total_tokens=out.prompt_tokens + out.completion_tokens,
    )

    metadata = ReflexMetadata(
        execution_path=out.execution_path,
        tier=out.tier,
        latency_ms=round(out.latency_ms, 2),
        confidence=round(out.confidence, 4),
        steps_executed=out.steps_executed,
        candidate_action=out.candidate_action,
    )

    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_ts,
        model=settings.MODEL_ID,
        choices=[choice],
        usage=usage,
        reflex_metadata=metadata,
    )

    # If client requested streaming SSE
    if request.stream:
        async def event_generator():
            # Initial chunk with role
            chunk_role = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": settings.MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk_role)}\n\n"

            # Content or Tool call chunk
            if tool_calls_models:
                tc_delta = [{
                    "index": 0,
                    "id": tool_calls_models[0].id,
                    "type": "function",
                    "function": {
                        "name": tool_calls_models[0].function.name,
                        "arguments": tool_calls_models[0].function.arguments,
                    },
                }]
                chunk_tc = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": settings.MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tc_delta},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk_tc)}\n\n"
            else:
                chunk_content = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": settings.MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": out.content},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk_content)}\n\n"

            # Finish reason chunk
            chunk_finish = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": settings.MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": out.finish_reason,
                }],
                "reflex_metadata": metadata.model_dump(),
            }
            yield f"data: {json.dumps(chunk_finish)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return response


@app.post("/admin/reload")
async def reload_runtime():
    """Dynamically reloads engine logic without reloading base model weights from disk."""
    global engine
    import importlib
    import src.schema.inspector as s_insp
    import src.schema as s_pkg
    import src.engine.native_tool_calling as e_native
    import src.engine.early_exit as e_early_exit
    import src.engine.runner as e_run
    import src.engine as e_pkg

    importlib.reload(s_insp)
    importlib.reload(s_pkg)
    importlib.reload(e_native)
    importlib.reload(e_early_exit)
    importlib.reload(e_run)
    importlib.reload(e_pkg)

    if engine is not None:
        # Rebinding the instance's class to the freshly reloaded class object
        # (rather than patching individual methods) correctly picks up
        # staticmethod/classmethod members too (e.g. _estimate_forward_passes,
        # _clean_display_text), which per-method __get__ rebinding would bind
        # incorrectly (an extra `self` argument).
        engine.__class__ = e_run.ReflexEngine
    return {"status": "success", "message": "Schema & engine modules reloaded successfully"}



def run_server():
    """Starts the uvicorn server directly."""
    settings = get_settings()
    print(f"Starting Project Reflex server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run(
        "src.server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()


