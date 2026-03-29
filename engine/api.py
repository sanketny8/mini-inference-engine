"""
OpenAI-compatible HTTP API using FastAPI.

Endpoints:
- POST /v1/chat/completions — chat completion (streaming + non-streaming)
- GET  /v1/models           — list available models
- GET  /health              — health check
- GET  /metrics             — engine stats
"""

import asyncio
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import InferenceEngine
from .sampler import SamplingParams

app = FastAPI(title="Mini Inference Engine", version="0.1.0")

# Global engine instance (initialized on startup)
_engine: Optional[InferenceEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-2.5-0.5b"
    messages: list[ChatMessage]
    temperature: float = Field(default=1.0, ge=0, le=2.0)
    top_p: float = Field(default=1.0, ge=0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = False
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


def format_prompt(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    prompt = format_prompt(request.messages)
    params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if request.stream:
        return StreamingResponse(
            _stream_generate(request_id, prompt, params, request.model),
            media_type="text/event-stream",
        )

    # Non-streaming
    result = await asyncio.to_thread(_engine.generate, prompt, params)

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["output"]),
                finish_reason="stop" if result.get("generated_tokens", 0) < params.max_tokens else "length",
            )
        ],
        usage=Usage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("generated_tokens", 0),
            total_tokens=result.get("prompt_tokens", 0) + result.get("generated_tokens", 0),
        ),
    )


async def _stream_generate(request_id: str, prompt: str, params: SamplingParams, model: str):
    """Stream tokens as SSE events."""
    seq_id = _engine.add_request(prompt, params)

    # Add EOS to stop tokens if not already
    eos_id = _engine.tokenizer.eos_token_id
    if eos_id is not None and eos_id not in params.stop_token_ids:
        params.stop_token_ids.append(eos_id)

    prev_len = 0

    while _engine.has_pending():
        await asyncio.to_thread(_engine.step)

        # Check if our sequence has new tokens
        for seq in _engine.scheduler.running + _engine.scheduler.finished:
            if seq.seq_id == seq_id and len(seq.generated_token_ids) > prev_len:
                new_tokens = seq.generated_token_ids[prev_len:]
                text = _engine.tokenizer.decode(new_tokens, skip_special_tokens=True)
                prev_len = len(seq.generated_token_ids)

                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-2.5-0.5b",
                "object": "model",
                "owned_by": "mini-inference-engine",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.get("/metrics")
async def metrics():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine.stats()


def create_app(engine: InferenceEngine) -> FastAPI:
    """Create FastAPI app with engine instance."""
    global _engine
    _engine = engine
    return app
