"""Start the OpenAI-compatible inference server."""

import argparse

import torch
import uvicorn

from engine.api import create_app
from engine.engine import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Mini Inference Engine Server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--num-cache-blocks", type=int, default=256)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    engine = InferenceEngine(
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        max_batch_size=args.max_batch_size,
        num_cache_blocks=args.num_cache_blocks,
    )

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
