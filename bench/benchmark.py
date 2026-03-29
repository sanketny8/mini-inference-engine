"""
Benchmark script for mini inference engine.

Measures:
- Throughput (tokens/second)
- Time to First Token (TTFT)
- Total generation time
- Memory usage
"""

import argparse
import time

import torch

from engine.engine import InferenceEngine
from engine.sampler import SamplingParams


def run_benchmark(
    model_name: str,
    prompts: list[str],
    max_tokens: int = 128,
    device: str = "cpu",
    dtype_str: str = "float32",
):
    dtype = getattr(torch, dtype_str)
    engine = InferenceEngine(model_name, device=device, dtype=dtype)

    print(f"\n{'='*60}")
    print(f"Benchmark: {model_name}")
    print(f"Device: {device} | Dtype: {dtype_str}")
    print(f"Prompts: {len(prompts)} | Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    params = SamplingParams(temperature=0, max_tokens=max_tokens)

    # --- Single prompt ---
    print("--- Single Prompt ---")
    start = time.perf_counter()
    result = engine.generate(prompts[0], params)
    elapsed = time.perf_counter() - start

    gen_tokens = result["generated_tokens"]
    ttft = result.get("ttft", 0)
    tps = gen_tokens / elapsed if elapsed > 0 else 0

    print(f"Prompt: {prompts[0][:60]}...")
    print(f"Generated: {gen_tokens} tokens")
    print(f"TTFT: {ttft*1000:.1f} ms")
    print(f"Total: {elapsed:.2f}s")
    print(f"Throughput: {tps:.1f} tok/s")
    print(f"Output: {result['output'][:100]}...")

    # --- Batch ---
    if len(prompts) > 1:
        print(f"\n--- Batch ({len(prompts)} prompts) ---")
        start = time.perf_counter()
        results = engine.generate_batch(prompts, params)
        elapsed = time.perf_counter() - start

        total_tokens = sum(r.get("generated_tokens", 0) for r in results)
        tps = total_tokens / elapsed if elapsed > 0 else 0

        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {tps:.1f} tok/s")

        for i, r in enumerate(results):
            print(f"  [{i}] {r.get('generated_tokens', 0)} tokens | {r.get('output', '')[:60]}...")

    # --- Stats ---
    stats = engine.stats()
    print(f"\n--- Engine Stats ---")
    print(f"Cache memory: {stats['cache_memory_mb']:.1f} MB")
    print(f"Free blocks: {stats['cache_free_blocks']}/{stats['cache_total_blocks']}")
    print(f"Total steps: {stats['total_steps']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mini inference engine")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    prompts = [
        "Explain what a transformer is in simple terms.",
        "Write a Python function to compute fibonacci numbers.",
        "What are the key differences between TCP and UDP?",
        "Describe the attention mechanism in neural networks.",
    ]

    run_benchmark(args.model, prompts, args.max_tokens, args.device, args.dtype)
