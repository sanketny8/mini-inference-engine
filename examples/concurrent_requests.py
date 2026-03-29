"""Concurrent requests example using continuous batching."""

import time

from engine.engine import InferenceEngine
from engine.sampler import SamplingParams


def main():
    engine = InferenceEngine("Qwen/Qwen2.5-0.5B-Instruct")

    prompts = [
        "What is attention in transformers?",
        "Explain gradient descent in one paragraph.",
        "Write a Python function to reverse a linked list.",
        "What is the difference between TCP and UDP?",
        "Describe the CAP theorem.",
        "What is a KV cache and why does it matter?",
        "Explain continuous batching for LLM inference.",
        "What is speculative decoding?",
    ]

    params = SamplingParams(temperature=0, max_tokens=64)

    print(f"Submitting {len(prompts)} concurrent requests...\n")
    start = time.perf_counter()
    results = engine.generate_batch(prompts, params)
    elapsed = time.perf_counter() - start

    total_tokens = 0
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        gen_tokens = result.get("generated_tokens", 0)
        total_tokens += gen_tokens
        print(f"[{i}] ({gen_tokens} tokens) {prompt[:50]}...")
        print(f"    → {result.get('output', '')[:80]}...\n")

    tps = total_tokens / elapsed if elapsed > 0 else 0
    print(f"{'='*60}")
    print(f"Total: {total_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")
    print(f"Stats: {engine.stats()}")


if __name__ == "__main__":
    main()
