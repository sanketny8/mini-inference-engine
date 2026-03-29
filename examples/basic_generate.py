"""Basic generation example."""

from engine.engine import InferenceEngine
from engine.sampler import SamplingParams


def main():
    engine = InferenceEngine("Qwen/Qwen2.5-0.5B-Instruct")

    # Greedy generation
    result = engine.generate(
        "Explain what a KV cache is in LLM inference, in 2 sentences.",
        SamplingParams(temperature=0, max_tokens=100),
    )
    print(f"Greedy: {result['output']}")
    print(f"  Tokens: {result['generated_tokens']}, TTFT: {result['ttft']*1000:.0f}ms")

    # Sampling
    result = engine.generate(
        "Write a haiku about distributed systems.",
        SamplingParams(temperature=0.8, top_p=0.9, max_tokens=50),
    )
    print(f"\nSampled: {result['output']}")

    print(f"\nEngine stats: {engine.stats()}")


if __name__ == "__main__":
    main()
