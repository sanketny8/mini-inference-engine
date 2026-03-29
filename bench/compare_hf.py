"""
Correctness verification: compare our engine's output with HuggingFace transformers.

Uses greedy decoding (temperature=0) for deterministic comparison.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.engine import InferenceEngine
from engine.sampler import SamplingParams


def compare(model_name: str, prompt: str, max_tokens: int = 32, device: str = "cpu"):
    dtype = torch.float32

    # --- HuggingFace baseline ---
    print(f"[HF] Loading {model_name}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    hf_model.eval()

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    hf_generated = hf_output[0][input_ids.shape[1]:]
    hf_text = hf_tokenizer.decode(hf_generated, skip_special_tokens=True)
    hf_ids = hf_generated.tolist()

    del hf_model
    torch.cuda.empty_cache() if device != "cpu" else None

    # --- Our engine ---
    print(f"[Engine] Loading {model_name}...")
    engine = InferenceEngine(model_name, device=device, dtype=dtype)

    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    result = engine.generate(prompt, params)

    engine_ids = result["output_token_ids"]
    engine_text = result["output"]

    # --- Compare ---
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*60}")

    print(f"\n[HF]     ({len(hf_ids)} tokens): {hf_text[:200]}")
    print(f"[Engine] ({len(engine_ids)} tokens): {engine_text[:200]}")

    # Token-by-token comparison
    min_len = min(len(hf_ids), len(engine_ids))
    mismatches = 0
    for i in range(min_len):
        if hf_ids[i] != engine_ids[i]:
            hf_tok = hf_tokenizer.decode([hf_ids[i]])
            eng_tok = hf_tokenizer.decode([engine_ids[i]])
            print(f"  MISMATCH at position {i}: HF='{hf_tok}' ({hf_ids[i]}) vs Engine='{eng_tok}' ({engine_ids[i]})")
            mismatches += 1

    if len(hf_ids) != len(engine_ids):
        print(f"  LENGTH DIFF: HF={len(hf_ids)} vs Engine={len(engine_ids)}")

    if mismatches == 0 and len(hf_ids) == len(engine_ids):
        print("\n  PASS: Token-for-token match!")
    else:
        print(f"\n  FAIL: {mismatches} mismatches out of {min_len} tokens")

    return mismatches == 0 and len(hf_ids) == len(engine_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare engine output with HuggingFace")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    prompts = [
        "What is 2 + 2?",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    results = []
    for prompt in prompts:
        passed = compare(args.model, prompt, args.max_tokens, args.device)
        results.append(passed)
        print()

    print(f"\n{'='*60}")
    print(f"Results: {sum(results)}/{len(results)} prompts matched")
    print(f"{'='*60}")
