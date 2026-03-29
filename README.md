# Mini Inference Engine

A from-scratch LLM inference engine in ~1500 lines of Python, implementing the core techniques used in production systems like vLLM and TGI.

```
Client (curl / Python) ──→ HTTP API (FastAPI, OpenAI-compatible)
                                    │
                              Scheduler (continuous batching)
                                    │
                              Engine (step loop)
                                    │
                           ┌────────┴─────────┐
                     Model (forward)     KV Cache (paged)
                           │
                     Sampler (greedy / top-k / top-p / temp)
                           │
                  [Speculative Decoder] (draft + target)
```

## What's Implemented

| Component | Description |
|-----------|-------------|
| **Transformer forward pass** | Qwen-2.5 architecture: RMSNorm, RoPE, GQA, SwiGLU FFN. Loads weights directly from HuggingFace. |
| **Paged KV cache** | Block-based memory management (vLLM-style PagedAttention). Fixed-size blocks, free list allocator, per-sequence page tables. |
| **Continuous batching** | Dynamic batching of prefill + decode operations. New requests join mid-generation. Preemption when memory is exhausted. |
| **Sampling** | Greedy, temperature, top-k, top-p (nucleus), repetition penalty. |
| **Speculative decoding** | Draft model (0.5B) proposes tokens, target model (1.5B) verifies in one pass. Reports acceptance rate. |
| **OpenAI-compatible API** | `/v1/chat/completions` with streaming (SSE), `/v1/models`, `/health`, `/metrics`. |

## Quick Start

```bash
pip install -r requirements.txt

# Basic generation
python examples/basic_generate.py

# Concurrent requests (continuous batching)
python examples/concurrent_requests.py

# Start OpenAI-compatible server
python serve.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000

# Query the server
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-0.5b",
    "messages": [{"role": "user", "content": "What is a KV cache?"}],
    "max_tokens": 100
  }'
```

## Project Structure

```
engine/
├── model.py          # Qwen-2.5 forward pass (RMSNorm, RoPE, GQA, SwiGLU)
├── kv_cache.py       # Paged KV cache with block allocator
├── scheduler.py      # Continuous batching scheduler
├── sampler.py        # Sampling strategies (greedy, top-k, top-p, temp)
├── speculative.py    # Speculative decoding (draft + verify)
├── engine.py         # Main inference loop
└── api.py            # FastAPI OpenAI-compatible server

bench/
├── benchmark.py      # Throughput, latency, memory benchmarks
└── compare_hf.py     # Token-for-token correctness vs HuggingFace

tests/
├── test_kv_cache.py  # Block allocator + paged cache tests
├── test_sampler.py   # Sampling distribution tests
└── test_scheduler.py # Continuous batching tests
```

## Benchmarks

```bash
# Throughput + latency
python bench/benchmark.py --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 64

# Correctness verification (greedy decode must match HF token-for-token)
python bench/compare_hf.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## Key Concepts

### Paged KV Cache

Instead of allocating one contiguous buffer per sequence, the cache is divided into fixed-size **blocks** (default: 16 tokens). A **block allocator** manages a free list, and each sequence maintains a **page table** mapping logical positions to physical blocks. This eliminates memory fragmentation and enables efficient memory sharing.

```
Sequence 0:  [Block 3] → [Block 7] → [Block 1]
Sequence 1:  [Block 0] → [Block 5]
Free:        [Block 2, Block 4, Block 6, ...]
```

### Continuous Batching

Traditional batching waits for all sequences in a batch to finish. Continuous batching allows new requests to **join mid-generation** and finished sequences to **leave immediately**. Each step processes a mix of prefill (new prompts) and decode (generating tokens) operations.

```
Step 1: [Prefill A] [Prefill B]
Step 2: [Decode A]  [Decode B]  [Prefill C]    ← C joins mid-batch
Step 3: [Decode A]  [Decode B]  [Decode C]
Step 4:             [Decode B]  [Decode C]      ← A finished, leaves
```

### Speculative Decoding

A small **draft model** (0.5B) generates K candidate tokens quickly. The larger **target model** (1.5B) then verifies all K tokens in a single forward pass. Accepted tokens are kept; on rejection, the target model's token is used instead. This can yield 1.5-2x speedup since the draft model runs K iterations but the target runs just one.

```
Draft (fast):   generates [t1, t2, t3, t4, t5]
Target (slow):  verifies all 5 in ONE forward pass
Result:         accepts [t1, t2, t3], rejects t4 → uses target's t4
```

## Tests

```bash
pytest tests/ -v
```

## Dependencies

- `torch` >= 2.0
- `transformers` >= 4.40
- `fastapi` + `uvicorn`
- `safetensors`, `numpy`

## Tech

Python, PyTorch, FastAPI, Qwen-2.5
