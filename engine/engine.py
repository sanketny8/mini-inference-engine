"""
Main inference engine tying model, KV cache, scheduler, and sampler together.

Provides:
- add_request() for submitting prompts
- step() for processing one batch iteration
- generate() for synchronous generation
"""

import time
from typing import Optional

import torch

from .kv_cache import CacheConfig, PagedKVCache
from .model import ModelConfig, QwenModel, load_model
from .sampler import SamplingParams, sample
from .scheduler import Scheduler, SchedulerBatch, SequenceState


class InferenceEngine:
    """
    Core inference engine with continuous batching and paged KV cache.

    Usage:
        engine = InferenceEngine("Qwen/Qwen2.5-0.5B-Instruct")
        request_id = engine.add_request("What is AI?")
        while engine.has_pending():
            engine.step()
        results = engine.get_results()
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        num_cache_blocks: int = 256,
        block_size: int = 16,
    ):
        print(f"[engine] Loading model: {model_name}")
        self.model, self.tokenizer, self.model_config = load_model(model_name, device, dtype)
        self.device = device
        self.dtype = dtype

        # Initialize paged KV cache
        cache_config = CacheConfig(
            block_size=block_size,
            num_blocks=num_cache_blocks,
            num_layers=self.model_config.num_hidden_layers,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            dtype=dtype,
            device=device,
        )
        self.kv_cache = PagedKVCache(cache_config)
        self.block_size = block_size

        # Initialize scheduler
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # Results storage
        self._results: dict[int, dict] = {}

        # Stats
        self.total_tokens_generated = 0
        self.total_steps = 0

        print(f"[engine] Ready. Cache: {num_cache_blocks} blocks x {block_size} tokens, device={device}")

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """Submit a prompt for generation. Returns request ID."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add EOS to stop tokens
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and eos_id not in sampling_params.stop_token_ids:
            sampling_params.stop_token_ids.append(eos_id)

        token_ids = self.tokenizer.encode(prompt)
        seq_id = self.scheduler.add_request(token_ids, sampling_params)
        return seq_id

    @torch.no_grad()
    def step(self):
        """Process one batch step: schedule → forward → sample → update."""
        batch = self.scheduler.schedule(
            num_free_blocks=self.kv_cache.num_free_blocks,
            block_size=self.block_size,
        )

        if batch.is_empty:
            return

        self.total_steps += 1
        finished_ids = set()

        # --- Process prefill sequences (one at a time for simplicity) ---
        for seq in batch.prefill_seqs:
            self._prefill(seq)
            self.scheduler.mark_running(seq)

            # Generate first token
            token_id = self._decode_one(seq)
            seq.generated_token_ids.append(token_id)
            seq.first_token_time = time.time()
            self.total_tokens_generated += 1

            if seq.is_finished():
                finished_ids.add(seq.seq_id)
                self._finalize(seq)

        # --- Process decode sequences ---
        for seq in batch.decode_seqs:
            token_id = self._decode_one(seq)
            seq.generated_token_ids.append(token_id)
            self.total_tokens_generated += 1

            if seq.is_finished():
                finished_ids.add(seq.seq_id)
                self._finalize(seq)

        self.scheduler.update_after_step(finished_ids)

    def _prefill(self, seq: SequenceState):
        """Run prefill forward pass and populate KV cache."""
        token_ids = seq.prompt_token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(token_ids), device=self.device).unsqueeze(0)

        # Allocate cache blocks
        self.kv_cache.allocate_sequence(seq.seq_id, len(token_ids))

        # Forward pass (no existing KV cache)
        logits, kv_caches = self.model(input_ids, positions, kv_caches=None)

        # Store KV cache
        for layer_idx, (k, v) in enumerate(kv_caches):
            # k, v: (1, num_kv_heads, seq_len, head_dim)
            k = k.squeeze(0).permute(1, 0, 2)  # (seq_len, num_kv_heads, head_dim)
            v = v.squeeze(0).permute(1, 0, 2)
            self.kv_cache.append_tokens(seq.seq_id, layer_idx, k, v)

        seq.prefill_done = True

    def _decode_one(self, seq: SequenceState) -> int:
        """Generate one token using KV cache."""
        # Get last token
        if seq.generated_token_ids:
            last_token = seq.generated_token_ids[-1]
        else:
            last_token = seq.prompt_token_ids[-1]

        # For the first decode after prefill, we already have logits
        # But for simplicity, we run a forward pass with the last generated token
        if not seq.generated_token_ids and seq.prefill_done:
            # Use prefill logits (already computed in _prefill, but we recalculate for clarity)
            pass

        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
        pos = seq.total_len - 1
        positions = torch.tensor([[pos]], dtype=torch.long, device=self.device)

        # Gather KV cache for all layers
        kv_caches = []
        for layer_idx in range(self.model_config.num_hidden_layers):
            k, v = self.kv_cache.get_kv(seq.seq_id, layer_idx)
            if k is not None:
                kv_caches.append((k, v))
            else:
                kv_caches.append(None)

        logits, new_kv = self.model(input_ids, positions, kv_caches=kv_caches)

        # Append new KV to cache
        for layer_idx, (k, v) in enumerate(new_kv):
            # Only the new token's KV (last position)
            k_new = k[:, :, -1:, :].squeeze(0).permute(1, 0, 2)  # (1, num_kv_heads, head_dim)
            v_new = v[:, :, -1:, :].squeeze(0).permute(1, 0, 2)
            self.kv_cache.append_tokens(seq.seq_id, layer_idx, k_new, v_new)

        # Sample
        last_logits = logits[0, -1, :]
        token_id = sample(last_logits, seq.sampling_params, seq.generated_token_ids)
        return token_id.item()

    def _finalize(self, seq: SequenceState):
        """Store result and free cache."""
        output_text = self.tokenizer.decode(seq.generated_token_ids, skip_special_tokens=True)
        self._results[seq.seq_id] = {
            "seq_id": seq.seq_id,
            "output": output_text,
            "output_token_ids": seq.generated_token_ids,
            "prompt_tokens": len(seq.prompt_token_ids),
            "generated_tokens": len(seq.generated_token_ids),
            "ttft": seq.first_token_time - seq.arrival_time if seq.first_token_time else 0,
            "total_time": time.time() - seq.arrival_time,
        }
        self.kv_cache.free_sequence(seq.seq_id)

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> dict:
        """Synchronous generation — submit prompt, run until done, return result."""
        seq_id = self.add_request(prompt, sampling_params)

        while self.has_pending():
            self.step()

        return self._results.get(seq_id, {})

    def generate_batch(
        self,
        prompts: list[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[dict]:
        """Generate for multiple prompts using continuous batching."""
        seq_ids = [self.add_request(p, sampling_params) for p in prompts]

        while self.has_pending():
            self.step()

        return [self._results.get(sid, {}) for sid in seq_ids]

    def has_pending(self) -> bool:
        return self.scheduler.has_pending()

    def get_results(self) -> dict:
        return dict(self._results)

    def stats(self) -> dict:
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_steps": self.total_steps,
            "cache_free_blocks": self.kv_cache.num_free_blocks,
            "cache_total_blocks": self.kv_cache.config.num_blocks,
            "cache_memory_mb": self.kv_cache.memory_usage_bytes() / (1024 * 1024),
            "scheduler_waiting": self.scheduler.num_waiting,
            "scheduler_running": self.scheduler.num_running,
        }
