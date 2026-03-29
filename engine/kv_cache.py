"""
Paged KV Cache with block-based memory management.

Implements vLLM-style PagedAttention:
- Fixed-size blocks (default 16 tokens per block)
- Block allocator with free list
- Per-sequence page tables mapping logical → physical blocks
- Pre-allocated GPU memory to avoid fragmentation
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CacheConfig:
    block_size: int = 16
    num_blocks: int = 256
    num_layers: int = 24
    num_kv_heads: int = 2
    head_dim: int = 64
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class BlockAllocator:
    """Manages allocation and deallocation of cache blocks."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: list[int] = list(range(num_blocks))
        self.ref_count: dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("Out of cache blocks")
        block_id = self.free_blocks.pop()
        self.ref_count[block_id] = 1
        return block_id

    def free(self, block_id: int):
        if block_id not in self.ref_count:
            return
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] <= 0:
            del self.ref_count[block_id]
            self.free_blocks.append(block_id)

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.free_blocks) >= num_blocks


class PagedKVCache:
    """
    Paged KV cache storing key/value tensors in fixed-size blocks.

    Memory layout:
        k_cache: (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        v_cache: (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.block_allocator = BlockAllocator(config.num_blocks)

        # Pre-allocate cache memory
        cache_shape = (
            config.num_layers,
            config.num_blocks,
            config.num_kv_heads,
            config.block_size,
            config.head_dim,
        )
        self.k_cache = torch.zeros(cache_shape, dtype=config.dtype, device=config.device)
        self.v_cache = torch.zeros(cache_shape, dtype=config.dtype, device=config.device)

        # Per-sequence page tables: seq_id -> list of block_ids
        self.page_tables: dict[int, list[int]] = {}
        # Per-sequence token count
        self.seq_lengths: dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int = 0):
        """Allocate initial blocks for a new sequence."""
        if seq_id in self.page_tables:
            raise ValueError(f"Sequence {seq_id} already allocated")

        num_blocks_needed = max(1, (num_tokens + self.config.block_size - 1) // self.config.block_size)

        if not self.block_allocator.can_allocate(num_blocks_needed):
            raise RuntimeError(f"Cannot allocate {num_blocks_needed} blocks for seq {seq_id}")

        blocks = [self.block_allocator.allocate() for _ in range(num_blocks_needed)]
        self.page_tables[seq_id] = blocks
        self.seq_lengths[seq_id] = 0

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        if seq_id not in self.page_tables:
            return
        for block_id in self.page_tables[seq_id]:
            self.block_allocator.free(block_id)
        del self.page_tables[seq_id]
        del self.seq_lengths[seq_id]

    def append_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Append new key/value tokens to the cache.

        Args:
            seq_id: sequence identifier
            layer_idx: transformer layer index
            keys: (num_new_tokens, num_kv_heads, head_dim)
            values: (num_new_tokens, num_kv_heads, head_dim)
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated")

        num_new = keys.shape[0]
        current_len = self.seq_lengths[seq_id]

        for i in range(num_new):
            pos = current_len + i
            block_idx = pos // self.config.block_size
            block_offset = pos % self.config.block_size

            # Allocate new block if needed
            while block_idx >= len(self.page_tables[seq_id]):
                if not self.block_allocator.can_allocate(1):
                    raise RuntimeError(f"Out of blocks while appending to seq {seq_id}")
                new_block = self.block_allocator.allocate()
                self.page_tables[seq_id].append(new_block)

            physical_block = self.page_tables[seq_id][block_idx]
            self.k_cache[layer_idx, physical_block, :, block_offset, :] = keys[i]
            self.v_cache[layer_idx, physical_block, :, block_offset, :] = values[i]

        if layer_idx == 0:
            self.seq_lengths[seq_id] = current_len + num_new

    def get_kv(self, seq_id: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve cached K/V tensors for a sequence.

        Returns:
            keys: (1, num_kv_heads, seq_len, head_dim)
            values: (1, num_kv_heads, seq_len, head_dim)
        """
        if seq_id not in self.page_tables:
            return None, None

        seq_len = self.seq_lengths[seq_id]
        if seq_len == 0:
            return None, None

        k_out = torch.zeros(
            seq_len, self.config.num_kv_heads, self.config.head_dim,
            dtype=self.config.dtype, device=self.config.device,
        )
        v_out = torch.zeros_like(k_out)

        for pos in range(seq_len):
            block_idx = pos // self.config.block_size
            block_offset = pos % self.config.block_size
            physical_block = self.page_tables[seq_id][block_idx]
            k_out[pos] = self.k_cache[layer_idx, physical_block, :, block_offset, :]
            v_out[pos] = self.v_cache[layer_idx, physical_block, :, block_offset, :]

        # Reshape to (1, num_kv_heads, seq_len, head_dim)
        k_out = k_out.permute(1, 0, 2).unsqueeze(0)
        v_out = v_out.permute(1, 0, 2).unsqueeze(0)
        return k_out, v_out

    @property
    def num_free_blocks(self) -> int:
        return self.block_allocator.num_free_blocks

    def memory_usage_bytes(self) -> int:
        return self.k_cache.nelement() * self.k_cache.element_size() * 2
