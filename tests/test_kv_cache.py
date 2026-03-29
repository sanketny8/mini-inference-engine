"""Tests for paged KV cache."""

import pytest
import torch

from engine.kv_cache import BlockAllocator, CacheConfig, PagedKVCache


class TestBlockAllocator:
    def test_allocate_and_free(self):
        alloc = BlockAllocator(num_blocks=4)
        assert alloc.num_free_blocks == 4

        b1 = alloc.allocate()
        assert alloc.num_free_blocks == 3

        alloc.free(b1)
        assert alloc.num_free_blocks == 4

    def test_allocate_all_blocks(self):
        alloc = BlockAllocator(num_blocks=3)
        blocks = [alloc.allocate() for _ in range(3)]
        assert alloc.num_free_blocks == 0
        assert len(set(blocks)) == 3

    def test_out_of_blocks(self):
        alloc = BlockAllocator(num_blocks=1)
        alloc.allocate()
        with pytest.raises(RuntimeError, match="Out of cache blocks"):
            alloc.allocate()

    def test_can_allocate(self):
        alloc = BlockAllocator(num_blocks=5)
        assert alloc.can_allocate(5)
        assert not alloc.can_allocate(6)
        alloc.allocate()
        assert not alloc.can_allocate(5)


class TestPagedKVCache:
    @pytest.fixture
    def cache(self):
        config = CacheConfig(
            block_size=4,
            num_blocks=8,
            num_layers=2,
            num_kv_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        return PagedKVCache(config)

    def test_allocate_and_free_sequence(self, cache):
        initial_free = cache.num_free_blocks
        cache.allocate_sequence(seq_id=0, num_tokens=4)
        assert cache.num_free_blocks == initial_free - 1

        cache.free_sequence(seq_id=0)
        assert cache.num_free_blocks == initial_free

    def test_allocate_multiple_blocks(self, cache):
        initial_free = cache.num_free_blocks
        cache.allocate_sequence(seq_id=0, num_tokens=10)  # needs 3 blocks (10/4=2.5, ceil=3)
        assert cache.num_free_blocks == initial_free - 3

    def test_append_and_retrieve(self, cache):
        cache.allocate_sequence(seq_id=0, num_tokens=1)

        # Append 2 tokens to layer 0
        keys = torch.randn(2, 2, 4)
        values = torch.randn(2, 2, 4)
        cache.append_tokens(seq_id=0, layer_idx=0, keys=keys, values=values)

        k, v = cache.get_kv(seq_id=0, layer_idx=0)
        assert k is not None
        assert k.shape == (1, 2, 2, 4)  # (1, num_kv_heads, seq_len, head_dim)

    def test_append_grows_blocks(self, cache):
        cache.allocate_sequence(seq_id=0, num_tokens=1)
        initial_free = cache.num_free_blocks

        # Append 5 tokens — should need a second block (block_size=4)
        keys = torch.randn(5, 2, 4)
        values = torch.randn(5, 2, 4)
        cache.append_tokens(seq_id=0, layer_idx=0, keys=keys, values=values)

        assert cache.num_free_blocks < initial_free

    def test_duplicate_allocate_raises(self, cache):
        cache.allocate_sequence(seq_id=0)
        with pytest.raises(ValueError, match="already allocated"):
            cache.allocate_sequence(seq_id=0)

    def test_get_kv_empty(self, cache):
        cache.allocate_sequence(seq_id=0)
        k, v = cache.get_kv(seq_id=0, layer_idx=0)
        assert k is None

    def test_memory_usage(self, cache):
        usage = cache.memory_usage_bytes()
        assert usage > 0
