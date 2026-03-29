"""Tests for sampling strategies."""

import torch

from engine.sampler import SamplingParams, sample


class TestSampler:
    def test_greedy(self):
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0])
        params = SamplingParams(temperature=0)
        token = sample(logits, params)
        assert token.item() == 1  # index of max value

    def test_top_k(self):
        torch.manual_seed(42)
        logits = torch.tensor([1.0, 5.0, 2.0, 3.0, 0.5, 0.1])
        params = SamplingParams(temperature=1.0, top_k=2)

        # Sample many times, should only get indices 1 or 3 (top 2)
        tokens = set()
        for _ in range(100):
            t = sample(logits.clone(), params)
            tokens.add(t.item())
        assert tokens.issubset({1, 3})

    def test_temperature_zero_is_greedy(self):
        logits = torch.tensor([0.1, 0.9, 0.5])
        params = SamplingParams(temperature=0)
        token = sample(logits, params)
        assert token.item() == 1

    def test_top_k_one_is_greedy(self):
        logits = torch.tensor([0.1, 0.9, 0.5])
        params = SamplingParams(top_k=1)
        token = sample(logits, params)
        assert token.item() == 1

    def test_repetition_penalty(self):
        logits = torch.tensor([5.0, 4.0, 3.0])
        params = SamplingParams(temperature=0, repetition_penalty=2.0)
        # Without penalty, picks index 0
        token_no_penalty = sample(logits.clone(), SamplingParams(temperature=0))
        assert token_no_penalty.item() == 0

        # With penalty on index 0, should pick index 1
        token_with_penalty = sample(logits.clone(), params, generated_ids=[0])
        assert token_with_penalty.item() == 1

    def test_batched_greedy(self):
        logits = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        params = SamplingParams(temperature=0)
        tokens = sample(logits, params)
        assert tokens.tolist() == [1, 2]
