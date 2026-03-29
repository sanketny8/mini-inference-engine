"""
Sampling strategies for token generation.

Supports greedy, temperature, top-k, top-p (nucleus), and repetition penalty.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 256
    stop_token_ids: list = field(default_factory=list)


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    generated_ids: Optional[list] = None,
) -> torch.Tensor:
    """
    Sample next token from logits.

    Args:
        logits: (batch, vocab_size) logits for the last position
        params: sampling parameters
        generated_ids: list of previously generated token IDs for repetition penalty
    Returns:
        (batch,) sampled token IDs
    """
    # Repetition penalty
    if params.repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if logits.dim() == 1:
                if logits[token_id] > 0:
                    logits[token_id] /= params.repetition_penalty
                else:
                    logits[token_id] *= params.repetition_penalty
            else:
                for b in range(logits.shape[0]):
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= params.repetition_penalty
                    else:
                        logits[b, token_id] *= params.repetition_penalty

    # Greedy
    if params.temperature == 0 or (params.top_k == 1):
        return logits.argmax(dim=-1)

    # Temperature
    if params.temperature != 1.0:
        logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0:
        top_k = min(params.top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Top-p (nucleus) filtering
    if 0 < params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= params.top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        # Scatter back
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    # Sample from distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
