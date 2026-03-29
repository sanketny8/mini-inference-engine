"""
Qwen-2.5 transformer forward pass implemented from scratch.

Loads weights from HuggingFace and implements:
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU FFN
- Prefill + decode modes with KV cache support
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    head_dim: int

    @classmethod
    def from_pretrained(cls, model_name: str) -> "ModelConfig":
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            head_dim=head_dim,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0, device: str = "cpu"):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    """Apply rotary position embeddings."""
    # x: (batch, num_heads, seq_len, head_dim)
    cos = cos[positions].unsqueeze(1)  # (batch, 1, seq_len, head_dim/2)
    sin = sin[positions].unsqueeze(1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos.repeat(1, 1, 1, 2)) + (rotated * sin.repeat(1, 1, 1, 2))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        # KV cache: append new k,v to cached
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # GQA: repeat KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), new_kv_cache


class FeedForward(nn.Module):
    """SwiGLU FFN used by Qwen-2.5."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, positions, mask=None, kv_cache=None):
        residual = x
        x = self.input_layernorm(x)
        x, new_kv_cache = self.self_attn(x, cos, sin, positions, mask, kv_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_kv_cache


class QwenModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.register_buffer("rope_cos", torch.zeros(config.max_position_embeddings, config.head_dim // 2))
        self.register_buffer("rope_sin", torch.zeros(config.max_position_embeddings, config.head_dim // 2))

    def _init_rope(self, device):
        cos, sin = precompute_rope_freqs(
            self.config.head_dim,
            self.config.max_position_embeddings,
            self.config.rope_theta,
            device=device,
        )
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[list] = None,
    ) -> tuple:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            positions: (batch, seq_len) position indices
            kv_caches: list of (k, v) tuples per layer, or None
        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: list of (k, v) tuples
        """
        hidden = self.embed_tokens(input_ids)

        # Build causal mask
        seq_len = input_ids.shape[1]
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)
            # Account for cached tokens
            if kv_caches and kv_caches[0] is not None:
                cached_len = kv_caches[0][0].shape[2]
                prefix = torch.zeros((seq_len, cached_len), device=input_ids.device)
                mask = torch.cat([prefix, mask], dim=-1)
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches else None
            hidden, new_cache = layer(hidden, self.rope_cos, self.rope_sin, positions, mask, layer_cache)
            new_kv_caches.append(new_cache)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        return logits, new_kv_caches


def load_model(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> tuple:
    """Load Qwen model weights from HuggingFace into our custom architecture."""
    from transformers import AutoModelForCausalLM

    config = ModelConfig.from_pretrained(model_name)
    model = QwenModel(config)

    # Load HF model to extract weights
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    hf_state = hf_model.state_dict()

    # Map HF weight names to our names
    state_dict = {}
    state_dict["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]
    state_dict["norm.weight"] = hf_state["model.norm.weight"]

    # lm_head — Qwen ties embed and lm_head
    if "lm_head.weight" in hf_state:
        state_dict["lm_head.weight"] = hf_state["lm_head.weight"]
    else:
        state_dict["lm_head.weight"] = hf_state["model.embed_tokens.weight"]

    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        our_prefix = f"layers.{i}"

        state_dict[f"{our_prefix}.input_layernorm.weight"] = hf_state[f"{prefix}.input_layernorm.weight"]
        state_dict[f"{our_prefix}.post_attention_layernorm.weight"] = hf_state[f"{prefix}.post_attention_layernorm.weight"]

        # Attention projections
        state_dict[f"{our_prefix}.self_attn.q_proj.weight"] = hf_state[f"{prefix}.self_attn.q_proj.weight"]
        state_dict[f"{our_prefix}.self_attn.q_proj.bias"] = hf_state[f"{prefix}.self_attn.q_proj.bias"]
        state_dict[f"{our_prefix}.self_attn.k_proj.weight"] = hf_state[f"{prefix}.self_attn.k_proj.weight"]
        state_dict[f"{our_prefix}.self_attn.k_proj.bias"] = hf_state[f"{prefix}.self_attn.k_proj.bias"]
        state_dict[f"{our_prefix}.self_attn.v_proj.weight"] = hf_state[f"{prefix}.self_attn.v_proj.weight"]
        state_dict[f"{our_prefix}.self_attn.v_proj.bias"] = hf_state[f"{prefix}.self_attn.v_proj.bias"]
        state_dict[f"{our_prefix}.self_attn.o_proj.weight"] = hf_state[f"{prefix}.self_attn.o_proj.weight"]

        # FFN
        state_dict[f"{our_prefix}.mlp.gate_proj.weight"] = hf_state[f"{prefix}.mlp.gate_proj.weight"]
        state_dict[f"{our_prefix}.mlp.up_proj.weight"] = hf_state[f"{prefix}.mlp.up_proj.weight"]
        state_dict[f"{our_prefix}.mlp.down_proj.weight"] = hf_state[f"{prefix}.mlp.down_proj.weight"]

    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model._init_rope(device)

    del hf_model
    torch.cuda.empty_cache() if device != "cpu" else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer, config
