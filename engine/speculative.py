"""
Speculative decoding: use a small draft model to propose tokens,
verified by the larger target model in a single forward pass.

Algorithm:
1. Draft model generates K tokens autoregressively
2. Target model scores all K tokens in one forward pass
3. Accept/reject each token based on probability comparison
4. On rejection, resample from adjusted distribution
"""

import torch
import torch.nn.functional as F

from .model import QwenModel, load_model
from .sampler import SamplingParams, sample


class SpeculativeDecoder:
    """
    Speculative decoding with a draft + target model pair.

    Usage:
        decoder = SpeculativeDecoder(
            draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        )
        tokens = decoder.generate(prompt_ids, max_tokens=100)
    """

    def __init__(
        self,
        draft_model_name: str,
        target_model_name: str,
        num_speculative_tokens: int = 5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_speculative = num_speculative_tokens
        self.device = device
        self.dtype = dtype

        print(f"[speculative] Loading draft model: {draft_model_name}")
        self.draft_model, self.tokenizer, self.draft_config = load_model(
            draft_model_name, device, dtype
        )

        print(f"[speculative] Loading target model: {target_model_name}")
        self.target_model, _, self.target_config = load_model(
            target_model_name, device, dtype
        )

        # Stats
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams = None,
    ) -> dict:
        """Generate tokens using speculative decoding."""
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0)

        prompt_ids = self.tokenizer.encode(prompt)
        generated = []

        # Initial sequence
        all_ids = list(prompt_ids)

        # KV caches
        draft_kv = None
        target_kv = None

        # Prefill both models
        input_tensor = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(all_ids), device=self.device).unsqueeze(0)

        _, draft_kv = self.draft_model(input_tensor, positions)
        _, target_kv = self.target_model(input_tensor, positions)

        while len(generated) < sampling_params.max_tokens:
            # --- Step 1: Draft generates K tokens ---
            draft_tokens = []
            draft_probs_list = []
            current_draft_kv = [
                (k.clone(), v.clone()) for k, v in draft_kv
            ]

            draft_input = torch.tensor([[all_ids[-1]]], dtype=torch.long, device=self.device)

            for _ in range(self.num_speculative):
                pos = len(all_ids) + len(draft_tokens) - 1
                draft_pos = torch.tensor([[pos]], dtype=torch.long, device=self.device)

                draft_logits, current_draft_kv = self.draft_model(
                    draft_input, draft_pos, kv_caches=current_draft_kv
                )

                draft_logit = draft_logits[0, -1, :]
                draft_prob = F.softmax(draft_logit, dim=-1)

                if sampling_params.temperature == 0:
                    token = draft_logit.argmax().item()
                else:
                    token = sample(draft_logit.clone(), sampling_params).item()

                draft_tokens.append(token)
                draft_probs_list.append(draft_prob)
                draft_input = torch.tensor([[token]], dtype=torch.long, device=self.device)

                if token in sampling_params.stop_token_ids:
                    break

            self.total_draft_tokens += len(draft_tokens)

            # --- Step 2: Target scores all draft tokens in one pass ---
            verify_ids = [all_ids[-1]] + draft_tokens
            verify_input = torch.tensor([verify_ids], dtype=torch.long, device=self.device)
            start_pos = len(all_ids) - 1
            verify_pos = torch.arange(
                start_pos, start_pos + len(verify_ids), device=self.device
            ).unsqueeze(0)

            target_logits, new_target_kv = self.target_model(
                verify_input, verify_pos, kv_caches=target_kv
            )

            # --- Step 3: Accept/reject ---
            num_accepted = 0

            for i, draft_token in enumerate(draft_tokens):
                target_prob = F.softmax(target_logits[0, i, :], dim=-1)
                draft_prob = draft_probs_list[i]

                if sampling_params.temperature == 0:
                    # Greedy: accept if target agrees
                    target_token = target_logits[0, i, :].argmax().item()
                    if target_token == draft_token:
                        num_accepted += 1
                    else:
                        # Reject: use target's token instead
                        generated.append(target_token)
                        all_ids.append(target_token)
                        break
                else:
                    # Stochastic: accept with probability min(1, target_p / draft_p)
                    p_target = target_prob[draft_token].item()
                    p_draft = draft_prob[draft_token].item()

                    if p_draft == 0:
                        accept = False
                    else:
                        accept_prob = min(1.0, p_target / p_draft)
                        accept = torch.rand(1).item() < accept_prob

                    if accept:
                        num_accepted += 1
                    else:
                        # Resample from adjusted distribution
                        adjusted = torch.clamp(target_prob - draft_prob, min=0)
                        adjusted = adjusted / adjusted.sum()
                        new_token = torch.multinomial(adjusted, 1).item()
                        generated.append(new_token)
                        all_ids.append(new_token)
                        break

                # Accepted — add token
                generated.append(draft_token)
                all_ids.append(draft_token)

                if draft_token in sampling_params.stop_token_ids:
                    break

            self.total_accepted_tokens += num_accepted

            # If all draft tokens accepted, sample one bonus token from target
            if num_accepted == len(draft_tokens) and not (
                draft_tokens and draft_tokens[-1] in sampling_params.stop_token_ids
            ):
                bonus_logit = target_logits[0, len(draft_tokens), :]
                if sampling_params.temperature == 0:
                    bonus_token = bonus_logit.argmax().item()
                else:
                    bonus_token = sample(bonus_logit, sampling_params).item()
                generated.append(bonus_token)
                all_ids.append(bonus_token)

            # Update target KV cache (truncate to accepted length)
            accepted_len = num_accepted + 1  # +1 for the last token position
            target_kv = [
                (k[:, :, :start_pos + accepted_len + 1, :],
                 v[:, :, :start_pos + accepted_len + 1, :])
                for k, v in new_target_kv
            ]

            # Rebuild draft KV cache from scratch up to current position
            rebuild_input = torch.tensor([all_ids], dtype=torch.long, device=self.device)
            rebuild_pos = torch.arange(len(all_ids), device=self.device).unsqueeze(0)
            _, draft_kv = self.draft_model(rebuild_input, rebuild_pos)

            # Check stop condition
            if generated and generated[-1] in sampling_params.stop_token_ids:
                break
            if len(generated) >= sampling_params.max_tokens:
                break

        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "output": output_text,
            "output_token_ids": generated,
            "num_tokens": len(generated),
            "acceptance_rate": self.acceptance_rate,
            "total_draft": self.total_draft_tokens,
            "total_accepted": self.total_accepted_tokens,
        }
