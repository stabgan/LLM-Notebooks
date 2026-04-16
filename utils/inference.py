"""Inference optimization utilities for Notebook 11: Inference Optimization.

Provides KV-cache management, weight quantization (4-bit/8-bit), and
speculative decoding — all implemented in MLX on Apple Silicon.

**Validates: Requirements 9.1–9.9**
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Tiny transformer model for demos and testing
# ---------------------------------------------------------------------------

class SimpleLMAttention(nn.Module):
    """Single-head attention with optional KV-cache support.

    💡 Supports both full-sequence (prefill) and single-token (decode) modes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def __call__(self, x: mx.array, cache: Optional[dict] = None):
        """Forward pass with optional KV-cache.

        Args:
            x: Input tensor [batch, seq, d_model].
            cache: Optional dict with 'k' and 'v' tensors from previous steps.

        Returns:
            (output, updated_cache) where cache is a dict with 'k', 'v'.
        """
        q = self.q_proj(x)  # [B, S, D]
        k = self.k_proj(x)  # [B, S, D]
        v = self.v_proj(x)  # [B, S, D]

        # Update cache
        if cache is not None and cache.get("k") is not None:
            k = mx.concatenate([cache["k"], k], axis=1)
            v = mx.concatenate([cache["v"], v], axis=1)

        new_cache = {"k": k, "v": v}

        # Attention: [B, S_q, D] @ [B, D, S_kv] -> [B, S_q, S_kv]
        scores = (q @ k.transpose(0, 2, 1)) / self.scale

        # Causal mask
        s_q = q.shape[1]
        s_kv = k.shape[1]
        # Mask: query position i can attend to key positions 0..(offset + i)
        offset = s_kv - s_q
        mask = mx.triu(mx.full((s_q, s_kv), -1e9), k=offset + 1)
        scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [B, S_q, D]
        out = self.o_proj(out)
        return out, new_cache


class SimpleLMBlock(nn.Module):
    """Transformer block: attention + FFN with residual connections."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = SimpleLMAttention(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array, cache: Optional[dict] = None):
        h = self.norm1(x)
        attn_out, new_cache = self.attn(h, cache=cache)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn_down(nn.silu(self.ffn_up(h)))
        return x, new_cache


class SimpleLM(nn.Module):
    """Minimal language model for inference optimization demos.

    ⚡ Small enough to run on any Apple Silicon device.
    🎯 Interview tip: this architecture mirrors real LLMs (embedding → blocks → head).
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = [SimpleLMBlock(d_model, d_ff) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache: Optional[list] = None):
        """Forward pass returning logits and updated caches.

        Args:
            input_ids: Token IDs [batch, seq].
            cache: Optional list of per-layer cache dicts.

        Returns:
            (logits [batch, seq, vocab], list_of_caches)
        """
        x = self.embed(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x, c = block(x, cache=layer_cache)
            new_caches.append(c)
        x = self.norm(x)
        logits = self.head(x)
        return logits, new_caches



# ---------------------------------------------------------------------------
# 12.1  KVCacheManager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Manage KV-cache for autoregressive generation with prefill + decode.

    💡 Prefill: process the full prompt in one forward pass, populate cache.
    ⚡ Decode: generate one token at a time using cached K, V — O(1) per token
       instead of O(n) recomputation.

    🎯 Interview tip: KV-cache trades memory for compute. For a 7B model with
    seq_len=2048, the cache is ~1GB in float16.

    ⚠️ Cache grows linearly with sequence length — monitor memory!
    """

    def __init__(self, model: SimpleLM):
        self.model = model
        self.cache: Optional[list] = None

    def reset(self):
        """Clear the KV-cache."""
        self.cache = None

    def prefill(self, prompt_ids: mx.array) -> mx.array:
        """Process the full prompt and populate the KV-cache.

        Args:
            prompt_ids: Token IDs [batch, seq].

        Returns:
            logits: [batch, seq, vocab] — logits for every prompt position.
        """
        logits, self.cache = self.model(prompt_ids, cache=None)
        mx.eval(logits, *[c["k"] for c in self.cache], *[c["v"] for c in self.cache])
        return logits

    def decode_step(self, token_id: mx.array) -> mx.array:
        """Generate logits for a single new token using the cache.

        Args:
            token_id: Single token ID [batch, 1].

        Returns:
            logits: [batch, 1, vocab] — logits for the new position.
        """
        assert self.cache is not None, "Must call prefill() before decode_step()"
        logits, self.cache = self.model(token_id, cache=self.cache)
        mx.eval(logits, *[c["k"] for c in self.cache], *[c["v"] for c in self.cache])
        return logits

    def generate(self, prompt_ids: mx.array, max_new_tokens: int) -> tuple:
        """Generate tokens autoregressively using KV-cache.

        Args:
            prompt_ids: [batch, seq] prompt token IDs.
            max_new_tokens: Number of tokens to generate.

        Returns:
            (generated_ids, all_logits) where generated_ids is [batch, max_new_tokens]
            and all_logits is a list of [batch, 1, vocab] tensors.
        """
        # Prefill
        logits = self.prefill(prompt_ids)
        # Take last token's logits for first generation step
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)  # [B, 1]
        mx.eval(next_token)

        generated = [next_token]
        all_logits = [logits[:, -1:, :]]

        # Decode loop
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_token)
            next_token = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(next_token)
            generated.append(next_token)
            all_logits.append(logits)

        generated_ids = mx.concatenate(generated, axis=1)
        return generated_ids, all_logits

    def memory_bytes(self) -> int:
        """Return total bytes used by the KV-cache."""
        if self.cache is None:
            return 0
        total = 0
        for c in self.cache:
            if c.get("k") is not None:
                total += c["k"].nbytes + c["v"].nbytes
        return total

    @staticmethod
    def generate_without_cache(model: SimpleLM, full_ids: mx.array) -> mx.array:
        """Full recomputation without cache (for equivalence verification).

        Args:
            model: The language model.
            full_ids: Complete sequence [batch, total_seq].

        Returns:
            logits: [batch, total_seq, vocab].
        """
        logits, _ = model(full_ids, cache=None)
        mx.eval(logits)
        return logits



# ---------------------------------------------------------------------------
# 12.3  Quantizer
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Container for a quantized weight tensor.

    Stores integer codes, per-group scale and zero-point for dequantization.
    """
    codes: mx.array       # Integer codes [n_groups, group_size]
    scale: mx.array        # Per-group scale [n_groups, 1]
    zero_point: mx.array   # Per-group zero point (min value) [n_groups, 1]
    orig_shape: tuple      # Original tensor shape
    bits: int              # Quantization bit width (4 or 8)
    group_size: int        # Group size used


class Quantizer:
    """Weight quantization with 4-bit and 8-bit support, configurable group size.

    💡 Min-max quantization maps each group's range to 2^N integer levels.
    ⚡ 4-bit: ~4x compression. 8-bit: ~2x compression. Minimal quality loss.

    🎯 Interview tip: group-wise quantization limits error to each group's range,
    giving much better accuracy than per-tensor quantization.

    ⚠️ Error bound per group: max_error ≤ (max_val - min_val) / (2^N - 1).
    """

    @staticmethod
    def _quantize(weights: mx.array, bits: int, group_size: int) -> QuantizedTensor:
        """Core quantization: map float weights to N-bit integers per group.

        Args:
            weights: Float tensor of any shape.
            bits: Bit width (4 or 8).
            group_size: Number of elements per quantization group.

        Returns:
            QuantizedTensor with codes, scale, zero_point.
        """
        assert bits in (4, 8), f"Supported bit widths: 4, 8. Got {bits}"
        orig_shape = weights.shape
        flat = weights.reshape(-1)

        # Pad to multiple of group_size
        n_elem = flat.shape[0]
        remainder = n_elem % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            flat = mx.concatenate([flat, mx.zeros(pad_size)])
        else:
            pad_size = 0

        # Reshape into groups
        grouped = flat.reshape(-1, group_size)  # [n_groups, group_size]

        # Per-group min/max
        g_min = mx.min(grouped, axis=-1, keepdims=True)   # [n_groups, 1]
        g_max = mx.max(grouped, axis=-1, keepdims=True)   # [n_groups, 1]

        n_levels = 2 ** bits - 1
        scale = (g_max - g_min) / n_levels  # [n_groups, 1]

        # Avoid division by zero for constant groups
        safe_scale = mx.where(scale == 0, mx.ones_like(scale), scale)

        # Quantize: code = round((x - min) / scale), clipped to [0, n_levels]
        codes = mx.round((grouped - g_min) / safe_scale)
        codes = mx.clip(codes, 0, n_levels).astype(mx.uint8)

        mx.eval(codes, scale, g_min)

        return QuantizedTensor(
            codes=codes,
            scale=scale,
            zero_point=g_min,
            orig_shape=orig_shape,
            bits=bits,
            group_size=group_size,
        )

    @staticmethod
    def quantize_4bit(weights: mx.array, group_size: int = 64) -> QuantizedTensor:
        """Quantize weights to 4-bit integers with group-wise scaling.

        Args:
            weights: Float weight tensor.
            group_size: Elements per quantization group (default 64).

        Returns:
            QuantizedTensor with 4-bit codes.
        """
        return Quantizer._quantize(weights, bits=4, group_size=group_size)

    @staticmethod
    def quantize_8bit(weights: mx.array, group_size: int = 64) -> QuantizedTensor:
        """Quantize weights to 8-bit integers with group-wise scaling.

        Args:
            weights: Float weight tensor.
            group_size: Elements per quantization group (default 64).

        Returns:
            QuantizedTensor with 8-bit codes.
        """
        return Quantizer._quantize(weights, bits=8, group_size=group_size)

    @staticmethod
    def dequantize(qtensor: QuantizedTensor) -> mx.array:
        """Dequantize back to float32.

        Args:
            qtensor: A QuantizedTensor from quantize_4bit or quantize_8bit.

        Returns:
            Reconstructed float32 tensor with original shape.
        """
        # Reconstruct: x_hat = code * scale + zero_point
        reconstructed = qtensor.codes.astype(mx.float32) * qtensor.scale + qtensor.zero_point
        # Flatten and trim padding, then reshape
        flat = reconstructed.reshape(-1)
        n_orig = 1
        for s in qtensor.orig_shape:
            n_orig *= s
        flat = flat[:n_orig]
        result = flat.reshape(qtensor.orig_shape)
        mx.eval(result)
        return result

    @staticmethod
    def compute_error_bound(weights: mx.array, bits: int, group_size: int) -> mx.array:
        """Compute the theoretical max error per group: (max - min) / (2^N - 1).

        Args:
            weights: Original float weights.
            bits: Bit width.
            group_size: Group size.

        Returns:
            Per-group error bounds [n_groups].
        """
        flat = weights.reshape(-1)
        n_elem = flat.shape[0]
        remainder = n_elem % group_size
        if remainder != 0:
            pad_size = group_size - remainder
            flat = mx.concatenate([flat, mx.zeros(pad_size)])

        grouped = flat.reshape(-1, group_size)
        g_min = mx.min(grouped, axis=-1)
        g_max = mx.max(grouped, axis=-1)
        n_levels = 2 ** bits - 1
        bounds = (g_max - g_min) / n_levels
        mx.eval(bounds)
        return bounds

    @staticmethod
    def compression_stats(weights: mx.array, qtensor: QuantizedTensor) -> dict:
        """Compute compression statistics.

        Returns:
            Dict with original_bytes, quantized_bytes, compression_ratio, max_error.
        """
        orig_bytes = weights.nbytes
        # Codes + scales + zero_points
        q_bytes = qtensor.codes.nbytes + qtensor.scale.nbytes + qtensor.zero_point.nbytes

        deq = Quantizer.dequantize(qtensor)
        max_error = mx.max(mx.abs(weights - deq)).item()

        return {
            "original_bytes": orig_bytes,
            "quantized_bytes": q_bytes,
            "compression_ratio": orig_bytes / max(q_bytes, 1),
            "max_error": max_error,
            "bits": qtensor.bits,
        }



# ---------------------------------------------------------------------------
# 12.5  SpeculativeDecoder
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Speculative decoding: draft with a small model, verify with a large model.

    💡 The draft model proposes N tokens cheaply. The target model verifies them
    in a single forward pass. Accepted tokens are guaranteed identical to what
    the target model would generate alone.

    ⚡ Typical speedup: 2-3x when draft model has high acceptance rate.

    🎯 Interview tip: speculative decoding gives EXACT same output as the target
    model — it's a pure speed optimization with no quality loss (for greedy).

    ⚠️ Acceptance rate depends on how well the draft model matches the target.
    """

    def __init__(self, draft_model: SimpleLM, target_model: SimpleLM):
        self.draft_model = draft_model
        self.target_model = target_model

    @staticmethod
    def draft(draft_model: SimpleLM, prompt_ids: mx.array, n_draft: int) -> tuple:
        """Generate n_draft candidate tokens using the draft model (greedy).

        Args:
            draft_model: Small, fast model.
            prompt_ids: [batch, seq] prompt tokens.
            n_draft: Number of draft tokens to generate.

        Returns:
            (draft_tokens [batch, n_draft], draft_logits list of [batch, 1, vocab])
        """
        cache = None
        # Prefill
        logits, cache = draft_model(prompt_ids, cache=None)
        mx.eval(logits, *[c["k"] for c in cache], *[c["v"] for c in cache])

        tokens = []
        draft_logits = []
        next_tok = mx.argmax(logits[:, -1:, :], axis=-1)  # [B, 1]
        mx.eval(next_tok)
        tokens.append(next_tok)
        draft_logits.append(logits[:, -1:, :])

        for _ in range(n_draft - 1):
            logits, cache = draft_model(next_tok, cache=cache)
            mx.eval(logits, *[c["k"] for c in cache], *[c["v"] for c in cache])
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)
            mx.eval(next_tok)
            tokens.append(next_tok)
            draft_logits.append(logits[:, -1:, :])

        draft_tokens = mx.concatenate(tokens, axis=1)  # [B, n_draft]
        return draft_tokens, draft_logits

    @staticmethod
    def verify(
        target_model: SimpleLM,
        prompt_ids: mx.array,
        draft_tokens: mx.array,
    ) -> tuple:
        """Verify draft tokens against the target model (greedy).

        Runs the target model on [prompt + draft_tokens] in one forward pass,
        then checks which draft tokens match what the target would generate.

        Args:
            target_model: Large, accurate model.
            prompt_ids: [batch, seq] original prompt.
            draft_tokens: [batch, n_draft] proposed tokens.

        Returns:
            (accepted_tokens [batch, n_accepted], n_accepted int)
        """
        n_draft = draft_tokens.shape[1]

        # Run target on full sequence: prompt + draft tokens
        full_ids = mx.concatenate([prompt_ids, draft_tokens], axis=1)
        target_logits, _ = target_model(full_ids, cache=None)
        mx.eval(target_logits)

        prompt_len = prompt_ids.shape[1]

        # Target's greedy choices at each position after the prompt
        # target_logits[:, prompt_len-1, :] predicts position prompt_len (first generated)
        # target_logits[:, prompt_len, :] predicts position prompt_len+1, etc.
        n_accepted = 0
        accepted = []

        for i in range(n_draft):
            target_token = mx.argmax(target_logits[:, prompt_len - 1 + i, :], axis=-1)  # [B]
            draft_token = draft_tokens[:, i]  # [B]
            mx.eval(target_token, draft_token)

            if mx.array_equal(target_token, draft_token):
                n_accepted += 1
                accepted.append(draft_token[:, None])  # [B, 1]
            else:
                # Reject: use target's token instead and stop
                accepted.append(target_token[:, None])
                n_accepted += 1  # We still get one correct token (the target's)
                break

        if len(accepted) > 0:
            accepted_tokens = mx.concatenate(accepted, axis=1)
        else:
            # Fallback: just use target's first prediction
            first_target = mx.argmax(target_logits[:, prompt_len - 1, :], axis=-1)
            accepted_tokens = first_target[:, None]
            n_accepted = 1

        mx.eval(accepted_tokens)
        return accepted_tokens, n_accepted

    def generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int,
        n_draft: int = 4,
    ) -> tuple:
        """Full speculative decoding generation loop.

        Args:
            prompt_ids: [batch, seq] prompt.
            max_new_tokens: Total tokens to generate.
            n_draft: Draft tokens per iteration.

        Returns:
            (generated_ids [batch, n_generated], stats dict)
        """
        generated = []
        total_draft = 0
        total_accepted = 0
        current_prompt = prompt_ids

        remaining = max_new_tokens
        while remaining > 0:
            draft_n = min(n_draft, remaining)

            # Draft
            draft_tokens, _ = self.draft(
                self.draft_model, current_prompt, draft_n,
            )
            total_draft += draft_n

            # Verify
            accepted, n_acc = self.verify(
                self.target_model, current_prompt, draft_tokens,
            )
            total_accepted += n_acc
            generated.append(accepted)
            remaining -= accepted.shape[1]

            # Update prompt for next iteration
            current_prompt = mx.concatenate([current_prompt, accepted], axis=1)
            mx.eval(current_prompt)

        if generated:
            generated_ids = mx.concatenate(generated, axis=1)
            # Trim to max_new_tokens
            generated_ids = generated_ids[:, :max_new_tokens]
        else:
            generated_ids = mx.zeros((prompt_ids.shape[0], 0), dtype=mx.int32)

        acceptance_rate = total_accepted / max(total_draft, 1)
        stats = {
            "total_draft": total_draft,
            "total_accepted": total_accepted,
            "acceptance_rate": acceptance_rate,
        }
        return generated_ids, stats

    @staticmethod
    def generate_target_only(
        target_model: SimpleLM,
        prompt_ids: mx.array,
        max_new_tokens: int,
    ) -> mx.array:
        """Generate tokens using only the target model (greedy, no cache).

        This is the reference implementation — speculative decoding must
        produce identical tokens for accepted positions.

        Args:
            target_model: The target model.
            prompt_ids: [batch, seq] prompt.
            max_new_tokens: Tokens to generate.

        Returns:
            generated_ids [batch, max_new_tokens].
        """
        current = prompt_ids
        generated = []

        for _ in range(max_new_tokens):
            logits, _ = target_model(current, cache=None)
            mx.eval(logits)
            next_tok = mx.argmax(logits[:, -1:, :], axis=-1)  # [B, 1]
            mx.eval(next_tok)
            generated.append(next_tok)
            current = mx.concatenate([current, next_tok], axis=1)
            mx.eval(current)

        return mx.concatenate(generated, axis=1)


# ---------------------------------------------------------------------------
# Helper: create matched draft/target model pair
# ---------------------------------------------------------------------------

def create_model_pair(
    vocab_size: int = 64,
    d_model_draft: int = 32,
    d_model_target: int = 64,
    n_layers_draft: int = 1,
    n_layers_target: int = 2,
    d_ff_draft: int = 64,
    d_ff_target: int = 128,
    seed: int = 42,
) -> tuple:
    """Create a (draft_model, target_model) pair for speculative decoding demos.

    Returns:
        (draft_model, target_model) both SimpleLM instances.
    """
    mx.random.seed(seed)
    draft = SimpleLM(vocab_size, d_model_draft, n_layers_draft, d_ff_draft)
    target = SimpleLM(vocab_size, d_model_target, n_layers_target, d_ff_target)
    mx.eval(draft.parameters())
    mx.eval(target.parameters())
    return draft, target
