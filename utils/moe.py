"""Mixture of Experts routing implementations in MLX.

Provides three routing strategies for MoE layers:
- MoERouter: Top-K learned routing (Mixtral, DeepSeek-V3 style)
- ExpertChoiceRouter: Experts choose tokens (V-MoE style)
- HashRouter: Deterministic hash-based routing (baseline)

All routers share the same interface:
    route(x) -> (indices [B, S, K], weights [B, S, K])
where weights sum to 1.0 along the last axis for every token.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """Configuration for a Mixture of Experts layer."""
    d_model: int = 64              # Model dimension
    num_experts: int = 8           # Total number of experts (N)
    num_active: int = 2            # Experts active per token (K)
    d_ff: int = 256                # FFN hidden dimension per expert
    has_shared_expert: bool = False # Shared expert (Gemma 4 / DeepSeek-V3 style)
    load_balance_weight: float = 0.01  # Auxiliary loss weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top_k(x: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Select top-k values and indices along the last axis.

    Args:
        x: Input tensor of any shape.
        k: Number of top elements to select.

    Returns:
        (values, indices) — each with last dim = k.
    """
    # argpartition gives indices where the k-th smallest element is in
    # position k.  We negate x to get top-k (largest) instead of bottom-k.
    indices = mx.argpartition(-x, kth=k - 1, axis=-1)[..., :k]
    assert indices.shape[-1] == k, (
        f"top_k indices last dim should be {k}, got {indices.shape[-1]}"
    )
    values = mx.take_along_axis(x, indices, axis=-1)
    assert values.shape == indices.shape, (
        f"top_k values shape {values.shape} != indices shape {indices.shape}"
    )
    return values, indices


# ---------------------------------------------------------------------------
# Top-K Router
# ---------------------------------------------------------------------------

class MoERouter(nn.Module):
    """Top-K gating router for Mixture of Experts.

    Routes each token to the top-k experts based on learned gating weights.
    Returns routing indices and normalised weights that sum to 1.0.

    This is the standard routing used in Mixtral and most MoE models.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_active = config.num_active
        # Gate: simple linear projection d_model -> num_experts (no bias)
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        # Cache last logits for load-balance loss
        self._last_logits: mx.array | None = None

    # ---- core routing ---------------------------------------------------

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route tokens to experts using Top-K gating.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.

        Returns:
            indices: Expert indices ``[batch, seq, k]``.
            weights: Routing weights ``[batch, seq, k]`` (sum to 1.0).
        """
        batch, seq, d_model = x.shape
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )

        # Step 1 — router logits
        router_logits = self.gate(x)  # [batch, seq, num_experts]
        assert router_logits.shape == (batch, seq, self.num_experts), (
            f"Router logits shape mismatch: {router_logits.shape}"
        )
        self._last_logits = router_logits

        # Step 2 — top-k selection
        top_k_logits, indices = top_k(router_logits, self.num_active)
        assert top_k_logits.shape == (batch, seq, self.num_active), (
            f"Top-K logits shape mismatch: {top_k_logits.shape}"
        )
        assert indices.shape == (batch, seq, self.num_active), (
            f"Top-K indices shape mismatch: {indices.shape}"
        )

        # Step 3 — softmax over selected experts only → weights sum to 1.0
        weights = mx.softmax(top_k_logits, axis=-1)
        assert weights.shape == (batch, seq, self.num_active), (
            f"Weights shape mismatch: {weights.shape}"
        )

        return indices, weights

    # ---- load-balance loss ----------------------------------------------

    def compute_load_balance_loss(self) -> mx.array:
        """Compute auxiliary load-balancing loss.

        Formula: ``num_experts × Σ(f_i × p_i)``
        where ``f_i`` = fraction of tokens routed to expert *i*
        and   ``p_i`` = mean routing probability for expert *i*.

        Must call :meth:`route` first to populate ``_last_logits``.
        """
        assert self._last_logits is not None, (
            "Call route() before computing load balance loss"
        )

        router_logits = self._last_logits  # [batch, seq, num_experts]
        batch, seq, num_experts = router_logits.shape

        # Flatten batch × seq
        logits_flat = router_logits.reshape(-1, num_experts)  # [N, E]
        N = logits_flat.shape[0]
        assert logits_flat.shape == (N, num_experts), (
            f"Flattened logits shape mismatch: {logits_flat.shape}"
        )

        # Routing decisions
        _, indices = top_k(logits_flat, self.num_active)  # [N, k]
        assert indices.shape == (N, self.num_active), (
            f"Indices shape mismatch: {indices.shape}"
        )

        # f_i — fraction of tokens routed to each expert
        one_hot = mx.zeros((N, num_experts))
        rows = mx.arange(N)
        for k_idx in range(self.num_active):
            expert_indices = indices[:, k_idx]  # [N]
            one_hot = one_hot.at[rows, expert_indices].add(1.0)

        f = mx.mean(one_hot, axis=0)  # [num_experts]
        assert f.shape == (num_experts,), f"f shape mismatch: {f.shape}"

        # p_i — mean routing probability for each expert
        probs = mx.softmax(logits_flat, axis=-1)  # [N, num_experts]
        assert probs.shape == (N, num_experts), (
            f"Probs shape mismatch: {probs.shape}"
        )
        p = mx.mean(probs, axis=0)  # [num_experts]
        assert p.shape == (num_experts,), f"p shape mismatch: {p.shape}"

        # Load-balance loss = num_experts × Σ(f_i × p_i)
        aux_loss = num_experts * mx.sum(f * p)
        return aux_loss

    # ---- convenience -----------------------------------------------------

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Convenience: ``route()`` is the forward pass."""
        return self.route(x)


# ---------------------------------------------------------------------------
# Expert Choice Router
# ---------------------------------------------------------------------------

class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing — experts choose tokens instead of tokens
    choosing experts.

    Each expert selects its top-C tokens from the full token set.
    Guarantees perfect load balance by construction.

    Used in: V-MoE (Google, 2022), some Switch Transformer variants.

    For comparison purposes the output is converted back to per-token
    ``(indices, weights)`` format matching the :class:`MoERouter` interface.
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_active = config.num_active
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route using Expert Choice: experts select their top-C tokens.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.

        Returns:
            indices: Expert indices ``[batch, seq, k]``.
            weights: Routing weights ``[batch, seq, k]`` (sum to 1.0).
        """
        batch, seq, d_model = x.shape
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )
        N_tokens = batch * seq

        # Step 1 — router logits
        router_logits = self.gate(x)  # [batch, seq, num_experts]
        assert router_logits.shape == (batch, seq, self.num_experts), (
            f"Router logits shape mismatch: {router_logits.shape}"
        )

        # Step 2 — reshape to [N_tokens, E] then transpose to [E, N_tokens]
        logits_flat = router_logits.reshape(N_tokens, self.num_experts)
        assert logits_flat.shape == (N_tokens, self.num_experts), (
            f"Flat logits shape mismatch: {logits_flat.shape}"
        )
        expert_logits = logits_flat.T  # [E, N]
        assert expert_logits.shape == (self.num_experts, N_tokens), (
            f"Expert logits shape mismatch: {expert_logits.shape}"
        )

        # Step 3 — each expert selects top-C tokens
        capacity = max(1, (self.num_active * N_tokens) // self.num_experts)

        assignment = mx.zeros((N_tokens, self.num_experts))
        for e in range(self.num_experts):
            expert_scores = expert_logits[e]  # [N_tokens]
            assert expert_scores.shape == (N_tokens,), (
                f"Expert {e} scores shape mismatch: {expert_scores.shape}"
            )

            if capacity < N_tokens:
                top_c_indices = mx.argpartition(
                    -expert_scores, kth=capacity - 1
                )[:capacity]
            else:
                top_c_indices = mx.arange(N_tokens)

            selected_scores = expert_scores[top_c_indices]
            selected_weights = mx.softmax(selected_scores, axis=-1)
            assignment = assignment.at[top_c_indices, e].add(selected_weights)

        mx.eval(assignment)

        # Step 4 — convert back to per-token top-k format
        top_k_weights, top_k_indices = top_k(assignment, self.num_active)
        assert top_k_weights.shape == (N_tokens, self.num_active), (
            f"Top-K weights shape mismatch: {top_k_weights.shape}"
        )
        assert top_k_indices.shape == (N_tokens, self.num_active), (
            f"Top-K indices shape mismatch: {top_k_indices.shape}"
        )

        # Normalise weights to sum to 1.0 per token.
        # Tokens not selected by any expert get uniform weights.
        weight_sums = mx.sum(top_k_weights, axis=-1, keepdims=True)
        unselected = weight_sums <= 0
        safe_sums = mx.where(unselected, mx.ones_like(weight_sums), weight_sums)
        weights = mx.where(
            mx.broadcast_to(unselected, top_k_weights.shape),
            mx.ones((N_tokens, self.num_active)) / self.num_active,
            top_k_weights / safe_sums,
        )
        assert weights.shape == (N_tokens, self.num_active), (
            f"Normalised weights shape mismatch: {weights.shape}"
        )

        # Reshape back to [batch, seq, k]
        indices = top_k_indices.reshape(batch, seq, self.num_active)
        weights = weights.reshape(batch, seq, self.num_active)
        assert indices.shape == (batch, seq, self.num_active), (
            f"Final indices shape mismatch: {indices.shape}"
        )
        assert weights.shape == (batch, seq, self.num_active), (
            f"Final weights shape mismatch: {weights.shape}"
        )

        return indices, weights

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.route(x)


# ---------------------------------------------------------------------------
# Hash Router
# ---------------------------------------------------------------------------

class HashRouter(nn.Module):
    """Hash-based routing — deterministic, no learned parameters.

    Assigns tokens to experts using position-based hashing.
    Useful as a baseline: if your learned router can't beat this,
    your routing is broken.

    Hash routing appears in Switch Transformer ablations and is used
    in some production systems for its zero-overhead routing.
    """

    _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_active = config.num_active

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Route using deterministic hashing on token positions.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.

        Returns:
            indices: Expert indices ``[batch, seq, k]``.
            weights: Uniform weights ``[batch, seq, k]`` (each = 1/k).
        """
        batch, seq, d_model = x.shape
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )

        # Position indices [batch, seq]
        positions = mx.broadcast_to(
            mx.arange(seq).reshape(1, seq), (batch, seq)
        )
        assert positions.shape == (batch, seq), (
            f"Positions shape mismatch: {positions.shape}"
        )

        # Batch offset so different batches get different routing
        batch_offsets = mx.broadcast_to(
            mx.arange(batch).reshape(batch, 1) * 1000003, (batch, seq)
        )
        base_hash = positions + batch_offsets
        assert base_hash.shape == (batch, seq), (
            f"Base hash shape mismatch: {base_hash.shape}"
        )

        # Generate k expert assignments using k different hash functions
        all_indices = []
        for k_idx in range(self.num_active):
            prime = self._PRIMES[k_idx % len(self._PRIMES)]
            hashed = mx.abs(
                (base_hash * prime + k_idx * 7919) % self.num_experts
            )
            assert hashed.shape == (batch, seq), (
                f"Hash {k_idx} shape mismatch: {hashed.shape}"
            )
            all_indices.append(hashed)

        indices = mx.stack(all_indices, axis=-1)  # [batch, seq, k]
        assert indices.shape == (batch, seq, self.num_active), (
            f"Indices shape mismatch: {indices.shape}"
        )

        # Handle collisions — shift duplicates to next expert
        for k_idx in range(1, self.num_active):
            for prev_idx in range(k_idx):
                collision = indices[..., k_idx] == indices[..., prev_idx]
                shifted = (indices[..., k_idx] + 1) % self.num_experts
                indices = indices.at[..., k_idx].add(
                    mx.where(
                        collision,
                        shifted - indices[..., k_idx],
                        mx.zeros_like(shifted),
                    )
                )

        assert indices.shape == (batch, seq, self.num_active), (
            f"Final indices shape mismatch: {indices.shape}"
        )

        # Uniform weights: each expert gets 1/k
        weights = mx.ones((batch, seq, self.num_active)) / self.num_active
        assert weights.shape == (batch, seq, self.num_active), (
            f"Weights shape mismatch: {weights.shape}"
        )

        return indices, weights

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return self.route(x)


# ---------------------------------------------------------------------------
# Expert FFN
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single expert feed-forward network with SwiGLU activation.

    Architecture (SwiGLU-style, used in LLaMA / Mixtral / Gemma 4):
        output = W2( SiLU(W1(x)) * W3(x) )

    Where:
        W1 (gate_proj):  d_model -> d_ff
        W3 (up_proj):    d_model -> d_ff
        W2 (down_proj):  d_ff -> d_model

    Input shape:  [batch, seq, d_model]
    Output shape: [batch, seq, d_model]
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        # SwiGLU uses three projections
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)  # W1
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)    # W3
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)  # W2

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the expert FFN.

        Args:
            x: Input tensor ``[batch, seq, d_model]`` or ``[N, d_model]``.

        Returns:
            Output tensor with same shape as input.
        """
        original_shape = x.shape
        d_model = original_shape[-1]
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )

        # SwiGLU: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
        gate = self.gate_proj(x)          # [..., d_ff]
        assert gate.shape[:-1] == x.shape[:-1], (
            f"Gate shape mismatch: {gate.shape}"
        )
        assert gate.shape[-1] == self.config.d_ff, (
            f"Gate last dim should be {self.config.d_ff}, got {gate.shape[-1]}"
        )

        up = self.up_proj(x)              # [..., d_ff]
        assert up.shape == gate.shape, (
            f"Up shape {up.shape} != gate shape {gate.shape}"
        )

        hidden = nn.silu(gate) * up       # [..., d_ff]
        assert hidden.shape == gate.shape, (
            f"Hidden shape {hidden.shape} != gate shape {gate.shape}"
        )

        output = self.down_proj(hidden)    # [..., d_model]
        assert output.shape == original_shape, (
            f"Output shape {output.shape} != input shape {original_shape}"
        )

        return output


# ---------------------------------------------------------------------------
# MoE Block
# ---------------------------------------------------------------------------

def _check_memory_budget(limit_bytes: int = 20 * 1024**3) -> bool:
    """Check if current Metal memory usage is under the budget.

    Returns True if within budget, False if over.
    """
    try:
        active = mx.metal.get_active_memory()
        return active < limit_bytes
    except Exception:
        # If Metal memory query is unavailable, assume OK
        return True


class MoEBlock(nn.Module):
    """Full Mixture of Experts block.

    Combines a router, multiple expert FFNs, and an optional shared expert.

    Forward pass:
        1. Route tokens to top-k experts via MoERouter
        2. Process tokens through selected experts
        3. Combine expert outputs with routing weights
        4. (Optional) Add shared expert output for all tokens
        5. Compute load balancing auxiliary loss

    Input shape:  [batch, seq, d_model]
    Output: (output [batch, seq, d_model], aux_loss scalar)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # Router
        self.router = MoERouter(config)

        # Expert FFNs
        self.experts = [ExpertFFN(config) for _ in range(config.num_experts)]

        # Shared expert (Gemma 4 / DeepSeek-V3 style)
        self.shared_expert: ExpertFFN | None = None
        if config.has_shared_expert:
            self.shared_expert = ExpertFFN(config)

    # ---- forward pass ---------------------------------------------------

    def __call__(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array]:
        """MoE forward pass with OOM recovery.

        Args:
            x: Input tensor ``[batch, seq, d_model]``.

        Returns:
            (output, aux_loss) where output has same shape as x.
        """
        batch, seq, d_model = x.shape
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )

        # Try standard (parallel) forward; fall back to sequential on OOM
        try:
            output, aux_loss = self._forward_parallel(x)
        except (MemoryError, RuntimeError):
            # OOM recovery: switch to sequential expert evaluation
            output, aux_loss = self._forward_sequential(x)

        assert output.shape == (batch, seq, d_model), (
            f"Output shape {output.shape} != input shape {(batch, seq, d_model)}"
        )
        return output, aux_loss

    # ---- parallel forward (default) -------------------------------------

    def _forward_parallel(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Standard forward: evaluate all active experts per token."""
        batch, seq, d_model = x.shape
        N = batch * seq  # total tokens

        # Step 1 — route tokens
        indices, weights = self.router.route(x)  # [B, S, k], [B, S, k]
        assert indices.shape == (batch, seq, self.config.num_active), (
            f"Indices shape mismatch: {indices.shape}"
        )
        assert weights.shape == (batch, seq, self.config.num_active), (
            f"Weights shape mismatch: {weights.shape}"
        )

        # Flatten to [N, d_model] for easier indexing
        x_flat = x.reshape(N, d_model)                    # [N, d_model]
        indices_flat = indices.reshape(N, self.config.num_active)  # [N, k]
        weights_flat = weights.reshape(N, self.config.num_active)  # [N, k]

        # Step 2 — compute expert outputs
        combined = mx.zeros_like(x_flat)  # [N, d_model]

        for slot in range(self.config.num_active):
            slot_indices = indices_flat[:, slot]    # [N] — which expert
            slot_weights = weights_flat[:, slot]    # [N] — weight for this slot

            for expert_id in range(self.config.num_experts):
                # Mask: which tokens go to this expert in this slot
                mask = slot_indices == expert_id  # [N] bool
                mask_float = mask.astype(mx.float32)  # [N]

                # Skip if no tokens routed here
                token_count = mx.sum(mask_float)
                mx.eval(token_count)
                if token_count.item() == 0:
                    continue

                # Process ALL tokens through expert, then mask out unselected
                # (avoids dynamic indexing issues in MLX)
                expert_out = self.experts[expert_id](x_flat)  # [N, d_model]
                assert expert_out.shape == x_flat.shape, (
                    f"Expert {expert_id} output shape {expert_out.shape} "
                    f"!= input shape {x_flat.shape}"
                )

                # Weight and mask: only keep tokens assigned to this expert
                w = (slot_weights * mask_float)[:, None]  # [N, 1]
                combined = combined + w * expert_out

        # Step 3 — shared expert (processes ALL tokens)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)  # [N, d_model]
            assert shared_out.shape == x_flat.shape, (
                f"Shared expert output shape {shared_out.shape} "
                f"!= input shape {x_flat.shape}"
            )
            combined = combined + shared_out

        # Reshape back to [batch, seq, d_model]
        output = combined.reshape(batch, seq, d_model)
        assert output.shape == (batch, seq, d_model), (
            f"Output shape {output.shape} != expected {(batch, seq, d_model)}"
        )

        # Step 4 — load balancing loss
        raw_loss = self.router.compute_load_balance_loss()
        aux_loss = self.config.load_balance_weight * raw_loss

        return output, aux_loss

    # ---- sequential forward (OOM fallback) ------------------------------

    def _forward_sequential(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """OOM-safe forward: evaluate one expert at a time, freeing memory
        between experts.  Slower but uses less peak memory.
        """
        batch, seq, d_model = x.shape
        N = batch * seq

        # Route
        indices, weights = self.router.route(x)
        x_flat = x.reshape(N, d_model)
        indices_flat = indices.reshape(N, self.config.num_active)
        weights_flat = weights.reshape(N, self.config.num_active)

        combined = mx.zeros_like(x_flat)

        for slot in range(self.config.num_active):
            slot_indices = indices_flat[:, slot]
            slot_weights = weights_flat[:, slot]

            for expert_id in range(self.config.num_experts):
                mask = slot_indices == expert_id
                mask_float = mask.astype(mx.float32)

                token_count = mx.sum(mask_float)
                mx.eval(token_count)
                if token_count.item() == 0:
                    continue

                # Sequential: evaluate and immediately accumulate
                expert_out = self.experts[expert_id](x_flat)
                w = (slot_weights * mask_float)[:, None]
                combined = combined + w * expert_out

                # Force evaluation to free intermediate memory
                mx.eval(combined)

        # Shared expert
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            combined = combined + shared_out
            mx.eval(combined)

        output = combined.reshape(batch, seq, d_model)

        raw_loss = self.router.compute_load_balance_loss()
        aux_loss = self.config.load_balance_weight * raw_loss

        return output, aux_loss
