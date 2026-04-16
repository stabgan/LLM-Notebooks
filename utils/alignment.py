"""Alignment utilities for Notebook 17: RLHF, DPO, and GRPO.

Provides a simple reward model built entirely in MLX that maps
(prompt, response) token sequences to scalar reward scores.

Architecture
------------
RewardModel wraps a small transformer base model and adds a linear
reward head that projects the last hidden state at the final token
position to a scalar.

    input_ids [batch, seq]
        → embedding + positional encoding  [batch, seq, d_model]
        → N × SimpleTransformerBlock       [batch, seq, d_model]
        → take last-token hidden state     [batch, d_model]
        → reward_head (Linear d_model→1)   [batch, 1]
"""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    """Configuration for the reward model.

    Parameters
    ----------
    d_model : int
        Hidden dimension of the transformer base model.
    n_heads : int
        Number of attention heads.  ``d_model`` must be divisible by ``n_heads``.
    n_layers : int
        Number of transformer blocks in the base model.
    vocab_size : int
        Vocabulary size for the token embedding table.
    max_seq_len : int
        Maximum sequence length (used for positional encoding).
    """

    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    vocab_size: int = 256
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )


# ---------------------------------------------------------------------------
# SimpleTransformerBlock — basic transformer block for the base model
# ---------------------------------------------------------------------------

class SimpleTransformerBlock(nn.Module):
    """A minimal pre-norm transformer block (attention + FFN).

    Uses multi-head self-attention with RMSNorm and a SiLU-gated FFN,
    matching the style used in modern small LLMs.

    Shapes
    ------
    Input  : ``[batch, seq, d_model]``
    Output : ``[batch, seq, d_model]``
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm layers
        self.attn_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

        # Multi-head self-attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # SiLU-gated FFN (gate + up → SiLU(gate) * up → down)
        d_ff = 4 * d_model
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.scale = math.sqrt(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_model]``

        Returns
        -------
        mx.array, shape ``[batch, seq, d_model]``
        """
        batch, seq, d = x.shape
        assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"

        # --- Self-attention with pre-norm ---
        residual = x
        x_norm = self.attn_norm(x)
        assert x_norm.shape == (batch, seq, self.d_model)

        # QKV projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        assert q.shape == (batch, seq, self.d_model)

        # Reshape to multi-head: [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        q = q.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        assert q.shape == (batch, self.n_heads, seq, self.head_dim)

        # Scaled dot-product attention with causal mask
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) / self.scale
        assert attn_weights.shape == (batch, self.n_heads, seq, seq)

        # Causal mask: prevent attending to future positions
        mask = mx.triu(mx.full((seq, seq), float("-inf")), k=1)
        attn_weights = attn_weights + mask
        attn_weights = mx.softmax(attn_weights, axis=-1)

        attn_out = attn_weights @ v
        assert attn_out.shape == (batch, self.n_heads, seq, self.head_dim)

        # Merge heads: [batch, n_heads, seq, head_dim] → [batch, seq, d_model]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
        attn_out = self.o_proj(attn_out)
        assert attn_out.shape == (batch, seq, self.d_model)

        x = residual + attn_out

        # --- FFN with pre-norm ---
        residual = x
        x_norm = self.ffn_norm(x)
        assert x_norm.shape == (batch, seq, self.d_model)

        gate = nn.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        ffn_out = self.down_proj(gate * up)
        assert ffn_out.shape == (batch, seq, self.d_model)

        x = residual + ffn_out
        assert x.shape == (batch, seq, self.d_model)
        return x


# ---------------------------------------------------------------------------
# RewardModel — transformer base + linear reward head
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Reward model that maps token sequences to scalar reward scores.

    Architecture: token embedding + sinusoidal positional encoding
    → N transformer blocks → last-token hidden state → linear head → scalar.

    This follows the standard reward model design from InstructGPT / RLHF:
    take a pretrained LM, replace the language-modelling head with a scalar
    projection, and train on preference pairs via the Bradley-Terry loss.

    Parameters
    ----------
    config : RewardModelConfig
        Model configuration.

    Shapes
    ------
    ``forward(input_ids)``
        Input  : ``[batch, seq]``  (integer token IDs)
        Output : ``[batch, 1]``    (scalar reward per sequence)
    """

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        self.pos_enc = self._make_sinusoidal_pe(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = [
            SimpleTransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ]

        # Final norm before reward head
        self.final_norm = nn.RMSNorm(config.d_model)

        # Reward head: project last hidden state to scalar
        self.reward_head = nn.Linear(config.d_model, 1, bias=True)

    # ------------------------------------------------------------------
    # Positional encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sinusoidal_pe(max_len: int, d_model: int) -> mx.array:
        """Create sinusoidal positional encoding table.

        Returns
        -------
        mx.array, shape ``[1, max_len, d_model]``
        """
        positions = mx.arange(max_len, dtype=mx.float32)[:, None]       # [max_len, 1]
        dims = mx.arange(0, d_model, 2, dtype=mx.float32)[None, :]     # [1, d_model//2]
        angles = positions / mx.power(10000.0, dims / d_model)          # [max_len, d_model//2]

        pe = mx.zeros((max_len, d_model))
        pe[:, 0::2] = mx.sin(angles)
        pe[:, 1::2] = mx.cos(angles)
        return pe[None, :, :]  # [1, max_len, d_model]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Compute reward scores for a batch of token sequences.

        Parameters
        ----------
        input_ids : mx.array, shape ``[batch, seq]``
            Integer token IDs.

        Returns
        -------
        mx.array, shape ``[batch, 1]``
            Scalar reward score for each sequence.
        """
        batch, seq = input_ids.shape
        assert seq <= self.config.max_seq_len, (
            f"Sequence length {seq} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # 1. Token embedding
        x = self.embed(input_ids)
        assert x.shape == (batch, seq, self.config.d_model), (
            f"Embedding shape {x.shape} != expected ({batch}, {seq}, {self.config.d_model})"
        )

        # 2. Add positional encoding
        x = x + self.pos_enc[:, :seq, :]
        assert x.shape == (batch, seq, self.config.d_model)

        # 3. Run through transformer blocks
        for block in self.blocks:
            x = block(x)
            assert x.shape == (batch, seq, self.config.d_model)

        # 4. Final norm
        x = self.final_norm(x)
        assert x.shape == (batch, seq, self.config.d_model)

        # 5. Extract last-token hidden state
        #    For reward modelling we use the hidden state at the *last* token
        #    position, which has seen the entire (prompt, response) via causal
        #    attention.
        last_hidden = x[:, -1, :]
        assert last_hidden.shape == (batch, self.config.d_model), (
            f"last_hidden shape {last_hidden.shape} != ({batch}, {self.config.d_model})"
        )

        # 6. Project to scalar reward
        reward = self.reward_head(last_hidden)
        assert reward.shape == (batch, 1), (
            f"reward shape {reward.shape} != ({batch}, 1)"
        )

        return reward

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------

    def compute_reward(self, input_ids: mx.array) -> mx.array:
        """Convenience wrapper returning a scalar reward per sequence.

        Parameters
        ----------
        input_ids : mx.array, shape ``[batch, seq]``

        Returns
        -------
        mx.array, shape ``[batch]``
            Scalar reward for each sequence (squeezed from ``[batch, 1]``).
        """
        return self.__call__(input_ids).squeeze(-1)


# ---------------------------------------------------------------------------
# DPO Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization.

    Parameters
    ----------
    beta : float
        KL penalty coefficient controlling the alignment-capability tradeoff.
        Higher β → more conservative updates (closer to reference).
    learning_rate : float
        Learning rate for the policy model optimizer.
    max_length : int
        Maximum sequence length for preference pairs.
    """

    beta: float = 0.1
    learning_rate: float = 1e-4
    max_length: int = 128

    def __post_init__(self) -> None:
        assert self.beta > 0, f"beta must be positive, got {self.beta}"
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"


# ---------------------------------------------------------------------------
# SimpleLM — small language model for DPO policy / reference
# ---------------------------------------------------------------------------

class SimpleLM(nn.Module):
    """A small language model that outputs logits over the vocabulary.

    Reuses ``SimpleTransformerBlock`` as the backbone and adds a linear
    language-modelling head that projects hidden states to vocab logits.

    Architecture::

        input_ids [batch, seq]
            → embedding + sinusoidal PE  [batch, seq, d_model]
            → N × SimpleTransformerBlock [batch, seq, d_model]
            → RMSNorm                    [batch, seq, d_model]
            → lm_head (Linear)           [batch, seq, vocab_size]

    Parameters
    ----------
    config : RewardModelConfig
        Model configuration (reuses the same config structure).
    """

    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Sinusoidal positional encoding (fixed)
        self.pos_enc = RewardModel._make_sinusoidal_pe(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = [
            SimpleTransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ]

        # Final norm + LM head
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass returning logits.

        Parameters
        ----------
        input_ids : mx.array, shape ``[batch, seq]``

        Returns
        -------
        mx.array, shape ``[batch, seq, vocab_size]``
        """
        batch, seq = input_ids.shape
        assert seq <= self.config.max_seq_len, (
            f"Sequence length {seq} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Embed + positional encoding
        x = self.embed(input_ids)
        assert x.shape == (batch, seq, self.config.d_model)
        x = x + self.pos_enc[:, :seq, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            assert x.shape == (batch, seq, self.config.d_model)

        # Norm + project to vocab
        x = self.final_norm(x)
        logits = self.lm_head(x)
        assert logits.shape == (batch, seq, self.config.vocab_size)
        return logits


# ---------------------------------------------------------------------------
# DPOTrainer — Direct Preference Optimization
# ---------------------------------------------------------------------------

class DPOTrainer:
    """Trainer implementing Direct Preference Optimization (DPO).

    DPO eliminates the need for a separate reward model by reparameterizing
    the reward as a function of the policy and reference model log-ratios:

        loss = -mean(log σ(β × (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log π_policy(y) - log π_ref(y).

    The reference model is kept frozen — only the policy model is updated.

    Parameters
    ----------
    config : DPOConfig
        DPO hyperparameters (beta, learning_rate, max_length).
    policy_model : SimpleLM
        The policy model to be optimized.
    reference_model : SimpleLM
        The frozen reference model (typically a copy of the initial policy).
    """

    def __init__(
        self,
        config: DPOConfig,
        policy_model: SimpleLM,
        reference_model: SimpleLM,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model

        # Freeze reference model — no gradient updates
        self.reference_model.freeze()

        # Optimizer for policy model only
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)

        # Checkpoint storage for NaN recovery
        self._last_good_params: dict | None = None

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_log_probs(model: SimpleLM, input_ids: mx.array) -> mx.array:
        """Compute per-sequence log probability under a model.

        For each sequence, sums the log-softmax of the predicted token at
        each position (teacher-forced), using positions 0..T-2 to predict
        tokens at positions 1..T-1.

        Parameters
        ----------
        model : SimpleLM
            Language model producing logits ``[batch, seq, vocab_size]``.
        input_ids : mx.array, shape ``[batch, seq]``
            Token IDs for the full sequence.

        Returns
        -------
        mx.array, shape ``[batch]``
            Total log probability for each sequence.
        """
        batch, seq = input_ids.shape
        assert seq >= 2, "Sequence must have at least 2 tokens for next-token prediction"

        # Forward pass → logits [batch, seq, vocab_size]
        logits = model(input_ids)
        assert logits.shape == (batch, seq, model.config.vocab_size)

        # Log-softmax over vocabulary
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        assert log_probs.shape == logits.shape

        # Shift: use positions 0..T-2 to predict tokens at 1..T-1
        # log_probs_shifted: [batch, seq-1, vocab_size]
        log_probs_shifted = log_probs[:, :-1, :]
        # labels: [batch, seq-1]
        labels = input_ids[:, 1:]

        assert log_probs_shifted.shape == (batch, seq - 1, model.config.vocab_size)
        assert labels.shape == (batch, seq - 1)

        # Gather log-probs of the actual next tokens
        # Use advanced indexing: for each (b, t), pick log_probs_shifted[b, t, labels[b, t]]
        batch_idx = mx.broadcast_to(
            mx.arange(batch)[:, None], labels.shape
        )
        seq_idx = mx.broadcast_to(
            mx.arange(seq - 1)[None, :], labels.shape
        )
        token_log_probs = log_probs_shifted[batch_idx, seq_idx, labels]
        assert token_log_probs.shape == (batch, seq - 1)

        # Sum over sequence to get per-sequence log probability
        seq_log_probs = mx.sum(token_log_probs, axis=-1)
        assert seq_log_probs.shape == (batch,)

        return seq_log_probs

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------

    def dpo_loss(
        self,
        chosen_ids: mx.array,
        rejected_ids: mx.array,
    ) -> mx.array:
        """Compute the DPO loss for a batch of preference pairs.

        loss = -mean(log σ(β × (log_ratio_chosen - log_ratio_rejected)))

        where log_ratio = log π_policy(y) - log π_ref(y).

        Parameters
        ----------
        chosen_ids : mx.array, shape ``[batch, seq]``
            Token IDs for the preferred (chosen) responses.
        rejected_ids : mx.array, shape ``[batch, seq]``
            Token IDs for the rejected responses.

        Returns
        -------
        mx.array, scalar
            The DPO loss (non-negative).
        """
        assert chosen_ids.shape == rejected_ids.shape, (
            f"chosen {chosen_ids.shape} != rejected {rejected_ids.shape}"
        )

        # Log-probs under policy
        log_pi_chosen = self.compute_log_probs(self.policy_model, chosen_ids)
        log_pi_rejected = self.compute_log_probs(self.policy_model, rejected_ids)

        # Log-probs under reference (no gradients)
        log_ref_chosen = mx.stop_gradient(
            self.compute_log_probs(self.reference_model, chosen_ids)
        )
        log_ref_rejected = mx.stop_gradient(
            self.compute_log_probs(self.reference_model, rejected_ids)
        )

        # Log-ratios
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected

        # DPO loss: -mean(log σ(β × (log_ratio_chosen - log_ratio_rejected)))
        logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        # log σ(x) = -softplus(-x) = x - softplus(x) — numerically stable
        log_sigmoid = logits - mx.logaddexp(mx.zeros_like(logits), logits)
        loss = -mx.mean(log_sigmoid)

        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _loss_fn(
        self,
        chosen_ids: mx.array,
        rejected_ids: mx.array,
    ) -> mx.array:
        """Internal loss function compatible with ``nn.value_and_grad``.

        ``nn.value_and_grad(model, fn)`` internally updates the model's
        parameters and then calls ``fn(*args)``.  So this function uses
        ``self.policy_model`` (whose weights are swapped in by the wrapper)
        and ``self.reference_model`` (frozen, unchanged).
        """
        # Log-probs under policy (weights are being differentiated)
        log_pi_chosen = self.compute_log_probs(self.policy_model, chosen_ids)
        log_pi_rejected = self.compute_log_probs(self.policy_model, rejected_ids)

        # Log-probs under reference (frozen, no gradients)
        log_ref_chosen = mx.stop_gradient(
            self.compute_log_probs(self.reference_model, chosen_ids)
        )
        log_ref_rejected = mx.stop_gradient(
            self.compute_log_probs(self.reference_model, rejected_ids)
        )

        # Log-ratios
        log_ratio_chosen = log_pi_chosen - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected

        # DPO loss: -mean(log σ(β × (log_ratio_chosen - log_ratio_rejected)))
        logits = self.config.beta * (log_ratio_chosen - log_ratio_rejected)
        log_sigmoid = logits - mx.logaddexp(mx.zeros_like(logits), logits)
        loss = -mx.mean(log_sigmoid)

        return loss

    def train_step(
        self,
        chosen_ids: mx.array,
        rejected_ids: mx.array,
    ) -> float:
        """Execute one DPO training step.

        1. Compute DPO loss
        2. Backward pass (policy model only — reference is frozen)
        3. Optimizer step
        4. NaN check: if loss is NaN, skip update and recover

        Parameters
        ----------
        chosen_ids : mx.array, shape ``[batch, seq]``
        rejected_ids : mx.array, shape ``[batch, seq]``

        Returns
        -------
        float
            The loss value for this step.
        """
        # Save checkpoint before update for NaN recovery
        if self._last_good_params is None:
            self._last_good_params = self.policy_model.parameters()

        # nn.value_and_grad(model, fn) → fn(*args) with model params swapped
        loss_and_grad_fn = nn.value_and_grad(self.policy_model, self._loss_fn)
        loss, grads = loss_and_grad_fn(chosen_ids, rejected_ids)

        # Force evaluation so we can inspect the loss value
        mx.eval(loss)
        loss_val = loss.item()

        # NaN / divergence check
        if math.isnan(loss_val) or math.isinf(loss_val):
            # Recover: reload last good params, reduce LR by 10×
            self.policy_model.load_weights(
                list(self._last_good_params.items())
                if isinstance(self._last_good_params, dict)
                else self._last_good_params
            )
            old_lr = self.optimizer.learning_rate
            new_lr = old_lr * 0.1
            self.optimizer = optim.Adam(learning_rate=new_lr)
            return float("nan")

        # Apply gradients (policy only — reference is frozen)
        self.optimizer.update(self.policy_model, grads)
        mx.eval(self.policy_model.parameters(), self.optimizer.state)

        # Save as last good checkpoint
        self._last_good_params = self.policy_model.parameters()

        return loss_val


# ---------------------------------------------------------------------------
# GRPO Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization.

    Parameters
    ----------
    group_size : int
        Number of responses to sample per prompt for group normalization.
        Must be ≥ 2 for meaningful normalization.
    beta : float
        KL penalty coefficient controlling drift from the reference policy.
    learning_rate : float
        Learning rate for the policy model optimizer.
    max_gen_len : int
        Maximum generation length for sampled responses.
    """

    group_size: int = 8
    beta: float = 0.1
    learning_rate: float = 1e-4
    max_gen_len: int = 32

    def __post_init__(self) -> None:
        assert self.group_size >= 2, (
            f"group_size must be >= 2 for normalization, got {self.group_size}"
        )
        assert self.beta > 0, f"beta must be positive, got {self.beta}"
        assert self.learning_rate > 0, (
            f"learning_rate must be positive, got {self.learning_rate}"
        )


# ---------------------------------------------------------------------------
# GRPOTrainer — Group Relative Policy Optimization
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Trainer implementing Group Relative Policy Optimization (GRPO).

    GRPO (DeepSeek, 2025) replaces learned value-function baselines with
    group-relative reward normalization.  For each prompt:

    1. Sample a *group* of N responses from the policy.
    2. Score each response with a reward function (can be rule-based).
    3. Normalize rewards within the group to mean ≈ 0, std ≈ 1.
    4. Update the policy via REINFORCE-style gradient weighted by
       normalized rewards, plus a KL penalty against the reference.

    Loss (simplified, without PPO clipping):

        L = -mean(r̂_i · log π_θ(y_i|x)) + β · KL(π_θ ‖ π_ref)

    where r̂_i = (r_i − μ_group) / (σ_group + ε).

    Parameters
    ----------
    config : GRPOConfig
        GRPO hyperparameters.
    policy_model : SimpleLM
        The policy model to be optimized.
    reference_model : SimpleLM
        Frozen reference model for KL penalty.
    reward_fn : callable
        Maps ``mx.array[batch, seq]`` → ``mx.array[batch]`` of scalar rewards.
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_model: SimpleLM,
        reference_model: SimpleLM,
        reward_fn,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_fn = reward_fn

        # Freeze reference model
        self.reference_model.freeze()

        # Optimizer for policy model
        self.optimizer = optim.Adam(learning_rate=config.learning_rate)

    # ------------------------------------------------------------------
    # Group sampling
    # ------------------------------------------------------------------

    def sample_group(
        self,
        prompt_ids: mx.array,
        n: int | None = None,
    ) -> mx.array:
        """Sample a group of N responses from the policy model.

        For each of the N group members, we autoregressively sample tokens
        by taking the argmax (greedy) of the policy logits at each step.
        The prompt is prepended to every response.

        Parameters
        ----------
        prompt_ids : mx.array, shape ``[1, prompt_len]``
            Token IDs for the prompt.
        n : int, optional
            Group size.  Defaults to ``config.group_size``.

        Returns
        -------
        mx.array, shape ``[n, prompt_len + max_gen_len]``
            Full sequences (prompt + generated continuation) for each
            group member.
        """
        if n is None:
            n = self.config.group_size
        assert n >= 2, f"Group size must be >= 2, got {n}"
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, (
            f"prompt_ids must be [1, prompt_len], got {prompt_ids.shape}"
        )

        prompt_len = prompt_ids.shape[1]
        max_gen_len = self.config.max_gen_len
        total_len = prompt_len + max_gen_len

        # Replicate prompt for all group members: [n, prompt_len]
        current = mx.broadcast_to(prompt_ids, (n, prompt_len))
        # We'll build the full sequence column by column
        generated_tokens = []

        for step in range(max_gen_len):
            # Forward pass on current sequence: [n, current_len, vocab]
            logits = self.policy_model(current)
            assert logits.ndim == 3

            # Take logits at the last position: [n, vocab]
            next_logits = logits[:, -1, :]

            # Sample from the distribution (categorical sampling)
            # Use Gumbel-max trick for differentiable-friendly sampling
            gumbel_noise = -mx.log(-mx.log(
                mx.random.uniform(shape=next_logits.shape) + 1e-10
            ) + 1e-10)
            next_tokens = mx.argmax(next_logits + gumbel_noise, axis=-1)
            assert next_tokens.shape == (n,)

            generated_tokens.append(next_tokens[:, None])  # [n, 1]

            # Append to current sequence
            current = mx.concatenate(
                [current, next_tokens[:, None]], axis=1
            )

        assert current.shape == (n, total_len), (
            f"Expected [{n}, {total_len}], got {current.shape}"
        )
        mx.eval(current)
        return current

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_group_rewards(self, group_ids: mx.array) -> mx.array:
        """Score each response in the group using the reward function.

        Parameters
        ----------
        group_ids : mx.array, shape ``[n, seq]``
            Token IDs for each group member.

        Returns
        -------
        mx.array, shape ``[n]``
            Raw reward score for each response.
        """
        assert group_ids.ndim == 2, (
            f"group_ids must be 2-D [n, seq], got ndim={group_ids.ndim}"
        )
        n = group_ids.shape[0]
        assert n >= 2, f"Need at least 2 group members, got {n}"

        rewards = self.reward_fn(group_ids)
        assert rewards.shape == (n,), (
            f"reward_fn must return [n], got {rewards.shape}"
        )
        return rewards

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_rewards(rewards: mx.array, eps: float = 1e-8) -> mx.array:
        """Normalize rewards to mean ≈ 0, std ≈ 1 within the group.

        Parameters
        ----------
        rewards : mx.array, shape ``[n]``
            Raw reward scores.
        eps : float
            Small constant for numerical stability.

        Returns
        -------
        mx.array, shape ``[n]``
            Normalized rewards with mean ≈ 0 and std ≈ 1.
        """
        assert rewards.ndim == 1, (
            f"rewards must be 1-D [n], got ndim={rewards.ndim}"
        )
        assert rewards.shape[0] >= 2, (
            f"Need at least 2 rewards for normalization, got {rewards.shape[0]}"
        )

        mean = mx.mean(rewards)
        std = mx.sqrt(mx.mean((rewards - mean) ** 2))
        normalized = (rewards - mean) / (std + eps)

        assert normalized.shape == rewards.shape
        return normalized

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def grpo_loss(
        self,
        prompt_ids: mx.array,
        group_ids: mx.array,
        normalized_rewards: mx.array,
    ) -> mx.array:
        """Compute the GRPO loss: policy gradient + KL penalty.

        Loss = -mean(r̂_i · log π_θ(y_i|x)) + β · KL(π_θ ‖ π_ref)

        where KL is estimated per-sequence as:
            KL_i = log π_θ(y_i) − log π_ref(y_i)

        Parameters
        ----------
        prompt_ids : mx.array, shape ``[1, prompt_len]``
            The prompt token IDs (unused in loss but kept for API clarity).
        group_ids : mx.array, shape ``[n, seq]``
            Full sequences (prompt + response) for each group member.
        normalized_rewards : mx.array, shape ``[n]``
            Group-normalized rewards (mean ≈ 0, std ≈ 1).

        Returns
        -------
        mx.array, scalar
            The GRPO loss.
        """
        n, seq = group_ids.shape
        assert normalized_rewards.shape == (n,), (
            f"normalized_rewards shape {normalized_rewards.shape} != ({n},)"
        )

        # Log-probs under policy and reference
        log_pi_policy = DPOTrainer.compute_log_probs(
            self.policy_model, group_ids
        )
        log_pi_ref = mx.stop_gradient(
            DPOTrainer.compute_log_probs(self.reference_model, group_ids)
        )
        assert log_pi_policy.shape == (n,)
        assert log_pi_ref.shape == (n,)

        # Stop gradient on normalized rewards (they are constants for
        # the policy gradient — only log_pi_policy carries gradients)
        r_hat = mx.stop_gradient(normalized_rewards)

        # Policy gradient term: -mean(r̂_i · log π_θ(y_i))
        policy_gradient = -mx.mean(r_hat * log_pi_policy)

        # KL penalty: β · mean(log π_θ(y_i) − log π_ref(y_i))
        kl_per_seq = log_pi_policy - log_pi_ref
        kl_penalty = self.config.beta * mx.mean(kl_per_seq)

        loss = policy_gradient + kl_penalty
        return loss


# ---------------------------------------------------------------------------
# Comparison visualization: RLHF vs DPO vs GRPO
# ---------------------------------------------------------------------------

def plot_alignment_method_comparison() -> None:
    """Create a grouped bar chart comparing RLHF, DPO, and GRPO.

    Compares five dimensions on a 1–5 scale (higher = better/easier):
      - Implementation simplicity
      - Data efficiency (less data needed = higher)
      - Training stability
      - Compute efficiency (fewer models/less VRAM = higher)
      - Empirical quality (reported results)

    Uses matplotlib only — no external dependencies.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    categories = [
        "Implementation\nSimplicity",
        "Data\nEfficiency",
        "Training\nStability",
        "Compute\nEfficiency",
        "Empirical\nQuality",
    ]
    # Scores on a 1–5 scale (higher = better)
    rlhf_scores = [1, 2, 2, 1, 4]   # complex, needs pairs, PPO unstable, 4 models, strong results
    dpo_scores  = [4, 2, 4, 4, 4]   # simple loss, needs pairs, stable, 2 models, strong results
    grpo_scores = [3, 5, 3, 3, 5]   # moderate, prompts only, policy grad variance, 2 models + sampling, excellent on reasoning

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, rlhf_scores, width, label="RLHF", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x, dpo_scores, width, label="DPO", color="#3498db", alpha=0.85)
    bars3 = ax.bar(x + width, grpo_scores, width, label="GRPO", color="#2ecc71", alpha=0.85)

    ax.set_ylabel("Score (1 = worst, 5 = best)", fontsize=11)
    ax.set_title("Alignment Methods Comparison: RLHF vs DPO vs GRPO", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5.8)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    plt.show()
