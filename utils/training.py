"""Training utilities for Notebook 07: Building GPT from Scratch.

Provides dataset download, tokenization, LR scheduling, gradient clipping,
checkpointing, evaluation, and text generation — all using MLX exclusively.

**Validates: Requirements 7.1–7.9**
"""

import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np


# ---------------------------------------------------------------------------
# 9.1  Dataset download & character tokenizer
# ---------------------------------------------------------------------------

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(
    data_dir: str = "data",
    val_fraction: float = 0.1,
) -> tuple[str, str]:
    """Download tiny_shakespeare and split into train / val text.

    💡 The dataset is ~1 MB of Shakespeare plays — perfect for a quick demo.

    Returns:
        (train_text, val_text) with a 90/10 split by default.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(filepath):
        print(f"⬇️  Downloading tiny_shakespeare to {filepath} …")
        urllib.request.urlretrieve(_TINY_SHAKESPEARE_URL, filepath)
        print("✅ Download complete.")
    else:
        print(f"✅ Found cached {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * (1 - val_fraction))
    return text[:split_idx], text[split_idx:]


class CharTokenizer:
    """Minimal character-level tokenizer.

    ⚡ Character-level keeps vocab tiny — great for fast iteration.
    """

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = {i: ch for ch, i in self._stoi.items()}

    def encode(self, s: str) -> list[int]:
        return [self._stoi[c] for c in s if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos.get(i, "?") for i in ids)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# 9.2  Cosine LR schedule with linear warmup  &  gradient clipping
# ---------------------------------------------------------------------------

def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    🎯 Interview tip: this is the standard schedule used by GPT-3, LLaMA, etc.

    During warmup (step < warmup_steps):
        lr = max_lr × (step / warmup_steps)

    After warmup:
        lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × step / max_steps))
    """
    if step < warmup_steps:
        return max_lr * (step / warmup_steps) if warmup_steps > 0 else max_lr
    # Cosine decay phase
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * step / max_steps))


def clip_grad_norm(
    grads: Any,
    max_norm: float,
) -> tuple[Any, float]:
    """Clip gradient global norm to *max_norm*.

    ⚠️ Pitfall: without clipping, a single bad batch can blow up training.

    Returns:
        (clipped_grads, original_norm)
    """
    # Flatten all gradient leaves into a list
    flat_grads = tree_flatten(grads)
    # Compute global L2 norm
    sum_sq = mx.array(0.0)
    for _, g in flat_grads:
        sum_sq = sum_sq + mx.sum(g * g)
    global_norm = mx.sqrt(sum_sq)
    mx.eval(global_norm)
    norm_val = global_norm.item()

    if norm_val > max_norm and norm_val > 0:
        scale = max_norm / norm_val
        clipped = tree_unflatten(
            [(k, g * scale) for k, g in flat_grads]
        )
        return clipped, norm_val
    return grads, norm_val


# ---------------------------------------------------------------------------
# 9.4  CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save / load model weights, optimizer state, and training step.

    💡 Enables resuming training and NaN recovery.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        step: int,
        path: str | None = None,
    ) -> str:
        """Persist model weights, optimizer state, and step counter."""
        if path is None:
            path = str(self.checkpoint_dir / f"ckpt_step_{step}")
        os.makedirs(path, exist_ok=True)

        # Model weights
        weights = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(os.path.join(path, "model.safetensors"), weights)

        # Optimizer state
        opt_state = dict(tree_flatten(optimizer.state))
        if opt_state:
            mx.save_safetensors(os.path.join(path, "optimizer.safetensors"), opt_state)

        # Metadata
        meta = {"step": step}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)

        return path

    def load(
        self,
        model: nn.Module,
        optimizer: Any,
        path: str,
    ) -> int:
        """Restore model weights, optimizer state, and return the step."""
        # Model weights
        weights = mx.load(os.path.join(path, "model.safetensors"))
        model.load_weights(list(weights.items()))

        # Optimizer state
        opt_path = os.path.join(path, "optimizer.safetensors")
        if os.path.exists(opt_path):
            opt_state = mx.load(opt_path)
            opt_flat = list(opt_state.items())
            optimizer.state = tree_unflatten(opt_flat)

        # Metadata
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        return meta["step"]

    def detect_nan(self, loss_val: float) -> bool:
        """Return True if loss is NaN or Inf."""
        return not math.isfinite(loss_val)


# ---------------------------------------------------------------------------
# 9.6  Evaluation loop  &  text generation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    val_data: mx.array,
    batch_size: int = 8,
    seq_len: int = 64,
    n_batches: int = 10,
) -> tuple[float, float]:
    """Compute validation loss and perplexity.

    🎯 Perplexity = exp(cross-entropy loss). Lower is better.

    Returns:
        (val_loss, perplexity)
    """
    total_loss = 0.0
    count = 0
    for _ in range(n_batches):
        if len(val_data) <= seq_len + 1:
            break
        ix = np.random.randint(0, len(val_data) - seq_len, size=(batch_size,))
        x = mx.stack([val_data[i : i + seq_len] for i in ix])
        y = mx.stack([val_data[i + 1 : i + seq_len + 1] for i in ix])
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss) if math.isfinite(avg_loss) else float("inf")
    return avg_loss, perplexity


def generate_text(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    """Auto-regressive text generation with temperature sampling.

    ⚡ Uses the model's max_seq_len as context window.
    """
    token_ids = tokenizer.encode(prompt)
    max_ctx = getattr(model, "max_seq_len", 256)

    for _ in range(max_tokens):
        ctx = token_ids[-max_ctx:]
        x = mx.array([ctx])
        logits = model(x)  # (1, T, vocab)
        logits = logits[0, -1, :] / max(temperature, 1e-8)
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)
        probs_np = np.array(probs)
        # Guard against negative / NaN probabilities
        probs_np = np.clip(probs_np, 0, None)
        total = probs_np.sum()
        if total <= 0 or not np.isfinite(total):
            next_token = int(np.random.randint(0, len(probs_np)))
        else:
            probs_np /= total
            next_token = int(np.random.choice(len(probs_np), p=probs_np))
        token_ids.append(next_token)

    return tokenizer.decode(token_ids)
