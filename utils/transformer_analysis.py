"""Transformer architecture deep-dive analysis utilities (Notebook 06 rework).

Provides:
- ActivationComparison: compute and plot ReLU, GELU, SiLU, SwiGLU, GeGLU
- NormalizationComparison: compute and plot LayerNorm, RMSNorm, DeepNorm
- gradient_flow_analysis: compare pre-norm vs post-norm gradient CV
- ParameterCounter: per-component parameter counts for any TransformerConfig
- estimate_memory: memory estimation for any config + dtype
- Weight initialization helpers (Xavier, Kaiming)
- Backpropagation walkthrough helpers

All implementations use MLX exclusively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ── TransformerConfig (shared) ──────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """Configuration for a standard transformer model."""
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 2048
    activation: str = "swiglu"  # relu | gelu | silu | swiglu | geglu
    norm_type: str = "rmsnorm"  # layernorm | rmsnorm | deepnorm

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_kv_heads <= self.n_heads, "n_kv_heads must be <= n_heads"


# ── Activation Functions ────────────────────────────────────────────────────

def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0.0)


def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


def _silu(x: mx.array) -> mx.array:
    return nn.silu(x)


def _swiglu(x: mx.array, w1: mx.array, w_gate: mx.array) -> mx.array:
    """SwiGLU: (x @ W1) * SiLU(x @ W_gate)."""
    return (x @ w1) * nn.silu(x @ w_gate)


def _geglu(x: mx.array, w1: mx.array, w_gate: mx.array) -> mx.array:
    """GeGLU: (x @ W1) * GELU(x @ W_gate)."""
    return (x @ w1) * nn.gelu(x @ w_gate)


class ActivationComparison:
    """Compare activation functions: ReLU, GELU, SiLU, SwiGLU, GeGLU.

    For element-wise activations (ReLU, GELU, SiLU), operates directly on input.
    For gated activations (SwiGLU, GeGLU), uses identity-initialized projections
    so the comparison is meaningful on the same input range.
    """

    NAMES = ["ReLU", "GELU", "SiLU", "SwiGLU", "GeGLU"]

    @staticmethod
    def compute(x: mx.array) -> Dict[str, mx.array]:
        """Compute all activations on the same 1-D input tensor.

        For gated variants, we use identity projections so the gate and
        linear paths both see the raw input — making the comparison fair.
        """
        d = x.shape[-1]
        eye = mx.eye(d)

        results = {
            "ReLU": _relu(x),
            "GELU": _gelu(x),
            "SiLU": _silu(x),
            "SwiGLU": _swiglu(x, eye, eye),
            "GeGLU": _geglu(x, eye, eye),
        }
        mx.eval(*results.values())
        return results

    @staticmethod
    def compute_derivatives_np(x_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute numerical derivatives for each activation (numpy)."""
        eps = 1e-5
        derivs: Dict[str, np.ndarray] = {}

        # ReLU derivative
        derivs["ReLU"] = (x_np > 0).astype(np.float32)

        # GELU derivative (numerical)
        x_p = mx.array(x_np + eps)
        x_m = mx.array(x_np - eps)
        g_p = nn.gelu(x_p)
        g_m = nn.gelu(x_m)
        mx.eval(g_p, g_m)
        derivs["GELU"] = (np.array(g_p) - np.array(g_m)) / (2 * eps)

        # SiLU derivative (numerical)
        s_p = nn.silu(x_p)
        s_m = nn.silu(x_m)
        mx.eval(s_p, s_m)
        derivs["SiLU"] = (np.array(s_p) - np.array(s_m)) / (2 * eps)

        # SwiGLU derivative — with identity projections, SwiGLU(x) = x * SiLU(x)
        swi = lambda v: np.array(v) * (np.array(v) / (1 + np.exp(-np.array(v))))
        derivs["SwiGLU"] = (swi(x_np + eps) - swi(x_np - eps)) / (2 * eps)

        # GeGLU derivative — with identity projections, GeGLU(x) = x * GELU(x)
        gelu_np = lambda v: v * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v**3)))
        geglu = lambda v: v * gelu_np(v)
        derivs["GeGLU"] = (geglu(x_np + eps) - geglu(x_np - eps)) / (2 * eps)

        return derivs

    @staticmethod
    def plot(x_np: np.ndarray, activations: Dict[str, mx.array],
             derivatives: Dict[str, np.ndarray] | None = None) -> plt.Figure:
        """Side-by-side plots of activations and their derivatives."""
        n_plots = 2 if derivatives else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        # Activation values
        ax = axes[0]
        for name, color in zip(ActivationComparison.NAMES, colors):
            y = np.array(activations[name]).flatten()
            ax.plot(x_np, y, label=name, color=color, linewidth=2)
        ax.set_title("Activation Functions", fontsize=14)
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        # Derivatives
        if derivatives:
            ax = axes[1]
            for name, color in zip(ActivationComparison.NAMES, colors):
                ax.plot(x_np, derivatives[name], label=f"{name}'", color=color, linewidth=2)
            ax.set_title("Activation Derivatives", fontsize=14)
            ax.set_xlabel("Input")
            ax.set_ylabel("Derivative")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return fig


# ── Normalization Comparison ────────────────────────────────────────────────

def _layer_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Manual LayerNorm: (x - mean) / sqrt(var + eps)."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """Manual RMSNorm: x / sqrt(mean(x^2) + eps)."""
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def _deep_norm(x: mx.array, sublayer_out: mx.array, alpha: float = 1.0,
               eps: float = 1e-5) -> mx.array:
    """DeepNorm: LayerNorm(alpha * x + sublayer(x)).

    For visualization without a sublayer, we treat sublayer_out as a
    scaled version of x.
    """
    combined = alpha * x + sublayer_out
    return _layer_norm(combined, eps)


class NormalizationComparison:
    """Compare normalization methods: LayerNorm, RMSNorm, DeepNorm."""

    NAMES = ["LayerNorm", "RMSNorm", "DeepNorm"]

    @staticmethod
    def compute(x: mx.array, alpha: float = 1.0) -> Dict[str, mx.array]:
        """Compute all normalizations on the same input."""
        sublayer_out = x * 0.1  # small perturbation as sublayer output
        results = {
            "LayerNorm": _layer_norm(x),
            "RMSNorm": _rms_norm(x),
            "DeepNorm": _deep_norm(x, sublayer_out, alpha=alpha),
        }
        mx.eval(*results.values())
        return results

    @staticmethod
    def plot(x_np: np.ndarray, norms: Dict[str, mx.array]) -> plt.Figure:
        """Plot normalization outputs."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71"]

        for (name, color) in zip(NormalizationComparison.NAMES, colors):
            y = np.array(norms[name]).flatten()
            ax.plot(x_np, y, label=name, color=color, linewidth=2)

        ax.set_title("Normalization Methods Comparison", fontsize=14)
        ax.set_xlabel("Input (sorted)")
        ax.set_ylabel("Normalized Output")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


# ── Gradient Flow Analysis (Pre-Norm vs Post-Norm) ─────────────────────────

class _SimpleFFN(nn.Module):
    """Minimal FFN for gradient analysis."""
    def __init__(self, d: int):
        super().__init__()
        self.w1 = nn.Linear(d, d * 4, bias=False)
        self.w2 = nn.Linear(d * 4, d, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)))


class _SimpleAttn(nn.Module):
    """Minimal single-head attention for gradient analysis."""
    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.scale = math.sqrt(d)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        scores = (q @ k.transpose(0, 2, 1)) / self.scale
        weights = mx.softmax(scores, axis=-1)
        return self.out(weights @ v)


class _PreNormBlock(nn.Module):
    """Pre-norm transformer block: x + sublayer(norm(x))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = _SimpleAttn(d)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = _SimpleFFN(d)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _PostNormBlock(nn.Module):
    """Post-norm transformer block: norm(x + sublayer(x))."""
    def __init__(self, d: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = _SimpleAttn(d)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = _SimpleFFN(d)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


def gradient_flow_analysis(
    depth: int = 6,
    d_model: int = 32,
    seq_len: int = 8,
    batch: int = 2,
    n_trials: int = 20,
) -> Dict[str, object]:
    """Compare per-layer activation-gradient norms for pre-norm vs post-norm.

    For a fixed-depth stack, measures ||∂L/∂h_l|| at each layer l using a
    zero-probe injection technique.  Pre-norm preserves a direct residual
    path so these activation gradients stay more uniform across layers.

    Returns dict with:
      - pre_norm_cv: coefficient of variation of activation-gradient norms
      - post_norm_cv: coefficient of variation of activation-gradient norms
      - pre_grad_norms: list of per-layer activation-gradient norms
      - post_grad_norms: list of per-layer activation-gradient norms
      - figure: matplotlib Figure with comparison plot
    """

    def _cv(values):
        arr = np.array(values)
        mean_val = np.mean(arr)
        if mean_val < 1e-12:
            return 0.0
        return float(np.std(arr) / mean_val)

    def _activation_grad_norms(block_cls, depth, dim, x):
        """Measure ||∂L/∂h_l|| at each layer l using probe injection.

        For each layer l, we:
        1. Forward through layers 0..l-1 (stop gradient)
        2. Add a probe tensor at layer l
        3. Forward through layers l..N-1
        4. Compute gradient of loss w.r.t. probe (= ∂L/∂h_l)
        """
        # Build the stack once
        layers = [block_cls(dim) for _ in range(depth)]
        head = nn.Linear(dim, 1, bias=False)
        mx.eval([p for l in layers for p in l.parameters().values()])
        mx.eval(head.parameters())

        # Forward pass to get intermediate activations (detached)
        intermediates = [x]
        h = x
        for layer in layers:
            h = layer(h)
            intermediates.append(mx.stop_gradient(h))

        norms = []
        for l in range(depth):
            h_l = intermediates[l]  # activation entering layer l (detached)

            # Build a function: probe -> loss, where we inject probe at layer l
            def _loss_from_probe(probe, _h_l=h_l, _l=l):
                h = _h_l + probe
                for layer in layers[_l:]:
                    h = layer(h)
                return head(h).mean()

            probe_zero = mx.zeros_like(h_l)
            grad_fn = mx.grad(_loss_from_probe)
            g = grad_fn(probe_zero)
            mx.eval(g)
            norm = float(mx.sqrt(mx.sum(g * g)).item())
            norms.append(norm)

        return norms

    all_pre_cvs: list[float] = []
    all_post_cvs: list[float] = []
    all_pre_norms: list[list[float]] = []
    all_post_norms: list[list[float]] = []

    for seed in range(n_trials):
        mx.random.seed(seed)
        x_data = mx.random.normal((batch, seq_len, d_model))

        pre_n = _activation_grad_norms(_PreNormBlock, depth, d_model, x_data)
        post_n = _activation_grad_norms(_PostNormBlock, depth, d_model, x_data)

        all_pre_norms.append(pre_n)
        all_post_norms.append(post_n)
        all_pre_cvs.append(_cv(pre_n))
        all_post_cvs.append(_cv(post_n))

    pre_cv = float(np.mean(all_pre_cvs))
    post_cv = float(np.mean(all_post_cvs))
    pre_norms = [float(np.mean([r[i] for r in all_pre_norms])) for i in range(depth)]
    post_norms = [float(np.mean([r[i] for r in all_post_norms])) for i in range(depth)]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    layers_idx = list(range(1, depth + 1))

    ax = axes[0]
    ax.bar(np.array(layers_idx) - 0.15, pre_norms, width=0.3,
           label=f"Pre-Norm (CV={pre_cv:.3f})", color="#3498db")
    ax.bar(np.array(layers_idx) + 0.15, post_norms, width=0.3,
           label=f"Post-Norm (CV={post_cv:.3f})", color="#e74c3c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||∂L/∂h_l|| (Activation Gradient Norm)")
    ax.set_title("Per-Layer Activation Gradient Norms")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if pre_norms and max(pre_norms) > 0:
        pre_normalized = np.array(pre_norms) / max(pre_norms)
    else:
        pre_normalized = np.array(pre_norms)
    if post_norms and max(post_norms) > 0:
        post_normalized = np.array(post_norms) / max(post_norms)
    else:
        post_normalized = np.array(post_norms)
    ax.plot(layers_idx, pre_normalized, "o-", label="Pre-Norm (normalized)", color="#3498db")
    ax.plot(layers_idx, post_normalized, "s-", label="Post-Norm (normalized)", color="#e74c3c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Gradient Norm")
    ax.set_title("Gradient Flow Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Pre-Norm vs Post-Norm Gradient Flow (depth={depth}, d={d_model})", fontsize=14)
    fig.tight_layout()

    return {
        "pre_norm_cv": pre_cv,
        "post_norm_cv": post_cv,
        "pre_mean_grad": float(np.mean(pre_norms)),
        "post_mean_grad": float(np.mean(post_norms)),
        "pre_grad_norms": pre_norms,
        "post_grad_norms": post_norms,
        "figure": fig,
    }


# ── Parameter Counter ───────────────────────────────────────────────────────

# Bytes per element for common dtypes
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


class ParameterCounter:
    """Count parameters per component and estimate memory for a TransformerConfig."""

    @staticmethod
    def count(config: TransformerConfig) -> Dict[str, int]:
        """Return per-component parameter counts.

        Components:
          - embedding: vocab_size * d_model
          - attention: per-layer Q, K, V, O projections (accounts for GQA)
          - ffn: per-layer feed-forward (accounts for gated variants)
          - normalization: per-layer norm weights
          - total: sum of all
        """
        d = config.d_model
        n = config.n_layers
        d_head = d // config.n_heads

        # Embedding
        embedding = config.vocab_size * d

        # Attention per layer: Q, O use n_heads; K, V use n_kv_heads (GQA)
        q_params = d * (config.n_heads * d_head)      # W_q
        k_params = d * (config.n_kv_heads * d_head)    # W_k
        v_params = d * (config.n_kv_heads * d_head)    # W_v
        o_params = (config.n_heads * d_head) * d       # W_o
        attn_per_layer = q_params + k_params + v_params + o_params
        attention = attn_per_layer * n

        # FFN per layer
        act = config.activation.lower()
        if act in ("swiglu", "geglu"):
            # Gated: W1 (d→d_ff), W_gate (d→d_ff), W2 (d_ff→d)
            ffn_per_layer = d * config.d_ff + d * config.d_ff + config.d_ff * d
        else:
            # Standard: W1 (d→d_ff), W2 (d_ff→d)
            ffn_per_layer = d * config.d_ff + config.d_ff * d
        ffn = ffn_per_layer * n

        # Normalization per layer: 2 norms (attn_norm, ffn_norm) + 1 final norm
        norm_type = config.norm_type.lower()
        if norm_type == "layernorm":
            norm_params_each = d * 2  # weight + bias
        else:
            norm_params_each = d  # weight only (RMSNorm, DeepNorm)
        normalization = norm_params_each * (2 * n + 1)  # 2 per layer + final

        total = embedding + attention + ffn + normalization

        return {
            "embedding": embedding,
            "attention": attention,
            "ffn": ffn,
            "normalization": normalization,
            "total": total,
        }

    @staticmethod
    def estimate_memory(config: TransformerConfig, dtype: str = "float32") -> Dict[str, float]:
        """Estimate memory in bytes for each component.

        Memory = params × bytes_per_element.
        """
        counts = ParameterCounter.count(config)
        bpe = DTYPE_BYTES.get(dtype, 4)

        memory = {}
        for key, count in counts.items():
            memory[key] = count * bpe

        return memory

    @staticmethod
    def plot(config: TransformerConfig, dtype: str = "float32") -> plt.Figure:
        """Stacked bar chart of parameter counts and memory."""
        counts = ParameterCounter.count(config)
        memory = ParameterCounter.estimate_memory(config, dtype)

        components = ["embedding", "attention", "ffn", "normalization"]
        param_vals = [counts[c] for c in components]
        mem_vals = [memory[c] / (1024**2) for c in components]  # MB

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

        # Parameter counts
        ax = axes[0]
        bottom = 0
        for comp, val, color in zip(components, param_vals, colors):
            ax.bar("Parameters", val, bottom=bottom, label=comp.capitalize(), color=color)
            bottom += val
        ax.set_ylabel("Parameter Count")
        ax.set_title(f"Parameter Breakdown (Total: {counts['total']:,})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Memory
        ax = axes[1]
        bottom = 0
        for comp, val, color in zip(components, mem_vals, colors):
            ax.bar(f"Memory ({dtype})", val, bottom=bottom, label=comp.capitalize(), color=color)
            bottom += val
        ax.set_ylabel("Memory (MB)")
        total_mb = memory["total"] / (1024**2)
        ax.set_title(f"Memory Breakdown (Total: {total_mb:.1f} MB)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            f"TransformerConfig: d={config.d_model}, layers={config.n_layers}, "
            f"heads={config.n_heads}, d_ff={config.d_ff}, vocab={config.vocab_size}",
            fontsize=11,
        )
        fig.tight_layout()
        return fig


# ── Weight Initialization ───────────────────────────────────────────────────

def xavier_init(shape: tuple, gain: float = 1.0) -> mx.array:
    """Xavier/Glorot uniform initialization.

    Appropriate for layers with sigmoid/tanh activations.
    Maintains variance across layers: Var(W) = 2 / (fan_in + fan_out).
    """
    fan_in, fan_out = shape[-2], shape[-1]
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return mx.random.uniform(-limit, limit, shape)


def kaiming_init(shape: tuple, nonlinearity: str = "relu") -> mx.array:
    """Kaiming/He initialization.

    Appropriate for layers with ReLU-family activations.
    Maintains variance: Var(W) = 2 / fan_in.
    """
    fan_in = shape[-2]
    if nonlinearity == "relu":
        gain = math.sqrt(2.0)
    elif nonlinearity in ("silu", "gelu"):
        gain = math.sqrt(2.0)  # approximate
    else:
        gain = 1.0
    std = gain / math.sqrt(fan_in)
    return mx.random.normal(shape) * std


def plot_init_distributions(d: int = 256) -> plt.Figure:
    """Compare Xavier vs Kaiming initialization distributions."""
    mx.random.seed(0)
    shape = (d, d)

    xavier_w = xavier_init(shape)
    kaiming_relu_w = kaiming_init(shape, "relu")
    kaiming_silu_w = kaiming_init(shape, "silu")
    mx.eval(xavier_w, kaiming_relu_w, kaiming_silu_w)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, w, title in zip(
        axes,
        [xavier_w, kaiming_relu_w, kaiming_silu_w],
        ["Xavier (Glorot)", "Kaiming (ReLU)", "Kaiming (SiLU)"],
    ):
        vals = np.array(w).flatten()
        ax.hist(vals, bins=60, density=True, alpha=0.7, color="#3498db")
        ax.set_title(f"{title}\nstd={np.std(vals):.4f}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Weight Initialization Comparison ({d}×{d})", fontsize=13)
    fig.tight_layout()
    return fig


# ── Backpropagation Walkthrough Helpers ─────────────────────────────────────

def backprop_walkthrough(d_model: int = 32, n_heads: int = 1, seq_len: int = 4) -> Dict[str, object]:
    """Step-by-step gradient flow through a single transformer block.

    Returns gradient norms for each sub-component:
    attention, ffn, residual connections, normalization.

    Args:
        d_model: model dimension
        n_heads: number of attention heads (used for documentation; internal
                 block uses single-head for simplicity)
        seq_len: sequence length for the test input
    """
    mx.random.seed(42)
    x = mx.random.normal((1, seq_len, d_model))

    block = _PreNormBlock(d_model)
    mx.eval(block.parameters())

    def loss_fn(model, x):
        return model(x).mean()

    loss, grads = nn.value_and_grad(block, loss_fn)(block, x)
    mx.eval(loss, grads)

    # Extract gradient norms per component
    component_norms = {}

    def _norm_of(d):
        total = 0.0
        if isinstance(d, mx.array):
            mx.eval(d)
            return float(mx.sqrt(mx.sum(d * d)).item())
        if isinstance(d, dict):
            for v in d.values():
                n = _norm_of(v)
                total += n * n
        return math.sqrt(total)

    component_norms["attn_norm (RMSNorm)"] = _norm_of(grads.get("norm1", {}))
    component_norms["attention (QKV+O)"] = _norm_of(grads.get("attn", {}))
    component_norms["ffn_norm (RMSNorm)"] = _norm_of(grads.get("norm2", {}))
    component_norms["ffn (W1+W2)"] = _norm_of(grads.get("ffn", {}))

    return {
        "loss": float(loss.item()),
        "component_grad_norms": component_norms,
    }
