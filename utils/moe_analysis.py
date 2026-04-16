"""Memory analysis and visualization for MoE vs dense model comparisons.

Provides functions to compute parameter counts, active parameters per token,
memory footprints, and generate comparison visualizations. All computations
are pure Python/NumPy — no MLX dependency needed for analysis.

Used by Notebook 15 (Mixture of Experts) for memory-focused analysis cells.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.moe import MoEConfig


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _expert_params(d_model: int, d_ff: int) -> int:
    """Parameter count for one SwiGLU expert FFN (3 matrices, no bias)."""
    return 3 * d_model * d_ff


def compute_moe_vs_dense_stats(
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute total params, active params per token, and memory for each model config.

    Each config dict must have:
        - "name": str — model label
        - "config": MoEConfig — the MoE configuration
        - "type": str — "Dense" or "MoE"

    For dense models, set num_experts=1 and num_active=1.

    Returns a list of dicts with computed statistics per model.
    """
    stats: list[dict[str, Any]] = []

    for entry in configs:
        name: str = entry["name"]
        cfg: MoEConfig = entry["config"]
        model_type: str = entry.get("type", "MoE" if cfg.num_experts > 1 else "Dense")

        expert_p = _expert_params(cfg.d_model, cfg.d_ff)

        # Total expert parameters
        total_expert_params = cfg.num_experts * expert_p
        active_expert_params = cfg.num_active * expert_p

        # Shared expert (always active)
        shared_params = expert_p if cfg.has_shared_expert else 0
        total_expert_params += shared_params
        active_expert_params += shared_params

        # Router parameters (gate weight matrix)
        router_params = cfg.d_model * cfg.num_experts if cfg.num_experts > 1 else 0
        total_params = total_expert_params + router_params

        # Memory at different precisions (bytes)
        bytes_per_param = {"float32": 4, "bfloat16": 2, "int8": 1, "int4": 0.5}
        memory_gb = {
            prec: total_params * bpp / (1024**3)
            for prec, bpp in bytes_per_param.items()
        }

        # Sparsity: fraction of expert params NOT active per token
        sparsity = (
            1.0 - (active_expert_params / total_expert_params)
            if total_expert_params > 0
            else 0.0
        )

        # Memory overhead ratio: total memory / memory if only active params existed
        overhead_ratio = (
            total_params / active_expert_params
            if active_expert_params > 0
            else 1.0
        )

        stats.append(
            {
                "name": name,
                "type": model_type,
                "num_experts": cfg.num_experts,
                "num_active": cfg.num_active,
                "has_shared": cfg.has_shared_expert,
                "d_model": cfg.d_model,
                "d_ff": cfg.d_ff,
                "total_params": total_params,
                "active_params": active_expert_params,
                "router_params": router_params,
                "sparsity": sparsity,
                "overhead_ratio": overhead_ratio,
                "memory_gb": memory_gb,
            }
        )

    return stats


# ---------------------------------------------------------------------------
# Visualisation: MoE vs Dense bar chart
# ---------------------------------------------------------------------------

def plot_moe_vs_dense_params(
    stats: list[dict[str, Any]],
) -> plt.Figure:
    """Bar chart comparing total params vs active params per token.

    Produces a 2-panel figure:
      Left  — grouped bars: total params vs active params per model.
      Right — memory overhead ratio (how much extra memory MoE needs
              relative to what it actually uses per token).

    Args:
        stats: Output of :func:`compute_moe_vs_dense_stats`.

    Returns:
        matplotlib Figure.
    """
    names = [s["name"] for s in stats]
    total_b = [s["total_params"] / 1e9 for s in stats]
    active_b = [s["active_params"] / 1e9 for s in stats]
    overhead = [s["overhead_ratio"] for s in stats]
    types = [s["type"] for s in stats]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(names))
    bw = 0.35

    # --- Panel 1: Total vs Active params ---
    c_total = ["#3498db" if t == "Dense" else "#e74c3c" for t in types]
    c_active = ["#85c1e9" if t == "Dense" else "#2ecc71" for t in types]

    axes[0].bar(x - bw / 2, total_b, bw, label="Total Params", color=c_total, edgecolor="white")
    axes[0].bar(x + bw / 2, active_b, bw, label="Active / Token", color=c_active, edgecolor="white")

    for i, (tp, ap) in enumerate(zip(total_b, active_b)):
        axes[0].text(i - bw / 2, tp + max(total_b) * 0.02, f"{tp:.1f}B",
                     ha="center", fontsize=7, fontweight="bold")
        axes[0].text(i + bw / 2, ap + max(total_b) * 0.02, f"{ap:.1f}B",
                     ha="center", fontsize=7, fontweight="bold")

    axes[0].set_ylabel("Parameters (Billions)")
    axes[0].set_title("💡 Total Params vs Active Params per Token")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Panel 2: Memory overhead ratio ---
    bar_colors = ["#3498db" if t == "Dense" else "#f39c12" for t in types]
    axes[1].bar(x, overhead, 0.55, color=bar_colors, edgecolor="white", alpha=0.9)
    for i, ov in enumerate(overhead):
        axes[1].text(i, ov + 0.15, f"{ov:.1f}×", ha="center", fontsize=9, fontweight="bold")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1× (no overhead)")
    axes[1].set_ylabel("Memory Overhead Ratio")
    axes[1].set_title("⚠️ Memory Overhead: Total / Active Params")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("MoE vs Dense: Parameter Efficiency & Memory Overhead",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Visualisation: MoE scaling curve
# ---------------------------------------------------------------------------

def plot_moe_scaling_curve(
    d_model: int,
    d_ff: int,
    expert_counts: list[int],
    k: int,
) -> plt.Figure:
    """Show how capacity (total params) vs cost (active params) scales with expert count.

    Produces a 2-panel figure:
      Left  — log-scale plot of total vs active params as N grows.
      Right — memory overhead ratio and sparsity as N grows.

    Args:
        d_model: Model dimension.
        d_ff: FFN hidden dimension per expert.
        expert_counts: List of expert counts to evaluate (e.g. [1,2,4,8,...]).
        k: Number of active experts per token.

    Returns:
        matplotlib Figure.
    """
    expert_p = _expert_params(d_model, d_ff)

    total_b = []
    active_b = []
    overhead = []
    sparsity = []

    for n in expert_counts:
        k_eff = min(k, n)
        tp = n * expert_p
        ap = k_eff * expert_p
        total_b.append(tp / 1e9)
        active_b.append(ap / 1e9)
        overhead.append(tp / ap if ap > 0 else 1.0)
        sparsity.append(1.0 - k_eff / n if n > 0 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Capacity vs Cost scaling ---
    axes[0].plot(expert_counts, total_b, "o-", color="#e74c3c", lw=2, label="Total params (capacity)")
    axes[0].plot(expert_counts, active_b, "s-", color="#2ecc71", lw=2, label=f"Active params/token (K={k})")
    axes[0].fill_between(expert_counts, active_b, total_b, alpha=0.12, color="#e74c3c",
                         label="Idle params (memory overhead)")
    axes[0].set_xlabel("Number of Experts (N)")
    axes[0].set_ylabel("Parameters (Billions)")
    axes[0].set_title(f"⚡ Capacity vs Cost Scaling (d={d_model}, d_ff={d_ff})")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Overhead ratio & sparsity ---
    ax2 = axes[1]
    color_oh = "#f39c12"
    color_sp = "#9b59b6"

    ax2.bar(range(len(expert_counts)), overhead, 0.4, color=color_oh, alpha=0.8, label="Overhead ratio")
    ax2.set_xlabel("Number of Experts (N)")
    ax2.set_ylabel("Memory Overhead Ratio (×)", color=color_oh)
    ax2.set_xticks(range(len(expert_counts)))
    ax2.set_xticklabels([str(n) for n in expert_counts])
    ax2.tick_params(axis="y", labelcolor=color_oh)

    ax2b = ax2.twinx()
    ax2b.plot(range(len(expert_counts)), [s * 100 for s in sparsity], "D-",
              color=color_sp, lw=2, label="Sparsity %")
    ax2b.set_ylabel("Sparsity (%)", color=color_sp)
    ax2b.tick_params(axis="y", labelcolor=color_sp)
    ax2b.set_ylim(0, 105)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax2.set_title(f"🎯 Memory Overhead & Sparsity (K={k})")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("MoE Scaling: How Expert Count Affects Efficiency",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
