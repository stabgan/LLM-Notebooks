"""Scaling law utilities for Notebook 18: Scaling Laws & Compute-Optimal Training.

Implements the Chinchilla scaling law L(N, D) = A/N^α + B/D^β + E
and compute-optimal allocation given a FLOP budget C ≈ 6ND.

All pure Python/numpy — no MLX needed for scaling law math.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ScalingLawParams:
    """Parameters for the Chinchilla scaling law.

    L(N, D) = A / N^alpha + B / D^beta + E

    Defaults are the calibrated constants from Hoffmann et al. (2022).
    """

    A: float = 406.4
    B: float = 410.7
    alpha: float = 0.34
    beta: float = 0.28
    E: float = 1.69

    def __post_init__(self) -> None:
        assert self.A > 0, f"A must be positive, got {self.A}"
        assert self.B > 0, f"B must be positive, got {self.B}"
        assert self.alpha > 0, f"alpha must be positive, got {self.alpha}"
        assert self.beta > 0, f"beta must be positive, got {self.beta}"
        assert self.E >= 0, f"E must be non-negative, got {self.E}"


@dataclass
class ComputeBudget:
    """Result of compute-optimal allocation for a given FLOP budget."""

    total_flops: float
    optimal_N: float
    optimal_D: float
    estimated_loss: float

    def __post_init__(self) -> None:
        assert self.total_flops > 0, f"total_flops must be positive, got {self.total_flops}"
        assert self.optimal_N > 0, f"optimal_N must be positive, got {self.optimal_N}"
        assert self.optimal_D > 0, f"optimal_D must be positive, got {self.optimal_D}"
        assert np.isfinite(self.estimated_loss), f"estimated_loss must be finite, got {self.estimated_loss}"


class ScalingLawPredictor:
    """Predicts cross-entropy loss and computes optimal N, D allocation.

    Implements:
      - predict_loss(N, D): L = A/N^α + B/D^β + E
      - estimate_training_flops(N, D): C = 6 * N * D
      - compute_optimal_allocation(C): closed-form N*, D* from Chinchilla
    """

    def __init__(self, params: ScalingLawParams | None = None) -> None:
        self.params = params or ScalingLawParams()

    def predict_loss(self, N: float, D: float) -> float:
        """Predict cross-entropy loss for model size N and data size D.

        L(N, D) = A / N^α + B / D^β + E

        Args:
            N: Number of model parameters (must be > 0).
            D: Number of training tokens (must be > 0).

        Returns:
            Predicted cross-entropy loss (nats).
        """
        assert N > 0, f"N must be positive, got {N}"
        assert D > 0, f"D must be positive, got {D}"

        p = self.params
        loss = p.A / (N ** p.alpha) + p.B / (D ** p.beta) + p.E

        assert np.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss > p.E, f"Loss {loss} should exceed irreducible loss E={p.E}"
        return float(loss)

    def estimate_training_flops(self, N: float, D: float) -> float:
        """Estimate total training FLOPs using the C ≈ 6ND rule.

        Args:
            N: Number of model parameters (must be > 0).
            D: Number of training tokens (must be > 0).

        Returns:
            Estimated FLOPs.
        """
        assert N > 0, f"N must be positive, got {N}"
        assert D > 0, f"D must be positive, got {D}"

        flops = 6.0 * N * D
        assert flops > 0, f"FLOPs must be positive, got {flops}"
        return float(flops)

    def compute_optimal_allocation(self, C: float) -> ComputeBudget:
        """Compute optimal model size N* and token count D* for budget C.

        Uses the closed-form solution from the Chinchilla derivation:
            N* = (αA / (βB · 6^β))^(1/(α+β)) · C^(β/(α+β))
            D* = C / (6 · N*)

        Args:
            C: Total compute budget in FLOPs (must be > 0).

        Returns:
            ComputeBudget with optimal_N, optimal_D, and estimated_loss.
        """
        assert C > 0, f"Compute budget C must be positive, got {C}"

        p = self.params

        # Closed-form optimal N*
        coeff = (p.alpha * p.A) / (p.beta * p.B * (6.0 ** p.beta))
        exponent = 1.0 / (p.alpha + p.beta)
        optimal_N = (coeff ** exponent) * (C ** (p.beta / (p.alpha + p.beta)))

        assert optimal_N > 0, f"optimal_N must be positive, got {optimal_N}"
        assert np.isfinite(optimal_N), f"optimal_N is not finite: {optimal_N}"

        # D* from the constraint C = 6 * N * D
        optimal_D = C / (6.0 * optimal_N)

        assert optimal_D > 0, f"optimal_D must be positive, got {optimal_D}"
        assert np.isfinite(optimal_D), f"optimal_D is not finite: {optimal_D}"

        # Verify budget conservation: |6*N*D - C| / C <= 0.10
        actual_flops = 6.0 * optimal_N * optimal_D
        relative_error = abs(actual_flops - C) / C
        assert relative_error <= 0.10, (
            f"Budget conservation violated: |6*N*D - C|/C = {relative_error:.6f} > 0.10"
        )

        estimated_loss = self.predict_loss(optimal_N, optimal_D)

        return ComputeBudget(
            total_flops=C,
            optimal_N=optimal_N,
            optimal_D=optimal_D,
            estimated_loss=estimated_loss,
        )


# ---------------------------------------------------------------------------
# Visualization & comparison utilities (Task 5.4)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_loss_vs_model_size(
    predictor: ScalingLawPredictor,
    D_fixed: float,
    N_range: np.ndarray,
) -> Figure:
    """Log-log plot of predicted loss vs model size N, holding D fixed.

    Args:
        predictor: A ScalingLawPredictor instance.
        D_fixed: Fixed number of training tokens.
        N_range: 1-D array of model sizes to evaluate.

    Returns:
        matplotlib Figure.
    """
    losses = np.array([predictor.predict_loss(float(n), D_fixed) for n in N_range])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(N_range, losses, "o-", color="tab:blue", linewidth=2, markersize=4)
    ax.axhline(predictor.params.E, color="gray", linestyle="--", label=f"Irreducible E={predictor.params.E}")
    ax.set_xlabel("Model parameters N")
    ax.set_ylabel("Predicted loss L(N, D)")
    ax.set_title(f"Loss vs Model Size (D = {D_fixed:.1e} tokens)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss_vs_data_size(
    predictor: ScalingLawPredictor,
    N_fixed: float,
    D_range: np.ndarray,
) -> Figure:
    """Log-log plot of predicted loss vs data size D, holding N fixed.

    Args:
        predictor: A ScalingLawPredictor instance.
        N_fixed: Fixed number of model parameters.
        D_range: 1-D array of token counts to evaluate.

    Returns:
        matplotlib Figure.
    """
    losses = np.array([predictor.predict_loss(N_fixed, float(d)) for d in D_range])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(D_range, losses, "s-", color="tab:orange", linewidth=2, markersize=4)
    ax.axhline(predictor.params.E, color="gray", linestyle="--", label=f"Irreducible E={predictor.params.E}")
    ax.set_xlabel("Training tokens D")
    ax.set_ylabel("Predicted loss L(N, D)")
    ax.set_title(f"Loss vs Data Size (N = {N_fixed:.1e} params)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def _kaplan_optimal_allocation(C: float):
    """Kaplan (2020) optimal allocation: N ∝ C^0.73, D ∝ C^0.27.

    Uses approximate proportionality constants calibrated so that
    6 * N * D ≈ C.
    """
    # N ∝ C^0.73 and D = C / (6N)
    # We pick a proportionality constant so 6*N*D = C.
    # N = k * C^0.73  =>  D = C/(6*k*C^0.73) = C^0.27 / (6*k)
    # 6*N*D = 6 * k*C^0.73 * C^0.27/(6*k) = C  (always satisfied)
    # So we just need a reasonable k.  Use k = 1/(6^0.73) so that at C=1 N≈D≈small.
    k = (1.0 / 6.0) ** 0.73
    N = k * (C ** 0.73)
    D = C / (6.0 * N)
    return N, D


def plot_kaplan_vs_chinchilla(C_range: np.ndarray) -> Figure:
    """Compare Kaplan (2020) vs Chinchilla (2022) optimal allocations.

    Plots optimal N and D/N ratio as a function of compute budget C for
    both scaling-law prescriptions.

    Args:
        C_range: 1-D array of compute budgets (FLOPs).

    Returns:
        matplotlib Figure with two subplots.
    """
    chinchilla = ScalingLawPredictor()

    chin_N, chin_D, kap_N, kap_D = [], [], [], []
    for C in C_range:
        cb = chinchilla.compute_optimal_allocation(float(C))
        chin_N.append(cb.optimal_N)
        chin_D.append(cb.optimal_D)
        kn, kd = _kaplan_optimal_allocation(float(C))
        kap_N.append(kn)
        kap_D.append(kd)

    chin_N = np.array(chin_N)
    chin_D = np.array(chin_D)
    kap_N = np.array(kap_N)
    kap_D = np.array(kap_D)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: optimal N vs C
    axes[0].loglog(C_range, chin_N, "o-", label="Chinchilla (2022)", color="tab:blue", linewidth=2, markersize=4)
    axes[0].loglog(C_range, kap_N, "s--", label="Kaplan (2020)", color="tab:red", linewidth=2, markersize=4)
    axes[0].set_xlabel("Compute budget C (FLOPs)")
    axes[0].set_ylabel("Optimal model size N*")
    axes[0].set_title("Optimal N* vs Compute")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # Right: D/N ratio vs C
    axes[1].semilogx(C_range, chin_D / chin_N, "o-", label="Chinchilla (2022)", color="tab:blue", linewidth=2, markersize=4)
    axes[1].semilogx(C_range, kap_D / kap_N, "s--", label="Kaplan (2020)", color="tab:red", linewidth=2, markersize=4)
    axes[1].set_xlabel("Compute budget C (FLOPs)")
    axes[1].set_ylabel("Tokens per parameter (D*/N*)")
    axes[1].set_title("Data-to-Model Ratio vs Compute")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def print_compute_budget_table(
    predictor: ScalingLawPredictor,
    C_values: list[float],
) -> None:
    """Print a formatted table of optimal allocations for given FLOP budgets.

    For each budget C, shows optimal N*, D*, D*/N* ratio, estimated loss,
    and approximate GPU-hours on an H100 (1 PFLOP/s).

    Args:
        predictor: A ScalingLawPredictor instance.
        C_values: List of compute budgets in FLOPs.
    """
    h100_flops_per_sec = 1e15  # ~1 PFLOP/s

    header = (
        f"{'Budget C':>14}  {'Optimal N*':>12}  {'Optimal D*':>12}  "
        f"{'D*/N*':>7}  {'Loss':>7}  {'H100-hours':>11}"
    )
    print(header)
    print("-" * len(header))

    for C in C_values:
        result = predictor.compute_optimal_allocation(C)
        ratio = result.optimal_D / result.optimal_N
        gpu_hours = C / h100_flops_per_sec / 3600.0
        print(
            f"{C:>14.1e}  {result.optimal_N:>12.2e}  {result.optimal_D:>12.2e}  "
            f"{ratio:>7.1f}  {result.estimated_loss:>7.4f}  {gpu_hours:>11.1f}"
        )
