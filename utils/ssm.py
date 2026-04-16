"""
State Space Model utilities for Notebook 16.

Provides SSM components: configs, discretization, scanning,
selective scan (Mamba-style), and visualization helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def plot_attention_vs_ssm_scaling(
    seq_lengths: list[int] | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Plot O(n²) attention vs O(n) SSM scaling for FLOPs and memory.

    Shows concrete numbers for why attention becomes a bottleneck
    at long sequence lengths and how SSMs avoid this.

    Parameters
    ----------
    seq_lengths : list[int], optional
        Sequence lengths to plot. Defaults to powers of 2 from 128 to 131072.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if seq_lengths is None:
        seq_lengths = [2**i for i in range(7, 18)]  # 128 to 131072

    d_model = 768  # typical hidden dim

    # Attention: O(n² * d) for QK^T, plus O(n² * d) for attn @ V
    attn_flops = [2 * (n**2) * d_model for n in seq_lengths]
    # SSM: O(n * d * d_state) — linear in sequence length
    d_state = 16  # typical SSM state dim
    ssm_flops = [2 * n * d_model * d_state for n in seq_lengths]

    # Memory: attention materializes n×n matrix; SSM keeps d_model×d_state state
    attn_mem_bytes = [n * n * 4 for n in seq_lengths]  # float32
    ssm_mem_bytes = [d_model * d_state * 4 for n in seq_lengths]  # constant!

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- FLOPs comparison ---
    ax = axes[0]
    ax.loglog(seq_lengths, attn_flops, "o-", color="#e74c3c", label="Attention O(n²d)", linewidth=2)
    ax.loglog(seq_lengths, ssm_flops, "s-", color="#2ecc71", label="SSM O(n·d·N)", linewidth=2)
    ax.set_xlabel("Sequence Length (n)", fontsize=12)
    ax.set_ylabel("FLOPs (log scale)", fontsize=12)
    ax.set_title("⚡ Compute: Attention vs SSM", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate the crossover region
    for n, af, sf in zip(seq_lengths, attn_flops, ssm_flops):
        if n == 4096:
            ratio = af / sf
            ax.annotate(
                f"{ratio:.0f}× gap at n={n:,}",
                xy=(n, af),
                xytext=(n * 0.15, af * 2),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                color="#e74c3c",
            )
            break

    # --- Memory comparison ---
    ax = axes[1]
    ax.loglog(seq_lengths, attn_mem_bytes, "o-", color="#e74c3c", label="Attention O(n²)", linewidth=2)
    ax.loglog(seq_lengths, ssm_mem_bytes, "s-", color="#2ecc71", label="SSM O(1) in seq len", linewidth=2)
    ax.set_xlabel("Sequence Length (n)", fontsize=12)
    ax.set_ylabel("Memory (bytes, log scale)", fontsize=12)
    ax.set_title("💾 Memory: Attention vs SSM", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate memory at 16k
    for n, am, sm in zip(seq_lengths, attn_mem_bytes, ssm_mem_bytes):
        if n == 16384:
            ax.annotate(
                f"Attn: {am / 1e9:.1f} GB\nSSM: {sm / 1e3:.0f} KB",
                xy=(n, am),
                xytext=(n * 0.08, am * 3),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                color="#333",
            )
            break

    fig.suptitle(
        "Why SSMs? The Quadratic Bottleneck of Attention",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig


def print_scaling_table(seq_lengths: list[int] | None = None) -> None:
    """Print a concrete table of attention vs SSM costs.

    Parameters
    ----------
    seq_lengths : list[int], optional
        Sequence lengths to tabulate.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 65536, 131072]

    d_model = 768
    d_state = 16

    print(f"{'Seq Len':>10} | {'Attn FLOPs':>14} | {'SSM FLOPs':>14} | {'Ratio':>8} | {'Attn Mem':>10} | {'SSM Mem':>10}")
    print("-" * 80)
    for n in seq_lengths:
        attn_f = 2 * n * n * d_model
        ssm_f = 2 * n * d_model * d_state
        ratio = attn_f / ssm_f
        attn_m = n * n * 4
        ssm_m = d_model * d_state * 4

        def fmt_bytes(b: float) -> str:
            if b >= 1e9:
                return f"{b / 1e9:.1f} GB"
            if b >= 1e6:
                return f"{b / 1e6:.1f} MB"
            if b >= 1e3:
                return f"{b / 1e3:.1f} KB"
            return f"{b:.0f} B"

        def fmt_flops(f: float) -> str:
            if f >= 1e12:
                return f"{f / 1e12:.1f} TFLOPs"
            if f >= 1e9:
                return f"{f / 1e9:.1f} GFLOPs"
            if f >= 1e6:
                return f"{f / 1e6:.1f} MFLOPs"
            return f"{f:.0f}"

        print(f"{n:>10,} | {fmt_flops(attn_f):>14} | {fmt_flops(ssm_f):>14} | {ratio:>7.0f}× | {fmt_bytes(attn_m):>10} | {fmt_bytes(ssm_m):>10}")


# ---------------------------------------------------------------------------
# MLX imports for SSM modules
# ---------------------------------------------------------------------------
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSMConfig:
    """Configuration for a State Space Model layer.

    Parameters
    ----------
    d_model : int
        Model / input dimension.
    d_state : int
        SSM hidden-state dimension *N* (typically 16 or 64).
    d_inner : int, optional
        Inner dimension after expansion.  Defaults to ``d_model * expand_factor``.
    expand_factor : int
        Expansion factor used to derive *d_inner* when it is not given.
    dt_rank : int, optional
        Rank of the Δ projection (used by SelectiveSSM later).
        Defaults to ``ceil(d_model / 16)``.
    """

    d_model: int = 64
    d_state: int = 16
    d_inner: int | None = None
    expand_factor: int = 2
    dt_rank: int | None = None
    d_conv: int = 4

    def __post_init__(self) -> None:
        if self.d_inner is None:
            self.d_inner = self.d_model * self.expand_factor
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)


# ---------------------------------------------------------------------------
# SimpleSSM — fixed-parameter SSM with ZOH discretization
# ---------------------------------------------------------------------------

class SimpleSSM(nn.Module):
    """A simple State Space Model with Zero-Order Hold discretization.

    Operates **per-channel**: one independent SSM of state dimension
    ``d_state`` is run for each of the ``d_inner`` channels.

    Parameters
    ----------
    config : SSMConfig
        Model configuration.

    Shapes
    ------
    Input  : ``[batch, seq, d_inner]``
    Output : ``[batch, seq, d_inner]``

    Internal state ``h`` : ``[batch, d_inner, d_state]``
    """

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        d_state = config.d_state

        # A — diagonal state-transition matrix, initialised with negative
        # values for stability (HiPPO-inspired: -1, -2, …, -N).
        # Shape: (d_inner, d_state)
        A_init = -mx.broadcast_to(
            mx.arange(1, d_state + 1, dtype=mx.float32)[None, :],
            (d_inner, d_state),
        )
        self.A_log = A_init  # store as-is; we'll use exp(delta * A) later

        # B — input-to-state projection, learnable.  Shape: (d_inner, d_state)
        self.B = mx.random.normal((d_inner, d_state)) * 0.01

        # C — state-to-output projection, learnable.  Shape: (d_inner, d_state)
        self.C = mx.random.normal((d_inner, d_state)) * 0.01

        # delta (Δ) — step size per channel, learnable.
        # Stored in log-space; we apply softplus to ensure positivity.
        # Shape: (d_inner,)
        self.log_delta = mx.zeros((d_inner,))

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    @staticmethod
    def discretize(
        A: mx.array,
        B: mx.array,
        delta: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Zero-Order Hold discretization.

        Parameters
        ----------
        A : mx.array, shape ``(d_inner, d_state)``
            Continuous-time diagonal state matrix (negative values).
        B : mx.array, shape ``(d_inner, d_state)``
            Continuous-time input matrix.
        delta : mx.array, shape ``(d_inner,)``
            Step sizes (positive).

        Returns
        -------
        A_bar : mx.array, shape ``(d_inner, d_state)``
            Discrete state matrix: ``exp(Δ × A)``.
        B_bar : mx.array, shape ``(d_inner, d_state)``
            Discrete input matrix: ``Δ × B``.
        """
        # Expand delta for broadcasting: (d_inner,) -> (d_inner, 1)
        delta_expanded = delta[:, None]
        assert delta_expanded.shape == (A.shape[0], 1), (
            f"delta_expanded shape {delta_expanded.shape} != ({A.shape[0]}, 1)"
        )

        # A_bar = exp(delta * A)  — element-wise for diagonal A
        A_bar = mx.exp(delta_expanded * A)
        assert A_bar.shape == A.shape, (
            f"A_bar shape {A_bar.shape} != A shape {A.shape}"
        )

        # B_bar = delta * B  — first-order ZOH approximation
        B_bar = delta_expanded * B
        assert B_bar.shape == B.shape, (
            f"B_bar shape {B_bar.shape} != B shape {B.shape}"
        )

        return A_bar, B_bar

    # ------------------------------------------------------------------
    # Sequential scan
    # ------------------------------------------------------------------

    def scan(self, x: mx.array) -> mx.array:
        """Run the sequential SSM scan.

        Recurrence
        ----------
        h[t] = A_bar × h[t-1] + B_bar × x[t]
        y[t] = C @ h[t]   (contract over d_state)

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_inner]``

        Returns
        -------
        y : mx.array, shape ``[batch, seq, d_inner]``
        """
        batch, seq_len, d_inner = x.shape
        d_state = self.config.d_state
        assert d_inner == self.config.d_inner, (
            f"Expected d_inner={self.config.d_inner}, got {d_inner}"
        )

        # --- Discretize ---
        delta = nn.softplus(self.log_delta)  # ensure positivity, (d_inner,)
        assert delta.shape == (d_inner,), (
            f"delta shape {delta.shape} != ({d_inner},)"
        )

        A_bar, B_bar = self.discretize(self.A_log, self.B, delta)
        # A_bar, B_bar: (d_inner, d_state)

        # --- Numerical stability check ---
        # A_bar should have magnitude <= 1 for stable recurrence
        # (guaranteed when A has negative values and delta > 0)

        # --- Sequential scan ---
        h = mx.zeros((batch, d_inner, d_state))  # hidden state
        outputs = []

        for t in range(seq_len):
            # x_t: (batch, d_inner) -> (batch, d_inner, 1) for broadcasting
            x_t = x[:, t, :]
            assert x_t.shape == (batch, d_inner), (
                f"x_t shape {x_t.shape} != ({batch}, {d_inner})"
            )

            # State update: h = A_bar * h + B_bar * x_t
            # A_bar: (d_inner, d_state) broadcast with h: (batch, d_inner, d_state)
            # B_bar: (d_inner, d_state) * x_t[:, :, None]: (batch, d_inner, 1)
            h = A_bar[None, :, :] * h + B_bar[None, :, :] * x_t[:, :, None]
            assert h.shape == (batch, d_inner, d_state), (
                f"h shape {h.shape} != ({batch}, {d_inner}, {d_state})"
            )

            # Output: y_t = sum(C * h, axis=-1) — contract over d_state
            y_t = mx.sum(self.C[None, :, :] * h, axis=-1)
            assert y_t.shape == (batch, d_inner), (
                f"y_t shape {y_t.shape} != ({batch}, {d_inner})"
            )
            outputs.append(y_t)

        # Stack along sequence dimension
        y = mx.stack(outputs, axis=1)  # (batch, seq, d_inner)
        assert y.shape == (batch, seq_len, d_inner), (
            f"Output shape {y.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        return y

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: discretize + sequential scan.

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_inner]``

        Returns
        -------
        y : mx.array, shape ``[batch, seq, d_inner]``
        """
        assert x.ndim == 3, f"Expected 3-D input [batch, seq, d_inner], got {x.ndim}-D"
        assert x.shape[-1] == self.config.d_inner, (
            f"Expected last dim = {self.config.d_inner}, got {x.shape[-1]}"
        )
        return self.scan(x)


# ---------------------------------------------------------------------------
# SelectiveSSM — Mamba-style SSM with input-dependent Δ, B, C
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Selective State Space Model (Mamba-style).

    Unlike SimpleSSM where A, B, C, Δ are fixed parameters, SelectiveSSM
    makes Δ, B, C **input-dependent** — projected from the input at each
    timestep.  This breaks the LTI (Linear Time-Invariant) assumption and
    gives the model the ability to selectively focus on or ignore parts of
    the input sequence.

    The core Mamba insight: by making parameters input-dependent, the model
    learns WHEN to update its state (large Δ → integrate input) and WHEN
    to ignore (small Δ → retain previous state).

    Parameters
    ----------
    config : SSMConfig
        Model configuration.

    Shapes
    ------
    Input  : ``[batch, seq, d_inner]``
    Output : ``[batch, seq, d_inner]``
    """

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        d_state = config.d_state
        dt_rank = config.dt_rank

        # A — diagonal state matrix, stored as log for parameterisation.
        # Initialised with negative values (-1, -2, …, -N) for stability.
        # Shape: (d_inner, d_state)
        A_init = -mx.broadcast_to(
            mx.arange(1, d_state + 1, dtype=mx.float32)[None, :],
            (d_inner, d_state),
        )
        self.A_log = A_init

        # Combined input projection: x -> (dt_input, B, C)
        # dt_input has dt_rank dims, B and C each have d_state dims.
        # Shape: Linear(d_inner -> dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state)

        # Δ projection: dt_rank -> d_inner  (broadcast to per-channel step size)
        self.dt_proj = nn.Linear(dt_rank, d_inner)

    # ------------------------------------------------------------------
    # Sequential scan
    # ------------------------------------------------------------------

    @staticmethod
    def sequential_scan(
        x: mx.array,
        A_bar: mx.array,
        B_bar: mx.array,
        C: mx.array,
    ) -> mx.array:
        """Run the SSM recurrence sequentially — O(n) serial.

        Recurrence (input-dependent parameters at each step):
            h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
            y[t] = sum(C[t] * h[t], axis=-1)

        Parameters
        ----------
        x     : [batch, seq, d_inner]       — input signal
        A_bar : [batch, seq, d_inner, d_state] — discretised state matrix
        B_bar : [batch, seq, d_inner, d_state] — discretised input matrix
        C     : [batch, seq, d_state]        — output projection (input-dep)

        Returns
        -------
        y : [batch, seq, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A_bar.shape[-1]

        assert A_bar.shape == (batch, seq_len, d_inner, d_state), (
            f"A_bar shape {A_bar.shape} != ({batch}, {seq_len}, {d_inner}, {d_state})"
        )
        assert B_bar.shape == (batch, seq_len, d_inner, d_state), (
            f"B_bar shape {B_bar.shape} != ({batch}, {seq_len}, {d_inner}, {d_state})"
        )
        assert C.shape == (batch, seq_len, d_state), (
            f"C shape {C.shape} != ({batch}, {seq_len}, {d_state})"
        )

        h = mx.zeros((batch, d_inner, d_state))
        outputs = []

        for t in range(seq_len):
            # x_t: (batch, d_inner) -> (batch, d_inner, 1) for broadcasting
            x_t = x[:, t, :]  # (batch, d_inner)
            assert x_t.shape == (batch, d_inner)

            # A_bar_t, B_bar_t: (batch, d_inner, d_state)
            A_bar_t = A_bar[:, t, :, :]
            B_bar_t = B_bar[:, t, :, :]

            # State update: h = A_bar_t * h + B_bar_t * x_t
            h = A_bar_t * h + B_bar_t * x_t[:, :, None]
            assert h.shape == (batch, d_inner, d_state)

            # Output: y_t = sum(C_t * h, axis=-1)
            # C_t: (batch, d_state) -> (batch, 1, d_state) for broadcasting
            C_t = C[:, t, :]  # (batch, d_state)
            y_t = mx.sum(C_t[:, None, :] * h, axis=-1)  # (batch, d_inner)
            assert y_t.shape == (batch, d_inner)

            outputs.append(y_t)

        y = mx.stack(outputs, axis=1)  # (batch, seq, d_inner)
        assert y.shape == (batch, seq_len, d_inner)
        return y

    # ------------------------------------------------------------------
    # Parallel scan (associative scan / prefix sum)
    # ------------------------------------------------------------------

    @staticmethod
    def parallel_scan(
        x: mx.array,
        A_bar: mx.array,
        B_bar: mx.array,
        C: mx.array,
    ) -> mx.array:
        """Run the SSM recurrence via parallel prefix sum — O(n log n) work.

        Uses the associative-scan formulation.  The recurrence
            h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
        can be written as an associative binary operator on tuples (a, b):
            (a2, b2) ∘ (a1, b1) = (a2 * a1,  a2 * b1 + b2)
        An inclusive prefix scan with this operator yields all hidden states
        in O(log n) parallel steps (Blelloch / Hillis-Steele).

        Parameters
        ----------
        x     : [batch, seq, d_inner]
        A_bar : [batch, seq, d_inner, d_state]
        B_bar : [batch, seq, d_inner, d_state]
        C     : [batch, seq, d_state]

        Returns
        -------
        y : [batch, seq, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A_bar.shape[-1]

        assert A_bar.shape == (batch, seq_len, d_inner, d_state)
        assert B_bar.shape == (batch, seq_len, d_inner, d_state)
        assert C.shape == (batch, seq_len, d_state)

        # Build initial tuples: a[t] = A_bar[t], b[t] = B_bar[t] * x[t]
        # a: (batch, seq, d_inner, d_state)
        # b: (batch, seq, d_inner, d_state)  — the "input contribution"
        a = A_bar
        b = B_bar * x[:, :, :, None]  # broadcast x over d_state

        assert a.shape == (batch, seq_len, d_inner, d_state)
        assert b.shape == (batch, seq_len, d_inner, d_state)

        # Hillis-Steele inclusive prefix scan: O(log n) steps
        stride = 1
        while stride < seq_len:
            # Elements at position i (where i >= stride) combine with i-stride
            # (a[i], b[i]) = (a[i] * a[i-s], a[i] * b[i-s] + b[i])
            a_shifted = mx.concatenate(
                [mx.ones((batch, stride, d_inner, d_state)), a[:, :-stride, :, :]],
                axis=1,
            )
            b_shifted = mx.concatenate(
                [mx.zeros((batch, stride, d_inner, d_state)), b[:, :-stride, :, :]],
                axis=1,
            )

            b = a * b_shifted + b
            a = a * a_shifted

            stride *= 2

        # After the scan, b[t] = h[t] (the hidden state at each position)
        h = b  # (batch, seq, d_inner, d_state)
        assert h.shape == (batch, seq_len, d_inner, d_state)

        # Output: y[t] = sum(C[t] * h[t], axis=-1)
        # C: (batch, seq, d_state) -> (batch, seq, 1, d_state)
        y = mx.sum(C[:, :, None, :] * h, axis=-1)  # (batch, seq, d_inner)
        assert y.shape == (batch, seq_len, d_inner)
        return y

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array, use_parallel: bool = False) -> mx.array:
        """Forward pass: project input-dependent params, discretize, scan.

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_inner]``
        use_parallel : bool
            If True, use parallel scan (O(n log n) work, O(log n) depth).
            If False (default), use sequential scan (O(n) work, O(n) depth).

        Returns
        -------
        y : mx.array, shape ``[batch, seq, d_inner]``
        """
        assert x.ndim == 3, f"Expected 3-D input [batch, seq, d_inner], got {x.ndim}-D"
        batch, seq_len, d_inner = x.shape
        assert d_inner == self.config.d_inner, (
            f"Expected d_inner={self.config.d_inner}, got {d_inner}"
        )
        d_state = self.config.d_state
        dt_rank = self.config.dt_rank

        # --- Project input to get dt, B, C (all input-dependent) ---
        # x_proj: (batch, seq, d_inner) -> (batch, seq, dt_rank + 2*d_state)
        x_dbc = self.x_proj(x)
        assert x_dbc.shape == (batch, seq_len, dt_rank + 2 * d_state), (
            f"x_dbc shape {x_dbc.shape} != ({batch}, {seq_len}, {dt_rank + 2 * d_state})"
        )

        # Split into dt_input, B, C
        dt_input = x_dbc[:, :, :dt_rank]                          # (batch, seq, dt_rank)
        B = x_dbc[:, :, dt_rank:dt_rank + d_state]                # (batch, seq, d_state)
        C = x_dbc[:, :, dt_rank + d_state:]                       # (batch, seq, d_state)

        assert dt_input.shape == (batch, seq_len, dt_rank)
        assert B.shape == (batch, seq_len, d_state)
        assert C.shape == (batch, seq_len, d_state)

        # --- Compute Δ (step size) per channel ---
        # dt_proj: (batch, seq, dt_rank) -> (batch, seq, d_inner)
        dt = nn.softplus(self.dt_proj(dt_input))  # ensure positivity
        assert dt.shape == (batch, seq_len, d_inner), (
            f"dt shape {dt.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        # --- Discretize: A_bar = exp(dt * A), B_bar = dt * B ---
        # A_log: (d_inner, d_state)
        # dt: (batch, seq, d_inner) -> (batch, seq, d_inner, 1) for broadcasting
        A = self.A_log  # (d_inner, d_state) — negative values
        A_bar = mx.exp(dt[:, :, :, None] * A[None, None, :, :])
        assert A_bar.shape == (batch, seq_len, d_inner, d_state), (
            f"A_bar shape {A_bar.shape} != ({batch}, {seq_len}, {d_inner}, {d_state})"
        )

        # B_bar = dt * B  (broadcast dt over d_state, B over d_inner)
        # dt: (batch, seq, d_inner, 1), B: (batch, seq, 1, d_state)
        B_bar = dt[:, :, :, None] * B[:, :, None, :]
        assert B_bar.shape == (batch, seq_len, d_inner, d_state), (
            f"B_bar shape {B_bar.shape} != ({batch}, {seq_len}, {d_inner}, {d_state})"
        )

        # --- Run scan ---
        scan_fn = self.parallel_scan if use_parallel else self.sequential_scan
        y = scan_fn(x, A_bar, B_bar, C)

        assert y.shape == (batch, seq_len, d_inner), (
            f"Output shape {y.shape} != ({batch}, {seq_len}, {d_inner})"
        )
        return y


# ---------------------------------------------------------------------------
# MambaBlock — full Mamba block with projection, conv1d, SSM, gating
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Full Mamba block: input projection → conv1d → selective SSM → SiLU gating → output projection.

    This is the core building block of the Mamba architecture.  It wraps
    :class:`SelectiveSSM` with:

    * **Input projection** — expands ``d_model`` to ``2 × d_inner`` (SSM branch + gate branch)
    * **Causal 1-D convolution** — provides local context mixing (like a small attention window)
    * **Selective SSM** — provides global context via linear recurrence (like attention but O(n))
    * **SiLU gating** — controls information flow between branches
    * **Output projection** — maps ``d_inner`` back to ``d_model``
    * **Pre-normalization** — RMSNorm before the block
    * **Residual connection** — skip connection around the block

    Parameters
    ----------
    config : SSMConfig
        Model configuration.  Uses ``d_model``, ``d_inner``, ``d_state``,
        ``d_conv``, ``dt_rank``, and ``expand_factor``.

    Shapes
    ------
    Input  : ``[batch, seq, d_model]``
    Output : ``[batch, seq, d_model]``

    Memory : O(batch × d_inner × d_state) — independent of sequence length.
    """

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_conv = config.d_conv

        # Pre-normalization
        self.norm = nn.RMSNorm(d_model)

        # Input projection: d_model -> 2 * d_inner (SSM branch + gate branch)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)

        # Causal 1-D depthwise convolution (local context mixing)
        # Weight shape: (d_inner, d_conv) — one kernel per channel
        self.conv_weight = mx.random.normal((d_inner, d_conv)) * 0.02
        self.conv_bias = mx.zeros((d_inner,))

        # Selective SSM (global context via linear recurrence)
        self.ssm = SelectiveSSM(config)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(d_inner, d_model)

    # ------------------------------------------------------------------
    # Causal 1-D convolution
    # ------------------------------------------------------------------

    def causal_conv1d(self, x: mx.array) -> mx.array:
        """Apply causal depthwise 1-D convolution.

        Pads the input on the **left** by ``d_conv - 1`` so that the output
        at position *t* depends only on inputs at positions ``t - d_conv + 1``
        through ``t`` (no future leakage).

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_inner]``

        Returns
        -------
        mx.array, shape ``[batch, seq, d_inner]``
        """
        batch, seq_len, d_inner = x.shape
        d_conv = self.config.d_conv
        assert d_inner == self.config.d_inner, (
            f"Expected d_inner={self.config.d_inner}, got {d_inner}"
        )

        # Causal padding: pad left by (d_conv - 1), no padding on right
        pad = mx.zeros((batch, d_conv - 1, d_inner))
        x_padded = mx.concatenate([pad, x], axis=1)
        assert x_padded.shape == (batch, seq_len + d_conv - 1, d_inner), (
            f"x_padded shape {x_padded.shape} != ({batch}, {seq_len + d_conv - 1}, {d_inner})"
        )

        # Depthwise convolution via shifted accumulation
        # conv_weight: (d_inner, d_conv) — one kernel per channel
        result = mx.zeros((batch, seq_len, d_inner))
        for k in range(d_conv):
            # x_padded[:, k:k+seq_len, :] is (batch, seq_len, d_inner)
            # conv_weight[:, k] is (d_inner,) — broadcasts over batch and seq
            result = result + x_padded[:, k : k + seq_len, :] * self.conv_weight[:, k]

        result = result + self.conv_bias
        assert result.shape == (batch, seq_len, d_inner), (
            f"conv1d output shape {result.shape} != ({batch}, {seq_len}, {d_inner})"
        )
        return result

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass of the full Mamba block.

        Pipeline::

            x ─→ RMSNorm ─→ in_proj ─→ split ─→ [x_proj | z]
                                                     │       │
                                                  conv1d   SiLU
                                                     │       │
                                                   SiLU      │
                                                     │       │
                                                    SSM      │
                                                     │       │
                                                     └── × ──┘
                                                         │
                                                      out_proj
                                                         │
                                                    + residual
                                                         │
                                                       output

        Parameters
        ----------
        x : mx.array, shape ``[batch, seq, d_model]``

        Returns
        -------
        mx.array, shape ``[batch, seq, d_model]``
        """
        assert x.ndim == 3, f"Expected 3-D input [batch, seq, d_model], got {x.ndim}-D"
        batch, seq_len, d_model = x.shape
        assert d_model == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {d_model}"
        )
        d_inner = self.config.d_inner
        residual = x

        # 1. Pre-normalization
        x_norm = self.norm(x)
        assert x_norm.shape == (batch, seq_len, d_model), (
            f"norm output shape {x_norm.shape} != ({batch}, {seq_len}, {d_model})"
        )

        # 2. Input projection: d_model -> 2 * d_inner, then split
        x_proj = self.in_proj(x_norm)  # (batch, seq, 2 * d_inner)
        assert x_proj.shape == (batch, seq_len, 2 * d_inner), (
            f"in_proj output shape {x_proj.shape} != ({batch}, {seq_len}, {2 * d_inner})"
        )
        x_ssm = x_proj[:, :, :d_inner]   # SSM branch
        z = x_proj[:, :, d_inner:]         # Gate branch
        assert x_ssm.shape == (batch, seq_len, d_inner), (
            f"x_ssm shape {x_ssm.shape} != ({batch}, {seq_len}, {d_inner})"
        )
        assert z.shape == (batch, seq_len, d_inner), (
            f"z shape {z.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        # 3. Causal 1-D convolution (local context mixing)
        x_conv = self.causal_conv1d(x_ssm)
        assert x_conv.shape == (batch, seq_len, d_inner), (
            f"conv1d output shape {x_conv.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        # 4. SiLU activation + Selective SSM (global context)
        x_ssm_out = self.ssm(nn.silu(x_conv))
        assert x_ssm_out.shape == (batch, seq_len, d_inner), (
            f"SSM output shape {x_ssm_out.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        # 5. SiLU gating: element-wise multiply with gated branch
        output = x_ssm_out * nn.silu(z)
        assert output.shape == (batch, seq_len, d_inner), (
            f"gated output shape {output.shape} != ({batch}, {seq_len}, {d_inner})"
        )

        # 6. Output projection: d_inner -> d_model
        output = self.out_proj(output)
        assert output.shape == (batch, seq_len, d_model), (
            f"out_proj output shape {output.shape} != ({batch}, {seq_len}, {d_model})"
        )

        # 7. Residual connection
        output = output + residual
        assert output.shape == (batch, seq_len, d_model), (
            f"final output shape {output.shape} != ({batch}, {seq_len}, {d_model})"
        )

        return output
