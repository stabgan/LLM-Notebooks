"""Attention optimization utilities for Notebook 12: Flash, Paged, and Ring Attention.

Provides online softmax, tiled flash attention, paged KV-cache management,
ring attention simulation, and benchmarking — all implemented in MLX on Apple Silicon.

**Validates: Requirements 10.1–10.9**
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Standard (reference) attention
# ---------------------------------------------------------------------------

def standard_attention(Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
    """Compute standard scaled dot-product attention: softmax(QK^T/√d)V.

    💡 This is the O(n²) reference implementation used to verify optimized variants.

    Args:
        Q: Query tensor [seq, d] or [batch, heads, seq, d].
        K: Key tensor, same shape as Q.
        V: Value tensor, same shape as Q.

    Returns:
        Attention output, same shape as Q.
    """
    d = Q.shape[-1]
    scale = math.sqrt(d)
    scores = (Q @ mx.transpose(K, list(range(K.ndim - 2)) + [K.ndim - 1, K.ndim - 2])) / scale
    weights = mx.softmax(scores, axis=-1)
    return weights @ V


# ---------------------------------------------------------------------------
# Online Softmax
# ---------------------------------------------------------------------------

def online_softmax(x: mx.array) -> mx.array:
    """Compute softmax using the online (streaming) algorithm.

    💡 Key insight: softmax can be computed incrementally over blocks using
    running max and running sum statistics — no need to see all values first.

    The algorithm processes elements one-at-a-time (or block-at-a-time),
    maintaining:
      - m: running maximum (for numerical stability)
      - l: running sum of exp(x_i - m)

    Final result: softmax(x)_i = exp(x_i - m) / l

    ⚡ This is the core algorithm that makes Flash Attention possible.

    Args:
        x: Input tensor of any shape. Softmax is applied along the last axis.

    Returns:
        Softmax output, same shape as x.
    """
    # Work along the last axis
    original_shape = x.shape
    if x.ndim == 1:
        x_flat = mx.reshape(x, (1, -1))
    else:
        # Flatten all but last dim
        x_flat = mx.reshape(x, (-1, x.shape[-1]))

    rows, cols = x_flat.shape

    # Process element-by-element along last axis using running stats
    m = mx.full((rows, 1), -1e30)  # running max
    l = mx.zeros((rows, 1))        # running sum of exp(x - m)

    for j in range(cols):
        x_j = x_flat[:, j : j + 1]  # [rows, 1]
        m_new = mx.maximum(m, x_j)
        # Rescale previous sum and add new element
        l = l * mx.exp(m - m_new) + mx.exp(x_j - m_new)
        m = m_new

    # Compute final softmax values
    result = mx.exp(x_flat - m) / l
    return mx.reshape(result, original_shape)


def online_softmax_blocked(x: mx.array, block_size: int = 4) -> mx.array:
    """Compute softmax using blocked online algorithm (processes chunks at a time).

    🎯 Interview tip: Flash Attention processes attention scores in blocks,
    not element-by-element. This blocked variant mirrors that approach.

    Args:
        x: Input tensor. Softmax applied along last axis.
        block_size: Number of elements to process per block.

    Returns:
        Softmax output, same shape as x.
    """
    original_shape = x.shape
    if x.ndim == 1:
        x_flat = mx.reshape(x, (1, -1))
    else:
        x_flat = mx.reshape(x, (-1, x.shape[-1]))

    rows, cols = x_flat.shape
    m = mx.full((rows, 1), -1e30)
    l = mx.zeros((rows, 1))

    for start in range(0, cols, block_size):
        end = min(start + block_size, cols)
        x_block = x_flat[:, start:end]  # [rows, block_width]

        # Block max
        block_max = mx.max(x_block, axis=-1, keepdims=True)  # [rows, 1]
        m_new = mx.maximum(m, block_max)

        # Rescale previous sum and add block contribution
        l = l * mx.exp(m - m_new) + mx.sum(mx.exp(x_block - m_new), axis=-1, keepdims=True)
        m = m_new

    result = mx.exp(x_flat - m) / l
    return mx.reshape(result, original_shape)


# ---------------------------------------------------------------------------
# Tiled Flash Attention
# ---------------------------------------------------------------------------

def tiled_attention(Q: mx.array, K: mx.array, V: mx.array,
                    block_size: int = 32) -> mx.array:
    """Compute attention using tiled (Flash Attention) algorithm.

    ⚡ Processes Q, K, V in blocks without materializing the full O(n²) matrix.
    Uses online softmax to incrementally accumulate the output.

    Memory: O(n × d + block_size²) vs O(n² + n × d) for standard attention.

    Algorithm:
        For each Q block i:
            For each KV block j:
                1. Compute S_ij = Q_i @ K_j^T / √d   (block_size × block_size tile)
                2. Update running max m and sum l (online softmax)
                3. Accumulate O_i += softmax_tile @ V_j (rescaled)
            Normalize O_i by final l

    Args:
        Q: Query tensor [seq, d].
        K: Key tensor [seq, d].
        V: Value tensor [seq, d].
        block_size: Tile size for blocking. Must be > 0.

    Returns:
        Attention output [seq, d], identical to standard attention within 1e-5.
    """
    seq_len, d = Q.shape
    scale = math.sqrt(d)
    num_blocks = math.ceil(seq_len / block_size)

    # Pad if seq_len not divisible by block_size
    pad_len = num_blocks * block_size - seq_len
    if pad_len > 0:
        Q = mx.concatenate([Q, mx.zeros((pad_len, d))], axis=0)
        K = mx.concatenate([K, mx.zeros((pad_len, d))], axis=0)
        V = mx.concatenate([V, mx.zeros((pad_len, d))], axis=0)

    return _tiled_attention_impl(Q, K, V, block_size, seq_len, d)


def _tiled_attention_impl(Q: mx.array, K: mx.array, V: mx.array,
                           block_size: int, orig_seq_len: int, d: int) -> mx.array:
    """Internal implementation of tiled flash attention with clean block accumulation.

    Handles padding by masking out positions beyond orig_seq_len so that
    padded zeros don't affect the softmax computation.
    """
    padded_len = Q.shape[0]
    num_blocks = padded_len // block_size
    scale = math.sqrt(d)

    # Build a mask for padded key positions: True = valid, False = padded
    # We need to mask out columns in S_ij that correspond to padded K positions
    valid_mask_k = mx.arange(padded_len) < orig_seq_len  # [padded_len]

    output_blocks = []

    for i in range(num_blocks):
        q_s = i * block_size
        q_e = q_s + block_size
        Q_i = Q[q_s:q_e, :]

        # Per-row running statistics for this Q block
        m_i = mx.full((block_size, 1), -1e30)
        l_i = mx.zeros((block_size, 1))
        O_i = mx.zeros((block_size, d))

        for j in range(num_blocks):
            k_s = j * block_size
            k_e = k_s + block_size
            K_j = K[k_s:k_e, :]
            V_j = V[k_s:k_e, :]

            # Tile scores
            S_ij = (Q_i @ K_j.T) / scale  # [block_size, block_size]

            # Mask out padded key positions with -inf
            kv_mask = valid_mask_k[k_s:k_e]  # [block_size] boolean
            # Also mask padded query positions (rows)
            qv_mask = mx.arange(q_s, q_e) < orig_seq_len  # [block_size]
            # Apply key mask: set padded columns to -inf
            kv_mask_2d = mx.reshape(kv_mask, (1, -1))  # [1, block_size]
            S_ij = mx.where(kv_mask_2d, S_ij, mx.full(S_ij.shape, -1e30))

            # Online softmax update
            row_max = mx.max(S_ij, axis=-1, keepdims=True)
            m_new = mx.maximum(m_i, row_max)

            rescale = mx.exp(m_i - m_new)
            O_i = O_i * rescale
            l_i = l_i * rescale

            P_ij = mx.exp(S_ij - m_new)
            O_i = O_i + P_ij @ V_j
            l_i = l_i + mx.sum(P_ij, axis=-1, keepdims=True)

            m_i = m_new

        # Normalize
        O_i = O_i / l_i
        output_blocks.append(O_i)

    O = mx.concatenate(output_blocks, axis=0)
    # Remove padding
    return O[:orig_seq_len, :]



def flash_memory_analysis(seq_len: int, d: int, block_size: int = 32) -> Dict[str, float]:
    """Analyze memory usage: standard attention vs flash attention.

    ⚠️ Standard attention materializes the full n×n matrix — O(n²) memory.
    Flash attention only needs one block_size × block_size tile at a time — O(n).

    Args:
        seq_len: Sequence length.
        d: Head dimension.
        block_size: Flash attention block size.

    Returns:
        Dict with memory estimates in bytes (float32).
    """
    bytes_per_elem = 4  # float32

    # Standard: stores full attention matrix + Q, K, V, O
    standard_attn_matrix = seq_len * seq_len * bytes_per_elem
    standard_qkvo = 4 * seq_len * d * bytes_per_elem
    standard_total = standard_attn_matrix + standard_qkvo

    # Flash: one tile at a time + Q, K, V, O + running stats
    flash_tile = block_size * block_size * bytes_per_elem
    flash_qkvo = 4 * seq_len * d * bytes_per_elem
    flash_stats = 2 * seq_len * bytes_per_elem  # m and l vectors
    flash_total = flash_tile + flash_qkvo + flash_stats

    return {
        "standard_attn_matrix_bytes": standard_attn_matrix,
        "standard_total_bytes": standard_total,
        "flash_tile_bytes": flash_tile,
        "flash_total_bytes": flash_total,
        "memory_ratio": standard_total / flash_total if flash_total > 0 else float("inf"),
        "standard_complexity": "O(n² + n×d)",
        "flash_complexity": "O(n×d + block_size²)",
    }


# ---------------------------------------------------------------------------
# Paged Attention — Block Manager
# ---------------------------------------------------------------------------

@dataclass
class KVBlock:
    """A fixed-size block for paged KV-cache storage.

    💡 Analogy: just like OS virtual memory pages, KV-cache is divided into
    fixed-size blocks that are allocated on demand — no wasted memory.
    """
    block_id: int
    block_size: int
    num_heads: int
    head_dim: int
    key_data: mx.array    # [block_size, num_heads, head_dim]
    value_data: mx.array  # [block_size, num_heads, head_dim]
    used: int = 0         # number of slots filled

    @property
    def is_full(self) -> bool:
        return self.used >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.used


class PagedAttentionBlockManager:
    """Block manager for paged KV-cache (vLLM-style).

    🎯 Interview tip: Paged Attention (vLLM, 2023) manages KV-cache like
    virtual memory — fixed-size blocks allocated on demand, reducing
    fragmentation when serving multiple concurrent requests.

    Args:
        block_size: Tokens per block (e.g., 16).
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        max_blocks: Maximum number of blocks in the pool.
    """

    def __init__(self, block_size: int = 16, num_heads: int = 4,
                 head_dim: int = 64, max_blocks: int = 256):
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks

        self._next_id = 0
        self._active_blocks: Dict[int, KVBlock] = {}
        self._free_ids: List[int] = []
        self._page_table: List[int] = []  # ordered list of block IDs for a sequence

    def allocate_block(self) -> KVBlock:
        """Allocate a new KV block from the pool.

        Returns:
            A fresh KVBlock with zeroed storage.

        Raises:
            RuntimeError: If no blocks are available.
        """
        if self._free_ids:
            bid = self._free_ids.pop()
        else:
            if self._next_id >= self.max_blocks:
                raise RuntimeError("⚠️ Block pool exhausted — no free blocks available")
            bid = self._next_id
            self._next_id += 1

        block = KVBlock(
            block_id=bid,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            key_data=mx.zeros((self.block_size, self.num_heads, self.head_dim)),
            value_data=mx.zeros((self.block_size, self.num_heads, self.head_dim)),
            used=0,
        )
        self._active_blocks[bid] = block
        self._page_table.append(bid)
        return block

    def free_block(self, block_id: int) -> None:
        """Return a block to the free pool.

        Args:
            block_id: ID of the block to free.
        """
        if block_id in self._active_blocks:
            del self._active_blocks[block_id]
            self._free_ids.append(block_id)
            if block_id in self._page_table:
                self._page_table.remove(block_id)

    def append_token(self, key: mx.array, value: mx.array) -> None:
        """Append a single token's KV to the cache, allocating a new block if needed.

        Args:
            key: Key vector [num_heads, head_dim].
            value: Value vector [num_heads, head_dim].
        """
        # Get current block or allocate new one
        if not self._page_table or self._active_blocks[self._page_table[-1]].is_full:
            self.allocate_block()

        block = self._active_blocks[self._page_table[-1]]
        slot = block.used

        # Write KV into the slot
        # We need to update the arrays — use concatenation for MLX immutability
        k_data = block.key_data
        v_data = block.value_data

        # Create updated slice
        k_row = mx.expand_dims(key, axis=0)   # [1, num_heads, head_dim]
        v_row = mx.expand_dims(value, axis=0)  # [1, num_heads, head_dim]

        if slot == 0:
            new_k = mx.concatenate([k_row, k_data[1:, :, :]], axis=0)
            new_v = mx.concatenate([v_row, v_data[1:, :, :]], axis=0)
        elif slot == self.block_size - 1:
            new_k = mx.concatenate([k_data[:slot, :, :], k_row], axis=0)
            new_v = mx.concatenate([v_data[:slot, :, :], v_row], axis=0)
        else:
            new_k = mx.concatenate([k_data[:slot, :, :], k_row, k_data[slot + 1:, :, :]], axis=0)
            new_v = mx.concatenate([v_data[:slot, :, :], v_row, v_data[slot + 1:, :, :]], axis=0)

        block.key_data = new_k
        block.value_data = new_v
        block.used = slot + 1

    def read_kv(self) -> Tuple[mx.array, mx.array]:
        """Read all cached KV pairs in sequence order.

        Returns:
            (keys, values) each of shape [total_tokens, num_heads, head_dim].
        """
        if not self._page_table:
            return (
                mx.zeros((0, self.num_heads, self.head_dim)),
                mx.zeros((0, self.num_heads, self.head_dim)),
            )

        k_parts = []
        v_parts = []
        for bid in self._page_table:
            block = self._active_blocks[bid]
            if block.used > 0:
                k_parts.append(block.key_data[: block.used, :, :])
                v_parts.append(block.value_data[: block.used, :, :])

        if not k_parts:
            return (
                mx.zeros((0, self.num_heads, self.head_dim)),
                mx.zeros((0, self.num_heads, self.head_dim)),
            )

        return mx.concatenate(k_parts, axis=0), mx.concatenate(v_parts, axis=0)

    @property
    def num_tokens(self) -> int:
        """Total number of tokens stored across all blocks."""
        return sum(
            self._active_blocks[bid].used
            for bid in self._page_table
            if bid in self._active_blocks
        )

    @property
    def num_active_blocks(self) -> int:
        return len(self._page_table)

    @property
    def utilization(self) -> float:
        """Fraction of allocated slots that are used."""
        total_slots = self.num_active_blocks * self.block_size
        if total_slots == 0:
            return 0.0
        return self.num_tokens / total_slots

    def stats(self) -> Dict[str, Any]:
        """Return block manager statistics."""
        return {
            "num_tokens": self.num_tokens,
            "num_active_blocks": self.num_active_blocks,
            "block_size": self.block_size,
            "utilization": self.utilization,
            "memory_bytes": self.num_active_blocks * self.block_size * self.num_heads * self.head_dim * 4 * 2,
        }


# ---------------------------------------------------------------------------
# Ring Attention Simulation
# ---------------------------------------------------------------------------

def simulate_ring(Q: mx.array, K: mx.array, V: mx.array,
                  num_devices: int) -> mx.array:
    """Simulate Ring Attention across virtual devices.

    🎯 Ring Attention (2024) distributes long sequences across devices in a
    ring topology. Each device holds a chunk of Q and circulates K/V chunks
    around the ring, enabling context lengths of 1M+ tokens.

    Algorithm:
        1. Split Q, K, V into num_devices chunks
        2. Each device computes partial attention with its local Q and
           the current K/V chunk
        3. K/V chunks are passed to the next device in the ring
        4. After num_devices steps, each device has seen all K/V chunks
        5. Combine partial results using online softmax statistics

    Args:
        Q: Query tensor [seq, d].
        K: Key tensor [seq, d].
        V: Value tensor [seq, d].
        num_devices: Number of virtual devices in the ring.

    Returns:
        Attention output [seq, d], identical to standard attention.
    """
    seq_len, d = Q.shape
    scale = math.sqrt(d)
    chunk_size = math.ceil(seq_len / num_devices)

    # Pad to make evenly divisible
    pad_len = chunk_size * num_devices - seq_len
    if pad_len > 0:
        Q = mx.concatenate([Q, mx.zeros((pad_len, d))], axis=0)
        K = mx.concatenate([K, mx.zeros((pad_len, d))], axis=0)
        V = mx.concatenate([V, mx.zeros((pad_len, d))], axis=0)

    padded_len = chunk_size * num_devices

    # Split into chunks per device
    Q_chunks = [Q[i * chunk_size:(i + 1) * chunk_size, :] for i in range(num_devices)]
    K_chunks = [K[i * chunk_size:(i + 1) * chunk_size, :] for i in range(num_devices)]
    V_chunks = [V[i * chunk_size:(i + 1) * chunk_size, :] for i in range(num_devices)]

    # Each device maintains its own output accumulator and softmax stats
    O_local = [mx.zeros((chunk_size, d)) for _ in range(num_devices)]
    m_local = [mx.full((chunk_size, 1), -1e30) for _ in range(num_devices)]
    l_local = [mx.zeros((chunk_size, 1)) for _ in range(num_devices)]

    # Ring communication: rotate K/V chunks num_devices times
    for step in range(num_devices):
        for dev in range(num_devices):
            # Which K/V chunk does this device see at this step?
            kv_idx = (dev + step) % num_devices
            K_remote = K_chunks[kv_idx]
            V_remote = V_chunks[kv_idx]

            # Compute attention scores for this tile
            S = (Q_chunks[dev] @ K_remote.T) / scale  # [chunk_size, chunk_size]

            # Online softmax update (same as flash attention inner loop)
            row_max = mx.max(S, axis=-1, keepdims=True)
            m_new = mx.maximum(m_local[dev], row_max)

            rescale = mx.exp(m_local[dev] - m_new)
            O_local[dev] = O_local[dev] * rescale
            l_local[dev] = l_local[dev] * rescale

            P = mx.exp(S - m_new)
            O_local[dev] = O_local[dev] + P @ V_remote
            l_local[dev] = l_local[dev] + mx.sum(P, axis=-1, keepdims=True)

            m_local[dev] = m_new

    # Normalize each device's output
    for dev in range(num_devices):
        O_local[dev] = O_local[dev] / l_local[dev]

    # Concatenate all device outputs
    O = mx.concatenate(O_local, axis=0)
    return O[:seq_len, :]


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_attention(seq_lengths: Optional[List[int]] = None,
                        d: int = 64, block_size: int = 32,
                        n_warmup: int = 3, n_runs: int = 5) -> List[Dict[str, Any]]:
    """Benchmark standard vs flash attention at various sequence lengths.

    ⚡ Measures wall-clock time for both implementations. Flash attention
    avoids materializing the O(n²) matrix, which matters at long sequences.

    Args:
        seq_lengths: List of sequence lengths to benchmark.
        d: Head dimension.
        block_size: Flash attention block size.
        n_warmup: Warmup iterations.
        n_runs: Timed iterations.

    Returns:
        List of dicts with timing results per sequence length.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    results = []
    for seq_len in seq_lengths:
        mx.random.seed(42)
        Q = mx.random.normal((seq_len, d))
        K = mx.random.normal((seq_len, d))
        V = mx.random.normal((seq_len, d))
        mx.eval(Q, K, V)

        # Warmup standard
        for _ in range(n_warmup):
            mx.eval(standard_attention(Q, K, V))

        # Time standard
        t0 = time.perf_counter()
        for _ in range(n_runs):
            mx.eval(standard_attention(Q, K, V))
        t_standard = (time.perf_counter() - t0) / n_runs

        # Warmup flash
        for _ in range(n_warmup):
            mx.eval(_tiled_attention_impl(Q, K, V, block_size, seq_len, d))

        # Time flash
        t0 = time.perf_counter()
        for _ in range(n_runs):
            mx.eval(_tiled_attention_impl(Q, K, V, block_size, seq_len, d))
        t_flash = (time.perf_counter() - t0) / n_runs

        mem = flash_memory_analysis(seq_len, d, block_size)

        results.append({
            "seq_len": seq_len,
            "standard_ms": t_standard * 1000,
            "flash_ms": t_flash * 1000,
            "speedup": t_standard / t_flash if t_flash > 0 else float("inf"),
            "memory_ratio": mem["memory_ratio"],
        })

    return results


# ---------------------------------------------------------------------------
# Convenience wrappers (matching spec interface names)
# ---------------------------------------------------------------------------

def benchmark_standard_vs_flash(
    seq_lengths: Optional[List[int]] = None,
    d_model: int = 64,
    block_size: int = 64,
    n_warmup: int = 3,
    n_runs: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark standard vs flash attention at various sequence lengths.

    ⚡ Convenience wrapper around benchmark_attention matching the spec interface.

    Args:
        seq_lengths: Sequence lengths to test (default: [512, 1024, 2048, 4096]).
        d_model: Head dimension.
        block_size: Flash attention block size.
        n_warmup: Warmup iterations.
        n_runs: Timed iterations.

    Returns:
        List of dicts with timing and memory results.
    """
    return benchmark_attention(
        seq_lengths=seq_lengths, d=d_model, block_size=block_size,
        n_warmup=n_warmup, n_runs=n_runs,
    )


def plot_benchmark(results: List[Dict[str, Any]]) -> None:
    """Plot benchmark results: latency comparison and memory savings.

    Args:
        results: Output from benchmark_standard_vs_flash or benchmark_attention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slens = [r["seq_len"] for r in results]
    std_times = [r["standard_ms"] for r in results]
    flash_times = [r["flash_ms"] for r in results]
    mem_ratios = [r["memory_ratio"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(slens, std_times, "o-", label="Standard", color="#e74c3c", linewidth=2)
    ax1.plot(slens, flash_times, "s-", label="Flash", color="#2ecc71", linewidth=2)
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("⚡ Attention Latency")
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(slens)), mem_ratios, color="#3498db", alpha=0.8)
    ax2.set_xticks(range(len(slens)))
    ax2.set_xticklabels([str(s) for s in slens])
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Memory Savings (x)")
    ax2.set_title("💡 Flash Attention Memory Savings")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("attention_benchmark.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\n⚡ Flash attention's advantage grows with sequence length!")
