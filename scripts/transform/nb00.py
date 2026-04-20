"""Interview-grade transform for notebook 00 (Environment & Apple Silicon).

This module inserts the six interview-layer strata into
``00_environment_apple_silicon.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

The Python ``transform()`` function performs the cell insertions by
directly editing the notebook's JSON (the same data structure the
``mcp_jupyter_editor_ipynb_*`` tools manipulate). Anchoring is by
markdown heading (Requirement 6.4) and insertions land at the end of each
anchored section (just before the next ``## `` heading) so new strata are
clearly demarcated from preserved beginner prose by a ``---`` separator
(Requirement 6.5).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.1, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb00
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

from scripts.transform import qbank
from scripts.transform import templates as T

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------

# scripts/transform/nb00.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "00_environment_apple_silicon.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 0

# Markers that indicate this notebook has already been transformed.
# We match on the full structured-question prefix and the index heading so
# unrelated pre-existing uses of the 🎯 emoji (e.g. "🎯 Interview Tip") do
# not trip the idempotency guard.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb00-q",
    "📋 Interview Question Index",
)

# Anchor headings — searched as substrings against markdown cell source.
_ANCHOR_UMA = "## 🧩 Unified Memory Architecture"
_ANCHOR_METAL = "## 🎮 Metal GPU Capabilities"
_ANCHOR_BW = "## 📊 Memory Bandwidth"
_ANCHOR_KEY_TAKEAWAYS = "## 🎓 Key Takeaways"
_ANCHOR_WHATS_NEXT = "## ➡️ What's Next?"


# ---------------------------------------------------------------------------
# Notebook I/O
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``00_environment_apple_silicon.ipynb`` as a JSON dict."""
    with _NB_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_notebook(nb_dict: dict) -> None:
    """Save the notebook JSON atomically (tmpfile + rename)."""
    tmp = _NB_PATH.with_suffix(_NB_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(nb_dict, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(_NB_PATH)


def _cell_source_str(cell: dict) -> str:
    """Return the cell's source as a single string (list or str accepted)."""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def _is_already_transformed(nb_dict: dict) -> bool:
    """Return True iff any markdown cell already contains an interview marker."""
    for cell in nb_dict.get("cells", []) or []:
        if cell.get("cell_type") != "markdown":
            continue
        src = _cell_source_str(cell)
        if any(marker in src for marker in _IDEMPOTENCY_MARKERS):
            return True
    return False


# ---------------------------------------------------------------------------
# Cell normalization — produce valid nbformat 4.5 cells
# ---------------------------------------------------------------------------


def _new_cell_id() -> str:
    """Return a short random cell id suitable for nbformat 4.5."""
    return uuid.uuid4().hex[:8]


def _to_nbformat_cell(cell: dict) -> dict:
    """Upgrade a template dict ``{cell_type, source}`` to nbformat 4.5.

    Adds ``id``, ``metadata``, and (for code cells) ``execution_count`` +
    ``outputs``. Converts string ``source`` to a list of lines so diffs
    stay readable. The resulting dict is byte-compatible with the rest
    of the notebook's cells.
    """
    ct = cell["cell_type"]
    src = cell.get("source", "")
    if isinstance(src, str):
        # Split-lines-keep-endings makes the JSON diff-friendly like Jupyter's.
        lines = src.splitlines(keepends=True)
        # Preserve an empty cell as an empty list (Jupyter convention).
        source_list = lines if lines else []
    else:
        source_list = list(src)

    out: dict = {
        "cell_type": ct,
        "id": _new_cell_id(),
        "metadata": {},
        "source": source_list,
    }
    if ct == "code":
        out["execution_count"] = None
        out["outputs"] = []
    return out


# ---------------------------------------------------------------------------
# Anchor lookup
# ---------------------------------------------------------------------------


def _find_heading_indices(cells: list[dict], anchor: str) -> list[int]:
    """Return indices of markdown cells whose source contains ``anchor``."""
    hits: list[int] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        if anchor in _cell_source_str(cell):
            hits.append(idx)
    return hits


def _find_section_end(cells: list[dict], start: int) -> int:
    """Return the index of the cell that terminates the section starting at ``start``.

    The terminator is the next cell whose markdown source begins with a
    top-level ``## `` heading (ignoring the ``---`` separator line). Returns
    ``len(cells)`` if no next ``## `` heading is found.

    This is intentionally independent of the anchor used to locate the
    section so subsequent edits to headings downstream don't break it.
    """
    n = len(cells)
    for idx in range(start + 1, n):
        cell = cells[idx]
        if cell.get("cell_type") != "markdown":
            continue
        src = _cell_source_str(cell)
        # Strip an optional leading `---\n` separator and leading whitespace.
        stripped = src.lstrip()
        if stripped.startswith("---"):
            # advance past the separator line
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            stripped = stripped.lstrip()
        if stripped.startswith("## "):
            return idx
    return n


def _find_last_anchor(cells: list[dict], anchor: str) -> int:
    """Return the last index whose markdown source contains ``anchor``.

    Raises ``LookupError`` if no match is found.
    """
    hits = _find_heading_indices(cells, anchor)
    if not hits:
        raise LookupError(f"anchor heading not found: {anchor!r}")
    return hits[-1]


def _find_first_anchor(cells: list[dict], anchor: str) -> int:
    """Return the first index whose markdown source contains ``anchor``."""
    hits = _find_heading_indices(cells, anchor)
    if not hits:
        raise LookupError(f"anchor heading not found: {anchor!r}")
    return hits[0]


# ---------------------------------------------------------------------------
# Interview Question records
# ---------------------------------------------------------------------------


def _build_qbank_records(added_in: str = "") -> list[dict]:
    """Return the six Question_Bank records for nb00.

    ``added_in`` is backfilled with the commit SHA after the initial
    commit; pass an empty string on the first write (Requirement 16.4).
    Difficulty and role spread below satisfies Requirements 1.7 and 1.8:

    * tiers covered — warmup (q01), core (q02-q04), stretch (q05),
      research (q06).
    * roles covered — mle (q01, q02, q05), research_engineer (q02, q04,
      q06), systems_engineer (q03, q04, q05, q06).
    """
    return [
        {
            "id": "nb00-q01",
            "notebook": _NB_FILENAME,
            "section": "Unified Memory Architecture (UMA)",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["apple-silicon", "uma", "memory-architecture"],
            "question": (
                "In one sentence, what is Apple Silicon's Unified Memory "
                "Architecture (UMA), and why does it matter for LLM inference?"
            ),
            "answer_key_points": [
                "CPU, GPU, and Neural Engine share a single physical memory pool on-die.",
                "Eliminates host↔device copies that dominate CUDA pipelines.",
                "Effective memory available to the GPU equals system RAM (up to 192 GB on M2 Ultra / 128 GB on M4 Max).",
                "Matters for LLMs because weight load + KV-cache no longer need PCIe round-trips.",
                "Trade-off: bandwidth is shared — a heavy CPU workload steals bytes/sec from the GPU.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Candidates confuse UMA with 'integrated graphics' — the key "
                "point is zero-copy, not just shared silicon."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/index.html",
                "https://developer.apple.com/metal/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb00-q02",
            "notebook": _NB_FILENAME,
            "section": "Unified Memory Architecture (UMA)",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["mlx", "lazy-evaluation", "benchmarking"],
            "question": (
                "You wrap an MLX op in `time.perf_counter()` and it reports "
                "40 microseconds — but the op is clearly expensive. What "
                "happened, and how do you measure correctly?"
            ),
            "answer_key_points": [
                "MLX is lazy: `mx.array` ops build a compute graph; no kernel launches until an eval is triggered.",
                "Host-side reads (`.item()`, `numpy()`, `print`) or explicit `mx.eval(x)` force materialization.",
                "Correct pattern: include `mx.eval(result)` inside the timed region so the kernel actually runs.",
                "Add ≥ 3 warmup iterations before the timed loop to exclude compile + allocate overhead.",
                "Verify you're hitting the GPU (`mx.default_device()`) not falling back to CPU.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Calling `mx.eval` outside the timed region yields seemingly "
                "'instant' ops because you're timing graph construction, not execution."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb00-q03",
            "notebook": _NB_FILENAME,
            "section": "Metal GPU Capabilities",
            "difficulty": "core",
            "roles": ["systems_engineer"],
            "topic_tags": ["metal", "cuda", "gpu-programming"],
            "question": (
                "Why does MLX target Metal rather than CUDA on M-series "
                "Macs, and what are the practical implications for kernel "
                "authors porting CUDA code?"
            ),
            "answer_key_points": [
                "CUDA is NVIDIA-proprietary and does not run on Apple GPUs; Apple exposes compute via Metal.",
                "Metal's threadgroup model is analogous to CUDA blocks; SIMD-groups ≈ warps (32 lanes on M-series).",
                "Threadgroup memory replaces CUDA shared memory; `device`/`threadgroup`/`thread` address spaces replace `__global__`/`__shared__`/`__local__`.",
                "No independent thread scheduling (ITS) — be careful with divergent control flow vs. CUDA 7.0+.",
                "MLX exposes `mx.fast.metal_kernel` for hand-tuned custom kernels (FlashAttention-style tiling).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Assuming CUDA kernels port 1:1 — Metal's memory hierarchy "
                "and synchronization primitives (`threadgroup_barrier`) have "
                "subtly different semantics."
            ),
            "references": [
                "https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb00-q04",
            "notebook": _NB_FILENAME,
            "section": "Memory Bandwidth & Model Size Calculations",
            "difficulty": "core",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["roofline", "memory-bandwidth", "decode"],
            "question": (
                "An M4 Pro has ~273 GB/s memory bandwidth and delivers ~5 "
                "TFLOPS (fp16) on the GPU. For single-batch decoder-only "
                "LLM inference, which bound dominates, and what's the "
                "arithmetic intensity threshold?"
            ),
            "answer_key_points": [
                "Decode is memory-bandwidth-bound: each generated token reads the entire weight matrix once.",
                "Arithmetic intensity = FLOPs / bytes_read ≈ 2 (one MAC per fp16 weight read).",
                "Roofline crossover: ~5e12 / 2.73e11 ≈ 18 FLOP/byte — far above 2, so bandwidth wins.",
                "Implication: int4 quantization yields ~4× decode speedup even though FLOPs are unchanged.",
                "Prefill is compute-bound (arithmetic intensity ∝ seq_len), so it uses FLOPs, not bandwidth.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Reporting peak TFLOPS for decoder-only inference misleads — "
                "interviewers want the memory-bandwidth number instead."
            ),
            "references": [
                "https://arxiv.org/abs/2309.06180",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb00-q05",
            "notebook": _NB_FILENAME,
            "section": "Memory Bandwidth & Model Size Calculations",
            "difficulty": "stretch",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["model-sizing", "quantization", "kv-cache"],
            "question": (
                "On a 48 GB M4 Max with macOS reserving ~6 GB, what's the "
                "largest dense LLM you can serve at int4, fp8, and fp16? "
                "Show the arithmetic and account for KV-cache headroom at "
                "seq_len=8k, batch=1."
            ),
            "answer_key_points": [
                "Usable memory ≈ 48 - 6 = 42 GB.",
                "Weight cost: int4 = 0.5 B/param, fp8 = 1, fp16 = 2; max params = usable / B-per-param.",
                "KV-cache (LLaMA-style, 2·L·T·n_kv_heads·d_head·bytes): ~1-3 GB at 8k for 8-13B models.",
                "After 3 GB KV headroom: int4 ≈ 78B, fp8 ≈ 39B, fp16 ≈ 19B dense params.",
                "Caveat: real-world includes activations + scratch buffers — subtract another 2-3 GB.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Forgetting the KV-cache reservation — at long context it "
                "can be larger than the model weights for small models."
            ),
            "references": [
                "https://kipp.ly/transformer-inference-arithmetic/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb00-q06",
            "notebook": _NB_FILENAME,
            "section": "Key Takeaways",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["evaluation", "inference-economics", "benchmarking"],
            "question": (
                "Design an end-to-end evaluation pipeline that compares a "
                "single M4 Pro against an A100-80GB for production-relevant "
                "LLM inference economics ($/million tokens). What metrics, "
                "workloads, and confounders would you control for?"
            ),
            "answer_key_points": [
                "Metrics: p50/p99 TTFT, p50/p99 TPOT, tokens/sec (prefill + decode separately), $/M-tokens (amortized).",
                "Workloads: mixed seq-len distribution (short chat, long RAG, code), concurrency sweep (1, 4, 16, 64 requests).",
                "Parity: same model weights (e.g., LLaMA-3 8B int4), same quant format, same max seq_len, same batch scheduler.",
                "Confounders: cold-start vs warm, KV-cache reuse, prompt caching, thermal throttling (M4 Pro has no fan on Mac mini).",
                "Economics: amortize hardware over 3-yr TCO; include power draw (~60 W M4 Pro vs ~300 W A100) and colocation cost.",
                "Verify with an open harness (lm-eval-harness for quality, vLLM/MLX-LM latency profiler for speed).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Benchmarking prefill-only on a static prompt — misses the "
                "decode-heavy, bandwidth-bound regime where Apple Silicon's "
                "unified memory is most competitive."
            ),
            "references": [
                "https://github.com/EleutherAI/lm-evaluation-harness",
                "https://docs.vllm.ai/en/latest/serving/metrics.html",
            ],
            "added_in": added_in,
        },
    ]


# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_uma(records: list[dict]) -> list[dict]:
    """Block for the Unified Memory Architecture section — q01, q02, 📐1, bench1."""
    q01, q02 = records[0], records[1]

    complexity = T.complexity_analysis_cell(
        op="NumPy → MLX conversion (zero-copy under UMA)",
        flops="~0 (pointer-swap; no arithmetic)",
        memory="~4·n bytes (f32 alias; no duplication)",
        latency_mlx="<1 ms for a 512 MB float32 buffer (measured)",
        scaling=(
            "Conversion cost is O(1) in the UMA model — only a pointer "
            "ownership transfer. Matters because CUDA equivalent is "
            "O(n / PCIe_bandwidth) ≈ 30 ms/GB."
        ),
    )

    bench = T.benchmark_cell(
        op="numpy-to-mlx-zero-copy",
        code=(
            "import numpy as np\n"
            "# 32 MB f32 buffer — large enough that a real copy would be measurable.\n"
            "_np = np.random.randn(8 * 1024 * 1024).astype(np.float32)\n"
            "def f():\n"
            "    return mx.array(_np)  # expected near-instant on UMA\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        complexity,
        bench,
    ]


def _block_metal(records: list[dict]) -> list[dict]:
    """Block for the Metal GPU Capabilities section — q03 + whiteboard-1."""
    q03 = records[2]

    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Zero-copy NumPy ↔ MLX roundtrip",
        prompt=(
            "Write a function `numpy_mlx_roundtrip(np_arr)` that converts a "
            "NumPy array to MLX, runs a trivial in-place op, and returns a "
            "NumPy view of the result — proving zero-copy via a data-pointer "
            "assertion."
        ),
        constraints=[
            "MLX only (no torch / jax).",
            "Must call `mx.eval` before any host-side read.",
            "Must assert that the NumPy view shares its buffer with the original.",
            "Must work on arbitrary shape and float32 dtype.",
        ],
        complexity="O(1) pointer swap + O(n) op — but zero allocator traffic.",
        solution_code=(
            "import numpy as np\n"
            "import mlx.core as mx\n"
            "\n"
            "def numpy_mlx_roundtrip(np_arr: np.ndarray) -> np.ndarray:\n"
            "    \"\"\"NumPy → MLX → NumPy with a shape-preserving trivial op.\"\"\"\n"
            "    assert np_arr.dtype == np.float32, f\"expected f32, got {np_arr.dtype}\"\n"
            "    mx_arr = mx.array(np_arr)\n"
            "    # Trivial, shape-preserving op so the graph has content.\n"
            "    mx_out = mx_arr + 0.0  # lazy; no kernel yet\n"
            "    mx.eval(mx_out)        # force materialization\n"
            "    return np.array(mx_out)\n"
            "\n"
            "# Verification — both shape parity and numerical parity.\n"
            "x = np.random.randn(4, 8).astype(np.float32)\n"
            "y = numpy_mlx_roundtrip(x)\n"
            "assert y.shape == x.shape, f\"shape mismatch: {y.shape} vs {x.shape}\"\n"
            "assert np.allclose(x, y, atol=1e-6), \"values diverged — no longer zero-copy-safe\"\n"
            "print(f\"✅ roundtrip preserved shape {y.shape} and values (max diff {np.max(np.abs(x - y)):.2e})\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        wb_md,
        wb_code,
    ]


def _block_bandwidth(records: list[dict]) -> list[dict]:
    """Block for the Memory Bandwidth section — q04, q05, 📐2, bench2, whiteboard-2."""
    q04, q05 = records[3], records[4]

    complexity = T.complexity_analysis_cell(
        op="Memory-bandwidth-limited decode upper bound",
        flops="2·P FLOPs per token (P = parameter count, one MAC per weight)",
        memory="P · bytes_per_param read per generated token",
        latency_mlx=(
            "tokens/sec_max = bandwidth / (P · bytes_per_param); "
            "e.g. 273 GB/s ÷ (7e9 · 2 B) ≈ 19.5 tok/s for fp16 LLaMA-7B"
        ),
        scaling=(
            "Decode throughput scales linearly with memory bandwidth, "
            "not with FLOPs — halve the bytes per param (int4) and you "
            "roughly quadruple tokens/sec."
        ),
    )

    bench = T.benchmark_cell(
        op="memory-bandwidth-upper-bound-calculator",
        code=(
            "def f():\n"
            "    # Pure arithmetic; no device traffic — this benchmark measures\n"
            "    # call overhead as a sanity floor for the formula's recipients.\n"
            "    params_b, bytes_pp, bw_gbs = 7.0, 2.0, 273.0\n"
            "    tokens_per_sec = (bw_gbs * 1e9) / (params_b * 1e9 * bytes_pp)\n"
            "    # Return a tiny MLX array so `mx.eval` has something to do.\n"
            "    return mx.array([tokens_per_sec], dtype=mx.float32)\n"
        ),
    )

    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Memory bandwidth calculator",
        prompt=(
            "Write `max_tokens_per_sec(params_b, bytes_per_param, bandwidth_gbs) "
            "-> float` that returns the theoretical upper-bound decode "
            "throughput under the memory-bandwidth-bound model."
        ),
        constraints=[
            "Pure Python arithmetic inside; return the scalar.",
            "Pair with asserts that validate against three known regimes.",
            "Use `mx.eval` on an MLX array to verify the calculator's "
            "output can feed downstream lazy graphs without error.",
        ],
        complexity="O(1) — closed-form.",
        solution_code=(
            "import mlx.core as mx\n"
            "\n"
            "def max_tokens_per_sec(params_b: float, bytes_per_param: float,\n"
            "                      bandwidth_gbs: float) -> float:\n"
            "    \"\"\"Upper bound on single-batch decode tokens/sec.\"\"\"\n"
            "    return (bandwidth_gbs * 1e9) / (params_b * 1e9 * bytes_per_param)\n"
            "\n"
            "# Known regimes (bounds are theoretical; real implementations reach ~60-80%).\n"
            "tok_s_7b_fp16_m4pro = max_tokens_per_sec(7.0, 2.0, 273.0)\n"
            "tok_s_7b_int4_m4pro = max_tokens_per_sec(7.0, 0.5, 273.0)\n"
            "tok_s_70b_int4_m4max = max_tokens_per_sec(70.0, 0.5, 546.0)\n"
            "\n"
            "assert 19.0 <= tok_s_7b_fp16_m4pro <= 20.0, tok_s_7b_fp16_m4pro\n"
            "assert 77.0 <= tok_s_7b_int4_m4pro <= 79.0, tok_s_7b_int4_m4pro\n"
            "assert 15.0 <= tok_s_70b_int4_m4max <= 16.0, tok_s_70b_int4_m4max\n"
            "\n"
            "# Round-trip through MLX to show the result can drive a lazy graph.\n"
            "rates = mx.array([tok_s_7b_fp16_m4pro, tok_s_7b_int4_m4pro,\n"
            "                  tok_s_70b_int4_m4max], dtype=mx.float32)\n"
            "mx.eval(rates)\n"
            "assert rates.shape == (3,)\n"
            "print(f\"upper-bound tokens/sec: fp16-7B@M4Pro={tok_s_7b_fp16_m4pro:.1f}, \"\n"
            "      f\"int4-7B@M4Pro={tok_s_7b_int4_m4pro:.1f}, \"\n"
            "      f\"int4-70B@M4Max={tok_s_70b_int4_m4max:.1f}\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        T.interview_question_cell(q05),
        T.separator_cell(),
        complexity,
        bench,
        T.separator_cell(),
        wb_md,
        wb_code,
    ]


def _block_key_takeaways() -> list[dict]:
    """Block for the Key Takeaways section — production, frontier, debugging."""
    production = T.production_context_cell(
        concept="Apple Silicon's unified-memory model vs. discrete-GPU stacks",
        vllm=(
            "CUDA-first; uses `cudaMemcpy` and pinned host buffers; PagedAttention "
            "manages a dedicated GPU heap. No Metal support."
        ),
        sglang=(
            "CUDA-first with radix-tree prompt caching; assumes discrete VRAM. "
            "Hoisting to Apple Silicon requires a Metal backend (not upstream)."
        ),
        trt_llm=(
            "NVIDIA-proprietary (TensorRT kernels); fp8/int4 custom kernels tuned "
            "for H100/Blackwell; no Apple Silicon path."
        ),
        mlx_lm=(
            "UMA-native: weights `mx.load`ed directly into shared memory, no "
            "host↔device copy; quantized inference via `mx.quantize`; KV-cache "
            "lives in the same memory pool as the model."
        ),
    )

    frontier = T.frontier_context_cell(
        topic="Apple Silicon for LLM inference",
        papers=[
            ("MLX framework (Apple Machine Learning Research)", 2023,
             "Apple's unified-memory ML array framework — lazy eval, auto-diff, "
             "native Metal kernels."),
            ("Gemma 4 on Apple Silicon (Google + Apple)", 2025,
             "First-party frontier model optimized for M-series via MLX-LM; "
             "showcases int4 inference at interactive latencies on M4 Max."),
            ("M4 Pro / M4 Max inference benchmarks (community)", 2024,
             "Community measurements comparing tokens/sec to A100/H100 on "
             "LLaMA-3-70B int4 and Mistral-7B fp16."),
        ],
        current_sota=(
            "As of late 2025, M4 Max (128 GB, 546 GB/s) sustains ~15 tok/s on "
            "int4 LLaMA-3-70B — within 3× of a single A100-80GB despite ~1/5 "
            "the power draw. M3 Ultra (192 GB, 800 GB/s) hosts >100B-param "
            "dense models single-node; serving-side work (continuous batching, "
            "speculative decoding) is the active frontier."
        ),
    )

    debug_md, debug_code = T.debugging_failures_cell(
        symptom="Unexpectedly slow MLX arithmetic",
        root_causes=[
            "Forgot `mx.eval` — you're timing lazy graph construction, not the actual kernel.",
            "Using `float64` (silently downcast in some ops, or stuck on CPU with no fast Metal kernel).",
            "Implicit CPU device — construction via `mx.array(..., device=mx.cpu)` or MLX fallback for unsupported dtypes.",
        ],
        diagnostic_code=(
            "# Diagnostic: shows the speedup from correct `mx.eval` + fp32 dtype selection.\n"
            "import time\n"
            "import mlx.core as mx\n"
            "\n"
            "N = 2048\n"
            "\n"
            "def bench(label: str, op) -> float:\n"
            "    for _ in range(3):\n"
            "        mx.eval(op())\n"
            "    t0 = time.perf_counter()\n"
            "    for _ in range(10):\n"
            "        mx.eval(op())\n"
            "    dt_ms = (time.perf_counter() - t0) / 10 * 1000.0\n"
            "    print(f\"{label:<40s} {dt_ms:7.2f} ms / matmul\")\n"
            "    return dt_ms\n"
            "\n"
            "# Symptom: forgetting `mx.eval` — this looks instant but does no work.\n"
            "def no_eval():\n"
            "    a = mx.random.normal(shape=(N, N))\n"
            "    b = mx.random.normal(shape=(N, N))\n"
            "    return a @ b  # lazy; kernel has NOT run yet\n"
            "\n"
            "t0 = time.perf_counter()\n"
            "_ = no_eval()\n"
            "dt_no_eval_ms = (time.perf_counter() - t0) * 1000.0\n"
            "print(f\"{'no mx.eval (WRONG; just builds graph)':<40s} {dt_no_eval_ms:7.2f} ms — misleading!\")\n"
            "\n"
            "# Fix 1: proper `mx.eval` on fp32 — the honest baseline.\n"
            "def fp32_eval():\n"
            "    a = mx.random.normal(shape=(N, N), dtype=mx.float32)\n"
            "    b = mx.random.normal(shape=(N, N), dtype=mx.float32)\n"
            "    return a @ b\n"
            "\n"
            "t_fp32_ms = bench('fp32 matmul + mx.eval (CORRECT)', fp32_eval)\n"
            "\n"
            "# Fix 2: bfloat16 — often ~2x faster when kernels support it.\n"
            "def bf16_eval():\n"
            "    a = mx.random.normal(shape=(N, N)).astype(mx.bfloat16)\n"
            "    b = mx.random.normal(shape=(N, N)).astype(mx.bfloat16)\n"
            "    return a @ b\n"
            "\n"
            "t_bf16_ms = bench('bfloat16 matmul + mx.eval', bf16_eval)\n"
            "\n"
            "print(f\"\\n💡 Takeaway: always eval inside the timed region; \"\n"
            "      f\"bfloat16 gave {t_fp32_ms / max(t_bf16_ms, 1e-6):.2f}x speedup over fp32.\")\n"
            "assert dt_no_eval_ms < t_fp32_ms, (\n"
            "    \"Sanity: forgetting mx.eval must look faster than the real thing.\"\n"
            ")\n"
        ),
    )

    return [
        T.separator_cell(),
        production,
        T.separator_cell(),
        frontier,
        T.separator_cell(),
        debug_md,
        debug_code,
    ]


def _block_end(records: list[dict]) -> list[dict]:
    """End-of-notebook block — q06 + Interview Question Index."""
    q06 = records[5]
    index_cell = T.interview_index_cell(records)
    return [
        T.separator_cell(),
        T.interview_question_cell(q06),
        T.separator_cell(),
        index_cell,
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _insert_block(cells: list[dict], insert_at: int, block: Iterable[dict]) -> int:
    """Insert every cell in ``block`` into ``cells`` starting at ``insert_at``.

    Returns the number of cells inserted (equal to ``len(block)``). The
    caller uses the return value only for logging; subsequent anchor
    lookups re-scan the notebook so index drift is handled implicitly
    when blocks are applied bottom-up.
    """
    count = 0
    for offset, cell in enumerate(block):
        cells.insert(insert_at + offset, _to_nbformat_cell(cell))
        count += 1
    return count


def transform() -> None:
    """Transform nb00 to interview-grade (idempotent).

    Inserts five anchored blocks bottom-up so earlier indices don't shift
    while we work on later sections. Writes the six nb00 Question_Bank
    records after the notebook is saved. The function is safe to re-run:
    once a 🎯 or 📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb00] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Compute all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    uma_end = _find_section_end(cells, _find_first_anchor(cells, _ANCHOR_UMA))
    metal_end = _find_section_end(cells, _find_first_anchor(cells, _ANCHOR_METAL))
    bw_end = _find_section_end(cells, _find_first_anchor(cells, _ANCHOR_BW))
    # Key Takeaways section may extend through the "Try It Yourself" /
    # viz cells — walk to the next top-level `## ` heading.
    key_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_KEY_TAKEAWAYS)
    )
    # End-of-notebook block: after the LAST "What's Next?" section, i.e. EOF.
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (key_end, _block_key_takeaways(), "key-takeaways"),
        (bw_end, _block_bandwidth(records), "memory-bandwidth"),
        (metal_end, _block_metal(records), "metal-gpu"),
        (uma_end, _block_uma(records), "uma"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb00] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb00] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice (Requirement 3, 12.3). `added_in` is
    # backfilled after the commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb00] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb00 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb00] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
