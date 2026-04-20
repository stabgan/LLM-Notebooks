"""Interview-grade transform for notebook 01 (MLX Fundamentals).

This module inserts the six interview-layer strata into
``01_mlx_fundamentals.ipynb`` and upserts the notebook's slice of
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

    .venv/bin/python -m scripts.transform.nb01
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

# scripts/transform/nb01.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "01_mlx_fundamentals.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 1

# Markers that indicate this notebook has already been transformed.
# We match on the full structured-question prefix and the index heading so
# unrelated pre-existing uses of the 🎯 emoji (e.g. "🎯 Key Takeaways") do
# not trip the idempotency guard.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb01-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
_ANCHOR_MLX_VS_PYTORCH = "## 🆚 MLX vs PyTorch"
_ANCHOR_ARRAY_CREATION = "## 📦 MLX Array Creation"
_ANCHOR_LAZY_EVAL = "## ⏳ Lazy Evaluation"
_ANCHOR_AUTODIFF = "## 🧮 Automatic Differentiation"
_ANCHOR_KEY_TAKEAWAYS = "## 🎯 Key Takeaways"



# ---------------------------------------------------------------------------
# Notebook I/O
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``01_mlx_fundamentals.ipynb`` as a JSON dict."""
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
        lines = src.splitlines(keepends=True)
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
        stripped = src.lstrip()
        if stripped.startswith("---"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            stripped = stripped.lstrip()
        if stripped.startswith("## "):
            return idx
    return n


def _find_first_anchor(cells: list[dict], anchor: str) -> int:
    """Return the first index whose markdown source contains ``anchor``.

    Raises ``LookupError`` if no match is found.
    """
    hits = _find_heading_indices(cells, anchor)
    if not hits:
        raise LookupError(f"anchor heading not found: {anchor!r}")
    return hits[0]



# ---------------------------------------------------------------------------
# Interview Question records
# ---------------------------------------------------------------------------


def _build_qbank_records(added_in: str = "") -> list[dict]:
    """Return the six Question_Bank records for nb01.

    ``added_in`` is backfilled with the commit SHA after the initial
    commit; pass an empty string on the first write (Requirement 16.4).

    Difficulty and role spread below satisfies Requirements 1.7 and 1.8:

    * tiers covered — warmup (q01), core (q02, q03, q04), stretch (q05),
      research (q06).
    * roles covered — mle (q01, q02, q04), research_engineer (q02, q05,
      q06), systems_engineer (q03, q04, q05, q06).

    Required topic coverage (per task brief):
        q01 — MLX lazy-eval semantics            (warmup, mle)
        q02 — mx.compile graph-capture rules     (core, mle+RE)
        q03 — shape-broadcasting semantics       (core, systems)
        q04 — mx.eval timing traps               (core, mle+systems)
        q05 — bfloat16 numerics                  (stretch, RE+systems)
        q06 — unified-memory zero-copy (internals) (research, RE+systems)
    """
    return [
        {
            "id": "nb01-q01",
            "notebook": _NB_FILENAME,
            "section": "MLX vs PyTorch: Lazy Evaluation",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["mlx", "lazy-evaluation", "semantics"],
            "question": (
                "Explain MLX's lazy evaluation model in one paragraph. "
                "What happens when you write `c = a + b` on two MLX arrays, "
                "and when does the addition actually run on the GPU?"
            ),
            "answer_key_points": [
                "`a + b` builds a node in a directed-acyclic compute graph; no kernel is dispatched yet.",
                "The node carries shape + dtype metadata (propagated statically) so downstream ops can be composed without running anything.",
                "Materialization happens on: `mx.eval(c)`, `c.item()`, `np.array(c)`, `print(c)`, or when another `mx.eval` consumes `c`.",
                "This lets the scheduler fuse chains like `sum(exp(softmax(x)))` into a single Metal kernel (Requirement 2.1).",
                "Consequence: timing a lazy op without `mx.eval` measures graph-construction, not execution — the #1 beginner trap.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'lazy means async like CUDA streams' — it's stronger: "
                "NO kernel is launched at all until materialization is forced."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb01-q02",
            "notebook": _NB_FILENAME,
            "section": "Lazy Evaluation & Graph Capture",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["mlx", "mx.compile", "graph-capture", "fusion"],
            "question": (
                "When you wrap a function with `mx.compile`, what exactly is "
                "captured, what Python-side behaviour becomes unsafe, and when "
                "does the graph get re-traced?"
            ),
            "answer_key_points": [
                "Capture happens on first call: MLX traces the function against symbolic placeholders matching the input shapes/dtypes, producing a fused kernel plan.",
                "Only MLX-array ops enter the graph; Python control flow based on `.item()` / host values is baked in at trace time and won't re-run per call.",
                "Python side effects (print, list.append, random.random) fire only during trace, not on subsequent calls — a classic foot-gun.",
                "Re-trace triggers: a new input shape, a new dtype, or a different set of keyword args; MLX caches one compiled version per input signature.",
                "Benefit: kernel fusion + reduced graph-construction overhead on the hot loop (typically 1.5–3× throughput on small ops).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Assuming `mx.compile` behaves like `torch.compile` with "
                "dynamic shape support — MLX recompiles on shape change."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/usage/compile.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb01-q03",
            "notebook": _NB_FILENAME,
            "section": "MLX Array Creation & Shape Semantics",
            "difficulty": "core",
            "roles": ["systems_engineer"],
            "topic_tags": ["broadcasting", "shape", "numpy-semantics"],
            "question": (
                "Walk me through MLX's broadcasting rules and give a concrete "
                "example where two shapes broadcast correctly, and one where "
                "they don't — with the exact error reasoning."
            ),
            "answer_key_points": [
                "Rule: align from the TRAILING axis; a dimension broadcasts if it is equal to the other OR equal to 1.",
                "Missing leading dims are treated as 1 (prepend-1 rule).",
                "Valid example: `(4, 1, 32) * (8, 32)` → result `(4, 8, 32)` (trailing 32==32, middle 1→8, leading 4 prepended).",
                "Invalid example: `(4, 3, 32) * (5, 32)` — middle axis 3 vs 5 are neither equal nor 1, so the op fails with a shape mismatch.",
                "Semantics match NumPy exactly; MLX performs shape inference lazily so errors surface at graph-construction, not at `mx.eval`.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Trying to broadcast `(N, D)` with `(D, N)` — the leading "
                "axis needs transpose, not broadcast; many candidates reach for reshape before diagnosing."
            ),
            "references": [
                "https://numpy.org/doc/stable/user/basics.broadcasting.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb01-q04",
            "notebook": _NB_FILENAME,
            "section": "Lazy Evaluation — Timing Traps",
            "difficulty": "core",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["benchmarking", "mx.eval", "timing"],
            "question": (
                "A candidate reports that an MLX matmul on a (2048, 2048) fp32 "
                "input 'ran in 45 microseconds' using `time.perf_counter`. "
                "Why is this wrong, and what are the three invariants every "
                "correct MLX benchmark must satisfy?"
            ),
            "answer_key_points": [
                "Without an explicit `mx.eval` inside the timed region, you're timing graph construction — the actual kernel has not launched.",
                "Invariant 1: call `mx.eval(result)` inside the timed loop so the host actually waits on the GPU.",
                "Invariant 2: at least 3 warmup iterations before the timed loop — compile, allocate, and JIT-cache populate.",
                "Invariant 3: measure N ≥ 5 timed iterations and report the mean (or median) — the first post-warmup iter can still be an outlier from scheduler-side cold caches.",
                "Optional: `mx.synchronize()` is implicit inside `mx.eval`; don't 'double-sync' by calling both — it just adds host-side overhead.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Calling `mx.eval` OUTSIDE the timed region but INSIDE a wrapping "
                "function — only the first iter actually runs the kernel; later iters "
                "hit the evaluated tensor and look instant."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb01-q05",
            "notebook": _NB_FILENAME,
            "section": "Automatic Differentiation — Numerics",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["bfloat16", "numerics", "mixed-precision"],
            "question": (
                "Compare float16, bfloat16, and float32 for LLM training on "
                "Apple Silicon. What are the bit-layouts, the representable "
                "range, and which failure modes show up in gradient accumulation?"
            ),
            "answer_key_points": [
                "fp32: 1 sign / 8 exp / 23 mantissa; range ~1e-38 .. 3e38; ~7 decimal digits.",
                "fp16: 1 / 5 / 10; range ~6e-5 .. 6.5e4; ~3 decimal digits — narrow range causes underflow on small gradients and overflow on softmax logits.",
                "bf16: 1 / 8 / 7; SAME exponent width as fp32 → same dynamic range; ~2–3 decimal digits of precision.",
                "bf16 is the modern default for LLM training because underflow/overflow vanish, at the cost of ~1 decimal digit of precision.",
                "Failure mode: summing many small updates in bf16 stalls (last-bit noise dominates); keep optimizer state in fp32 (mixed-precision pattern) or use Kahan summation.",
                "On M-series: bf16 kernels are hardware-accelerated on the GPU for matmul; the CPU path may fall back to fp32 for unsupported ops.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Claiming fp16 and bf16 have the same dynamic range — they differ "
                "by ~10^34 in max magnitude because of the 5-bit vs 8-bit exponent."
            ),
            "references": [
                "https://arxiv.org/abs/1905.12322",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb01-q06",
            "notebook": _NB_FILENAME,
            "section": "Key Takeaways — Internals",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["unified-memory", "zero-copy", "mlx-internals"],
            "question": (
                "MLX claims 'zero-copy' NumPy interop on Apple Silicon. Dig "
                "into the internals: when is the guarantee actually exact, "
                "when does MLX silently copy, and how would you verify it "
                "from user code?"
            ),
            "answer_key_points": [
                "Zero-copy is a direct consequence of Unified Memory: both CPU and GPU share the same physical allocator, so a pointer swap suffices.",
                "Exactly zero-copy when: dtype matches a supported MLX dtype (f32/f16/bf16/i32/i8/u8/bool), the NumPy array is contiguous (C-order), and no endianness swap is needed.",
                "MLX *will* copy when: the array is non-contiguous (strided view), dtype needs casting, or the array originates from a non-Metal allocator (e.g., mmap'd file on certain paths).",
                "Verification: compare `array.__array_interface__['data'][0]` before and after `mx.array(x)` and `np.array(mx_x)` — same pointer ⇒ zero-copy.",
                "Reverse direction is stricter: `np.array(mx_arr)` is zero-copy only when MLX's internal buffer is aligned to NumPy's stride expectations; otherwise MLX emits a contiguous copy.",
                "Systems upshot: a 30 GB model-weight load costs ~0 bytes of host↔device traffic on UMA vs ~100 ms via PCIe on a discrete GPU — this is the primary speed-up source for MLX weight loading.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Assuming 'UMA = always zero-copy' — strided / non-contiguous "
                "views ALWAYS force a copy to satisfy MLX's contiguity invariant."
            ),
            "references": [
                "https://ml-explore.github.io/mlx/build/html/index.html",
                "https://numpy.org/doc/stable/reference/arrays.interface.html",
            ],
            "added_in": added_in,
        },
    ]



# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_mlx_vs_pytorch(records: list[dict]) -> list[dict]:
    """Block for the 'MLX vs PyTorch' section — q01 (warmup, lazy-eval).

    Q01 is the warmup-tier lazy-eval question; it slots here because the
    MLX-vs-PyTorch section is the reader's first encounter with the
    eager↔lazy distinction.
    """
    q01 = records[0]
    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
    ]


def _block_array_creation(records: list[dict]) -> list[dict]:
    """Block for the 'MLX Array Creation' section.

    Contents: q03 (shape-broadcasting), whiteboard-a (broadcast-checker),
    📐-1 + benchmark (dtype memory footprint comparison).
    """
    q03 = records[2]

    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Broadcasted shape-check function",
        prompt=(
            "Write `broadcast_shape(a_shape, b_shape) -> tuple[int, ...]` "
            "that returns the NumPy/MLX-compatible broadcast shape of two "
            "input shapes, or raises `ValueError` with a diagnostic message "
            "when they are incompatible. Validate against MLX by actually "
            "broadcasting two arrays of each shape."
        ),
        constraints=[
            "Pure Python — operate on the shape tuples only.",
            "Follow the trailing-axis rule: a dim broadcasts if equal OR 1.",
            "Raise `ValueError` with both shapes in the message when incompatible.",
            "Must call `mx.eval` on the broadcasted MLX arrays for one valid case to prove the shape is materializable.",
            "Must include at least one `assert` that validates your formula against MLX's runtime behaviour.",
        ],
        complexity=(
            "O(max(len(a_shape), len(b_shape))) — one pass over the aligned axes."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "\n"
            "def broadcast_shape(a_shape: tuple[int, ...],\n"
            "                    b_shape: tuple[int, ...]) -> tuple[int, ...]:\n"
            "    \"\"\"Return broadcast shape of ``a_shape`` and ``b_shape`` or raise.\"\"\"\n"
            "    # Align from the trailing axis: reverse, compare, re-reverse.\n"
            "    ra, rb = list(reversed(a_shape)), list(reversed(b_shape))\n"
            "    out: list[int] = []\n"
            "    for i in range(max(len(ra), len(rb))):\n"
            "        da = ra[i] if i < len(ra) else 1\n"
            "        db = rb[i] if i < len(rb) else 1\n"
            "        if da == db:\n"
            "            out.append(da)\n"
            "        elif da == 1:\n"
            "            out.append(db)\n"
            "        elif db == 1:\n"
            "            out.append(da)\n"
            "        else:\n"
            "            raise ValueError(\n"
            "                f\"shapes {a_shape} and {b_shape} are not broadcast-compatible \"\n"
            "                f\"at reversed axis {i}: {da} vs {db}\"\n"
            "            )\n"
            "    return tuple(reversed(out))\n"
            "\n"
            "# Unit cases: verify the formula against NumPy/MLX semantics.\n"
            "assert broadcast_shape((4, 1, 32), (8, 32)) == (4, 8, 32), \\\n"
            "    broadcast_shape((4, 1, 32), (8, 32))\n"
            "assert broadcast_shape((3,), (4, 3)) == (4, 3)\n"
            "assert broadcast_shape((1,), ()) == (1,)\n"
            "assert broadcast_shape((), ()) == ()\n"
            "\n"
            "# Validate against MLX's actual broadcasting runtime.\n"
            "a = mx.ones((4, 1, 32), dtype=mx.float32)\n"
            "b = mx.ones((8, 32), dtype=mx.float32)\n"
            "c = a + b  # lazy\n"
            "mx.eval(c)  # force materialization\n"
            "assert c.shape == broadcast_shape(a.shape, b.shape) == (4, 8, 32), \\\n"
            "    f\"MLX shape {c.shape} disagrees with formula {broadcast_shape(a.shape, b.shape)}\"\n"
            "\n"
            "# Incompatible case must raise with both shapes in the message.\n"
            "try:\n"
            "    broadcast_shape((4, 3, 32), (5, 32))\n"
            "except ValueError as exc:\n"
            "    msg = str(exc)\n"
            "    assert \"(4, 3, 32)\" in msg and \"(5, 32)\" in msg, msg\n"
            "    print(f\"✅ incompatible case correctly raised: {msg}\")\n"
            "else:\n"
            "    raise AssertionError(\"expected ValueError on (4,3,32) vs (5,32)\")\n"
            "\n"
            "print(\"✅ broadcast_shape matches MLX runtime on all 4 compatible cases.\")\n"
        ),
    )

    complexity = T.complexity_analysis_cell(
        op="Dtype memory footprint (fp32 vs fp16 vs bf16)",
        flops="~0 (allocation only; no arithmetic)",
        memory=(
            "shape.numel() * {4, 2, 2} bytes for {fp32, fp16, bf16} — "
            "half the bytes for half-precision, identical between fp16 and bf16"
        ),
        latency_mlx=(
            "allocation ≈ 1e-5 s for 1M elements; the measurable speed "
            "difference shows up in BANDWIDTH-bound downstream ops, not "
            "allocation itself"
        ),
        scaling=(
            "Decode-time memory traffic scales linearly with bytes-per-elem; "
            "moving from fp32 to bf16 halves the bytes/sec the scheduler "
            "must push and typically doubles throughput of memory-bound kernels."
        ),
    )

    bench = T.benchmark_cell(
        op="dtype-allocation-and-sum",
        code=(
            "shape = (2048, 2048)  # 4M elements\n"
            "\n"
            "def f():\n"
            "    # bf16 path — hardware-accelerated matmul on M-series.\n"
            "    x = mx.random.normal(shape=shape, dtype=mx.float32).astype(mx.bfloat16)\n"
            "    # A bandwidth-bound op (reduction) so the dtype actually matters.\n"
            "    return mx.sum(x)\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
    ]


def _block_lazy_eval(records: list[dict]) -> list[dict]:
    """Block for the 'Lazy Evaluation' section.

    Contents: q04 (timing traps), q02 (mx.compile), whiteboard-b
    (fix-timing-bug), 📐-2 + benchmark (mx.compile speedup), 🛠️ debugging
    (async-host-sync traps).
    """
    q02, q04 = records[1], records[3]

    # Whiteboard challenge: fix a lazy-eval timing bug.
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Fix the lazy-eval timing bug",
        prompt=(
            "The snippet below claims to benchmark an MLX matmul but reports "
            "microseconds — three orders of magnitude too fast. Rewrite it so "
            "the reported ms/call is CORRECT, and explain (in a comment) what "
            "was wrong."
        ),
        constraints=[
            "Add explicit `mx.eval` inside the timed region.",
            "Include ≥ 3 warmup iterations before measurement.",
            "Do N ≥ 5 timed iterations and report the mean.",
            "Preserve the original `f()` definition — only the harness changes.",
            "Assert that the fixed timing is ≥ 10× the buggy timing (the delta is the bug).",
        ],
        complexity="O(N · matmul_cost) — N is the timed iteration count (N=10 is plenty).",
        solution_code=(
            "import time\n"
            "import mlx.core as mx\n"
            "\n"
            "SHAPE = (2048, 2048)\n"
            "\n"
            "def f():\n"
            "    \"\"\"Original op-under-test — unchanged.\"\"\"\n"
            "    a = mx.random.normal(shape=SHAPE, dtype=mx.float32)\n"
            "    b = mx.random.normal(shape=SHAPE, dtype=mx.float32)\n"
            "    return a @ b\n"
            "\n"
            "# BUG: no mx.eval inside the loop — we're timing graph construction.\n"
            "t0 = time.perf_counter()\n"
            "for _ in range(10):\n"
            "    _ = f()\n"
            "buggy_ms = (time.perf_counter() - t0) / 10 * 1000.0\n"
            "\n"
            "# FIX: warmup → timed loop WITH mx.eval → mean.\n"
            "for _ in range(3):        # warmup: compile + allocate\n"
            "    y = f()\n"
            "    mx.eval(y)\n"
            "\n"
            "N = 10\n"
            "t0 = time.perf_counter()\n"
            "for _ in range(N):\n"
            "    y = f()\n"
            "    mx.eval(y)            # forces the kernel to actually run\n"
            "fixed_ms = (time.perf_counter() - t0) / N * 1000.0\n"
            "\n"
            "print(f\"buggy: {buggy_ms:.4f} ms/call (measuring graph build)\")\n"
            "print(f\"fixed: {fixed_ms:.4f} ms/call (measuring real kernel)\")\n"
            "print(f\"ratio: {fixed_ms / max(buggy_ms, 1e-9):.1f}x — the delta is the bug.\")\n"
            "\n"
            "# Assertion: the fix is meaningfully slower than the bug.\n"
            "assert fixed_ms > buggy_ms * 10, (\n"
            "    f\"expected ≥ 10x delta between buggy={buggy_ms:.4f} and fixed={fixed_ms:.4f}\"\n"
            ")\n"
        ),
    )

    complexity = T.complexity_analysis_cell(
        op="mx.compile graph capture + fusion",
        flops=(
            "Same as uncompiled: compile does NOT reduce FLOPs, it reduces "
            "per-op scheduling overhead"
        ),
        memory=(
            "Unchanged or slightly higher on first call (cached graph plan); "
            "steady-state same as eager"
        ),
        latency_mlx=(
            "First call: ~5–30 ms trace overhead; steady-state: 1.5–3× speedup "
            "for elementwise chains with many small ops"
        ),
        scaling=(
            "The win is in ops/second not FLOPs/second — best for tight loops "
            "over small tensors; matmul-dominated code sees little benefit "
            "because the kernel is already fused."
        ),
    )

    bench = T.benchmark_cell(
        op="mx.compile speedup on a fused elementwise chain",
        code=(
            "x_const = mx.random.normal(shape=(512, 512), dtype=mx.float32)\n"
            "\n"
            "def chain(x):\n"
            "    # Intentionally fusion-friendly: many small elementwise ops.\n"
            "    return mx.sum(mx.exp(mx.tanh(x) * 0.5) + mx.sin(x))\n"
            "\n"
            "_compiled = mx.compile(chain)\n"
            "\n"
            "def f():\n"
            "    return _compiled(x_const)\n"
        ),
    )

    # 🛠️ Debugging & Failure Modes cell — async-host-sync traps
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "`print(x)` shows `array(…not evaluated…)` OR a host-sync hangs "
            "forever after a long MLX loop"
        ),
        root_causes=[
            "Reading a lazy array's repr without `mx.eval` — MLX will show the "
            "unevaluated placeholder representation instead of values.",
            "Building an unboundedly long graph in a Python loop (each iter "
            "adds nodes without materializing) — `mx.eval` at the end then "
            "materializes an enormous DAG and appears to hang.",
            "Mixing `mx.eval(x)` with an `x.item()` read that preceded it — "
            "the `.item()` triggered its OWN eval on an incomplete graph; a "
            "later `.item()` races the scheduler.",
        ],
        diagnostic_code=(
            "# Diagnostic: shows (a) unevaluated repr symptom and (b) the fix.\n"
            "import mlx.core as mx\n"
            "import time\n"
            "\n"
            "# Symptom (a): long chain WITHOUT periodic eval.\n"
            "x = mx.ones((256,), dtype=mx.float32)\n"
            "t0 = time.perf_counter()\n"
            "for _ in range(200):\n"
            "    x = x + 1.0  # lazy — graph depth grows to 200\n"
            "    # Intentionally NO eval here\n"
            "mx.eval(x)       # one big eval at the very end\n"
            "dt_deep = (time.perf_counter() - t0) * 1000.0\n"
            "print(f\"deep graph (200 ops, 1 eval): {dt_deep:.2f} ms\")\n"
            "\n"
            "# Fix: periodic eval keeps the graph bounded — same FLOPs, less overhead.\n"
            "x = mx.ones((256,), dtype=mx.float32)\n"
            "t0 = time.perf_counter()\n"
            "for i in range(200):\n"
            "    x = x + 1.0\n"
            "    if (i + 1) % 20 == 0:  # eval every 20 ops\n"
            "        mx.eval(x)\n"
            "mx.eval(x)\n"
            "dt_periodic = (time.perf_counter() - t0) * 1000.0\n"
            "print(f\"periodic eval (every 20 ops):   {dt_periodic:.2f} ms\")\n"
            "\n"
            "# Both produce identical values — the bug is latency, not correctness.\n"
            "expected = float(200 + 1)  # 200 += 1 applied to ones(1) → 201.0\n"
            "assert abs(x[0].item() - expected) < 1e-5, x[0].item()\n"
            "print(f\"✅ values identical ({x[0].item():.1f}); fix saves \"\n"
            "      f\"{dt_deep - dt_periodic:+.2f} ms on this workload.\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
        T.separator_cell(),
        debug_md,
        debug_code,
    ]


def _block_autodiff(records: list[dict]) -> list[dict]:
    """Block for 'Automatic Differentiation' — q05 (bfloat16 numerics, stretch)."""
    q05 = records[4]
    return [
        T.separator_cell(),
        T.interview_question_cell(q05),
    ]


def _block_key_takeaways(records: list[dict]) -> list[dict]:
    """Block for 'Key Takeaways' — q06 (zero-copy research), 🏭 production, 🔭 frontier."""
    q06 = records[5]

    production = T.production_context_cell(
        concept="Lazy evaluation + graph fusion in modern LLM inference stacks",
        vllm=(
            "Eager by default (PyTorch); fuses via CUDA Graphs for decode, "
            "and uses PagedAttention for KV memory — no MLX-style "
            "trace-everything-lazily model."
        ),
        sglang=(
            "RadixAttention caches prefix KV across requests; built on PyTorch, "
            "so the 'graph' is really a set of optimized CUDA kernels, not a "
            "deferred DAG."
        ),
        trt_llm=(
            "Compiles the full model to a static TensorRT engine ahead of time; "
            "no runtime re-tracing, but any shape change requires rebuild. "
            "Closest spiritual cousin to `mx.compile`."
        ),
        mlx_lm=(
            "Leans on MLX's default laziness — kernels fuse naturally, and "
            "`mx.compile` around the per-token decode step removes Python-loop "
            "overhead without giving up dynamic shapes across prompts."
        ),
    )

    frontier = T.frontier_context_cell(
        topic="MLX internals and lazy compilation on Apple Silicon",
        papers=[
            (
                "MLX: An Array Framework for Apple Silicon (ml-explore)",
                2023,
                "Apple's open-source array framework — lazy eval, auto-diff, "
                "Metal-native kernels, unified-memory-first.",
            ),
            (
                "MLX-LM engineering notes + community benchmarks",
                2024,
                "`mx.compile` + int4 quantization drive LLaMA-3-70B decode to "
                "~15 tok/s on M4 Max; per-token overhead dominated by Python "
                "until `mx.compile` landed.",
            ),
            (
                "Gemma 2/3/4 on Apple Silicon (via MLX)",
                2025,
                "First-party ports of frontier models to MLX; showcase "
                "bf16 matmul + compiled decode loops for interactive latency "
                "on M4 Max / M3 Ultra.",
            ),
            (
                "Mamba-2 / SSM work on MLX (community)",
                2025,
                "State-space models exploit lazy-eval differently — the scan "
                "kernel is where compile-driven fusion pays off most because "
                "each state update is a chain of tiny ops.",
            ),
        ],
        current_sota=(
            "As of late 2025, MLX 0.18+ supports bf16 matmul, `mx.compile` "
            "with shape-polymorphic recompilation, and int4 quantization — "
            "close to vLLM feature parity on a single-node Apple Silicon box. "
            "Active frontier: dynamic-shape graph capture without per-shape "
            "recompile, and unified-memory spillover to SSD for 100B+ models."
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q06),
        T.separator_cell(),
        production,
        T.separator_cell(),
        frontier,
    ]


def _block_end(records: list[dict]) -> list[dict]:
    """End-of-notebook block — Interview Question Index only."""
    index_cell = T.interview_index_cell(records)
    return [
        T.separator_cell(),
        index_cell,
    ]



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _insert_block(cells: list[dict], insert_at: int, block: Iterable[dict]) -> int:
    """Insert every cell in ``block`` into ``cells`` starting at ``insert_at``.

    Returns the number of cells inserted. The caller uses the return value
    only for logging; anchors are resolved against the original list
    before any insertions happen, and blocks are applied bottom-up so
    earlier indices stay valid.
    """
    count = 0
    for offset, cell in enumerate(block):
        cells.insert(insert_at + offset, _to_nbformat_cell(cell))
        count += 1
    return count


def transform() -> None:
    """Transform nb01 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the six nb01 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb01] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    mlx_pt_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_MLX_VS_PYTORCH)
    )
    array_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_ARRAY_CREATION)
    )
    lazy_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_LAZY_EVAL)
    )
    autodiff_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_AUTODIFF)
    )
    key_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_KEY_TAKEAWAYS)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (key_end, _block_key_takeaways(records), "key-takeaways"),
        (autodiff_end, _block_autodiff(records), "autodiff"),
        (lazy_end, _block_lazy_eval(records), "lazy-eval"),
        (array_end, _block_array_creation(records), "array-creation"),
        (mlx_pt_end, _block_mlx_vs_pytorch(records), "mlx-vs-pytorch"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb01] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb01] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb01] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb01 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb01] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
