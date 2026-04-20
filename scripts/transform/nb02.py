"""Interview-grade transform for notebook 02 (Math Foundations).

This module inserts the six interview-layer strata into
``02_math_foundations.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): softmax Jacobian, cross-entropy gradient, KL
divergence properties, matrix-calculus conventions. The additions target
the four natural anchors already present in the notebook ("Matrix
Multiplication", "Softmax", "Cross-Entropy Loss", "Key Takeaways") plus
an end-of-notebook Interview Question Index.

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.1, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb02
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

# scripts/transform/nb02.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "02_math_foundations.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 2

# Markers that indicate this notebook has already been transformed.
# Matching the structured-question prefix plus the index heading avoids
# tripping on pre-existing incidental uses of the 🎯 emoji (e.g.
# "🎯 Interview tip:" inline hints already present in the notebook).
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb02-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# The `---\n` prefix the notebook uses is tolerated by _find_first_anchor.
_ANCHOR_MATMUL = "## 📊 Matrix Multiplication"
_ANCHOR_SOFTMAX = "## 🌟 Softmax"
_ANCHOR_CROSS_ENTROPY = "## 📊 Cross-Entropy Loss"
_ANCHOR_KEY_TAKEAWAYS = "## 🔑 Key Takeaways"


# ---------------------------------------------------------------------------
# Notebook I/O
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``02_math_foundations.ipynb`` as a JSON dict."""
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
    """Upgrade a template dict ``{cell_type, source}`` to nbformat 4.5."""
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
    """Return the index of the cell that terminates the section at ``start``.

    The terminator is the next cell whose markdown source begins with a
    top-level ``## `` heading (ignoring an optional ``---`` separator line).
    Returns ``len(cells)`` if no next ``## `` heading is found.
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
    """Return the seven Question_Bank records for nb02.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle              — q01, q02, q03, q04
        research_engineer— q02, q03, q04, q05, q06, q07
        systems_engineer — q03, q04, q05, q06, q07

    Topic coverage (task brief):
        q01 — Matrix-calculus conventions (numerator vs denominator layout)
        q02 — Softmax numerical stability (log-sum-exp trick)
        q03 — Softmax Jacobian (closed form)
        q04 — Cross-entropy + softmax combined gradient
        q05 — KL divergence properties (asymmetry, non-neg, zero-iff-equal)
        q06 — Entropy / mutual information in LLM training
        q07 — Log-sum-exp numerical stability (streaming softmax / Flash-Attn)
    """
    return [
        {
            "id": "nb02-q01",
            "notebook": _NB_FILENAME,
            "section": "Matrix Multiplication — Matrix Calculus Conventions",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["matrix-calculus", "conventions", "autograd"],
            "question": (
                "What is the difference between numerator-layout and "
                "denominator-layout matrix calculus, and which convention "
                "does PyTorch / MLX autograd actually use?"
            ),
            "answer_key_points": [
                "Numerator layout: ∂y/∂x has shape (dim(y), dim(x)) — the numerator's dimensions come first.",
                "Denominator layout: ∂y/∂x has shape (dim(x), dim(y)) — the denominator's dimensions come first. The two layouts are transposes of each other.",
                "Autograd frameworks (PyTorch, MLX, JAX) expose `grad(y) w.r.t. x` as having the SAME shape as `x`, not a Jacobian — this matches reverse-mode AD's vector-Jacobian-product semantics.",
                "For a scalar loss `L` and parameter tensor `W` with shape `(m, n)`, `∂L/∂W` also has shape `(m, n)` so SGD can write `W -= lr * grad`.",
                "Implication: when you write a matrix-calc derivation on paper, you must match your convention to the framework's or you'll get a transposed update rule — a classic source of off-by-transpose bugs.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Claiming 'autograd returns the Jacobian' — it returns a "
                "vector-Jacobian product shaped like the parameter, never a "
                "full dense Jacobian (which for a 1B-param model would be a "
                "~4 EB matrix)."
            ),
            "references": [
                "https://en.wikipedia.org/wiki/Matrix_calculus",
                "https://pytorch.org/docs/stable/notes/autograd.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q02",
            "notebook": _NB_FILENAME,
            "section": "Softmax — Numerical Stability",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["softmax", "numerical-stability", "log-sum-exp"],
            "question": (
                "The naive softmax `exp(x) / sum(exp(x))` blows up on "
                "attention logits near magnitude 50. Derive the "
                "max-subtraction / log-sum-exp trick from first principles "
                "and explain *exactly* why it is mathematically equivalent."
            ),
            "answer_key_points": [
                "Multiply numerator and denominator by `exp(-c)` for any constant `c` — the ratio is unchanged: softmax(x)_i = exp(x_i - c) / Σ_j exp(x_j - c).",
                "Pick c = max(x): all shifted inputs are ≤ 0, so every `exp(x_j - c)` is in (0, 1]; no overflow.",
                "At least one term equals exp(0) = 1, so the denominator is ≥ 1 — zero underflow for the largest entry (the one that matters most).",
                "log-sum-exp: log(Σ exp(x_i)) = c + log(Σ exp(x_i - c)); same shift argument, lets us compute log-softmax stably.",
                "Combined `log_softmax(x) = x - logsumexp(x)` avoids computing exp then log (a double-rounding source); most frameworks fuse it.",
                "Interview tip: ALWAYS reach for this in code — fp16 logits overflow at ~11.0, bf16 at ~88.7; un-shifted softmax is a latent NaN bomb.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Shifting by the mean instead of the max — the trick relies "
                "on ALL shifted inputs being ≤ 0 so every exp is bounded by 1; "
                "mean-shifting leaves the largest entry positive, which can still overflow."
            ),
            "references": [
                "https://en.wikipedia.org/wiki/LogSumExp",
                "https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q03",
            "notebook": _NB_FILENAME,
            "section": "Softmax — Jacobian (Closed Form)",
            "difficulty": "core",
            "roles": ["mle", "research_engineer", "systems_engineer"],
            "topic_tags": ["softmax", "jacobian", "backprop"],
            "question": (
                "Derive the Jacobian of softmax(x) ∈ ℝⁿ w.r.t. x ∈ ℝⁿ from "
                "scratch. Why does the diagonal vs off-diagonal structure "
                "matter for efficient backprop?"
            ),
            "answer_key_points": [
                "Let s = softmax(x), so s_i = exp(x_i) / Σ_k exp(x_k). Use quotient rule for ∂s_i/∂x_j.",
                "Diagonal (i == j): ∂s_i/∂x_i = s_i(1 - s_i).",
                "Off-diagonal (i ≠ j): ∂s_i/∂x_j = -s_i · s_j.",
                "Compact form: J = diag(s) - s · sᵀ  (an n × n rank-(n-1) matrix).",
                "Key property: J is symmetric (since s_i·s_j == s_j·s_i) and singular (sum of each row is 0 because Σ s_k = 1 is invariant to a uniform shift in x).",
                "Efficient backprop: you almost NEVER materialize J. Instead use the identity (J · v)_i = s_i · (v_i - sᵀv) — linear in n, not quadratic.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Writing the Jacobian as J = diag(s) - diag(s²); the "
                "off-diagonal piece is the OUTER PRODUCT s·sᵀ, not a "
                "diagonal elementwise square."
            ),
            "references": [
                "https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q04",
            "notebook": _NB_FILENAME,
            "section": "Cross-Entropy Loss — Combined Gradient with Softmax",
            "difficulty": "core",
            "roles": ["mle", "research_engineer", "systems_engineer"],
            "topic_tags": ["cross-entropy", "softmax", "gradient", "fused-kernel"],
            "question": (
                "For loss `L = -log softmax(logits)_y` (cross-entropy with "
                "one-hot target `y`), derive ∂L/∂logits. Explain why "
                "production kernels NEVER multiply explicitly by the softmax "
                "Jacobian."
            ),
            "answer_key_points": [
                "Chain rule: ∂L/∂logits = (∂L/∂s) · (∂s/∂logits), where s = softmax(logits).",
                "∂L/∂s is a one-hot vector: -1/s_y at position y, zero elsewhere.",
                "Multiplying by J = diag(s) - ssᵀ and simplifying gives the famous closed form: ∂L/∂logits = s - y  (s is the softmax output, y is the one-hot target).",
                "No n×n Jacobian ever appears in the final formula — it cancels algebraically. Cost drops from O(n²) to O(n).",
                "Numerical bonus: computed via log_softmax, the gradient is `exp(log_softmax) - one_hot(y)` — avoids computing exp twice and stays fp16/bf16-safe.",
                "Every production CE+softmax kernel (vLLM, TRT-LLM, MLX-LM, PyTorch's `cross_entropy`) exploits this fusion; hand-rolled 'softmax then CE' on logits is both slower and less stable.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Computing softmax then CE separately and backpropping through both "
                "layers — this both wastes FLOPs (the Jacobian-target-onehot cancellation "
                "is lost) and introduces a log-of-exp rounding pair that fp16 can't tolerate."
            ),
            "references": [
                "https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative",
                "https://cs231n.github.io/linear-classify/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q05",
            "notebook": _NB_FILENAME,
            "section": "Key Takeaways — KL Divergence",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["kl-divergence", "information-theory", "alignment"],
            "question": (
                "State the three defining properties of KL(p ‖ q), give a "
                "concrete example where KL(p ‖ q) ≠ KL(q ‖ p), and explain "
                "why RLHF/DPO uses the 'reverse' direction."
            ),
            "answer_key_points": [
                "Property 1 — non-negative: KL(p ‖ q) ≥ 0 (Gibbs' inequality, from Jensen applied to the convex −log).",
                "Property 2 — zero iff equal: KL(p ‖ q) = 0 ⇔ p = q almost everywhere.",
                "Property 3 — asymmetric: KL is NOT a metric (no triangle inequality, not symmetric); it's a *premetric*.",
                "Asymmetry example: p = [0.9, 0.1], q = [0.1, 0.9] → KL(p‖q) = 0.9·log(9) + 0.1·log(1/9) ≈ 1.76 nats, KL(q‖p) ≈ 1.76 nats (symmetric here by luck); try p=[1,0,0], q=[1/3,1/3,1/3]: KL(p‖q)=log(3)≈1.10, KL(q‖p)=∞ because q has mass where p has none.",
                "Zero-probability bin is the killer: KL(p‖q) blows up when q_i=0 but p_i>0 — this is why 'sample from p, score under q' requires q to have FULL support over p.",
                "RLHF / DPO use KL(π_policy ‖ π_ref) (policy on the left) so the loss is finite: the reference model is trained first and has full support; the policy just needs to stay close.",
                "Forward KL (p‖q) is 'mean-seeking' (zero-avoiding); reverse KL (q‖p) is 'mode-seeking' (zero-forcing) — MLE training minimizes forward KL of data ‖ model; RLHF minimizes reverse KL of policy ‖ reference.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Calling KL 'a distance' — it's not. It fails BOTH symmetry and the "
                "triangle inequality; it's only a divergence. Candidates who write "
                "`kl(p,q) == kl(q,p)` get filtered out."
            ),
            "references": [
                "https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence",
                "https://arxiv.org/abs/2305.18290",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q06",
            "notebook": _NB_FILENAME,
            "section": "Key Takeaways — Entropy & Mutual Information",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["entropy", "mutual-information", "perplexity"],
            "question": (
                "Connect the dots: what do entropy H(X), cross-entropy "
                "H(p, q), KL(p ‖ q), and model perplexity all have to do with "
                "each other? Give the exact identity."
            ),
            "answer_key_points": [
                "Entropy: H(p) = -Σ p_i log p_i. The self-information of the data distribution; in nats if log=ln, bits if log₂.",
                "Cross-entropy: H(p, q) = -Σ p_i log q_i. The avg # bits/nats needed to encode samples from p using a code optimized for q.",
                "The fundamental identity: H(p, q) = H(p) + KL(p ‖ q). Cross-entropy EQUALS entropy PLUS KL.",
                "LLM training minimizes H(p_data, p_model); since H(p_data) is fixed, this is exactly minimizing KL(p_data ‖ p_model).",
                "Perplexity = exp(cross-entropy per token). 'The model is choosing among PPL tokens on average' — a PPL of 10 means the model is as uncertain as if uniformly picking one of 10 options per position.",
                "Mutual information: I(X; Y) = H(X) - H(X|Y) = KL(p(x,y) ‖ p(x)p(y)). Measures how much knowing Y reduces uncertainty about X; appears in InfoNCE, contrastive losses, and information-bottleneck arguments.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Confusing perplexity with cross-entropy — perplexity is the "
                "EXPONENTIAL of the per-token CE, not CE itself. A 2× drop in "
                "CE is an exponential (not 2×) drop in perplexity."
            ),
            "references": [
                "https://en.wikipedia.org/wiki/Cross_entropy",
                "https://en.wikipedia.org/wiki/Mutual_information",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb02-q07",
            "notebook": _NB_FILENAME,
            "section": "Softmax — Log-Sum-Exp in Streaming / Flash-Attention",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["log-sum-exp", "flash-attention", "streaming", "online-softmax"],
            "question": (
                "Flash-Attention computes softmax 'online' — one tile at a "
                "time, without materializing the full attention matrix. "
                "Derive the tile-merge formula for log-sum-exp and explain "
                "why this is the crux that makes FA-2 fp16-safe."
            ),
            "answer_key_points": [
                "Given two partial running stats (m_a, ℓ_a) and (m_b, ℓ_b) — running max and sum-of-exp relative to that max — the merged (m, ℓ) is: m = max(m_a, m_b); ℓ = exp(m_a - m)·ℓ_a + exp(m_b - m)·ℓ_b.",
                "At every merge you ADJUST the previously-accumulated ℓ by exp(old_max − new_max) ≤ 1, so no rescale ever grows.",
                "This keeps every intermediate bounded in (0, ℓ·e], i.e. representable in fp16 (max ~6.5e4) even when raw logits exceed the fp16 exponent range.",
                "Output `O` tiles follow the same correction: O_new = O_old · exp(m_old - m) + ΔO_new; the rescale ratio is shared with the ℓ update — one extra multiply per tile.",
                "Memory win: FA-2 never stores the (N × N) softmax matrix — only O(N · d_head) in SRAM; wall-clock win follows because HBM bandwidth was the bottleneck.",
                "fp16-safety is NOT automatic — DeepSeek-V3 and Gemma still promote the softmax accumulator to fp32 even inside FA-3 tiles because bf16/fp16 precision at ℓ > 10⁴ degrades the tail of the distribution.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Assuming 'fp16 softmax = safe softmax' — tile merging keeps "
                "range OK, but precision on the long-tail probabilities (where "
                "temperature/sampling actually matters) requires an fp32 accumulator."
            ),
            "references": [
                "https://arxiv.org/abs/2205.14135",
                "https://arxiv.org/abs/2307.08691",
                "https://tridao.me/publications/flash2/flash2.pdf",
            ],
            "added_in": added_in,
        },
    ]


# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_matmul(records: list[dict]) -> list[dict]:
    """Block for 'Matrix Multiplication' — q01 (warmup, matrix-calc conventions).

    Light-touch insertion: just the warmup question — matrix calc lives here
    because matmul is the first time the reader meets (∂y/∂W) concretely.
    """
    q01 = records[0]
    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
    ]


def _block_softmax(records: list[dict]) -> list[dict]:
    """Block for the 'Softmax' section.

    Contents: q02 (numerical stability), q03 (Jacobian), q07 (streaming LSE),
    whiteboard-A (numerically-stable log-softmax), 📐-1 (softmax FLOPs/mem)
    with benchmark.
    """
    q02, q03, q07 = records[1], records[2], records[6]

    # Whiteboard A — numerically-stable log-softmax that matches mx.log(mx.softmax(·))
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Numerically-stable log-softmax",
        prompt=(
            "Implement `stable_log_softmax(x, axis=-1)` from scratch using "
            "the log-sum-exp trick and assert it matches "
            "`mx.log(mx.softmax(x, axis=-1))` within fp32 epsilon on a hard "
            "case where the largest logit is ~100 (naive softmax would "
            "overflow to `inf`)."
        ),
        constraints=[
            "Use only `mx.exp`, `mx.sum`, `mx.log`, `mx.max` — do NOT call `mx.softmax`.",
            "Subtract `x.max(axis=axis, keepdims=True)` before exp (Requirement 4.5 — complexity).",
            "Include at least one `assert` comparing against `mx.log(mx.softmax(x))` on a non-pathological input (max ~1).",
            "Include one additional `assert` showing your implementation is FINITE on an overflow case (max ≈ 100) while the naive formula isn't.",
            "Call `mx.eval` on any lazy MLX result before converting to a Python scalar for comparison.",
        ],
        complexity=(
            "O(n) per axis — three reductions (max, sum-of-exp, log) plus "
            "one elementwise subtraction."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "import numpy as np\n"
            "\n"
            "def stable_log_softmax(x: mx.array, axis: int = -1) -> mx.array:\n"
            "    \"\"\"Log-softmax via max-subtraction. Equivalent to `x - logsumexp(x)`.\"\"\"\n"
            "    # c = max(x) along the reduction axis (kept for broadcast).\n"
            "    c = mx.max(x, axis=axis, keepdims=True)\n"
            "    shifted = x - c\n"
            "    # log Σ exp(shifted) — shifted ≤ 0 so each exp is in (0, 1]. No overflow.\n"
            "    lse = c + mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))\n"
            "    return x - lse\n"
            "\n"
            "# Sanity: matches built-in on a normal input.\n"
            "x_ok = mx.random.normal(shape=(4, 10), dtype=mx.float32)\n"
            "ours = stable_log_softmax(x_ok, axis=-1)\n"
            "ref = mx.log(mx.softmax(x_ok, axis=-1))\n"
            "mx.eval(ours, ref)\n"
            "delta = float(mx.max(mx.abs(ours - ref)).item())\n"
            "assert delta < 1e-5, f\"mismatch vs mx.softmax: max|Δ|={delta}\"\n"
            "print(f\"✅ matches mx.log(mx.softmax(x)) within {delta:.2e}\")\n"
            "\n"
            "# Overflow case: logits with max ≈ 100.\n"
            "# Naive: exp(100) ≈ 2.7e43 — FINE in fp32 (fp32 max ≈ 3.4e38 is close but reachable),\n"
            "# but exp(104) overflows. So push just over the fp32 edge.\n"
            "x_hard = mx.array([[0.0, 50.0, 100.0, 105.0]], dtype=mx.float32)\n"
            "naive = mx.exp(x_hard) / mx.sum(mx.exp(x_hard), axis=-1, keepdims=True)\n"
            "naive_log = mx.log(naive)\n"
            "ours_hard = stable_log_softmax(x_hard, axis=-1)\n"
            "mx.eval(naive_log, ours_hard)\n"
            "\n"
            "naive_np = np.array(naive_log)\n"
            "ours_np = np.array(ours_hard)\n"
            "# Naive must produce at least one non-finite entry; ours must be fully finite.\n"
            "assert not np.all(np.isfinite(naive_np)), (\n"
            "    f\"expected naive to overflow on hard case, got {naive_np}\"\n"
            ")\n"
            "assert np.all(np.isfinite(ours_np)), (\n"
            "    f\"stable_log_softmax produced non-finite values: {ours_np}\"\n"
            ")\n"
            "print(f\"✅ overflow-case: naive log-softmax = {naive_np[0].tolist()}\")\n"
            "print(f\"   stable_log_softmax          = {ours_np[0].tolist()}\")\n"
        ),
    )

    # 📐 Complexity analysis + benchmark — softmax over last dim.
    complexity = T.complexity_analysis_cell(
        op="softmax(x, axis=-1) on a (B, N) tensor",
        flops=(
            "~5·B·N elementwise ops: one `max` reduction (B·N), one subtract "
            "(B·N), one `exp` (B·N), one `sum` reduction (B·N), one divide (B·N)"
        ),
        memory=(
            "2·B·N·bytes-per-elem (input + output). No O(N²) materialization "
            "per row — softmax is purely elementwise on the normalized axis"
        ),
        latency_mlx=(
            "Memory-bound on modern GPUs: peak ~HBM-bandwidth-limited. On M4 "
            "Pro / fp32, ~20–80 μs for B=1, N=32k (vocab-size). Fuses with "
            "matmul output in `mx.compile`"
        ),
        scaling=(
            "Softmax is BANDWIDTH-bound, not compute-bound — halving bytes "
            "(fp32→fp16) roughly halves wall-time. This is why bf16 decoders "
            "see ~2× softmax speed-up even though the FLOP count is identical."
        ),
    )

    bench = T.benchmark_cell(
        op="softmax over last dim on (8, 4096) fp32 (attention-like shape)",
        code=(
            "x_const = mx.random.normal(shape=(8, 4096), dtype=mx.float32)\n"
            "\n"
            "def f():\n"
            "    return mx.softmax(x_const, axis=-1)\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        T.interview_question_cell(q07),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
    ]


def _block_cross_entropy(records: list[dict]) -> list[dict]:
    """Block for the 'Cross-Entropy Loss' section.

    Contents: q04 (combined gradient), whiteboard-B (softmax Jacobian
    verified via `mx.grad`), 📐-2 (cross-entropy over batch × vocab) with
    benchmark, and the 🛠️ debugging cell (overflow + empty batch + zero-bin
    KL).
    """
    q04 = records[3]

    # Whiteboard B — derive softmax Jacobian, implement, validate via mx.grad
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Derive softmax Jacobian and validate via `mx.grad`",
        prompt=(
            "Implement `softmax_jacobian(x)` returning the n×n closed-form "
            "Jacobian J = diag(s) - s·sᵀ where s = softmax(x). Validate it "
            "against MLX's autodiff by comparing `J @ v` to the "
            "vector-Jacobian product obtained from `mx.grad`."
        ),
        constraints=[
            "`x` is a 1-D MLX array of length n; return an (n, n) MLX array.",
            "Use only `mx.softmax`, `mx.diag`, `mx.outer` (or broadcasting equivalents) — no Python loops over n.",
            "Compare `J @ v` against `(∂(vᵀ·s)/∂x)` obtained via `mx.grad`. This is the vJp identity the autograd engine uses.",
            "Assert max-abs-difference < 1e-5 on a random input and a random projection vector `v`.",
            "Call `mx.eval` on all lazy results before asserting.",
        ],
        complexity=(
            "O(n²) time & memory for the dense Jacobian; the vJp on the "
            "right-hand side is O(n)."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "\n"
            "def softmax_jacobian(x: mx.array) -> mx.array:\n"
            "    \"\"\"Closed-form Jacobian of softmax: J = diag(s) - s·sᵀ.\"\"\"\n"
            "    assert x.ndim == 1, f\"expected 1-D input, got shape {x.shape}\"\n"
            "    s = mx.softmax(x, axis=-1)\n"
            "    # diag(s) - outer(s, s). Broadcasting gives the outer product.\n"
            "    J = mx.diag(s) - s[:, None] * s[None, :]\n"
            "    return J\n"
            "\n"
            "# Random test case.\n"
            "mx.random.seed(0)\n"
            "n = 16\n"
            "x = mx.random.normal(shape=(n,), dtype=mx.float32)\n"
            "v = mx.random.normal(shape=(n,), dtype=mx.float32)\n"
            "\n"
            "# Closed-form J @ v.\n"
            "J = softmax_jacobian(x)\n"
            "jv_closed = J @ v\n"
            "mx.eval(jv_closed)\n"
            "\n"
            "# Autograd: grad of (v · softmax(x)) w.r.t. x == Jᵀ @ v == J @ v (J symmetric).\n"
            "def fn(x_):\n"
            "    s_ = mx.softmax(x_, axis=-1)\n"
            "    return mx.sum(v * s_)\n"
            "\n"
            "jv_autograd = mx.grad(fn)(x)\n"
            "mx.eval(jv_autograd)\n"
            "\n"
            "delta = float(mx.max(mx.abs(jv_closed - jv_autograd)).item())\n"
            "assert delta < 1e-5, f\"closed-form disagrees with mx.grad: max|Δ|={delta}\"\n"
            "print(f\"✅ softmax_jacobian matches mx.grad within {delta:.2e}\")\n"
            "\n"
            "# Bonus: verify symmetry and row-sum-to-zero (singularity) properties.\n"
            "assert float(mx.max(mx.abs(J - J.T)).item()) < 1e-6, \"J should be symmetric\"\n"
            "assert float(mx.max(mx.abs(mx.sum(J, axis=-1))).item()) < 1e-6, (\n"
            "    \"J rows should sum to 0 (softmax shift-invariance)\"\n"
            ")\n"
            "print(\"✅ J is symmetric and each row sums to 0 (softmax shift-invariance)\")\n"
        ),
    )

    # 📐 Complexity analysis + benchmark — cross-entropy over batch × vocab.
    complexity = T.complexity_analysis_cell(
        op="cross-entropy loss over (B, V) logits with integer targets",
        flops=(
            "~4·B·V (log-softmax over V) + B (gather + negate). Softmax "
            "Jacobian is NEVER materialized — `∂L/∂logits = s - onehot(y)`"
        ),
        memory=(
            "Forward: 2·B·V bytes working set. Backward: B·V bytes for the "
            "gradient (same shape as logits). With typical LLaMA vocab "
            "V=128k and batch=8, that's ~4 MB in fp32 per token position"
        ),
        latency_mlx=(
            "Dominated by the log-softmax reduction: ~BANDWIDTH-limited. For "
            "B=8, V=32k fp32 on M4 Pro: ~200–600 μs. Fused kernels (vLLM, "
            "MLX-LM) cut this by ~2×"
        ),
        scaling=(
            "At V ≈ 128k and sequence length L ≈ 4k, the per-token CE loss "
            "dominates the decoder tail; this is why frontier trainers shard "
            "the vocab across TP ranks and compute the loss in tensor-parallel."
        ),
    )

    bench = T.benchmark_cell(
        op="cross-entropy on (B=8, V=32000) logits + int32 targets",
        code=(
            "B, V = 8, 32_000\n"
            "logits_const = mx.random.normal(shape=(B, V), dtype=mx.float32)\n"
            "targets_const = mx.random.randint(0, V, shape=(B,))\n"
            "\n"
            "def f():\n"
            "    # Numerically stable CE: log_softmax then gather target logit.\n"
            "    ls = logits_const - mx.logsumexp(logits_const, axis=-1, keepdims=True)\n"
            "    # Gather the target log-prob per row, then negate and mean.\n"
            "    idx = mx.arange(B)\n"
            "    return -mx.mean(ls[idx, targets_const])\n"
        ),
    )

    # 🛠️ Debugging & Failure Modes — softmax overflow, empty batch, zero-prob KL.
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "NaN losses, `inf` softmax outputs, or `Infinity` KL — the three "
            "classic numerical traps of the math-layer"
        ),
        root_causes=[
            "Softmax overflow: un-shifted `exp(logits)` on logits where one "
            "entry > ~88.7 (bf16) or ~11.0 (fp16) → `inf / inf` → NaN.",
            "Empty batch: cross-entropy over `(0, V)` logits hits a `0/0` in "
            "the mean; many kernels happily return NaN instead of raising.",
            "Zero-probability bin in KL(p ‖ q): any position with q_i == 0 "
            "but p_i > 0 produces `log(0) = -inf` → the divergence is "
            "infinite. Fix by smoothing q (add ε then renormalize) or by "
            "masking out positions where p_i == 0.",
        ],
        diagnostic_code=(
            "# Reproduce each symptom, then show the fix.\n"
            "import mlx.core as mx\n"
            "import numpy as np\n"
            "\n"
            "# -- Symptom 1: softmax overflow -----------------------------------\n"
            "bad_logits = mx.array([0.0, 50.0, 100.0, 200.0], dtype=mx.float32)\n"
            "# Naive: exp(200) ≈ 7e86 — overflows fp32 (max ~3.4e38).\n"
            "naive_num = mx.exp(bad_logits)\n"
            "mx.eval(naive_num)\n"
            "print(f\"naive exp:       {np.array(naive_num).tolist()}\")  # inf\n"
            "# Fix: subtract the max before exp.\n"
            "shifted = bad_logits - mx.max(bad_logits)\n"
            "stable_num = mx.exp(shifted)\n"
            "stable_softmax = stable_num / mx.sum(stable_num)\n"
            "mx.eval(stable_softmax)\n"
            "print(f\"stable softmax:  {np.array(stable_softmax).tolist()}\")\n"
            "assert np.all(np.isfinite(np.array(stable_softmax))), \"stable softmax must be finite\"\n"
            "\n"
            "# -- Symptom 2: empty-batch cross-entropy --------------------------\n"
            "empty_logits = mx.zeros((0, 5), dtype=mx.float32)\n"
            "empty_targets = mx.zeros((0,), dtype=mx.int32)\n"
            "# Defensive wrapper — return 0.0 for empty batch instead of NaN.\n"
            "def safe_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:\n"
            "    if logits.shape[0] == 0:\n"
            "        return mx.array(0.0, dtype=mx.float32)\n"
            "    ls = logits - mx.logsumexp(logits, axis=-1, keepdims=True)\n"
            "    idx = mx.arange(logits.shape[0])\n"
            "    return -mx.mean(ls[idx, targets])\n"
            "safe_loss = safe_cross_entropy(empty_logits, empty_targets)\n"
            "mx.eval(safe_loss)\n"
            "assert float(safe_loss.item()) == 0.0, f\"expected 0.0, got {safe_loss.item()}\"\n"
            "print(f\"safe CE on empty batch: {safe_loss.item()}\")\n"
            "\n"
            "# -- Symptom 3: zero-probability bin in KL -------------------------\n"
            "p = mx.array([0.5, 0.5, 0.0])  # third bin has zero mass\n"
            "q = mx.array([0.1, 0.1, 0.8])  # full support\n"
            "# Forward KL is FINE: the p_i=0 bin contributes 0·log(0/0.8) := 0.\n"
            "# Reverse KL(q || p) blows up: q has mass on bin 3 but p_i=0 there.\n"
            "def kl(a: mx.array, b: mx.array, eps: float = 0.0) -> mx.array:\n"
            "    b_safe = b + eps\n"
            "    b_safe = b_safe / mx.sum(b_safe)  # renormalize after smoothing\n"
            "    # Mask 0·log(0) := 0 to follow the convention.\n"
            "    term = mx.where(a > 0, a * (mx.log(a) - mx.log(b_safe)), mx.zeros_like(a))\n"
            "    return mx.sum(term)\n"
            "kl_pq_naive = kl(p, q, eps=0.0)       # forward: finite\n"
            "kl_qp_naive = kl(q, p, eps=0.0)       # reverse: inf because p has a 0\n"
            "mx.eval(kl_pq_naive, kl_qp_naive)\n"
            "print(f\"KL(p||q) naive: {float(kl_pq_naive.item()):.4f}  (finite)\")\n"
            "kl_qp_naive_v = float(kl_qp_naive.item())\n"
            "print(f\"KL(q||p) naive: {kl_qp_naive_v}  ← inf, as expected\")\n"
            "assert not np.isfinite(kl_qp_naive_v), \"reverse KL with zero bin should be inf\"\n"
            "# Fix: smooth `p` with a small ε so log(p) is always finite.\n"
            "kl_qp_smoothed = kl(q, p, eps=1e-8)\n"
            "mx.eval(kl_qp_smoothed)\n"
            "assert np.isfinite(float(kl_qp_smoothed.item())), \"ε-smoothed KL must be finite\"\n"
            "print(f\"KL(q||p) smoothed (ε=1e-8): {float(kl_qp_smoothed.item()):.4f}\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q04),
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


def _block_key_takeaways(records: list[dict]) -> list[dict]:
    """Block for the 'Key Takeaways' section.

    Contents: q05 (KL properties), q06 (entropy / MI), 🏭 production
    (fused CE + FA softmax in vLLM/TRT-LLM/MLX-LM), 🔭 frontier (2024–2026
    works on streaming softmax + numerical tricks).
    """
    q05, q06 = records[4], records[5]

    production = T.production_context_cell(
        concept="Fused softmax + cross-entropy + Flash-Attention softmax",
        vllm=(
            "Uses PyTorch's `F.cross_entropy` (fused log-softmax + NLL in a "
            "single CUDA kernel) for training; inference path uses "
            "FlashAttention-2 whose implicit online softmax keeps fp16 safe "
            "and avoids materializing the (N × N) attention matrix."
        ),
        sglang=(
            "Inherits PyTorch's fused CE kernel. Its RadixAttention pre-"
            "computes softmax normalization constants once per shared prefix "
            "then reuses them — a cache-oriented twist on the LSE trick."
        ),
        trt_llm=(
            "Builds a single fused kernel `softmax_cross_entropy` at engine-"
            "compile time; attention uses FMHA/FMHCA kernels with online "
            "softmax baked into the tile loop. No user-visible softmax op "
            "exists at runtime."
        ),
        mlx_lm=(
            "`mx.softmax` + `mx.logsumexp` fuse automatically under "
            "`mx.compile`; attention uses the Metal-native SDPA kernel which, "
            "like FA-2, streams softmax tile-by-tile — bf16-safe on M-series."
        ),
    )

    frontier = T.frontier_context_cell(
        topic="Numerical tricks in modern attention + softmax",
        papers=[
            (
                "FlashAttention-2: Faster Attention with Better Parallelism (Dao, 2023)",
                2023,
                "Formalized the online-softmax tile-merge identity and showed "
                "2× wall-clock over FA-1 on A100; baseline for every 2024+ "
                "attention kernel including MLX's SDPA.",
            ),
            (
                "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision (Shah et al., 2024)",
                2024,
                "Hopper-specific; uses fp8 for attention ops but promotes the "
                "softmax ACCUMULATOR (m, ℓ) to fp32 — the same rule DeepSeek-V3 "
                "follows for its attention rewrite.",
            ),
            (
                "DeepSeek-V3 Technical Report (DeepSeek, 2024)",
                2024,
                "Documents the 'fp32 accumulator in softmax' rule explicitly "
                "as a stability requirement at frontier scale (671 B params, "
                "trained in fp8 + bf16). The log-sum-exp tile-merge is un-"
                "changed from FA-2.",
            ),
            (
                "Online Softmax + Streaming LLMs (community, 2024–2025)",
                2025,
                "StreamingLLM, Windowed Attention, and sink tokens all reuse "
                "the LSE tile-merge so they can discard old KV without "
                "re-normalizing softmax on the surviving tokens — the merge "
                "formula is the only reason streaming doesn't corrupt probabilities.",
            ),
            (
                "Gemma 2/3/4 engineering notes on MLX (Apple / community, 2025)",
                2025,
                "Port of Gemma to MLX inherits the bf16 softmax + fp32 "
                "accumulator rule; `mx.softmax` on M-series auto-promotes the "
                "reduction for bf16 inputs to preserve tail precision.",
            ),
        ],
        current_sota=(
            "As of late 2025, every frontier attention kernel (FA-2, FA-3, "
            "MLX-SDPA, TRT-LLM FMHA, vLLM V1 paged attention) uses the exact "
            "log-sum-exp tile-merge formula from FA-2; the only variation is "
            "fp8 vs bf16 vs fp16 storage and where exactly the fp32 "
            "accumulator lives. Active frontier: softmax-free attention "
            "(ReLU² attention, polynomial kernels) for sub-quadratic "
            "attention — the trade-off is expressivity vs LSE's provable "
            "numerical stability."
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q05),
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
    """Transform nb02 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb02 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb02] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    matmul_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_MATMUL)
    )
    softmax_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_SOFTMAX)
    )
    ce_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_CROSS_ENTROPY)
    )
    key_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_KEY_TAKEAWAYS)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (key_end, _block_key_takeaways(records), "key-takeaways"),
        (ce_end, _block_cross_entropy(records), "cross-entropy"),
        (softmax_end, _block_softmax(records), "softmax"),
        (matmul_end, _block_matmul(records), "matmul"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb02] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb02] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb02] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb02 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb02] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
