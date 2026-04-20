"""Interview-grade transform for notebook 05 (Self-Attention from Scratch).

This module inserts the six interview-layer strata into
``05_self_attention.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): why √d_head scaling (derived from the variance of
the Q·K product under unit-variance inputs, plus softmax saturation
without scaling), attention as weighted retrieval (Q/K/V as query-key-
value databases), causal-mask implementation (fill with −∞ BEFORE
softmax), attention memory formula O(B·H·T²) and its pathology past 8k,
attention FLOPs O(B·H·T²·d_head), cross-attention vs self-attention,
and scaled-dot-product vs additive attention.

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.1, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb05
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

# scripts/transform/nb05.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "05_self_attention.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 5

# Markers that indicate this notebook has already been transformed.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb05-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# Both anchors are resolved against the cell list BEFORE any insertion so
# insertions can proceed bottom-up without invalidating anchors.
#
# "### Putting It All Together: `scaled_dot_product_attention()`" — the
#   end of Section 1 (single-head SDPA). We insert just before the next
#   "## Section 2" heading.
# "## Section 3: Causal Masking" — after this whole section (and before
#   Section 4: Multi-Head Attention) we insert the causal-mask block.
_ANCHOR_SDPA = "Putting It All Together: `scaled_dot_product_attention()`"
_ANCHOR_CAUSAL = "## Section 3: Causal Masking"


# ---------------------------------------------------------------------------
# Notebook I/O (raw JSON; Requirement 19.4 permits this for .ipynb edits)
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``05_self_attention.ipynb`` as a JSON dict."""
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
    """Return the seven Question_Bank records for nb05.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle               — q01, q02, q03, q05
        research_engineer — q01, q02, q04, q05, q06, q07
        systems_engineer  — q03, q04, q06, q07

    Topic coverage (task brief — LLD-4):
        q01 — Why √d_head scaling (derive from variance of Q·K; show
              softmax saturation without it)
        q02 — Attention as weighted retrieval (Q/K/V as a soft
              key-value database)
        q03 — Causal masking implementation (fill to -∞ BEFORE softmax)
        q04 — Attention memory formula O(B·H·T²) and its pathology past 8k
        q05 — Cross-attention vs self-attention (encoder-decoder vs
              decoder-only, Q-source vs KV-source)
        q06 — Scaled-dot-product vs additive attention (Bahdanau/Luong →
              Vaswani; parameter count, parallelism, quality)
        q07 — Attention FLOPs O(B·H·T²·d_head) and the arithmetic-intensity
              argument behind FlashAttention
    """
    return [
        {
            "id": "nb05-q01",
            "notebook": _NB_FILENAME,
            "section": "Scaled Dot-Product — Why √d_head?",
            "difficulty": "warmup",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["scaling", "softmax", "variance", "derivation"],
            "question": (
                "Derive the √d_head scaling factor in "
                "`softmax(Q·Kᵀ / √d_head)·V`. Assume q and k are "
                "independent, zero-mean, unit-variance vectors of "
                "dimension d. What is Var(q·k), and what goes wrong "
                "to the softmax if you forget to divide by √d?"
            ),
            "answer_key_points": [
                "Setup: q, k ∈ ℝ^d with q_i, k_i i.i.d. zero-mean unit-variance. The dot product is `q·k = Σᵢ q_i · k_i`. Each term is a product of two independent unit-variance zero-mean variables → E[q_i·k_i] = 0, Var(q_i·k_i) = Var(q_i)·Var(k_i) + 0 = 1.",
                "Sum of d independent unit-variance terms: `Var(q·k) = d`, `Std(q·k) = √d`. So the raw logit `q·k` has scale √d — NOT 1. Without rescaling, the entries of `Q·Kᵀ` grow as √d_head and saturate the softmax at large d.",
                "Softmax saturation pathology: for logits with std σ, the softmax output is dominated by the largest entry once σ ≳ log(T) (T = sequence length). At d_head=128, σ = √128 ≈ 11.3 — the softmax is a near-delta function. Gradient of softmax at a near-delta is ~0 ⇒ attention can't learn.",
                "Fix: divide by √d_head BEFORE softmax. `(q·k) / √d_head` has variance 1 again; logits stay in the regime where softmax is smooth and gradients are non-vanishing. The scaling factor is a variance-preservation reparameterisation, not a learning-rate hack.",
                "Historical note: this is why Vaswani et al. (2017) call it 'SCALED' dot-product attention. Bahdanau/Luong (2014/2015) used additive attention `tanh(Wq+Wk)` which has bounded outputs — they didn't need rescaling because `tanh` is O(1) regardless of d.",
                "Alternative scalings that DON'T work: dividing by d (too aggressive — logits shrink with dimension), using LayerNorm inside Q·K (what 'QK-norm' / `use_qk_norm=True` does in Qwen-2.5/Gemma-2 — a more aggressive form; works but changes learning dynamics), or just reducing the learning rate (misses that the problem is in the softmax, not the optimizer).",
                "Beyond the basic story: QK-norm (applying LayerNorm to Q and K before the dot product) is the 2024 frontier fix — it handles the variance issue AND the training-instability problem at scale where the √d scaling is necessary but not SUFFICIENT at 100B+ params (Gemma-2, Qwen-2.5 both ship with QK-norm).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'we divide by √d_head so the softmax doesn't "
                "overflow numerically'. It's not about float overflow — "
                "it's about softmax SATURATION killing the gradient. "
                "Numerical overflow is handled separately by the "
                "max-subtract trick inside `mx.softmax` / `F.softmax`."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://lilianweng.github.io/posts/2018-06-24-attention/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q02",
            "notebook": _NB_FILENAME,
            "section": "Attention as Weighted Retrieval",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["attention", "intuition", "qkv", "retrieval"],
            "question": (
                "Explain attention as a (soft) key-value database lookup. "
                "What does each of Q, K, V represent semantically, and "
                "what does `softmax(Q·Kᵀ/√d)·V` compute in that analogy? "
                "Why is this 'soft' retrieval rather than a hard lookup?"
            ),
            "answer_key_points": [
                "Analogy: a token is asking a QUESTION (its query vector q). Every token in the context offers a KEY (how it advertises itself) and a VALUE (what it actually contains). The query goes to each key, gets a relevance score, and receives back a weighted mix of values.",
                "Q = queries. Per-token: 'what information am I looking for?' Shape (B, T, H, d_head). Produced by `X · W_Q`.",
                "K = keys. Per-token: 'what do I advertise that I have?' Shape (B, T, H, d_head). Produced by `X · W_K`. In self-attention K comes from the SAME sequence as Q; in cross-attention it comes from a DIFFERENT sequence (encoder output).",
                "V = values. Per-token: 'what's the payload I return if someone matches me?' Shape (B, T, H, d_v). Produced by `X · W_V`.",
                "The computation: `scores = Q·Kᵀ / √d_head` — (T, T) matrix of query-key dot-product similarities. `weights = softmax(scores, axis=-1)` — each row is a probability distribution over keys. `output = weights · V` — each output row is a convex combination of value rows, weighted by how much that key matched its query.",
                "'Soft' retrieval: a hard database lookup returns ONE value — the one whose key exactly matches. Soft retrieval returns a WEIGHTED SUM of all values, where the weights are a smooth function of key-similarity. This makes attention differentiable (the weights are softmax, not argmax), which is the whole reason it trains by gradient descent.",
                "Why this was a breakthrough: pre-2017 NN memory (neural Turing machines, memory networks) used softer but heavier lookup mechanisms. Vaswani's 'attention is all you need' showed that this ONE primitive — Q/K/V dot-product + softmax — is sufficient to replace recurrence AND convolution for sequence modeling. The receptive field is all-pairs; the operation is parallel; the gradients flow everywhere.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Answering 'Q, K, V are all the same thing — just the "
                "input projected through three matrices'. They ARE "
                "three projections of the same input in SELF-attention, "
                "but the three W matrices are learned INDEPENDENTLY and "
                "encode three semantically distinct roles. In CROSS-"
                "attention Q comes from a different sequence than K/V "
                "entirely."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://distill.pub/2016/augmented-rnns/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q03",
            "notebook": _NB_FILENAME,
            "section": "Causal Masking — Implementation Details",
            "difficulty": "core",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["causal-mask", "autoregressive", "softmax", "inference"],
            "question": (
                "Walk through the EXACT implementation of causal masking. "
                "Where in the pipeline do you apply the mask, what do "
                "you fill masked positions with, and what goes wrong if "
                "you apply the mask AFTER softmax instead of before?"
            ),
            "answer_key_points": [
                "Where: in `softmax(Q·Kᵀ/√d + M)·V` the mask `M` is added to the pre-softmax logit matrix. Positions (i, j) where i < j (j is strictly in the future of i) get `M[i,j] = -∞`; positions i ≥ j get `M[i,j] = 0` (no change).",
                "Concrete MLX/Torch idiom: `scores = mx.where(mask == 0, mx.array(-1e9, dtype=scores.dtype), scores)`. The sentinel `-1e9` rather than literal `-∞` sidesteps NaN issues from `exp(-∞) · 0` in corner cases (e.g. when an entire row is masked) — every production framework uses a large-negative-finite value.",
                "Pre-softmax is REQUIRED because softmax normalizes its input: `softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`. Setting `x_masked = -∞` makes `exp(x_masked) = 0`, which removes the masked entry from BOTH the numerator AND the denominator — the remaining entries form a valid probability distribution summing to 1.",
                "Apply AFTER softmax instead: the softmax has already summed UNMASKED+MASKED entries into the denominator. Zero-ing out the masked entries after the fact leaves the denominator too large — the unmasked entries now sum to < 1. The 'attention weights' no longer form a probability distribution; downstream `weights · V` is under-weighted and the model silently loses ~half of its attention signal at early positions.",
                "Symptom at inference time: an 'after-softmax' mask doesn't cause NaN or crash — the model just generates plausible-looking but degraded text, because attention weights are uniformly damped. This is one of the hardest-to-catch causal-attention bugs.",
                "Training-time consequence of a MISSING causal mask: the model cheats — at each position it sees the full context including future tokens. Training loss drops to near-zero (overfitting), but at inference time (no future tokens available) outputs are garbage. Classic 'train-test mismatch' bug.",
                "Production concern: at inference with KV cache, the mask is RECTANGULAR not square — a single new query attends to T existing cached keys. MLX-LM / vLLM generate the mask on-demand with the correct rectangular shape; getting this wrong (using a square mask of size T when Q has only 1 row) wastes compute but still produces correct output.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Filling the mask with `0.0` (instead of `-∞` or a "
                "large-negative finite) before softmax. Zero is "
                "ADDITIVE neutrality — it means 'no change to this "
                "score' — so a mask of zeros is a no-op. This is the "
                "single most common first-time causal-mask bug."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://peterbloem.nl/blog/transformers",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q04",
            "notebook": _NB_FILENAME,
            "section": "Attention Memory — O(B·H·T²)",
            "difficulty": "core",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["memory", "complexity", "long-context", "attention-matrix"],
            "question": (
                "Derive the working-set memory formula for the "
                "attention matrix: `O(B · H · T²)` elements. At "
                "B=1, H=32, T=8192, bf16, how many bytes is that? "
                "What's the pathology that kicks in past ~8k context "
                "and what does FlashAttention change?"
            ),
            "answer_key_points": [
                "Derivation: the raw attention scores matrix `Q·Kᵀ` has shape (B, H, T, T). Each head computes its own T × T similarity matrix. Total elements: `B · H · T · T = B · H · T²`. The softmax output has the same shape. Both are materialized in HBM by default.",
                "Concrete bytes at B=1, H=32, T=8192, bf16: `1 · 32 · 8192² · 2 = 4,294,967,296` bytes = 4.29 GiB. PER LAYER. At 32 layers that's 137 GiB — larger than the weights of most open models.",
                "Pathology: memory GROWS QUADRATICALLY with T. Double the context from 8k → 16k and attention-matrix memory QUADRUPLES (17 GiB/layer). This is why decoder-only models historically capped at 2k-8k context — it's not compute, it's the attention-matrix materialization.",
                "FlashAttention (Dao et al., 2022) changes: it NEVER materializes the full T × T attention matrix in HBM. Instead it tiles Q, K, V into SRAM-sized blocks, computes softmax online using the online-softmax recurrence (Milakov & Gimelshein 2018), and streams blocks through SRAM. Working set becomes O(B · H · T · d_head) — LINEAR in T, not quadratic.",
                "FlashAttention is NOT an approximation. The final output is bit-wise identical (up to float associativity) to standard attention. The win is PURELY memory-access pattern: HBM reads drop from O(T²) to O(T), which is why it's 2-4× faster in wall-clock despite having the same FLOP count.",
                "2024 frontier: FlashAttention-3 (2024) adds FP8 + warp specialization + async execution on Hopper GPUs. vLLM, SGLang, TensorRT-LLM all ship FA2/FA3. MLX has a native `mx.fast.scaled_dot_product_attention` that plays the same role on Apple Silicon.",
                "Training vs inference: in TRAINING we also need the T² attention matrix for the backward pass (gradients flow through the softmax). FlashAttention recomputes it during the backward from SRAM rather than caching in HBM — trades FLOPs for memory. At INFERENCE with KV cache the 'attention matrix' is rectangular (1 × T for decode) — the quadratic issue is the PREFILL phase only.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Quoting the complexity as O(T²) and stopping there — "
                "forgetting the `B · H` multipliers. At H=32 heads and "
                "B=4 batch, the attention matrix is 128× the size of a "
                "single (T, T) matrix. Interviewers test this by asking "
                "'what actually fits on an 80 GB A100 at 64k context?'"
            ),
            "references": [
                "https://arxiv.org/abs/2205.14135",
                "https://arxiv.org/abs/2307.08691",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q05",
            "notebook": _NB_FILENAME,
            "section": "Cross-Attention vs Self-Attention",
            "difficulty": "stretch",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["cross-attention", "self-attention", "encoder-decoder"],
            "question": (
                "Distinguish self-attention from cross-attention. "
                "Where does each one get its Q, K, V from? Why did "
                "decoder-only models (GPT) eliminate cross-attention "
                "even though the original Transformer paper used it "
                "heavily?"
            ),
            "answer_key_points": [
                "Self-attention: Q, K, V all come from the SAME sequence. `Q = X·W_Q, K = X·W_K, V = X·W_V` with the same `X`. Every token attends to every other token in its own sequence (subject to mask).",
                "Cross-attention: Q comes from one sequence (usually the decoder's current hidden state), K and V come from a DIFFERENT sequence (usually the encoder output). `Q = X_dec · W_Q, K = X_enc · W_K, V = X_enc · W_V`. The decoder token asks a query against the encoder's key-value database.",
                "The original Transformer (Vaswani 2017) was encoder-decoder for translation: English sentence → encoder self-attention → encoder output. Target French token → decoder self-attention (over generated prefix) → cross-attention over the encoder output → next French token. Three attention blocks per decoder layer.",
                "Why decoder-only models killed cross-attention: (a) it's a SEPARATE block with SEPARATE K/V projections, doubling parameter count per layer; (b) it requires a two-stage pipeline (encode first, then decode) that's awkward for streaming / autoregressive serving; (c) the 2019+ wave of research (GPT-2) showed that a big enough decoder-only model can do every 'encoder-decoder' task by PREPENDING the source sequence as a prompt — self-attention over the full prompt + response reproduces cross-attention's effect.",
                "Where cross-attention is still alive in 2024–2025: (a) multimodal models — image/audio encoders produce a small set of key-value tokens that the text decoder cross-attends to (Flamingo, Llava-Next, Whisper); (b) retrieval-augmented generation with a separate retriever-produced KV source (RETRO, earlier Atlas); (c) speculative-decoding drafters that cross-attend to the target model's hidden states; (d) encoder-decoder variants (T5, Bart, Whisper) are still used for translation / ASR.",
                "Cost note: in cross-attention the K/V are precomputed ONCE from the encoder (or vision encoder) and REUSED across every decoder step — it looks expensive but the amortized cost per generated token is low. That's what makes multimodal decoders practical.",
                "Implementation in production: vLLM supports cross-attention via encoder-decoder runner (used for Whisper); SGLang handles it via its multi-modal path; MLX-LM supports encoder-decoder models via the flash cross-attention primitive. Self-attention remains the hot path.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'cross-attention is just self-attention over a "
                "concatenated sequence'. They're different ops — in "
                "cross-attention Q and K come from DIFFERENT projection "
                "matrices (W_Q_dec vs W_K_enc) AND from different input "
                "sequences. Concatenate-then-self-attend would share "
                "the projections and is not an equivalent computation."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/2204.14198",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q06",
            "notebook": _NB_FILENAME,
            "section": "Scaled-Dot-Product vs Additive Attention",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["sdpa", "additive-attention", "bahdanau", "luong"],
            "question": (
                "Compare scaled-dot-product attention to additive "
                "(Bahdanau) attention. Write the score formula for "
                "each, compare parameter counts, and explain why "
                "dot-product-plus-scaling won even though Bahdanau "
                "preceded it and was theoretically no worse."
            ),
            "answer_key_points": [
                "Additive / Bahdanau (2014): `score(q, k) = vᵀ · tanh(W_q·q + W_k·k)`. Explicit trainable MLP computes the similarity. Parameters: W_q (d×d), W_k (d×d), v (d) — 2·d² + d extra params PER ATTENTION HEAD.",
                "Luong multiplicative (2015): `score(q, k) = qᵀ · W · k` — dot product with a trainable bilinear form. Cheaper than Bahdanau but still has d² extra params. Luong also proposed `score = qᵀ·k` (plain dot) but with no scaling it's unstable at large d.",
                "Scaled dot-product / Vaswani (2017): `score(q, k) = (qᵀ·k) / √d_head`. ZERO extra parameters beyond what W_Q/W_K already supply; the scaling factor is a deterministic constant. This is the only one with both O(1) per-pair compute AND zero per-head parameters.",
                "Parameter count winner: SDPA — 0 extra params per head. Bahdanau adds ~2·d_head² per head. At d_head=128, H=32 heads, 32 layers that's 32·32·2·128² = ~33M parameters Bahdanau would cost just for attention-score computation, on top of W_Q/W_K/W_V/W_O.",
                "Parallelism winner: SDPA — it's ONE matmul `Q·Kᵀ` over (T, T) output. Bahdanau requires a broadcast-add `W_q·q + W_k·k` (T, T, d) and then a `tanh` and then a `vᵀ` reduction — THREE separate ops with intermediate tensors of shape (T, T, d). SDPA's single matmul is what plays cleanly to modern GEMM hardware (Tensor Cores, Apple AMX).",
                "Why SDPA won historically: on GPUs with big matmul units Bahdanau's three-stage pipeline is memory-bound, while SDPA is compute-bound — factor ~10× throughput advantage at d_head=64. Once the 2017 paper showed that with √d_head scaling the QUALITY is indistinguishable, the engineering case was overwhelming.",
                "What Bahdanau still wins at: differentiability through NON-DIFFERENTIABLE keys. If you want keys to be discrete structures (e.g. typed database entries) and learn the similarity function end-to-end, a trainable MLP score function is more expressive than a fixed dot product. This is why some structured-prediction / graph-attention variants still use additive scoring.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Claiming 'SDPA is faster because dot products are "
                "cheaper than MLPs'. The real win is that SDPA is ONE "
                "big matmul — which maps directly to the "
                "matmul-accelerator hardware — while additive "
                "attention's three-step pipeline generates "
                "intermediate tensors that dominate the memory-access "
                "budget. FLOP count is not the bottleneck."
            ),
            "references": [
                "https://arxiv.org/abs/1409.0473",
                "https://arxiv.org/abs/1508.04025",
                "https://arxiv.org/abs/1706.03762",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb05-q07",
            "notebook": _NB_FILENAME,
            "section": "Attention FLOPs & Arithmetic Intensity",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["flops", "arithmetic-intensity", "flash-attention", "roofline"],
            "question": (
                "Derive the FLOP count for one layer of attention: "
                "`O(B · H · T² · d_head)`. Compute the arithmetic "
                "intensity (FLOPs / byte of data moved through HBM) at "
                "T=8192, d_head=128 and explain why vanilla attention "
                "is memory-bandwidth-bound rather than compute-bound "
                "on modern GPUs. This is the roofline argument behind "
                "FlashAttention."
            ),
            "answer_key_points": [
                "FLOP derivation: `Q·Kᵀ` is a (T, d_head) × (d_head, T) matmul — 2 · T² · d_head FLOPs per head. Then `softmax(...)·V` is a (T, T) × (T, d_head) matmul — another 2 · T² · d_head FLOPs. Per head per layer: 4 · T² · d_head. Across H heads and batch B: `4 · B · H · T² · d_head` FLOPs.",
                "Concrete at B=1, H=32, T=8192, d_head=128: 4 · 1 · 32 · 8192² · 128 ≈ 1.1 TFLOPs per attention layer. Per forward (32 layers): 35 TFLOPs from attention alone. Modern LLMs are attention-dominated past 16k context.",
                "HBM memory traffic (vanilla): load Q (B·H·T·d_head bf16), K (same), V (same); write full scores matrix (B·H·T·T bf16); read back, write softmax output (B·H·T·T); read for V-matmul; write final output (B·H·T·d_head). Dominant term: two reads and two writes of the T² matrix — 4·B·H·T² bytes in bf16.",
                "Arithmetic intensity: FLOPs / byte = (4 · B · H · T² · d_head) / (4 · B · H · T² · 2) = d_head / 2. At d_head=128 that's 64 FLOPs/byte. Ridge point of an H100 is ~1000 FLOPs/byte (800 TFLOPS bf16 / 3 TB/s HBM). 64 ≪ 1000 ⇒ attention is memory-bandwidth-bound at this d_head.",
                "FlashAttention's insight: never MATERIALIZE the T² matrix in HBM. Tile into SRAM-sized blocks and keep partial softmax statistics. HBM traffic becomes O(B·H·T·d_head) — linear in T. New arithmetic intensity: (4·B·H·T²·d_head) / (B·H·T·d_head · constant) = O(T). At T=8192 this is O(8192) — far above the ridge point. Attention becomes compute-bound.",
                "Measured consequences: FlashAttention-2 on H100 runs attention at ~70% of peak bf16 compute; vanilla attention gets ~10% because it's waiting on HBM. The 2-4× wall-clock speedup FlashAttention reports is NOT a FLOP reduction — it's a memory-access reduction.",
                "Why this matters for serving: at large batch (B) the ridge-point gap shrinks (more FLOPs per byte moved), so even vanilla attention becomes compute-bound at B=64+. FlashAttention's win is biggest at B=1 — single-request serving, exactly the latency-sensitive path. This is why vLLM/SGLang/TRT-LLM all hard-dep on FA2/FA3.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Answering 'attention is compute-bound because it's "
                "O(T²)'. It IS O(T²), but the arithmetic intensity "
                "(64 FLOP/byte at d_head=128) is below the hardware's "
                "ridge point (~1000 FLOP/byte for H100 bf16), so the "
                "bottleneck is HBM bandwidth, not TFLOPS. Roofline "
                "analysis distinguishes 'compute cost' from 'compute-"
                "bound'."
            ),
            "references": [
                "https://arxiv.org/abs/2205.14135",
                "https://horace.io/brrr_intro.html",
                "https://github.com/Dao-AILab/flash-attention",
            ],
            "added_in": added_in,
        },
    ]


# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_sdpa(records: list[dict]) -> list[dict]:
    """Block for the end of Section 1 (scaled_dot_product_attention).

    Contents: q01 (√d_head scaling derivation), q02 (attention as
    weighted retrieval), whiteboard-A (implement SDPA from scratch +
    verify against `mx.fast.scaled_dot_product_attention` numerically),
    📐-1 (attention FLOPs O(B·H·T²·d_head), measure latency & GFLOPS).
    """
    q01, q02 = records[0], records[1]

    # --- Whiteboard A — implement SDPA, verify vs mx.fast.SDPA. ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title=(
            "Scaled dot-product attention from scratch — verify vs "
            "mx.fast.scaled_dot_product_attention"
        ),
        prompt=(
            "Implement `sdpa(Q, K, V, mask=None)` that computes "
            "`softmax((Q·Kᵀ)/√d_head + mask)·V` directly in MLX. "
            "Then verify your implementation matches MLX's fused "
            "`mx.fast.scaled_dot_product_attention` to within float32 "
            "tolerance on random inputs at (B=2, H=8, T=64, d_head=32)."
        ),
        constraints=[
            "Use MLX throughout — no numpy, torch, jax. Shapes must match the MLX "
            "fused kernel's expectations: Q, K, V are (B, H, T, d_head).",
            "Apply the √d_head scaling BEFORE softmax. Your matmul should be "
            "`Q @ K.transpose(..., -2, -1)` producing (B, H, T, T) scores.",
            "Support an optional additive mask of shape broadcastable to "
            "(B, H, T, T). Passing `mask=None` means no masking.",
            "Assert your output matches `mx.fast.scaled_dot_product_attention("
            "Q, K, V, scale=1/sqrt(d_head), mask=None)` with `max |diff| < 1e-4` "
            "at float32 — fused kernels have different accumulation order.",
            "Use `mx.eval` on both outputs before taking the numeric diff.",
        ],
        complexity=(
            "Compute: O(B · H · T² · d_head). Memory (materialized "
            "scores matrix): O(B · H · T²) — this is the term FlashAttention "
            "eliminates."
        ),
        solution_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "\n"
            "def sdpa(Q: mx.array, K: mx.array, V: mx.array,\n"
            "         mask: mx.array | None = None) -> mx.array:\n"
            "    \"\"\"Scaled dot-product attention: softmax((Q·Kᵀ)/√d + mask)·V.\n"
            "\n"
            "    Q, K, V all shape (B, H, T, d_head). Returns (B, H, T, d_head).\n"
            "    \"\"\"\n"
            "    d_head = Q.shape[-1]\n"
            "    scale = 1.0 / math.sqrt(d_head)\n"
            "    # Q·Kᵀ — matmul on last two dims.\n"
            "    scores = (Q @ K.swapaxes(-2, -1)) * scale  # (B, H, T, T)\n"
            "    if mask is not None:\n"
            "        scores = scores + mask\n"
            "    weights = mx.softmax(scores, axis=-1)\n"
            "    return weights @ V  # (B, H, T, d_head)\n"
            "\n"
            "# Random Q/K/V at a production-ish shape. Names are prefixed with\n"
            "# an underscore so they don't collide with the notebook's existing\n"
            "# Q, K, V (3-D shape) defined in Section 1.\n"
            "_B, _H, _T, _d_head = 2, 8, 64, 32\n"
            "mx.random.seed(0)\n"
            "_Qwb = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "_Kwb = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "_Vwb = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "mx.eval(_Qwb, _Kwb, _Vwb)\n"
            "\n"
            "# Our reference implementation.\n"
            "_out_ours = sdpa(_Qwb, _Kwb, _Vwb, mask=None)\n"
            "\n"
            "# MLX's fused kernel — same math, different accumulation / layout.\n"
            "_scale = 1.0 / math.sqrt(_d_head)\n"
            "_out_fast = mx.fast.scaled_dot_product_attention(\n"
            "    _Qwb, _Kwb, _Vwb, scale=_scale, mask=None\n"
            ")\n"
            "mx.eval(_out_ours, _out_fast)\n"
            "\n"
            "# Numeric agreement: fused kernels reorder the sum, so 1e-5-ish drift is\n"
            "# expected at float32. A 1e-4 tolerance separates 'same math' from\n"
            "# 'implementation bug'.\n"
            "_diff = float(mx.max(mx.abs(_out_ours - _out_fast)).item())\n"
            "assert _diff < 1e-4, (\n"
            "    f\"sdpa disagrees with mx.fast.sdpa by {_diff:.4e} (>1e-4)\"\n"
            ")\n"
            "\n"
            "# Sanity: both outputs have the expected shape.\n"
            "assert _out_ours.shape == (_B, _H, _T, _d_head)\n"
            "assert _out_fast.shape == (_B, _H, _T, _d_head)\n"
            "\n"
            "# Attention weights form a probability distribution: each row sums to 1.\n"
            "_weights = mx.softmax((_Qwb @ _Kwb.swapaxes(-2, -1)) * _scale, axis=-1)\n"
            "_row_sums = mx.sum(_weights, axis=-1)\n"
            "mx.eval(_row_sums)\n"
            "_row_err = float(mx.max(mx.abs(_row_sums - 1.0)).item())\n"
            "assert _row_err < 1e-5, f\"softmax rows don't sum to 1: {_row_err:.4e}\"\n"
            "\n"
            "print(f\"✅ sdpa(Q, K, V) shape: {_out_ours.shape}\")\n"
            "print(f\"✅ max |ours - mx.fast.sdpa| = {_diff:.4e}  (< 1e-4)\")\n"
            "print(f\"✅ softmax rows sum to 1 within {_row_err:.4e}\")\n"
        ),
    )

    # --- 📐-1 Complexity cell — attention FLOPs O(B·H·T²·d_head). ---
    complexity = T.complexity_analysis_cell(
        op="Attention compute — O(B · H · T² · d_head) FLOPs per layer",
        flops=(
            "4 · B · H · T² · d_head per layer. Two matmuls: Q·Kᵀ "
            "(2·B·H·T²·d_head) and softmax-output·V (same). At B=1, "
            "H=32, T=2048, d_head=128 that's ~2.1 GFLOPs/layer"
        ),
        memory=(
            "Working set includes the (B, H, T, T) scores tensor — "
            "O(B·H·T²) elements. At B=1, H=32, T=2048, bf16 that's "
            "256 MiB; at T=8192 it's 4.3 GiB — the quadratic pathology"
        ),
        latency_mlx=(
            "M4 Pro, mx.fast.scaled_dot_product_attention, fp32, "
            "(B=1, H=32, T=2048, d_head=128): ~12–25 ms/call. Measured "
            "below against the naive (materialized-scores) path"
        ),
        scaling=(
            "Compute is quadratic in T. Doubling context from 8k to 16k "
            "QUADRUPLES attention FLOPs and MATERIALIZED memory. "
            "FlashAttention keeps the compute (still O(T²·d_head)) but "
            "makes memory LINEAR in T — see NB12 for the full story."
        ),
    )

    bench_src = (
        "# Benchmark: attention FLOPs O(B·H·T²·d_head) across T\n"
        "# Measures mx.fast.SDPA latency at (B=1, H=32, d_head=128) bf16\n"
        "# across T ∈ {256, 512, 1024, 2048} and reports derived GFLOP/s.\n"
        "# Underscore-prefixed names avoid colliding with the notebook's\n"
        "# existing Q, K, V, B, T, d_model globals from Section 1.\n"
        "import math\n"
        "import time\n"
        "import mlx.core as mx\n"
        "\n"
        "_B, _H, _d_head = 1, 32, 128\n"
        "_scale = 1.0 / math.sqrt(_d_head)\n"
        "\n"
        "def bench_sdpa(T: int, n_iter: int = 10, n_warmup: int = 3) -> tuple[float, float]:\n"
        "    \"\"\"Return (ms_per_call, measured_GFLOPS) for (B, H, T, d_head) bf16.\"\"\"\n"
        "    mx.random.seed(0)\n"
        "    _Q = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    _K = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    _V = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    mx.eval(_Q, _K, _V)\n"
        "\n"
        "    # Warmup — Requirement 5.3.\n"
        "    for _ in range(n_warmup):\n"
        "        _y = mx.fast.scaled_dot_product_attention(_Q, _K, _V, scale=_scale, mask=None)\n"
        "        mx.eval(_y)\n"
        "\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(n_iter):\n"
        "        _y = mx.fast.scaled_dot_product_attention(_Q, _K, _V, scale=_scale, mask=None)\n"
        "        mx.eval(_y)\n"
        "    dt_ms = (time.perf_counter() - t0) / n_iter * 1000.0\n"
        "\n"
        "    # Analytic FLOPs: 4 · B · H · T² · d_head (two matmuls).\n"
        "    flops = 4.0 * _B * _H * T * T * _d_head\n"
        "    gflops = flops / (dt_ms * 1e-3) / 1e9\n"
        "    return dt_ms, gflops\n"
        "\n"
        "print(f\"SDPA benchmark at B={_B}, H={_H}, d_head={_d_head}, bf16:\")\n"
        "print(f\"{'T':>6} | {'ms/call':>10} | {'GFLOP/s':>10} | {'T² ratio':>10}\")\n"
        "print(\"-\" * 48)\n"
        "_base_t = None\n"
        "_base_ms = None\n"
        "for _T in (256, 512, 1024, 2048):\n"
        "    _dt_ms, _gflops = bench_sdpa(_T)\n"
        "    if _base_t is None:\n"
        "        _base_t, _base_ms = _T, _dt_ms\n"
        "        _ratio = 1.0\n"
        "    else:\n"
        "        # Analytic expectation: doubling T quadruples FLOPs ⇒ ~4× ms.\n"
        "        _ratio = _dt_ms / _base_ms\n"
        "    print(f\"{_T:>6} | {_dt_ms:>10.3f} | {_gflops:>10.1f} | {_ratio:>10.2f}×\")\n"
        "\n"
        "# The 'T² ratio' column validates the O(T²) compute scaling:\n"
        "# going from T=256 → T=2048 (8×) should give ~64× (8²) — if kernels\n"
        "# are memory-bandwidth-bound the ratio is sub-quadratic; if they're\n"
        "# compute-bound it tracks the analytic T² curve.\n"
        "\n"
        "# Final correctness assertion: results are materialized via mx.eval.\n"
        "_ = bench_sdpa(512)\n"
        "print(\"\\n💡 FLOPs grow as O(T²·d_head); doubling context 4×es compute.\")\n"
    )
    bench = {"cell_type": "code", "source": bench_src}

    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
    ]


def _block_causal(records: list[dict]) -> list[dict]:
    """Block for the end of Section 3 (Causal Masking).

    Contents: q03 (causal masking), q04 (attention memory O(B·H·T²)),
    q05 (cross- vs self-attention), q06 (SDPA vs additive), q07
    (attention FLOPs & arithmetic intensity), whiteboard-B (implement
    & verify a causal mask: assert post-softmax upper triangle is exactly
    0), 📐-2 (attention memory O(B·H·T²) with T ∈ {512, 1024, 2048,
    4096} measurements), 🏭 production (vLLM/SGLang/TRT-LLM/MLX-LM all
    on FlashAttention-family kernels + implicit causal mask), 🔭
    frontier (FlashAttention-3, DeepSeek-V3 MLA, streaming attention in
    o1), 🛠️ debugging (softmax overflow without scaling, forgot causal
    mask at inference, shape mismatch between Q and K).
    """
    q03, q04, q05, q06, q07 = (
        records[2],
        records[3],
        records[4],
        records[5],
        records[6],
    )

    # --- Whiteboard B — causal mask + assert upper-triangle is zero ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Causal mask from scratch — assert post-softmax upper triangle is 0",
        prompt=(
            "Build a causal mask of shape (T, T) that sets future "
            "positions to `-∞` BEFORE softmax. Apply it inside an "
            "SDPA call, then prove it works by asserting that every "
            "element of the post-softmax attention weight matrix "
            "ABOVE the diagonal is EXACTLY 0 (not just small)."
        ),
        constraints=[
            "Use MLX throughout — no numpy. Build the mask with `mx.triu` "
            "(upper-triangular = future positions) or equivalently `mx.tril`.",
            "Fill future positions with `-1e9` (NOT `-math.inf` — mixing inf "
            "with bf16/fp16 produces NaN on some kernels). Every production "
            "framework uses a large-negative-finite sentinel.",
            "Compute softmax of the masked logits and assert `weights[i, j] == 0` "
            "EXACTLY for every (i, j) with j > i. `exp(-1e9) < 1e-308` — "
            "underflows to 0 in both fp32 and bf16. Assertion is on bit-exact 0.",
            "Additionally assert each ROW sums to 1.0 within 1e-5 — the mask "
            "preserves the probability-distribution property of softmax.",
            "Use `mx.eval` on weights before taking the assertions.",
        ],
        complexity=(
            "Mask build: O(T²) one-time. Apply: O(B · H · T²) elementwise "
            "add to scores, amortized into the existing scores matmul in "
            "production kernels."
        ),
        solution_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "\n"
            "def causal_mask(T: int, dtype=mx.float32) -> mx.array:\n"
            "    \"\"\"Return an additive causal mask of shape (T, T).\n"
            "\n"
            "    mask[i, j] = 0   if j <= i (past or present; allowed)\n"
            "    mask[i, j] = -1e9 if j > i (future; blocked).\n"
            "\n"
            "    We use -1e9 as sentinel rather than -inf so bf16 kernels\n"
            "    don't produce NaN in edge cases (e.g. entirely-masked rows).\n"
            "    \"\"\"\n"
            "    # mx.triu(..., k=1) keeps entries strictly above the diagonal.\n"
            "    upper = mx.triu(mx.ones((T, T), dtype=dtype), k=1)\n"
            "    return upper * mx.array(-1e9, dtype=dtype)\n"
            "\n"
            "def sdpa_causal(Q: mx.array, K: mx.array, V: mx.array) -> tuple[mx.array, mx.array]:\n"
            "    \"\"\"SDPA with an implicit causal mask. Returns (output, weights).\n"
            "\n"
            "    Q, K, V shape (B, H, T, d_head). Weights shape (B, H, T, T).\n"
            "    \"\"\"\n"
            "    d_head = Q.shape[-1]\n"
            "    T = Q.shape[-2]\n"
            "    scale = 1.0 / math.sqrt(d_head)\n"
            "    scores = (Q @ K.swapaxes(-2, -1)) * scale  # (B, H, T, T)\n"
            "    scores = scores + causal_mask(T, dtype=scores.dtype)\n"
            "    weights = mx.softmax(scores, axis=-1)\n"
            "    return weights @ V, weights\n"
            "\n"
            "# Random (B=1, H=4, T=8, d_head=16) — small enough to inspect by eye.\n"
            "# Names prefixed with underscore to avoid colliding with the notebook's\n"
            "# existing global Q, K, V (3-D, used by Section 1's scaled_dot_product_attention).\n"
            "_B, _H, _T, _d_head = 1, 4, 8, 16\n"
            "mx.random.seed(0)\n"
            "_Q = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "_K = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "_V = mx.random.normal(shape=(_B, _H, _T, _d_head))\n"
            "mx.eval(_Q, _K, _V)\n"
            "\n"
            "_out, _weights = sdpa_causal(_Q, _K, _V)\n"
            "mx.eval(_out, _weights)\n"
            "\n"
            "# Property 1: every STRICTLY UPPER-TRIANGULAR entry of weights is EXACTLY 0.\n"
            "# exp(-1e9) underflows to 0 in both fp32 and bf16 — this is bit-exact zero.\n"
            "_weights_list = _weights.tolist()\n"
            "_max_future = 0.0\n"
            "for _b in range(_B):\n"
            "    for _h in range(_H):\n"
            "        for _i in range(_T):\n"
            "            for _j in range(_i + 1, _T):\n"
            "                _v = _weights_list[_b][_h][_i][_j]\n"
            "                if _v > _max_future:\n"
            "                    _max_future = _v\n"
            "assert _max_future == 0.0, (\n"
            "    f\"causal mask leaked: max upper-triangle weight = {_max_future}\"\n"
            ")\n"
            "\n"
            "# Property 2: each row of weights sums to 1.0 (valid probability distribution).\n"
            "_row_sums = mx.sum(_weights, axis=-1)\n"
            "mx.eval(_row_sums)\n"
            "_row_err = float(mx.max(mx.abs(_row_sums - 1.0)).item())\n"
            "assert _row_err < 1e-5, (\n"
            "    f\"row sums deviate from 1.0 by {_row_err:.4e} — mask broke softmax\"\n"
            ")\n"
            "\n"
            "# Property 3: row 0 attends ONLY to position 0 (the first token sees only itself).\n"
            "_row0 = _weights_list[0][0][0]\n"
            "assert _row0[0] == 1.0, f\"row 0 should be [1, 0, 0, ..., 0], got {_row0}\"\n"
            "assert all(_v == 0.0 for _v in _row0[1:])\n"
            "\n"
            "print(f\"✅ weights shape: {_weights.shape}\")\n"
            "print(f\"✅ max upper-triangle weight (future-leak): {_max_future}\")\n"
            "print(f\"✅ row sums within {_row_err:.4e} of 1.0\")\n"
            "print(f\"✅ row 0 = {_row0[:4]}...  (attends only to position 0)\")\n"
        ),
    )

    # --- 📐-2 Complexity cell — attention MEMORY O(B·H·T²) ---
    complexity = T.complexity_analysis_cell(
        op="Attention memory — O(B · H · T²) materialized scores matrix",
        flops=(
            "Not the focus here; see the 📐-1 cell above for the FLOP "
            "count. Memory work is dominated by HBM traffic, not arithmetic"
        ),
        memory=(
            "`B · H · T · T · bytes_per_elem` for the scores matrix "
            "(+ same for softmax output in the backward pass). At B=1, "
            "H=32, T=8192, bf16 that's 4.3 GiB PER LAYER — larger than "
            "most model weights"
        ),
        latency_mlx=(
            "M4 Pro, (B=1, H=8, d_head=64) bf16 via mx.fast.SDPA: T=512 "
            "→ ~1 ms, T=2048 → ~15 ms, T=4096 → ~60 ms. Memory measured "
            "via `mx.metal.get_peak_memory()` below"
        ),
        scaling=(
            "Memory grows as O(T²) — doubling context QUADRUPLES "
            "attention-matrix footprint. This is the single wall that "
            "kept decoder-only models at ≤ 8k context until FlashAttention "
            "(2022) showed how to avoid materializing the T²  matrix at all. "
            "Modern kernels keep memory LINEAR in T; vanilla attention does not."
        ),
    )

    mem_bench_src = (
        "# Benchmark: attention memory scales as O(T²) at fixed (B, H, d_head)\n"
        "# Measures peak MLX memory via mx.metal.get_peak_memory() at T ∈ {512,\n"
        "# 1024, 2048, 4096} — should grow approximately 4× per doubling of T.\n"
        "# All module-level names are underscore-prefixed to avoid colliding\n"
        "# with the notebook's existing globals (Q, K, V, B, H, T, d_model).\n"
        "import math\n"
        "import time\n"
        "import mlx.core as mx\n"
        "\n"
        "# Smaller H and d_head here so T=4096 fits comfortably on an 18 GB M4 Pro.\n"
        "_B, _H, _d_head = 1, 8, 64\n"
        "_scale = 1.0 / math.sqrt(_d_head)\n"
        "\n"
        "def bench_mem(T: int) -> tuple[float, float]:\n"
        "    \"\"\"Return (ms_per_call, peak_mib) for one mx.fast.SDPA call at size T.\"\"\"\n"
        "    # Reset MLX's peak-memory counter and GC any cached arrays first.\n"
        "    # Prefer the new mx.reset_peak_memory / mx.get_peak_memory API; fall\n"
        "    # back to the legacy mx.metal.* API for older MLX builds.\n"
        "    _reset = getattr(mx, \"reset_peak_memory\", None) or getattr(\n"
        "        getattr(mx, \"metal\", None), \"reset_peak_memory\", None\n"
        "    )\n"
        "    _get_peak = getattr(mx, \"get_peak_memory\", None) or getattr(\n"
        "        getattr(mx, \"metal\", None), \"get_peak_memory\", None\n"
        "    )\n"
        "    if _reset is not None:\n"
        "        try:\n"
        "            _reset()\n"
        "        except Exception:\n"
        "            pass  # best-effort\n"
        "\n"
        "    mx.random.seed(0)\n"
        "    _Q = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    _K = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    _V = mx.random.normal(shape=(_B, _H, T, _d_head)).astype(mx.bfloat16)\n"
        "    mx.eval(_Q, _K, _V)\n"
        "\n"
        "    # Warmup — Requirement 5.3.\n"
        "    for _ in range(3):\n"
        "        _y = mx.fast.scaled_dot_product_attention(_Q, _K, _V, scale=_scale, mask=None)\n"
        "        mx.eval(_y)\n"
        "\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(5):\n"
        "        _y = mx.fast.scaled_dot_product_attention(_Q, _K, _V, scale=_scale, mask=None)\n"
        "        mx.eval(_y)\n"
        "    dt_ms = (time.perf_counter() - t0) / 5.0 * 1000.0\n"
        "\n"
        "    peak_bytes = _get_peak() if _get_peak is not None else 0\n"
        "    peak_mib = peak_bytes / (1024 * 1024)\n"
        "    return dt_ms, peak_mib\n"
        "\n"
        "print(f\"Attention memory scaling at B={_B}, H={_H}, d_head={_d_head}, bf16:\")\n"
        "print(f\"{'T':>6} | {'ms/call':>10} | {'peak MiB':>12} | {'analytic T² MiB':>16}\")\n"
        "print(\"-\" * 60)\n"
        "for _T in (512, 1024, 2048, 4096):\n"
        "    _dt_ms, _peak_mib = bench_mem(_T)\n"
        "    # Analytic T² term at bf16: B · H · T² · 2 bytes.\n"
        "    _t2_bytes = _B * _H * _T * _T * 2\n"
        "    _t2_mib = _t2_bytes / (1024 * 1024)\n"
        "    print(f\"{_T:>6} | {_dt_ms:>10.2f} | {_peak_mib:>12.1f} | {_t2_mib:>16.2f}\")\n"
        "\n"
        "# Key observation:  mx.fast.SDPA uses a FlashAttention-style path that\n"
        "# does NOT materialize the full T² scores matrix — so the measured peak\n"
        "# memory grows LINEARLY with T rather than quadratically. The 'analytic\n"
        "# T² MiB' column shows what vanilla attention WOULD have allocated for\n"
        "# the scores tensor alone at each T.\n"
        "\n"
        "# Final sanity assertion: the last call actually produced a real array.\n"
        "mx.random.seed(0)\n"
        "_Qs = mx.random.normal(shape=(_B, _H, 128, _d_head)).astype(mx.bfloat16)\n"
        "_out = mx.fast.scaled_dot_product_attention(_Qs, _Qs, _Qs, scale=_scale)\n"
        "mx.eval(_out)\n"
        "assert _out.shape == (_B, _H, 128, _d_head)\n"
        "print(\"\\n💡 FlashAttention keeps memory LINEAR; vanilla attention is O(T²).\")\n"
    )
    mem_bench = {"cell_type": "code", "source": mem_bench_src}

    # --- 🏭 Production cell ---
    production = T.production_context_cell(
        concept="How production inference stacks compute attention",
        vllm=(
            "Uses FlashAttention-2 (via xformers or the flash-attn wheel) "
            "for prefill, PagedAttention for decode. Causal mask is "
            "IMPLICIT in the FA2 kernel — the kernel skips blocks entirely "
            "above the diagonal rather than materializing a T×T mask "
            "tensor. At T=128k the saved memory is ~130 GiB vs vanilla"
        ),
        sglang=(
            "Uses FlashAttention-3 when available (Hopper), FA2 otherwise. "
            "RadixAttention prefix cache exploits the fact that two "
            "requests sharing a prefix have IDENTICAL K/V and causal mask "
            "for the shared portion. Mask is constructed on-demand in the "
            "kernel, never stored"
        ),
        trt_llm=(
            "Emits a fused multi-head-attention kernel (fMHA) with the "
            "causal mask passed as a flag, not a tensor. Generates "
            "specialized kernels per (d_head, causal, alibi) combination "
            "at engine-build time; the mask is baked into control flow. "
            "Supports FP8 attention on H100/H200"
        ),
        mlx_lm=(
            "Routes self-attention through `mx.fast.scaled_dot_product_"
            "attention` which is FlashAttention-family on Metal (shader-"
            "level tiled softmax, no T² materialization). Causal mask is "
            "an additive tensor or a boolean flag depending on the call "
            "site; on UMA the scores tensor would land in system RAM, "
            "making the 'don't materialize T²' win even more material "
            "than on discrete-GPU platforms"
        ),
    )

    # --- 🔭 Frontier cell ---
    frontier = T.frontier_context_cell(
        topic="Attention kernels and long-context attention (2024–2026)",
        papers=[
            (
                "FlashAttention-3: Fast and Accurate Attention with Asynchrony "
                "and Low-Precision (Shah et al.)",
                2024,
                "Three techniques for Hopper GPUs: warp specialization "
                "(producer/consumer warps), interleaved matmul+softmax "
                "(hide softmax latency under matmul), and FP8 attention "
                "with incoherent processing. 1.5-2× faster than FA2 at "
                "bf16, 2.4× at FP8. Ships in vLLM / SGLang / TRT-LLM.",
            ),
            (
                "DeepSeek-V3 Technical Report (DeepSeek)",
                2024,
                "Multi-head Latent Attention (MLA) compresses KV cache "
                "by projecting K and V into a shared low-rank latent "
                "space — ~10× smaller KV cache than GQA at comparable "
                "quality. Pairs with FlashAttention-family kernel at "
                "the top of the stack.",
            ),
            (
                "Tensor-parallel attention in DeepSeek-V3 / LLaMA-3.1 "
                "(Meta, DeepSeek)",
                2024,
                "Shards attention heads across GPUs with explicit "
                "all-reduce collectives at the O-projection boundary. "
                "Attention's per-head independence (no cross-head "
                "dependencies) makes this sharding trivially correct — "
                "the same reason multi-head attention was designed in "
                "the first place.",
            ),
            (
                "o1 / o3 System Cards (OpenAI)",
                2024,
                "Streaming attention during chain-of-thought reasoning: "
                "the attention mask is EXTENDED incrementally as the "
                "model emits reasoning tokens, with the earliest scratch "
                "tokens sometimes EVICTED from the KV cache to stay "
                "within the context budget. Requires precise handling "
                "of the causal-mask position-indexing at eviction "
                "boundaries.",
            ),
            (
                "Native Sparse Attention (NSA) / MoBA-style attention (2025)",
                2025,
                "Learned block-sparse attention: the model chooses at "
                "runtime which K-blocks to attend to, skipping O(T) "
                "per-block rather than O(T²) per-token. Gets "
                "sub-quadratic effective compute while keeping the "
                "full attention API. Active research frontier as of "
                "late 2025.",
            ),
        ],
        current_sota=(
            "As of late 2025 the production default is FlashAttention-3 "
            "(Hopper) or FlashAttention-2 (Ampere and older) for the "
            "kernel itself, GQA / MLA for the head layout, YaRN-scaled "
            "RoPE for positional encoding, and implicit (kernel-level) "
            "causal masking. Active frontiers: (i) FP8 / FP4 attention "
            "(FA3 FP8 is already in vLLM), (ii) native sparse / learned-"
            "block attention (NSA, MoBA), (iii) streaming attention with "
            "KV eviction for reasoning models (o1-style)."
        ),
    )

    # --- 🛠️ Debugging cell ---
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "Three attention bugs — softmax overflow without √d scaling, "
            "missing causal mask at inference, shape mismatch between Q and K"
        ),
        root_causes=[
            "Softmax saturation without `/ √d_head` scaling: at d_head=128 "
            "the scores have std ≈ √128 ≈ 11.3. Softmax collapses onto the "
            "argmax entry — output is ~one-hot, gradients near 0, model "
            "won't train. Fix: ALWAYS divide by √d_head (or `d_head_scale` "
            "= 1/√d_head) BEFORE softmax. Diagnostic: print "
            "`scores.std()` at the first layer — if it exceeds ~3, you "
            "missed the scaling.",
            "Causal mask forgotten at inference (present at training): "
            "the model was trained with a causal mask and learned to not "
            "look at future tokens; at inference you forgot the mask and "
            "the model sees 'future' positions (which at decode time "
            "happen to be zero-padded positions beyond the current "
            "length). Symptom: generation 'works' but quality is "
            "materially worse than training loss would predict. Fix: use "
            "ONE `attention(Q, K, V, causal=True)` API that threads the "
            "mask through every call site; never pass `causal=False` "
            "unless you're doing encoder-style bidirectional attention.",
            "Shape mismatch between Q and K: Q is (B, T_q, H, d_head), "
            "K is (B, T_k, H, d_head) — T_q and T_k can differ (cross-"
            "attention, KV-cache decode where T_q=1 and T_k=cached). "
            "Common bug: the attention kernel assumes T_q == T_k, "
            "broadcasting incorrectly or indexing out of bounds. Fix: "
            "always use `Q.shape[-2]` and `K.shape[-2]` as independent "
            "parameters; never hardcode `T = Q.shape[-2]` and use it "
            "for the K dimension.",
        ],
        diagnostic_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "\n"
            "# All module-level names prefixed with underscore so this cell doesn't\n"
            "# leak Q, K, V, T, H, d_head over the notebook's pre-existing Section 1\n"
            "# / Section 2 globals.\n"
            "\n"
            "# -- Symptom 1: softmax saturation without √d scaling --------\n"
            "# At d_head=128 the raw dot-product scores have std ~ √128;\n"
            "# softmax becomes near-one-hot and attention weights lose\n"
            "# information. Demonstrate by measuring the ENTROPY of the\n"
            "# attention weights with vs without the scaling factor.\n"
            "_d_head = 128\n"
            "_T = 32\n"
            "mx.random.seed(0)\n"
            "_q = mx.random.normal(shape=(_T, _d_head))\n"
            "_k = mx.random.normal(shape=(_T, _d_head))\n"
            "mx.eval(_q, _k)\n"
            "\n"
            "_scores_unscaled = _q @ _k.swapaxes(-2, -1)\n"
            "_scores_scaled = _scores_unscaled / math.sqrt(_d_head)\n"
            "mx.eval(_scores_unscaled, _scores_scaled)\n"
            "\n"
            "_w_unscaled = mx.softmax(_scores_unscaled, axis=-1)\n"
            "_w_scaled = mx.softmax(_scores_scaled, axis=-1)\n"
            "mx.eval(_w_unscaled, _w_scaled)\n"
            "\n"
            "# Entropy per row: H(w) = -Σ w_i · log(w_i). Uniform → log(T) ≈ 3.47;\n"
            "# one-hot → 0. Scaled softmax should be closer to uniform.\n"
            "def _row_entropy(w: mx.array) -> float:\n"
            "    eps = 1e-12\n"
            "    ent = -mx.sum(w * mx.log(w + eps), axis=-1)\n"
            "    mx.eval(ent)\n"
            "    return float(mx.mean(ent).item())\n"
            "\n"
            "_h_unscaled = _row_entropy(_w_unscaled)\n"
            "_h_scaled = _row_entropy(_w_scaled)\n"
            "print(f\"[1] scores std: unscaled={float(mx.std(_scores_unscaled).item()):.2f}, \"\n"
            "      f\"scaled={float(mx.std(_scores_scaled).item()):.2f}\")\n"
            "print(f\"    mean softmax entropy: unscaled={_h_unscaled:.3f}, \"\n"
            "      f\"scaled={_h_scaled:.3f}  (uniform={math.log(_T):.3f})\")\n"
            "assert _h_scaled > _h_unscaled, (\n"
            "    \"scaled softmax must be more uniform (higher entropy) than unscaled\"\n"
            ")\n"
            "print(\"    → symptom: without /√d, softmax saturates; gradients vanish.\")\n"
            "\n"
            "# -- Symptom 2: forgot causal mask at inference -----------------\n"
            "# Compare attention output WITH and WITHOUT the causal mask at a\n"
            "# sequence length where future positions carry signal.\n"
            "_T2 = 16\n"
            "_H2 = 2\n"
            "_d_head2 = 32\n"
            "_scale2 = 1.0 / math.sqrt(_d_head2)\n"
            "mx.random.seed(1)\n"
            "_Q2 = mx.random.normal(shape=(1, _H2, _T2, _d_head2))\n"
            "_K2 = mx.random.normal(shape=(1, _H2, _T2, _d_head2))\n"
            "_V2 = mx.random.normal(shape=(1, _H2, _T2, _d_head2))\n"
            "mx.eval(_Q2, _K2, _V2)\n"
            "\n"
            "def _sdpa_naive(Q, K, V, causal: bool):\n"
            "    scores = (Q @ K.swapaxes(-2, -1)) * _scale2\n"
            "    if causal:\n"
            "        mask = mx.triu(mx.ones((_T2, _T2), dtype=scores.dtype), k=1) * (-1e9)\n"
            "        scores = scores + mask\n"
            "    return mx.softmax(scores, axis=-1) @ V\n"
            "\n"
            "_out_masked = _sdpa_naive(_Q2, _K2, _V2, causal=True)\n"
            "_out_open = _sdpa_naive(_Q2, _K2, _V2, causal=False)\n"
            "mx.eval(_out_masked, _out_open)\n"
            "\n"
            "# Row 0 is where they MUST differ most: with the mask, token 0 sees\n"
            "# only itself; without, token 0 sees everything — completely different\n"
            "# output for row 0.\n"
            "_row_diff = float(\n"
            "    mx.mean(mx.abs(_out_masked[0, 0, 0] - _out_open[0, 0, 0])).item()\n"
            ")\n"
            "print(f\"[2] mean |out_masked[0] - out_open[0]| at first token: {_row_diff:.4f}\")\n"
            "assert _row_diff > 0.05, (\n"
            "    \"causal mask should materially change the first-token output\"\n"
            ")\n"
            "print(\"    → symptom: model cheats at training, degrades at inference.\")\n"
            "\n"
            "# -- Symptom 3: shape mismatch between Q and K (cross-attention) --\n"
            "# Build Q with T_q != T_k — the classic KV-cache decode shape\n"
            "# where Q has 1 'new' row and K has T_k cached rows.\n"
            "_Tq, _Tk = 1, 32\n"
            "_Q3 = mx.random.normal(shape=(1, _H2, _Tq, _d_head2))\n"
            "_K3 = mx.random.normal(shape=(1, _H2, _Tk, _d_head2))\n"
            "_V3 = mx.random.normal(shape=(1, _H2, _Tk, _d_head2))\n"
            "mx.eval(_Q3, _K3, _V3)\n"
            "\n"
            "# Correct computation: scores shape is (B, H, T_q, T_k) — asymmetric.\n"
            "_scores3 = (_Q3 @ _K3.swapaxes(-2, -1)) * _scale2\n"
            "assert _scores3.shape == (1, _H2, _Tq, _Tk), (\n"
            "    f\"expected scores shape (1, {_H2}, {_Tq}, {_Tk}); got {_scores3.shape}\"\n"
            ")\n"
            "_out3 = mx.softmax(_scores3, axis=-1) @ _V3\n"
            "mx.eval(_out3)\n"
            "assert _out3.shape == (1, _H2, _Tq, _d_head2), (\n"
            "    f\"output shape must be (1, {_H2}, {_Tq}, {_d_head2}); got {_out3.shape}\"\n"
            ")\n"
            "print(f\"[3] scores shape (asymmetric Q/K): {tuple(_scores3.shape)}  ✅\")\n"
            "print(f\"    output shape after attention: {tuple(_out3.shape)}  ✅\")\n"
            "print(\"    → fix: never hardcode T = Q.shape[-2] for the K-dim.\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        T.interview_question_cell(q05),
        T.separator_cell(),
        T.interview_question_cell(q06),
        T.separator_cell(),
        T.interview_question_cell(q07),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        mem_bench,
        T.separator_cell(),
        production,
        T.separator_cell(),
        frontier,
        T.separator_cell(),
        debug_md,
        debug_code,
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

    Returns the number of cells inserted. Anchors are resolved against the
    original list before any insertions happen, and blocks are applied
    bottom-up so earlier indices stay valid.
    """
    count = 0
    for offset, cell in enumerate(block):
        cells.insert(insert_at + offset, _to_nbformat_cell(cell))
        count += 1
    return count


def transform() -> None:
    """Transform nb05 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb05 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb05] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    sdpa_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_SDPA)
    )
    causal_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_CAUSAL)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (causal_end, _block_causal(records), "causal"),
        (sdpa_end, _block_sdpa(records), "sdpa"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb05] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb05] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb05] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb05 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb05] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
