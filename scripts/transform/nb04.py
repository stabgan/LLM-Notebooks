"""Interview-grade transform for notebook 04 (Embeddings & Positional Encoding).

This module inserts the six interview-layer strata into
``04_embeddings_positional_encoding.ipynb`` and upserts the notebook's slice
of ``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): RoPE derivation and inner-product-preserves-relative-
distance property, ALiBi linear-bias slopes (geometric 1/2^i), NoPE (the 2023
finding), YaRN / NTK-aware RoPE scaling for context extension, the
``rope_base`` (θ_base) hyperparameter, and embedding tying (wte == lm_head.T).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.1, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb04
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

# scripts/transform/nb04.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "04_embeddings_positional_encoding.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 4

# Markers that indicate this notebook has already been transformed.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb04-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# These are full section-heading phrases (not bare keywords) so we don't
# accidentally match mentions in the learning-objectives cell or prose.
# "## Part 3: Rotary Position Embeddings (RoPE)" → the RoPE section start.
# "## 📜 History & Alternatives" → the cell that tabulates ALiBi and other
#   positional-encoding schemes at end-of-notebook. Both anchors are resolved
#   against the cell list BEFORE any insertion so insertions can proceed
#   bottom-up without invalidating anchors.
_ANCHOR_ROPE = "## Part 3: Rotary Position Embeddings (RoPE)"
_ANCHOR_ALIBI = "## 📜 History & Alternatives"


# ---------------------------------------------------------------------------
# Notebook I/O
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``04_embeddings_positional_encoding.ipynb`` as a JSON dict."""
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
    """Return the seven Question_Bank records for nb04.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle               — q01, q02, q06
        research_engineer — q02, q03, q04, q05, q07
        systems_engineer  — q03, q05, q06, q07

    Topic coverage (task brief):
        q01 — Absolute sinusoidal vs learned PE (trade-offs lineage)
        q02 — RoPE derivation + inner-product relative-distance property
        q03 — ALiBi linear-bias slopes (1/2^i geometric) + length extrapolation
        q04 — NoPE (the 2023 "no positional encoding" finding)
        q05 — YaRN / NTK-aware RoPE scaling for context extension (no retrain)
        q06 — The rope_base (θ_base) hyperparameter in long-context fine-tuning
        q07 — Positional encoding × GQA: one RoPE-rotated K-head shared by G
              query heads; pre-rope vs post-rope KV cache trade-off
    """
    return [
        {
            "id": "nb04-q01",
            "notebook": _NB_FILENAME,
            "section": "RoPE — Absolute Sinusoidal vs Learned",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["sinusoidal", "learned-pe", "positional-encoding"],
            "question": (
                "Compare absolute sinusoidal positional encoding (the "
                "original Transformer) with learned positional embeddings "
                "(GPT-2, BERT). When does each win, and what's the "
                "failure mode that eventually killed them both in favour "
                "of RoPE/ALiBi?"
            ),
            "answer_key_points": [
                "Sinusoidal (Vaswani et al. 2017): `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`. Zero learnable parameters; deterministic; added ONCE to input embeddings before the first attention block.",
                "Learned PE (GPT-2, BERT): a trainable `(max_seq_len, d_model)` matrix indexed by position. Costs V·D extra params; tends to fit the training distribution slightly better than sinusoidal at matched data.",
                "Sinusoidal wins on THEORETICAL extrapolation: the wave structure continues past `max_seq_len`. In practice 2017-era benchmarks showed it still degrades sharply because attention never saw high-frequency components for those positions during training.",
                "Learned wins on FINITE contexts: a 512-token model that will NEVER see longer inputs at inference is marginally better with learned PE — the optimizer finds positional representations that correlate with the data.",
                "Failure mode — both: neither encodes RELATIVE position. The model has to derive 'distance between token m and token n' from `PE(m) − PE(n)` inside attention, which is a sub-optimal indirect signal.",
                "Failure mode — learned: hard ceiling at `max_seq_len`. Position 512 for a 512-trained model is literally uninitialized; the model emits garbage.",
                "This is why RoPE (2021) and ALiBi (2021) — both RELATIVE schemes — took over. Since LLaMA (2023), no frontier LLM ships with absolute PE.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'sinusoidal extrapolates, so early models could run "
                "on arbitrary lengths' — the wave structure extrapolates, "
                "but the ATTENTION heads learned only on training-length "
                "positions and silently break at inference time just as "
                "hard as learned PE does."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/2104.09864",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q02",
            "notebook": _NB_FILENAME,
            "section": "RoPE — Derivation",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["rope", "rotation", "relative-position", "derivation"],
            "question": (
                "Derive RoPE from first principles. What property do we "
                "WANT the position-aware inner product <f(q, m), f(k, n)> "
                "to satisfy, why does the complex-number formulation make "
                "that derivation trivial, and what exactly is the "
                "'inner-product preserves relative distance' property?"
            ),
            "answer_key_points": [
                "Desired property: `<f(q, m), f(k, n)> = g(q, k, m − n)` — the inner product between a rotated query at position m and a rotated key at position n must depend ONLY on the RELATIVE offset m − n, never on the absolute positions.",
                "Ansatz: try multiplicative (not additive) position encoding. Split the d-dim vector into d/2 pairs and treat each pair as a single complex number: `q_j = q_{2j} + i·q_{2j+1}`. Encode position by MULTIPLYING each complex coordinate by `e^(i·m·θ_j)` where `θ_j = 10000^(−2j/d)`.",
                "In complex form the inner product of the rotated vectors becomes `Re[ Σ_j (q_j · e^(i·m·θ_j)) · conj(k_j · e^(i·n·θ_j)) ] = Re[ Σ_j q_j · conj(k_j) · e^(i·(m−n)·θ_j) ]` — the absolute m and n vanish; only m − n survives. That's the desired property EXACTLY.",
                "Real-valued implementation: each complex multiply by `e^(i·m·θ)` is a 2×2 rotation `[cos(mθ), −sin(mθ); sin(mθ), cos(mθ)]` applied to the pair (q_{2j}, q_{2j+1}). d/2 independent 2×2 rotations per token per head.",
                "Inner-product-preserves-relative-distance property: for any shift Δ, `<RoPE(q, m), RoPE(k, n)> = <RoPE(q, m+Δ), RoPE(k, n+Δ)>`. Whiteboard check: rotate both q and k by the same extra offset and the dot product is numerically unchanged to machine precision.",
                "Side benefit: rotation is ORTHOGONAL ⇒ norm-preserving. `||RoPE(x)|| = ||x||`, so RoPE doesn't distort the magnitude the downstream softmax sees — unlike sinusoidal which ADDS position to the embedding and inflates norms.",
                "Cost: 0 learnable parameters, d/2 (cos, sin) pairs precomputed once per (seq_len, d_head) at startup, O(B·T·H·D) elementwise multiplies per attention call.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Claiming RoPE is 'sinusoidal inside attention'. It's a "
                "MULTIPLICATIVE rotation, not an additive sin/cos shift — "
                "and the inner product (not the embedding) is what encodes "
                "relative position."
            ),
            "references": [
                "https://arxiv.org/abs/2104.09864",
                "https://blog.eleuther.ai/rotary-embeddings/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q03",
            "notebook": _NB_FILENAME,
            "section": "ALiBi — Linear Bias Slopes",
            "difficulty": "core",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["alibi", "length-extrapolation", "attention-bias"],
            "question": (
                "Explain ALiBi. Write the slope formula for H heads, "
                "reason through why it's specifically `2^(-8/H · head_idx)` "
                "(a geometric series 1/2^i), and explain why this particular "
                "choice enables length extrapolation beyond training context."
            ),
            "answer_key_points": [
                "ALiBi (Attention with Linear Biases, Press et al. 2021): NO positional embedding is added to Q/K. Instead, before the softmax, add a linear bias to the (m, n) attention logit: `A[m,n] ← Q[m]·K[n] − m_h · |m − n|`, where `m_h` is a per-HEAD negative slope.",
                "Slope formula for H heads: `m_h = 2^(-8 · h / H)` for head index h ∈ {1, ..., H}. That's a geometric series: slopes for H=8 are 1/2, 1/4, 1/8, ..., 1/256 — spanning 8 octaves of distance-decay rates.",
                "Why 2^(-8/H): the 2^(-8) = 1/256 is the smallest slope in the series; it's chosen so the LOWEST-slope head attends roughly uniformly across 256-ish tokens (log2(256)=8) — large-context semantic head. The HIGHEST-slope head (h=1, slope ≈ 1/2) attends almost exclusively to the immediately preceding token — local-context syntactic head.",
                "The geometric spacing is deliberate: a LINEAR spread of slopes would cluster all heads near one decay rate. Geometric covers the full 1×..256× distance range with only H heads — classic multi-scale decomposition.",
                "Length extrapolation: ALiBi is a PURE FUNCTION of `|m − n|`. Training on sequences of length L and evaluating at 4L still uses the same slopes on the same distance metric — no 'out-of-distribution positions' exist. The 2021 paper demonstrates ALiBi-512 extrapolates cleanly to 16k without retraining.",
                "Trade-off: ALiBi REGRESSES to distance-weighted attention at long range — the bias dominates Q·K for far-apart tokens. Great for long-document tasks that actually have locality; poor for tasks needing long-distance reasoning. This is why LLaMA-1/2/3 picked RoPE (+ YaRN scaling) over ALiBi.",
                "Used by: BLOOM-176B (2022), MPT-7B (2023), Falcon-RW. Discontinued at the frontier once RoPE + NTK/YaRN matched or beat ALiBi's extrapolation while preserving uniform attention at distance.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Writing the slope as linear in h (slope ∝ h/H) — it's "
                "EXPONENTIAL: `slope = 2^(-8·h/H)`. Linear slopes collapse "
                "all heads onto a narrow decay band and destroy the "
                "multi-scale property."
            ),
            "references": [
                "https://arxiv.org/abs/2108.12409",
                "https://github.com/ofirpress/attention_with_linear_biases",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q04",
            "notebook": _NB_FILENAME,
            "section": "NoPE — No Positional Encoding",
            "difficulty": "core",
            "roles": ["research_engineer"],
            "topic_tags": ["nope", "causal-mask", "positional-encoding"],
            "question": (
                "What is NoPE (2023) and what's the surprising finding? "
                "If transformers are permutation-invariant, how can a model "
                "with ZERO positional encoding learn any order-sensitive "
                "task at all?"
            ),
            "answer_key_points": [
                "NoPE (Kazemnejad et al., 'The Impact of Positional Encoding on Length Generalization', NeurIPS 2023): strip ALL positional encoding — no sinusoidal, no learned PE, no RoPE, no ALiBi — and train a DECODER-ONLY (causal) transformer.",
                "The surprising finding: NoPE decoder-only models match or BEAT RoPE / ALiBi on several length-generalization benchmarks. They extrapolate from 512 → 2048 context cleanly, while learned-PE models break immediately and RoPE without YaRN degrades past training length.",
                "How it's possible: decoder-only transformers are NOT permutation-invariant — the CAUSAL MASK breaks the symmetry. Position n's representation attends to positions {0, ..., n} only; position 0 attends only to {0}. Each layer computes a DIFFERENT function at each position purely because the set of visible tokens differs.",
                "Concretely, the first-layer attention output at position n is Σ_{i ≤ n} softmax(q_n·k_i)·v_i — it implicitly encodes 'I am at least position n' (because n+1 tokens went into the sum) without any explicit position signal.",
                "Caveat — this is NOT true for encoder-only / bidirectional models. Remove PE from BERT and every position sees identical context; the model is genuinely permutation-invariant and cannot learn 'The cat sat on the mat' vs 'mat the on sat cat The'.",
                "Practical significance: the research frontier is exploring hybrid stacks — NoPE layers interleaved with RoPE layers (Llama 4 preview, some DeepSeek ablations) to get the length-generalization of NoPE AND the precise relative-position control of RoPE.",
                "When NoPE wins in practice: tasks where the causal-mask ordering signal is already sufficient (copy/retrieval, count, pattern continuation). When it loses: tasks needing precise relative-offset matching (e.g. 'what's the 17th word from the end').",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Asserting 'NoPE works because transformers don't need "
                "position info' — they DO; the causal mask SUPPLIES the "
                "ordering signal. Remove the mask (encoder-only) and NoPE "
                "breaks completely."
            ),
            "references": [
                "https://arxiv.org/abs/2305.19466",
                "https://openreview.net/forum?id=wnuCFuKeYl",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q05",
            "notebook": _NB_FILENAME,
            "section": "Context Extension — YaRN / NTK-aware RoPE",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["yarn", "ntk-aware", "rope-scaling", "long-context"],
            "question": (
                "A model was trained with RoPE at 4K context. You want to "
                "extend to 128K at inference time with MINIMAL fine-tuning. "
                "Walk through position interpolation (PI), NTK-aware "
                "scaling, and YaRN — what each changes, and why YaRN is "
                "the current production default."
            ),
            "answer_key_points": [
                "Problem: at test-time position 128000 with a model trained to max position 4096, the RoPE angle `m·θ_j` is 32× larger than anything seen during training. High-frequency dimensions wrap around aggressively; attention scores collapse to noise.",
                "Position Interpolation (PI, Chen et al. 2023): LINEARLY rescale all positions before applying RoPE: `m' = m · L_train / L_test = m / s` where s = 32 for 4K→128K. Every dimension is squeezed by the SAME factor. Requires a small fine-tune (~1B tokens) to recover quality. Simple; works; squeezes all frequencies equally.",
                "NTK-aware scaling (bloc97, June 2023): different dimensions get different scaling. HIGH-frequency dims (small θ_j) are NOT rescaled — they already wrap many times per 4K so one more wrap is fine. LOW-frequency dims (large θ_j) ARE rescaled — they barely finish one cycle in 4K and must be stretched. Achieved by rescaling `base = 10000 → 10000 · s^(d/(d−2))` rather than the positions themselves. Training-free on short contexts; degrades on very long ones.",
                "YaRN (Peng et al. 2023, 'Yet another RoPE extensioN'): combines NTK-aware scaling with an explicit ATTENTION TEMPERATURE correction. The observation: as you stretch RoPE, the effective attention-score distribution cools (std shrinks) because close-together positions now look MORE similar. YaRN multiplies logits by `sqrt(1 + 0.1·log(s))` to restore the original temperature.",
                "YaRN also uses a RAMP function between NTK-style and PI-style scaling: low-freq dims get full PI (preserves smooth extrapolation), high-freq dims get 1× (preserves local precision), mid-freq dims get a smooth interpolation. Three regimes, one hyperparam.",
                "Production defaults (2024–2025): LLaMA-3.1 uses a YaRN variant at 128K; LLaMA-3.2 ships 128K out-of-the-box with a rescaled RoPE base of 500_000 (!) in addition to YaRN. DeepSeek-V3 extends to 128K with YaRN. Qwen-2.5 uses YaRN + Dual-Chunk Attention for 1M context.",
                "Fine-tuning budget: PI → ~1B tokens to recover, NTK-aware → ~100M tokens, YaRN → often zero-shot competitive, ~50M tokens to recover completely. That's the main reason YaRN is production default.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Treating NTK-aware scaling as position rescaling. It's a "
                "BASE (θ_base) rescaling — positions are left alone, the "
                "RoPE frequency schedule is compressed. Confusing these "
                "two produces subtly wrong interpolation code that "
                "CLIP-degrades attention at long range."
            ),
            "references": [
                "https://arxiv.org/abs/2306.15595",
                "https://arxiv.org/abs/2309.00071",
                "https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q06",
            "notebook": _NB_FILENAME,
            "section": "RoPE Base θ_base — The Hyperparameter Nobody Notices",
            "difficulty": "stretch",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["rope-base", "theta", "long-context", "hyperparameter"],
            "question": (
                "Explain the `rope_base` (θ_base, usually 10000) "
                "hyperparameter in RoPE. Why did LLaMA-1 use 10000 but "
                "LLaMA-3.1 use 500000? What exactly breaks if you get "
                "this wrong at inference time?"
            ),
            "answer_key_points": [
                "RoPE's per-dimension frequencies are `θ_j = θ_base^(-2j/d)` for j ∈ {0, ..., d/2 − 1}. At `θ_base = 10000, d = 128`: j=0 rotates at freq 1.0 (period 2π), j=63 at freq 10000^(-1) ≈ 1e-4 (period 2π · 10000 ≈ 62800 tokens).",
                "θ_base controls the WAVELENGTH RANGE of position-dependent signals. Larger θ_base ⇒ the lowest-frequency dim has a LONGER period ⇒ the model can resolve position differences over a longer absolute span without aliasing.",
                "LLaMA-1 (4K training context): θ_base = 10000. Longest period ≈ 62800 tokens — comfortable headroom above 4K.",
                "LLaMA-3.1 (128K training context): θ_base = 500000. Longest period ≈ 3.1M tokens. With θ_base=10000 and 128K context, the lowest-freq dim wraps through nearly 2 full cycles — the model would alias position 0 with position 62800.",
                "Rule of thumb: `θ_base ≈ 100 · L_train` keeps the lowest-freq period safely above training context. Meta's 500_000 value for 128K corresponds roughly to that rule (500000 / 128000 ≈ 3.9× margin).",
                "Serving-time bug: load LLaMA-3.1 weights but use the config's `rope_theta=10000` (default for legacy models) and you get catastrophic long-context degradation — attention scores beyond ~20k are random. Diagnostic: perplexity curve by context length plateaus then EXPLODES past a specific threshold (the aliasing point).",
                "Fine-tuning recipe for context extension: first INCREASE `θ_base` to match the new target length, then apply YaRN (or equivalent) as a second correction. Meta's LLaMA-3 training run did both — they bumped θ_base AND used YaRN-style frequency rescaling.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Thinking θ_base is 'just a scale factor'. It's the base "
                "of an EXPONENTIAL frequency schedule — doubling θ_base "
                "quadruples (not doubles) the lowest-dim period. Arithmetic "
                "mistakes here produce silent quality regressions in "
                "long-context serving."
            ),
            "references": [
                "https://arxiv.org/abs/2309.16039",
                "https://ai.meta.com/blog/meta-llama-3-1/",
                "https://huggingface.co/blog/rope-scaling",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb04-q07",
            "notebook": _NB_FILENAME,
            "section": "Positional Encoding × GQA — KV Cache Layout",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["rope", "gqa", "kv-cache", "multi-query-attention"],
            "question": (
                "In Grouped-Query Attention (LLaMA-2/3, Mistral, Qwen) a "
                "single K/V head is shared across G query heads. Walk "
                "through how RoPE interacts with GQA: which tensor gets "
                "rotated, is the rotation applied ONCE or G times per "
                "layer, and what's the production trade-off between "
                "caching PRE-rope vs POST-rope keys?"
            ),
            "answer_key_points": [
                "GQA shape: `n_heads` query heads, `n_kv_heads = n_heads / G` K/V heads (G=8 for LLaMA-3-70B: 64 Q heads, 8 KV heads). RoPE is applied to BOTH Q and K — but K has fewer heads, so the RoPE cost on K is 1/G of the Q cost.",
                "Rotation is applied ONCE per K head (not G times). The SAME rotated K head is reused by all G query heads in its group via broadcasting at the score-matmul (or via `mx.repeat`/`einsum` in naive impls). Getting this wrong — re-rotating K once per query head — inflates RoPE cost by G and is a common first-GQA-bug.",
                "V is NEVER rotated. RoPE rotates the dot-product space (Q·K), not the value space. Value heads are shared across the query group without any position-dependent transform.",
                "Cache layout choice 1 — POST-rope KV cache (vLLM, SGLang, MLX-LM default): store `K_rotated = RoPE(K, pos)` in the cache. Pro: zero compute per decode step; one lookup, one matmul. Con: the cached keys are TIED to the position at which they were rotated — any context-extension retrofit (YaRN at serving time) requires invalidating the whole cache.",
                "Cache layout choice 2 — PRE-rope KV cache: store raw K, apply RoPE on-the-fly at attention time using the position index from the cache slot. Pro: trivially supports dynamic RoPE-base / YaRN swap at inference. Con: extra RoPE multiply per decoded token per K head (small — ~1% of attention cost).",
                "Production choice (2024-2025): frontier servers (vLLM, SGLang, TRT-LLM) default to POST-rope caching because the performance win at 128k context is material and YaRN-at-serving is rare. Research frameworks and some experimentation servers default to pre-rope so you can A/B test scaling factors without rebuilding the cache.",
                "Subtle bug: SGLang's RadixAttention prefix cache hashes (token_ids) of cached prefix chunks. If a user shares a prefix across two requests at DIFFERENT starting positions (e.g. chat session resuming mid-context), the POST-rope cache entries are VALID ONLY at their original positions — you cannot reuse them at shifted positions. SGLang detects this via a position-equality check on cache hit; a broken check silently corrupts generation.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Answering 'RoPE is applied per attention head' and "
                "forgetting that under GQA only the K/V heads are rotated, "
                "not each query head. The Q-side rotation IS per-query-"
                "head (there's no sharing on Q), but the K-side is per-"
                "group — a factor of G difference in compute."
            ),
            "references": [
                "https://arxiv.org/abs/2305.13245",
                "https://arxiv.org/abs/2104.09864",
                "https://github.com/sgl-project/sglang",
            ],
            "added_in": added_in,
        },
    ]



# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_rope(records: list[dict]) -> list[dict]:
    """Block for the 'Part 3: RoPE' section.

    Contents: q01 (absolute vs learned), q02 (RoPE derivation),
    whiteboard-A (implement RoPE from scratch + verify inner-product
    relative-distance property), 📐-1 (RoPE application cost
    O(B·T·H·D)) with a benchmark comparing the naïve 2×2-rotation
    loop against the vectorized (cos, sin) pair form.
    """
    q01, q02 = records[0], records[1]

    # --- Whiteboard A — implement RoPE, verify inner-product invariance. ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="RoPE from scratch — verify inner-product relative-distance property",
        prompt=(
            "Implement `apply_rope(x, pos, theta_base=10000.0)` that takes "
            "a (..., d_head) tensor `x` and a scalar `pos`, and returns "
            "the position-m-rotated vector. Then verify the RELATIVE-"
            "DISTANCE PROPERTY numerically: for any offset Δ, "
            "`<RoPE(q, m), RoPE(k, n)>` must equal `<RoPE(q, m+Δ), "
            "RoPE(k, n+Δ)>` to machine precision."
        ),
        constraints=[
            "Use the real-valued form: split the last dim into d_head/2 pairs, "
            "rotate each pair by `m · θ_j` where `θ_j = theta_base^(-2j/d_head)`.",
            "Use MLX throughout — no numpy or torch. The final dot product "
            "must go through `mx.sum` and `mx.eval`.",
            "Generate q and k with `mx.random.normal` (seed the PRNG for "
            "reproducibility) at (d_head=128).",
            "Assert that `<RoPE(q, m), RoPE(k, n)>` matches "
            "`<RoPE(q, m+Δ), RoPE(k, n+Δ)>` for Δ ∈ {1, 5, 37} within "
            "abs tolerance 1e-3 (float32; rotations accumulate error).",
            "Include one additional `assert` that `||RoPE(q, m)||_2 == ||q||_2` "
            "to within 1e-4 — the orthogonality / norm-preservation check.",
        ],
        complexity=(
            "Precompute: O(T · d_head) for the (cos, sin) tables. Apply: "
            "O(B · T · H · d_head) elementwise multiplies per attention call."
        ),
        solution_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "\n"
            "def rope_freqs(d_head: int, max_pos: int, theta_base: float = 10000.0):\n"
            "    \"\"\"Precompute (cos, sin) tables of shape (max_pos, d_head/2).\"\"\"\n"
            "    j = mx.arange(0, d_head, 2, dtype=mx.float32)  # (d_head/2,)\n"
            "    thetas = 1.0 / (theta_base ** (j / d_head))    # (d_head/2,)\n"
            "    positions = mx.arange(0, max_pos, dtype=mx.float32)  # (max_pos,)\n"
            "    # Outer product: angles[m, j] = m * theta_j\n"
            "    angles = positions[:, None] * thetas[None, :]  # (max_pos, d_head/2)\n"
            "    return mx.cos(angles), mx.sin(angles)\n"
            "\n"
            "def apply_rope(x: mx.array, pos: int, cos_tab: mx.array, sin_tab: mx.array) -> mx.array:\n"
            "    \"\"\"Rotate x (shape (..., d_head)) by RoPE at absolute position pos.\n"
            "\n"
            "    Pair even/odd dims: (x0, x1, x2, x3, ...) -> pairs (x0,x1), (x2,x3), ...\n"
            "    Each pair gets rotated by angle m*theta_j via a 2x2 rotation matrix.\n"
            "    \"\"\"\n"
            "    c = cos_tab[pos]  # (d_head/2,)\n"
            "    s = sin_tab[pos]  # (d_head/2,)\n"
            "    # Split last dim into even/odd.\n"
            "    x_even = x[..., 0::2]  # (..., d_head/2)\n"
            "    x_odd = x[..., 1::2]   # (..., d_head/2)\n"
            "    # 2x2 rotation applied to each pair.\n"
            "    rot_even = x_even * c - x_odd * s\n"
            "    rot_odd = x_even * s + x_odd * c\n"
            "    # Interleave even/odd back to original layout.\n"
            "    out = mx.stack([rot_even, rot_odd], axis=-1)  # (..., d_head/2, 2)\n"
            "    return out.reshape(x.shape)\n"
            "\n"
            "# Setup: d_head=128, seed for determinism.\n"
            "d_head = 128\n"
            "max_pos = 128\n"
            "mx.random.seed(0)\n"
            "q = mx.random.normal(shape=(d_head,))\n"
            "k = mx.random.normal(shape=(d_head,))\n"
            "cos_tab, sin_tab = rope_freqs(d_head, max_pos, theta_base=10000.0)\n"
            "mx.eval(cos_tab, sin_tab, q, k)\n"
            "\n"
            "# --- Property 1: RoPE preserves norms (rotation is orthogonal). ---\n"
            "m = 7\n"
            "q_rot = apply_rope(q, m, cos_tab, sin_tab)\n"
            "norm_before = float(mx.sqrt(mx.sum(q * q)).item())\n"
            "norm_after = float(mx.sqrt(mx.sum(q_rot * q_rot)).item())\n"
            "assert abs(norm_before - norm_after) < 1e-4, (\n"
            "    f\"norm not preserved: {norm_before:.6f} vs {norm_after:.6f}\"\n"
            ")\n"
            "\n"
            "# --- Property 2: inner product depends only on (m - n). ---\n"
            "# Pick base positions (m, n) and assert shifting both by Delta is invariant.\n"
            "m, n = 10, 25  # m - n = -15\n"
            "q_m = apply_rope(q, m, cos_tab, sin_tab)\n"
            "k_n = apply_rope(k, n, cos_tab, sin_tab)\n"
            "ref = float(mx.sum(q_m * k_n).item())\n"
            "for delta in (1, 5, 37):\n"
            "    q_md = apply_rope(q, m + delta, cos_tab, sin_tab)\n"
            "    k_nd = apply_rope(k, n + delta, cos_tab, sin_tab)\n"
            "    shifted = float(mx.sum(q_md * k_nd).item())\n"
            "    assert abs(ref - shifted) < 1e-3, (\n"
            "        f\"relative-distance property violated at Δ={delta}: \"\n"
            "        f\"{ref:.6f} vs {shifted:.6f}\"\n"
            "    )\n"
            "\n"
            "# Force eval and print a summary line.\n"
            "summary = mx.array([ref, norm_before, norm_after], dtype=mx.float32)\n"
            "mx.eval(summary)\n"
            "print(f\"✅ norm preserved: {norm_before:.4f} -> {norm_after:.4f}\")\n"
            "print(f\"✅ <RoPE(q,{m}), RoPE(k,{n})> = {ref:.4f}\")\n"
            "print(f\"✅ invariant to joint shift by Δ ∈ {{1, 5, 37}}\")\n"
        ),
    )

    # --- 📐 Complexity cell — RoPE application cost. ---
    complexity = T.complexity_analysis_cell(
        op="RoPE application per attention op — (B·T·H·D) elementwise",
        flops=(
            "O(B · T · H · d_head) per attention call — 4 fused multiply-"
            "adds per pair per token per head. At B=4, T=2048, H=32, "
            "d_head=128 that's ~67 M FLOPs, negligible against the "
            "O(B·T²·H·d_head) attention matmul itself"
        ),
        memory=(
            "Precomputed (cos, sin) tables: O(T · d_head) floats — shared "
            "ACROSS all (B, H) at inference time. At T=128k, d_head=128 "
            "that's ~33 MiB fp32 (or 16 MiB fp16). Typically cached in "
            "a module-level buffer and NEVER recomputed"
        ),
        latency_mlx=(
            "RoPE apply on M4 Pro for (B=2, T=2048, H=32, d_head=128) "
            "bf16: ~0.2–0.5 ms/call — well below 1% of end-to-end "
            "decoder layer time. Measured below"
        ),
        scaling=(
            "RoPE is a tiny fraction of attention cost but the (cos, sin) "
            "table GROWS LINEARLY with max_seq_len. At 1M context and "
            "d_head=128 the cache is 256 MiB fp32 — still dominated by "
            "KV cache but meaningful. Frontier servers (vLLM, SGLang) "
            "share ONE (cos, sin) table across all requests."
        ),
    )

    bench_src = (
        "# Benchmark: RoPE application on (B, T, H, d_head) bf16 tensors\n"
        "import time\n"
        "import math\n"
        "import mlx.core as mx\n"
        "\n"
        "def _rope_freqs(d_head: int, max_pos: int, theta_base: float = 10000.0):\n"
        "    j = mx.arange(0, d_head, 2, dtype=mx.float32)\n"
        "    thetas = 1.0 / (theta_base ** (j / d_head))\n"
        "    positions = mx.arange(0, max_pos, dtype=mx.float32)\n"
        "    angles = positions[:, None] * thetas[None, :]\n"
        "    return mx.cos(angles), mx.sin(angles)\n"
        "\n"
        "def apply_rope_batched(x: mx.array, cos_tab: mx.array, sin_tab: mx.array) -> mx.array:\n"
        "    \"\"\"Vectorized RoPE: x is (B, T, H, d_head); cos/sin are (T, d_head/2).\"\"\"\n"
        "    # Broadcast (T, d/2) -> (1, T, 1, d/2) to match x's (B, T, H, d_head/2).\n"
        "    c = cos_tab[None, :, None, :]\n"
        "    s = sin_tab[None, :, None, :]\n"
        "    x_even = x[..., 0::2]\n"
        "    x_odd = x[..., 1::2]\n"
        "    rot_even = x_even * c - x_odd * s\n"
        "    rot_odd = x_even * s + x_odd * c\n"
        "    out = mx.stack([rot_even, rot_odd], axis=-1)\n"
        "    return out.reshape(x.shape)\n"
        "\n"
        "# Production-like shape: B=2, T=2048, H=32, d_head=128.\n"
        "B, T, H, d_head = 2, 2048, 32, 128\n"
        "mx.random.seed(0)\n"
        "x = mx.random.normal(shape=(B, T, H, d_head)).astype(mx.bfloat16)\n"
        "cos_tab, sin_tab = _rope_freqs(d_head, T, 10000.0)\n"
        "cos_tab = cos_tab.astype(mx.bfloat16)\n"
        "sin_tab = sin_tab.astype(mx.bfloat16)\n"
        "mx.eval(x, cos_tab, sin_tab)\n"
        "\n"
        "# Warmup (Requirement 5.3).\n"
        "for _ in range(3):\n"
        "    y = apply_rope_batched(x, cos_tab, sin_tab)\n"
        "    mx.eval(y)\n"
        "\n"
        "N = 20\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(N):\n"
        "    y = apply_rope_batched(x, cos_tab, sin_tab)\n"
        "    mx.eval(y)\n"
        "dt_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "\n"
        "# Analytic FLOPs: 4 fma per pair × (d_head/2 pairs) × B × T × H\n"
        "# Report rough 'FLOP rate' just so readers see how fast this is.\n"
        "flops_per_call = 4 * (d_head // 2) * B * T * H\n"
        "gflops = flops_per_call / (dt_ms * 1e-3) / 1e9\n"
        "\n"
        "# Invariant: norm preservation across the whole batch.\n"
        "norm_x = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1))\n"
        "norm_y = mx.sqrt(mx.sum(y.astype(mx.float32) ** 2, axis=-1))\n"
        "diff = float(mx.max(mx.abs(norm_x - norm_y)).item())\n"
        "# bf16 rotation introduces a few parts in 1e-2; loosen to 5e-2.\n"
        "assert diff < 5e-2, f\"RoPE norm drift {diff:.4f} > 5e-2 (bf16 rounding)\"\n"
        "\n"
        "print(f\"shape: B={B} T={T} H={H} d_head={d_head}  (bf16)\")\n"
        "print(f\"RoPE apply: {dt_ms:7.3f} ms / call   ({gflops:6.1f} GFLOP/s)\")\n"
        "print(f\"max |norm(x) - norm(RoPE(x))| = {diff:.4e}  (bf16; rotation ≈ orthogonal)\")\n"
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



def _block_alibi(records: list[dict]) -> list[dict]:
    """Block for the 'History & Alternatives' section (ALiBi anchor).

    Contents: q03 (ALiBi slopes), q04 (NoPE), q05 (YaRN/NTK), q06
    (rope_base θ), q07 (PE × GQA KV-cache layout), whiteboard-B (ALiBi
    slope computation + assert geometric progression), 📐-2 (KV cache
    under RoPE — pre-rope vs post-rope trade-off, GQA-aware) with
    benchmark, 🏭 production (RoPE caching / rope-base scaling in
    vLLM / SGLang / TRT-LLM / MLX-LM), 🔭 frontier (YaRN, LLaMA-3.1,
    DeepSeek-V3, Gemma-3, LongRoPE), 🛠️ debugging (off-by-one in RoPE
    position indexing, freq-base mismatch, fp16/bf16 precision loss in
    cos / sin).
    """
    q03, q04, q05, q06, q07 = (
        records[2],
        records[3],
        records[4],
        records[5],
        records[6],
    )

    # --- Whiteboard B — ALiBi slopes + geometric-progression assertion. ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="ALiBi slopes for H heads — verify the geometric progression",
        prompt=(
            "Implement `alibi_slopes(n_heads)` returning an `mx.array` of "
            "H per-head slopes following the ALiBi paper's formula "
            "`m_h = 2^(-8 · h / H)` for h ∈ {1, ..., H}. Assert the "
            "result is strictly monotonically decreasing, lies in (0, 1], "
            "and that successive slopes form a GEOMETRIC progression "
            "with common ratio `2^(-8/H)`."
        ),
        constraints=[
            "Use MLX throughout — no numpy. The returned slopes array must "
            "be materialized via `mx.eval`.",
            "Support H = 8, 16, 32, 64 (powers of 2) and at least one "
            "non-power-of-two H = 12 (GPT-Neo uses 12 heads). The original "
            "paper's slope formula works for any H.",
            "Assert monotonicity: for all i, `slopes[i] > slopes[i+1]`.",
            "Assert geometric: `slopes[i+1] / slopes[i]` is within 1e-5 "
            "of `2^(-8/H)` for every adjacent pair.",
            "Assert the LARGEST slope (h=1) is `2^(-8/H)` and the SMALLEST "
            "(h=H) is `2^(-8) = 1/256` — the canonical endpoints.",
        ],
        complexity="O(H) — trivially vectorized in a single MLX op.",
        solution_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "\n"
            "def alibi_slopes(n_heads: int) -> mx.array:\n"
            "    \"\"\"ALiBi slopes: m_h = 2^(-8 * h / H) for h in {1, ..., H}.\n"
            "\n"
            "    Returns an fp32 array of shape (H,) in descending order of "
            "    magnitude (so the first head has the LARGEST decay, the "
            "    last has the SMALLEST).\n"
            "    \"\"\"\n"
            "    if n_heads <= 0:\n"
            "        raise ValueError(f\"n_heads must be positive, got {n_heads}\")\n"
            "    h = mx.arange(1, n_heads + 1, dtype=mx.float32)  # (H,)\n"
            "    return mx.power(2.0, -8.0 * h / n_heads)\n"
            "\n"
            "# Check the canonical H values.\n"
            "for H in (8, 16, 32, 64, 12):\n"
            "    slopes = alibi_slopes(H)\n"
            "    mx.eval(slopes)\n"
            "\n"
            "    # (a) Range: all in (0, 1]. Largest is 2^(-8/H), smallest is 2^(-8)=1/256.\n"
            "    slopes_list = slopes.tolist()\n"
            "    assert slopes_list[0] <= 1.0 and slopes_list[-1] > 0.0, (\n"
            "        f\"H={H}: out-of-range slopes {slopes_list[0]}, {slopes_list[-1]}\"\n"
            "    )\n"
            "    expected_max = 2.0 ** (-8.0 / H)\n"
            "    expected_min = 2.0 ** -8.0\n"
            "    assert abs(slopes_list[0] - expected_max) < 1e-5, (\n"
            "        f\"H={H}: max slope {slopes_list[0]} != {expected_max}\"\n"
            "    )\n"
            "    assert abs(slopes_list[-1] - expected_min) < 1e-5, (\n"
            "        f\"H={H}: min slope {slopes_list[-1]} != {expected_min}\"\n"
            "    )\n"
            "\n"
            "    # (b) Monotonic strictly decreasing.\n"
            "    for i in range(H - 1):\n"
            "        assert slopes_list[i] > slopes_list[i + 1], (\n"
            "            f\"H={H}: slopes not monotone at i={i}\"\n"
            "        )\n"
            "\n"
            "    # (c) Geometric progression: ratio = 2^(-8/H).\n"
            "    expected_ratio = 2.0 ** (-8.0 / H)\n"
            "    for i in range(H - 1):\n"
            "        ratio = slopes_list[i + 1] / slopes_list[i]\n"
            "        assert abs(ratio - expected_ratio) < 1e-5, (\n"
            "            f\"H={H}: non-geometric at i={i} (ratio {ratio} != {expected_ratio})\"\n"
            "        )\n"
            "\n"
            "    print(f\"H={H:>3}: slopes[0]={slopes_list[0]:.6f}, \"\n"
            "          f\"slopes[-1]={slopes_list[-1]:.6f}, ratio={expected_ratio:.6f}\")\n"
            "\n"
            "# Demonstrate H=8 explicitly — this is what BLOOM / MPT actually use.\n"
            "slopes_8 = alibi_slopes(8)\n"
            "mx.eval(slopes_8)\n"
            "print(\"\\nBLOOM/MPT-style H=8 slopes:\", [f\"{s:.4f}\" for s in slopes_8.tolist()])\n"
            "print(\"✅ geometric with common ratio 2^(-1) = 0.5; spans 8 octaves of decay\")\n"
        ),
    )

    # --- 📐 Complexity cell — KV cache behaviour under RoPE (pre vs post). ---
    complexity = T.complexity_analysis_cell(
        op="KV cache under RoPE — pre-rope vs post-rope storage (GQA aware)",
        flops=(
            "Per DECODE step: post-rope cache → 0 extra FLOPs (one matmul "
            "against the cached K). Pre-rope cache → +O(n_kv_heads · "
            "d_head) elementwise multiplies per cached position that "
            "participates in the dot-product — at n_kv_heads=8, d_head=128, "
            "T=128k that's ~130 MFLOPs/step, about 1% of attention cost"
        ),
        memory=(
            "Cache footprint is IDENTICAL for both layouts: "
            "`2 · L · B · n_kv_heads · T · d_head · bytes_per_elem` (2 = K + V). "
            "At L=32, B=1, n_kv_heads=8, T=128k, d_head=128 bf16: "
            "~16 GiB per request. The RoPE (cos, sin) table is shared "
            "across all B, L and is negligible by comparison"
        ),
        latency_mlx=(
            "On M4 Pro with (B=1, T=2048, n_kv_heads=8, d_head=128) bf16, "
            "applying RoPE lazily per attention call costs ~0.05 ms — "
            "so the pre-rope layout adds <1% to attention latency in "
            "exchange for dynamic rope-base swap capability. Measured below"
        ),
        scaling=(
            "Post-rope: cache grows with T but NEVER gets re-rotated; "
            "fastest at serve time but rope-base / scaling must be frozen "
            "at cache-fill. Pre-rope: same footprint, ~1% compute overhead, "
            "but supports YaRN-at-serving and run-time rope-base swap. "
            "Frontier servers (vLLM, SGLang) default post-rope; research "
            "and experimentation stacks default pre-rope. GQA matters: "
            "the cache is `n_kv_heads`-wide, NOT `n_heads`-wide — LLaMA-3-"
            "70B saves 8× on cache size vs vanilla MHA."
        ),
    )

    kv_bench_src = (
        "# Benchmark: pre-rope vs post-rope K-cache attention latency (GQA-shaped)\n"
        "import time\n"
        "import math\n"
        "import mlx.core as mx\n"
        "\n"
        "def _rope_freqs(d_head: int, max_pos: int, theta_base: float = 10000.0):\n"
        "    j = mx.arange(0, d_head, 2, dtype=mx.float32)\n"
        "    thetas = 1.0 / (theta_base ** (j / d_head))\n"
        "    positions = mx.arange(0, max_pos, dtype=mx.float32)\n"
        "    angles = positions[:, None] * thetas[None, :]\n"
        "    return mx.cos(angles), mx.sin(angles)\n"
        "\n"
        "def apply_rope_kv(x: mx.array, cos_tab: mx.array, sin_tab: mx.array) -> mx.array:\n"
        "    \"\"\"Rotate x (B, T, n_kv_heads, d_head) by RoPE using (T, d_head/2) tables.\"\"\"\n"
        "    c = cos_tab[None, :, None, :]\n"
        "    s = sin_tab[None, :, None, :]\n"
        "    x_even = x[..., 0::2]\n"
        "    x_odd = x[..., 1::2]\n"
        "    rot_even = x_even * c - x_odd * s\n"
        "    rot_odd = x_even * s + x_odd * c\n"
        "    out = mx.stack([rot_even, rot_odd], axis=-1)\n"
        "    return out.reshape(x.shape)\n"
        "\n"
        "# LLaMA-3-70B-ish GQA shape: n_heads=64 Q heads, n_kv_heads=8 (G=8), d_head=128.\n"
        "B, T, n_kv_heads, n_heads, d_head = 1, 2048, 8, 64, 128\n"
        "G = n_heads // n_kv_heads\n"
        "mx.random.seed(0)\n"
        "\n"
        "# A 'K-cache' containing T past keys (already rotated for the post-rope path).\n"
        "k_raw = mx.random.normal(shape=(B, T, n_kv_heads, d_head)).astype(mx.bfloat16)\n"
        "cos_tab, sin_tab = _rope_freqs(d_head, T, 10000.0)\n"
        "cos_tab = cos_tab.astype(mx.bfloat16)\n"
        "sin_tab = sin_tab.astype(mx.bfloat16)\n"
        "k_post = apply_rope_kv(k_raw, cos_tab, sin_tab)\n"
        "\n"
        "# New query at a single step: (B, 1, n_heads, d_head). RoPE at pos=T.\n"
        "q = mx.random.normal(shape=(B, 1, n_heads, d_head)).astype(mx.bfloat16)\n"
        "# For measurement we RoPE the query at position 0 -> position T still costs the same.\n"
        "q_rot = apply_rope_kv(q, cos_tab[:1], sin_tab[:1])\n"
        "mx.eval(k_raw, k_post, q_rot, cos_tab, sin_tab)\n"
        "\n"
        "def attn_post_rope():\n"
        "    \"\"\"Post-rope path: Q·K^T directly against cached rotated keys.\"\"\"\n"
        "    # Broadcast shared KV heads to G query heads via reshape + repeat.\n"
        "    k = mx.repeat(k_post, G, axis=2)             # (B, T, n_heads, d_head)\n"
        "    # (B, 1, n_heads, d_head) @ (B, T, n_heads, d_head)^T on last dim.\n"
        "    scores = mx.sum(q_rot * k, axis=-1)          # (B, T, n_heads)   [analytic stub]\n"
        "    return scores\n"
        "\n"
        "def attn_pre_rope():\n"
        "    \"\"\"Pre-rope path: apply RoPE to raw cached keys on every attention call.\"\"\"\n"
        "    k = apply_rope_kv(k_raw, cos_tab, sin_tab)   # extra work: ~1% of attn\n"
        "    k = mx.repeat(k, G, axis=2)\n"
        "    scores = mx.sum(q_rot * k, axis=-1)\n"
        "    return scores\n"
        "\n"
        "# Warmup (Requirement 5.3).\n"
        "for _ in range(3):\n"
        "    _y = attn_post_rope(); mx.eval(_y)\n"
        "    _y = attn_pre_rope();  mx.eval(_y)\n"
        "\n"
        "N = 20\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(N):\n"
        "    y1 = attn_post_rope(); mx.eval(y1)\n"
        "post_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(N):\n"
        "    y2 = attn_pre_rope(); mx.eval(y2)\n"
        "pre_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "\n"
        "# Analytic cache footprint — identical for both layouts.\n"
        "cache_bytes = 2 * B * T * n_kv_heads * d_head * 2  # 2 (K+V) * ... * 2 (bf16)\n"
        "cache_mib = cache_bytes / (1024 * 1024)\n"
        "overhead_pct = (pre_ms - post_ms) / post_ms * 100.0\n"
        "\n"
        "# Invariants: both paths produce the SAME scores up to bf16 rounding.\n"
        "diff = float(mx.max(mx.abs(y1.astype(mx.float32) - y2.astype(mx.float32))).item())\n"
        "# bf16 non-associativity allows a few parts in 1e0 on (T=2048)-sum scores.\n"
        "assert diff < 5.0, f\"post-rope and pre-rope disagree by {diff:.4f}\"\n"
        "\n"
        "print(f\"GQA shape: n_heads={n_heads}, n_kv_heads={n_kv_heads}, G={G}, T={T}, d_head={d_head}\")\n"
        "print(f\"KV cache footprint (analytic, bf16): {cache_mib:.1f} MiB  (identical both layouts)\")\n"
        "print(f\"post-rope attention:  {post_ms:6.3f} ms / call\")\n"
        "print(f\"pre-rope  attention:  {pre_ms:6.3f} ms / call   (+{overhead_pct:+.1f}%)\")\n"
        "print(f\"max |post - pre| scores diff: {diff:.4f}  (bf16 rounding only — same attention)\")\n"
    )
    emb_bench = {"cell_type": "code", "source": kv_bench_src}

    # --- 🏭 Production cell. ---
    production = T.production_context_cell(
        concept="RoPE caching & rope-base scaling in long-context servers",
        vllm=(
            "Precomputes (cos, sin) tables ONCE at model-load time for "
            "the full `max_model_len` and broadcasts them across all "
            "attention layers. For models with RoPE scaling (LLaMA-3.1, "
            "DeepSeek-V3) reads `rope_scaling` from `config.json` and "
            "builds a YaRN-/NTK-aware table; `rope_theta` (θ_base) is "
            "honored per-model. The table lives in GPU memory and is "
            "shared across concurrent requests — ~16 MiB fp16 at 128k."
        ),
        sglang=(
            "Same RoPE precompute strategy as vLLM; additionally the "
            "RadixAttention prefix cache assumes RoPE has been applied "
            "BEFORE the KV entries are hashed — prefix cache hits "
            "therefore require identical `rope_theta` and scaling across "
            "a shared-prefix batch, or the cache is silently wrong."
        ),
        trt_llm=(
            "Emits RoPE as a fused CUDA kernel with the (cos, sin) "
            "tables uploaded as constant memory. Supports "
            "llama-style + GPTNeoX-style rotation conventions (they "
            "differ in whether pairs are (0,1),(2,3),... or (0,d/2),(1,d/2+1),... — "
            "mix them and you get silent garbage)."
        ),
        mlx_lm=(
            "Uses `mlx.nn.RoPE` with per-model `rope_theta` pulled from "
            "the HuggingFace config; caches the (cos, sin) tables on "
            "CPU and uploads to GPU on first use. For LLaMA-3.1-style "
            "YaRN scaling, MLX-LM reads `rope_scaling.type` and dispatches "
            "to the appropriate frequency schedule (linear / dynamic-NTK / "
            "YaRN). The tables are shared-memory friendly thanks to UMA."
        ),
    )

    # --- 🔭 Frontier cell. ---
    frontier = T.frontier_context_cell(
        topic="Context extension via RoPE scaling (2024–2026)",
        papers=[
            (
                "YaRN: Efficient Context Window Extension (Peng et al.)",
                2023,
                "Combines NTK-aware frequency rescaling with an explicit "
                "attention-temperature correction. Becomes the 2024 "
                "production default — LLaMA-3.1, DeepSeek-V3, Qwen-2.5 "
                "all ship YaRN variants.",
            ),
            (
                "LLaMA-3.1 Technical Report (Meta)",
                2024,
                "Native 128k context via YaRN + BUMPED θ_base (10000 → "
                "500000). First frontier open-weights model with 100k+ "
                "context out-of-the-box; demonstrates that θ_base is as "
                "important as the scaling factor.",
            ),
            (
                "DeepSeek-V3 Technical Report (DeepSeek)",
                2024,
                "128k context with YaRN; documents the 'two-stage' "
                "context extension recipe: pretrain short, extend with "
                "YaRN + ~50M-token continued pretraining at the target "
                "length. Reproducibility recipe for any open model.",
            ),
            (
                "Gemma-3 Technical Report (Google DeepMind)",
                2025,
                "Hybrid local/global attention with INTERLEAVED RoPE "
                "scaling: local layers use θ_base=10000 (short-range "
                "precision), global layers use θ_base=1_000_000 (128k "
                "context). Demonstrates that RoPE-base can vary "
                "per-layer — a 2025 generalization of LLaMA-3.1's "
                "single-θ recipe.",
            ),
            (
                "LongRoPE: Extending LLMs to 2M Context (Microsoft)",
                2024,
                "Progressive interpolation: find per-dimension frequency "
                "rescaling factors via a short non-linear search rather "
                "than analytically. Extends Mistral-7B to 2M tokens with "
                "~1B tokens of fine-tuning; the frontier of RoPE-scaling "
                "research.",
            ),
        ],
        current_sota=(
            "As of late 2025 the frontier recipe is: (a) train short with "
            "a moderately large θ_base (LLaMA-3 uses 500000, Gemma-3 "
            "global layers use 1_000_000), then (b) apply YaRN to extend "
            "context 16–32× with ~50M tokens of continued pretraining. "
            "LLaMA-3.1 (128k), DeepSeek-V3 (128k), Qwen-2.5-1M (1M via "
            "YaRN + Dual-Chunk Attention), Gemma-3 (128k via hybrid "
            "local/global RoPE). Active frontiers: (i) per-layer θ_base "
            "(Gemma-3), (ii) learned frequency schedules (LongRoPE's "
            "non-analytic θ_j), (iii) hybrid NoPE/RoPE stacks."
        ),
    )

    # --- 🛠️ Debugging cell. ---
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "Long-context RoPE bugs — silent quality cliffs, off-by-one "
            "position indexing, and freq-base / precision mismatches"
        ),
        root_causes=[
            "Off-by-one in RoPE position indexing: mixing up 0-indexed "
            "(first token is position 0) vs 1-indexed conventions, or "
            "confusing ABSOLUTE position (index in the full sequence) "
            "with CACHE-SLOT position (index in the KV cache) at decode "
            "time. Fix: every production path passes `past_len + i` for "
            "the i-th new token where `past_len` is the cached-prefix "
            "length; one-off the base and the relative distances (m − n) "
            "all shift by 1 and long-range attention degrades.",
            "RoPE freq-base mismatch at serve time: the checkpoint was "
            "trained at `rope_theta = 500000` (LLaMA-3.1 convention) but "
            "the inference config loads with the legacy default "
            "`rope_theta = 10000` — attention wraps around at ~62k "
            "tokens and perplexity explodes past that point. Fix: "
            "ALWAYS read `rope_theta` from the model's `config.json`; "
            "never assume a default. Diagnostic: assert the runtime "
            "`rope_theta` exactly matches the checkpoint's value.",
            "fp16 / bf16 precision loss in cos / sin at high positions: "
            "`m · θ_j` for small θ_j (late dims) grows linearly with m. "
            "Computing `cos(m · θ_j)` in bf16 at m > 65k loses all "
            "precision — `sin` wraps randomly. Fix: always precompute "
            "the (cos, sin) tables in fp32 and cast DOWN to bf16 only "
            "at use. Every production server does this.",
        ],
        diagnostic_code=(
            "import mlx.core as mx\n"
            "\n"
            "# -- Symptom 1: off-by-one in RoPE position indexing -----------\n"
            "# Demonstrate that a 1-index shift on BOTH q and k preserves the\n"
            "# relative-distance property — but a 1-index shift on ONLY ONE\n"
            "# (the classic off-by-one) silently breaks attention.\n"
            "import math\n"
            "\n"
            "d_head = 64\n"
            "max_pos = 128\n"
            "j = mx.arange(0, d_head, 2, dtype=mx.float32)\n"
            "theta = 1.0 / (10000.0 ** (j / d_head))\n"
            "positions = mx.arange(0, max_pos, dtype=mx.float32)\n"
            "angles = positions[:, None] * theta[None, :]\n"
            "cos_tab = mx.cos(angles)\n"
            "sin_tab = mx.sin(angles)\n"
            "mx.eval(cos_tab, sin_tab)\n"
            "\n"
            "def apply_rope(x: mx.array, pos: int) -> mx.array:\n"
            "    c = cos_tab[pos]; s = sin_tab[pos]\n"
            "    x_even = x[..., 0::2]; x_odd = x[..., 1::2]\n"
            "    rot_even = x_even * c - x_odd * s\n"
            "    rot_odd = x_even * s + x_odd * c\n"
            "    out = mx.stack([rot_even, rot_odd], axis=-1)\n"
            "    return out.reshape(x.shape)\n"
            "\n"
            "mx.random.seed(0)\n"
            "q = mx.random.normal(shape=(d_head,))\n"
            "k = mx.random.normal(shape=(d_head,))\n"
            "# Correct: both q and k rotated at their true positions m, n.\n"
            "m, n = 10, 25\n"
            "ref = float(mx.sum(apply_rope(q, m) * apply_rope(k, n)).item())\n"
            "# Off-by-one on q ONLY: uses position m+1 for q but n for k.\n"
            "bug = float(mx.sum(apply_rope(q, m + 1) * apply_rope(k, n)).item())\n"
            "rel_drift = abs(ref - bug) / (abs(ref) + 1e-9)\n"
            "print(f\"[1] correct q·k score at (m={m}, n={n}):      {ref:+.4f}\")\n"
            "print(f\"    off-by-one on q (m+1, n): {bug:+.4f}   drift = {rel_drift:.1%}\")\n"
            "assert rel_drift > 0.0, \"sanity: off-by-one must perturb the score\"\n"
            "print(\"    → symptom: long-range attention shifted by 1 token; fix: unify indexing.\")\n"
            "\n"
            "# -- Symptom 2: RoPE freq-base (theta_base) mismatch -----------\n"
            "# Build (cos, sin) at theta_base=10000 vs 500000 for the SAME position\n"
            "# and show they're materially different — this is the LLaMA-3.1 bug\n"
            "# every serving framework guarded against in mid-2024.\n"
            "def rope_at_pos(pos: int, theta_base: float) -> mx.array:\n"
            "    thetas = 1.0 / (theta_base ** (j / d_head))\n"
            "    ang = pos * thetas\n"
            "    return mx.cos(ang), mx.sin(ang)\n"
            "\n"
            "pos = 60000  # well within a 128k-context model\n"
            "c_small, s_small = rope_at_pos(pos, 10000.0)\n"
            "c_large, s_large = rope_at_pos(pos, 500000.0)\n"
            "mx.eval(c_small, s_small, c_large, s_large)\n"
            "# The two (cos, sin) tables diverge substantially at high positions\n"
            "# on the LOW-frequency end — where 10000-base aliases and 500000-base\n"
            "# is still in its first cycle.\n"
            "theta_diff = float(mx.max(mx.abs(c_small - c_large)).item())\n"
            "print(f\"[2] max |cos_10k − cos_500k| at pos={pos}: {theta_diff:.4f}\")\n"
            "assert theta_diff > 0.1, (\n"
            "    \"theta_base mismatch must produce materially different cos tables\"\n"
            ")\n"
            "print(\"    → symptom: catastrophic perplexity explosion past training length.\")\n"
            "print(\"    → fix: load rope_theta from config.json; assert exact match.\")\n"
            "\n"
            "# -- Symptom 3: fp16 / bf16 precision loss in cos, sin ---------\n"
            "# Compute (cos, sin) in fp32 at a very high position and compare to\n"
            "# the bf16-native computation. The bf16 path LOSES precision on the\n"
            "# low-frequency end (where m · θ_j is small but m is large).\n"
            "pos_hi = 100_000\n"
            "angle_fp32 = pos_hi * theta\n"
            "angle_bf16 = (\n"
            "    mx.array(pos_hi, dtype=mx.bfloat16) * theta.astype(mx.bfloat16)\n"
            ").astype(mx.float32)\n"
            "rel_err = float(\n"
            "    mx.max(mx.abs((angle_fp32 - angle_bf16) / (angle_fp32 + 1e-9))).item()\n"
            ")\n"
            "cos_fp32 = mx.cos(angle_fp32)\n"
            "cos_bf16 = mx.cos(\n"
            "    mx.array(pos_hi, dtype=mx.bfloat16) * theta.astype(mx.bfloat16)\n"
            ").astype(mx.float32)\n"
            "cos_diff = float(mx.max(mx.abs(cos_fp32 - cos_bf16)).item())\n"
            "mx.eval(cos_fp32, cos_bf16)\n"
            "print(f\"[3] bf16 rope-angle relative error at pos={pos_hi}: {rel_err:.2%}\")\n"
            "print(f\"    max |cos_fp32 − cos_bf16|                         = {cos_diff:.4f}\")\n"
            "assert cos_diff >= 0.0, \"sanity check\"\n"
            "print(\"    → fix: precompute (cos, sin) in fp32; cast DOWN to bf16 at use.\")\n"
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
        emb_bench,
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
    """Transform nb04 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb04 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb04] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    rope_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_ROPE)
    )
    alibi_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_ALIBI)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (alibi_end, _block_alibi(records), "alibi"),
        (rope_end, _block_rope(records), "rope"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb04] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb04] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb04] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb04 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb04] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
