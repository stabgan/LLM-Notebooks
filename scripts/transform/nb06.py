"""Interview-grade transform for notebook 06 (Transformer Architecture).

This module inserts the six interview-layer strata into
``06_transformer_architecture.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): pre-norm vs post-norm (where the LayerNorm/RMSNorm
sits relative to the residual add — pre-norm = `x + sublayer(norm(x))`,
post-norm = `norm(x + sublayer(x))`), parameter-count formulas (the
Karpathy `12·n_layers·d_model²` estimate — derive from Q/K/V/O
projections at `4·d²` plus MLP at `8·d²` under the classic 4× FFN
expansion), residual connections (why deep networks need them —
vanishing-gradient argument plus the additive-identity-path), LayerNorm
vs RMSNorm (RMSNorm drops mean-centering; equivalent quality, ~10-15%
faster because one less reduction pass over the hidden dim), FFN
expansion ratio (4× standard, 8/3× with SwiGLU in LLaMA to keep
parameter count matched), weight tying (`wte ↔ lm_head` saves
`vocab_size · d_model` parameters and improves small-model quality).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.2, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb06
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

# scripts/transform/nb06.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "06_transformer_architecture.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 6

# Markers that indicate this notebook has already been transformed.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb06-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# Both anchors are resolved against the cell list BEFORE any insertion so
# insertions can proceed bottom-up without invalidating anchors.
#
# "## Layer Normalization: LayerNorm vs RMSNorm" — after the RMSNorm-
#   from-scratch code block and before the common-error / residual
#   sections. We insert the norm-focused interview block (q01, q02, q04
#   + 📐-1 latency norm bench + whiteboard-A RMSNorm verification).
# "## Residual Connections" — after the residual narrative and
#   demonstration, before the Complete TransformerBlock. We insert the
#   architecture block (q03, q05, q06, q07 + 📐-2 param-memory bench +
#   whiteboard-B parameter-count + 🏭 + 🔭 + 🛠️).
_ANCHOR_NORM = "## Layer Normalization: LayerNorm vs RMSNorm"
_ANCHOR_RESIDUAL = "## Residual Connections"


# ---------------------------------------------------------------------------
# Notebook I/O (raw JSON; Requirement 19.4 permits this for .ipynb edits)
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``06_transformer_architecture.ipynb`` as a JSON dict."""
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
    """Return the seven Question_Bank records for nb06.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle               — q01, q02, q03, q04, q06
        research_engineer — q01, q02, q03, q05, q06, q07
        systems_engineer  — q02, q04, q05, q07

    Topic coverage (task brief — LLD-4):
        q01 — LayerNorm vs RMSNorm (derivation; RMSNorm drops mean-
              centering; equivalent quality, ~10-15% faster)
        q02 — Pre-norm vs post-norm (where the norm sits relative to
              the residual; training stability; gradient flow)
        q03 — Residual connections (why they're critical; vanishing-
              gradient argument; deep-network trainability)
        q04 — Parameter count of a transformer block — derive the
              Karpathy `12·d_model²` estimate from Q/K/V/O + MLP
        q05 — FFN expansion ratio (4× in GPT, 8/3× with SwiGLU in
              LLaMA to match params under gating)
        q06 — Weight tying (wte ↔ lm_head) — memory win and quality
              argument
        q07 — DeepNorm / Sub-LN / QK-norm — the 2022–2025 frontier on
              training stability at 100B+ params
    """
    return [
        {
            "id": "nb06-q01",
            "notebook": _NB_FILENAME,
            "section": "LayerNorm vs RMSNorm — Derivation",
            "difficulty": "warmup",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["layernorm", "rmsnorm", "normalization", "derivation"],
            "question": (
                "Write the LayerNorm formula and the RMSNorm formula "
                "side by side. What exactly does RMSNorm drop, and "
                "why does RMSNorm match LayerNorm quality on modern "
                "LLMs despite removing that step?"
            ),
            "answer_key_points": [
                "LayerNorm (Ba et al. 2016): `y = γ · (x - μ) / σ + β`, where μ = mean(x, axis=-1), σ = std(x, axis=-1), γ and β are learned per-feature (shape (d,)). TWO reductions over the hidden dim (mean, then variance), a subtract, a divide, an elementwise multiply, and an add — ~5 elementwise passes.",
                "RMSNorm (Zhang & Sennrich 2019): `y = γ · x / RMS(x)`, where `RMS(x) = sqrt(mean(x², axis=-1) + eps)`. No mean subtraction, no β (no shift). ONE reduction (mean of squares), a sqrt, a divide, a multiply. ~3 elementwise passes — ~10-15% fewer ops and one fewer reduction.",
                "What RMSNorm drops: the RE-CENTERING step (`x - μ`) and the LEARNED SHIFT (`β`). Both were thought essential in 2016 (centering controls the mean of activations, shift gives the layer a learnable bias) but empirically neither moves the needle at LLM scale.",
                "Why RMSNorm is equivalent in quality: for high-dimensional Gaussian-like activations, `Var(x) ≈ E[x²] - (E[x])² ≈ E[x²]` because the mean is already near zero (forced by the previous layer's LayerNorm or residual accumulation). So `RMS(x) ≈ std(x)` and the division does substantially the same variance-normalization work. The learned β was found redundant with the bias in the following Linear layer — both are learnable shifts of the same axis.",
                "Empirical validation: LLaMA, LLaMA-2, LLaMA-3, LLaMA-3.1, Mistral, Qwen, Gemma, DeepSeek-V3 — every 2023+ major open-weights LLM ships with RMSNorm. GPT-3 used LayerNorm; GPT-4 and successors are inferred to use RMSNorm (OpenAI hasn't published but the throughput pattern fits).",
                "Throughput win at scale: one fewer reduction pass over d_model means ~15% less HBM traffic for the norm and one fewer synchronization point. At d_model=8192, batch 2048, the cumulative savings across all ~80 norms per forward is measurable (single-digit % end-to-end) — worth it at scale, negligible at small scale.",
                "Beyond basic story: PyTorch 2.1+ ships a FUSED RMSNorm kernel (torch.nn.functional.rms_norm); Flash-LayerNorm (Triton) exists for LayerNorm; MLX ships `mx.fast.rms_norm` / `mx.fast.layer_norm` as fused Metal kernels. The kernel-fusion difference is what actually gets you the 10-15% speedup — the op-count arg alone doesn't close the gap without a fused kernel.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'RMSNorm is faster because it has fewer ops'. "
                "The op count alone doesn't help without a fused kernel "
                "— PyTorch with nn.LayerNorm spends most of the time in "
                "memory traffic, not arithmetic. RMSNorm's wall-clock win "
                "comes from fused Metal/CUDA kernels that exploit the "
                "ONE-reduction property to avoid a second pass over HBM."
            ),
            "references": [
                "https://arxiv.org/abs/1607.06450",
                "https://arxiv.org/abs/1910.07467",
                "https://arxiv.org/abs/2302.13971",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q02",
            "notebook": _NB_FILENAME,
            "section": "Pre-norm vs Post-norm",
            "difficulty": "core",
            "roles": ["mle", "research_engineer", "systems_engineer"],
            "topic_tags": ["pre-norm", "post-norm", "residual", "training-stability"],
            "question": (
                "Write the pre-norm and post-norm block formulas side "
                "by side. Why did the 2017 Transformer paper use "
                "post-norm but every LLM since GPT-2 uses pre-norm? "
                "What changes about the gradient path?"
            ),
            "answer_key_points": [
                "Post-norm (Vaswani 2017): `y = Norm(x + Sublayer(x))`. The norm sits AFTER the residual add. Every sublayer output is renormalized along with the residual path.",
                "Pre-norm (GPT-2, 2019 onwards): `y = x + Sublayer(Norm(x))`. The norm sits INSIDE the residual, applied to x BEFORE the sublayer. The residual path `x + ...` is untouched by norm — gradients flow through the `+` identity directly.",
                "Gradient-path argument (the reason pre-norm won): in post-norm, the gradient going BACK through a block multiplies by the Jacobian of Norm at every step. Stacking 30+ post-norm blocks → gradient norm can blow up OR vanish exponentially (depending on the LayerNorm scaling) — requires aggressive LR warmup and limits depth.",
                "In pre-norm, the residual connection `x + ...` has gradient 1.0 along the identity path regardless of depth. The gradient to `x` in `y = x + f(Norm(x))` is `1 + df/dx`, which is always at least `1` in expectation. You can stack 100+ pre-norm blocks without warmup and they train stably from step 0.",
                "Cost: pre-norm's residual stream grows in magnitude (each block ADDS to it without renormalization). After 30 blocks the residual's variance is ~30× the input variance. The final output norm (one more Norm at the end) renormalizes it before the lm_head — pre-norm models all have a final `lm_norm` that post-norm models don't need.",
                "Post-norm IS NOT OBSOLETE in 2024: recent work (DeepNorm, 2022; Sub-LN, 2024) re-habilitates post-norm for 100B+ models by carefully scaling the residual branch (e.g., `y = Norm(α · x + Sublayer(x))` with `α` tuned to the depth). These schemes trade pre-norm's gradient stability for post-norm's output-scale stability — an active research direction.",
                "Interview-grade summary: pre-norm wins at SCALE (training 30+ layers is stable without tuning); post-norm + DeepNorm wins at DEPTH (100+ layers, once you're willing to pay for the warmup and per-depth α). GPT-2/3/4, LLaMA, Mistral, Qwen, Gemma, DeepSeek-V3 are all pre-norm. The 2017 original transformer was 6 layers — depth where post-norm was fine.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'pre-norm is more stable because the norm is "
                "applied first'. The order of ops isn't what makes it "
                "stable — it's that the RESIDUAL identity path is "
                "preserved (gradient of `x` is 1.0 at every depth). "
                "Moving Norm anywhere else in the block, as long as it's "
                "OUTSIDE the residual branch, gets the same benefit."
            ),
            "references": [
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/2002.04745",
                "https://arxiv.org/abs/2203.00555",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q03",
            "notebook": _NB_FILENAME,
            "section": "Residual Connections — Why They're Critical",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["residual", "vanishing-gradient", "deep-networks", "identity-path"],
            "question": (
                "Why do residual connections (`x + f(x)`) make deep "
                "networks trainable? Write down the backward-pass "
                "gradient for a stack of N residual blocks and "
                "explain WHY the identity-path gradient dominates."
            ),
            "answer_key_points": [
                "Setup: a single residual block computes `y = x + f(x)`, where `f` is the sublayer (attention or FFN with its norm). The Jacobian of `y` w.r.t. `x` is `I + f'(x)` — identity PLUS the sublayer Jacobian.",
                "Stack N blocks: `x_0 → x_1 = x_0 + f_1(x_0) → ... → x_N = x_{N-1} + f_N(x_{N-1})`. Backward: `dL/dx_0 = dL/dx_N · Π_{i=1..N} (I + f_i'(x_{i-1}))`.",
                "Expand the product via distributivity: `Π (I + f_i') = I + Σ f_i' + Σ_{i≠j} f_i'·f_j' + ...`. The FIRST term is identity `I` — present in every path, never decays with depth. Gradient to x_0 always includes an UNATTENUATED copy of the output gradient.",
                "Without residuals (`y = f(x)`): backward is `dL/dx_0 = dL/dx_N · Π f_i'(x_{i-1})`. This is a pure product. If each Jacobian has spectral norm < 1 (common for well-initialized layers with saturating activations), the gradient DECAYS exponentially: ‖dL/dx_0‖ ≤ ρ^N · ‖dL/dx_N‖ for ρ < 1. At N=30, ρ=0.9: gradient is `(0.9)^30 ≈ 0.04`× — 25× attenuation. At N=100: `≈ 2.7e-5` — effectively zero signal to x_0.",
                "With residuals: even if every `f_i'` has spectral norm 0 (dead sublayers), the identity path survives: `dL/dx_0 = dL/dx_N`. The network gracefully DEGRADES to a shallower network rather than failing outright. Every extra residual block can make the model no worse than the shallower version — the theoretical basis for 'deeper is never worse'.",
                "Historical context: ResNet (He et al. 2015) introduced residuals for 152-layer CNNs; training a plain 20-layer net was already hard in 2015. The Transformer (2017) adopted the same idea. Without residuals, GPT-3's 96 layers would be untrainable; LLaMA-3's 80 layers likewise.",
                "Corollary for inference stability: because every residual block's output is CLOSE to its input (f(x) is a small perturbation), activations don't drift catastrophically. The magnitude of the residual stream grows ~sqrt(n_layers) — predictable, bounded, easy to tune norms for.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Answering 'residuals help by providing extra "
                "parameters'. They have ZERO extra parameters — they're "
                "just the `+` op on the forward pass. The benefit is "
                "entirely on the BACKWARD pass, through the identity-"
                "path Jacobian. Saying 'they let the network skip "
                "layers' is closer but still vague — the precise story "
                "is the additive-identity gradient term."
            ),
            "references": [
                "https://arxiv.org/abs/1512.03385",
                "https://arxiv.org/abs/1603.05027",
                "https://arxiv.org/abs/1706.03762",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q04",
            "notebook": _NB_FILENAME,
            "section": "Parameter Count of a Transformer Block",
            "difficulty": "core",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["parameter-count", "karpathy-formula", "attention", "mlp"],
            "question": (
                "Derive the parameter count of a single transformer "
                "block as a function of d_model (assume n_heads · "
                "d_head = d_model, FFN expansion = 4, no biases). "
                "Show how you arrive at the Karpathy "
                "`12 · n_layers · d_model²` estimate for the whole "
                "model (ignoring embedding and norms)."
            ),
            "answer_key_points": [
                "Attention projections: Q, K, V, O each map `d_model → d_model` (because n_heads · d_head = d_model). Four Linear layers with no bias, each with `d_model²` params. Attention total: `4 · d_model²` per block.",
                "MLP (classic GPT, 4× expansion, no gating): two Linears `d_model → 4·d_model → d_model`. Params: `d_model · (4·d_model) + (4·d_model) · d_model = 8·d_model²`. MLP total: `8 · d_model²` per block.",
                "Attention + MLP: `4·d² + 8·d² = 12 · d_model²` per block. Multiply by `n_layers` for the backbone: `12 · n_layers · d_model²`. This is the Karpathy estimate (nanoGPT, Let's build GPT from scratch).",
                "At d_model=768 (GPT-2 small), 12 layers: 12 · 12 · 768² = 85M params — matches the published 124M minus the embedding (50257 · 768 = 39M) ⇒ 85M backbone is correct.",
                "What the formula IGNORES: (a) the embedding (`vocab_size · d_model` — often comparable to the backbone at small scale); (b) the lm_head (same shape as embedding; half the cost if tied, see q06); (c) layer norms (each LN/RMSNorm has `d_model` or `2·d_model` params — negligible, ~0.01% of total); (d) positional embeddings if learned (absolute-position has `max_seq_len · d_model`, zero for RoPE).",
                "Modifications in modern LLMs: (a) SwiGLU MLP has THREE Linears of total shape `3 · d_model · d_ff` — to keep 12·d² matched, `d_ff = 8/3 · d_model` (not 4·d_model). See q05. Formula holds; (b) GQA with `n_kv_heads < n_heads` shrinks K and V: attention becomes `2·d² + 2·(d·d_kv)` where `d_kv = d_model · n_kv_heads/n_heads` — saves up to ~3·d² per block at 8:1 GQA ratio.",
                "Scaling checks: LLaMA-3-8B has d=4096, L=32 → backbone ≈ 12·32·4096² = 6.4B params. LLaMA-3-70B has d=8192, L=80 → 12·80·8192² = 64.4B — tight match. The formula is the single cheapest model-scale sanity check.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Forgetting the `W_O` output projection in attention. "
                "Beginners count 3 · d² (Q, K, V) and come up short. "
                "W_O is the matrix that recombines the per-head outputs "
                "back into d_model — a full d×d matrix and a required "
                "part of the block. Q/K/V/O = FOUR projections, not "
                "three."
            ),
            "references": [
                "https://karpathy.github.io/2022/06/29/llm-examples/",
                "https://github.com/karpathy/nanoGPT",
                "https://arxiv.org/abs/2302.13971",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q05",
            "notebook": _NB_FILENAME,
            "section": "FFN Expansion Ratio — 4× vs 8/3× with SwiGLU",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["ffn", "swiglu", "expansion-ratio", "gating"],
            "question": (
                "Why is the classic GPT FFN expansion `d_ff = 4 · "
                "d_model`, but LLaMA uses `d_ff = 8/3 · d_model` with "
                "SwiGLU? Show the parameter-matching derivation and "
                "explain why the 8/3 looks weird but is the "
                "mathematically right choice."
            ),
            "answer_key_points": [
                "Classic GPT FFN (no gating): `y = W_2 · gelu(W_1 · x)`. Two matrices: `W_1: d → 4·d`, `W_2: 4·d → d`. Parameters: `d · 4d + 4d · d = 8·d²`. Expansion 4× is a Vaswani-paper constant, chosen empirically for best compute-quality tradeoff.",
                "SwiGLU FFN (LLaMA, PaLM, DeepSeek): `y = W_3 · (silu(W_1 · x) ⊙ (W_2 · x))`. THREE matrices instead of two. `W_1: d → d_ff`, `W_2: d → d_ff`, `W_3: d_ff → d`. Parameters: `3 · d · d_ff`.",
                "To keep the SwiGLU FFN parameter-matched to the classic GPT 4× FFN: `3 · d · d_ff = 8 · d²` ⇒ `d_ff = 8/3 · d ≈ 2.67 · d`. Concretely: LLaMA-3-8B has d=4096, d_ff=14336 ≈ (8/3)·4096 · 1.3 (rounded up to a multiple of 256 for hardware); pre-rounding `2.67·d = 10923`.",
                "Why match parameters, not expansion ratio: the COMPUTE-QUALITY tradeoff is set by the parameter count per layer, not the hidden dim of the FFN. SwiGLU at `d_ff=8/3·d` has the SAME parameters as GELU at `d_ff=4·d` and empirically MATCHES or BEATS the quality. Chinchilla-optimal scaling is defined in total-params-per-token.",
                "Why SwiGLU wins at equal parameters: the GATING mechanism (`silu(u) ⊙ v`) is more expressive than a fixed nonlinearity. The `silu(u)` branch acts as a soft gate over the `v` branch — the FFN can learn to SELECTIVELY activate directions in feature space. GELU applies the same nonlinearity uniformly. PaLM (Chowdhery et al. 2022) published +0.5-1.0 pp improvement on common benchmarks at matched params.",
                "Why the 8/3 looks weird: it's an IRRATIONAL scaling that breaks the clean 'multiple of 4' convention of classic transformers. In practice every LLM rounds UP to the nearest multiple of 128, 256, or 512 for Metal / CUDA tile alignment — LLaMA-3-8B uses d_ff=14336 (multiple of 256); DeepSeek-V3 uses d_ff=18432 (multiple of 256). The 8/3 is a derivation target, not the final number.",
                "2024 frontier: Gemma-2 uses SwiGLU at 8/3×; Qwen-2.5 uses SwiGLU at 8/3×. GeGLU (gated GELU) is an alternative that some papers (GLaM, T5-v1.1) use — same 3-matrix structure, GELU instead of SiLU in the gate, slightly worse at scale but better early in training. SwiGLU has won in 2024–2025 deployments.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Answering 'SwiGLU is faster because 8/3 < 4 — fewer "
                "params'. The whole point is that SwiGLU at 8/3 has the "
                "SAME parameter count as GELU at 4 (matched by "
                "construction). It's not faster per-param; it's the "
                "same parameter budget delivering better quality "
                "because gating is more expressive."
            ),
            "references": [
                "https://arxiv.org/abs/2002.05202",
                "https://arxiv.org/abs/2302.13971",
                "https://arxiv.org/abs/2204.02311",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q06",
            "notebook": _NB_FILENAME,
            "section": "Weight Tying — wte ↔ lm_head",
            "difficulty": "stretch",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["weight-tying", "embedding", "lm-head", "parameter-efficiency"],
            "question": (
                "What is weight tying between `wte` (word-token "
                "embedding) and `lm_head`? How much memory does it "
                "save, and why does it improve quality on small models "
                "but is sometimes dropped at large scale?"
            ),
            "answer_key_points": [
                "Definition: in most LLMs, `wte` (`vocab_size × d_model` embedding lookup) and `lm_head` (`d_model × vocab_size` output projection from hidden to logits) have the SAME shape modulo transpose. Weight tying sets `lm_head.weight = wte.weight.T` — one tensor, used twice.",
                "Memory win: WITHOUT tying you store `2 · vocab_size · d_model` params. WITH tying you store `vocab_size · d_model`. Savings: `vocab_size · d_model`. At vocab=128k (LLaMA-3), d=4096: 128k · 4096 · 2 bytes (bf16) = 1 GB saved per copy. Non-trivial at small-model scale.",
                "Quality argument (Press & Wolf 2017, Inan et al. 2017): the embedding matrix `E` maps tokens to a d-dim space; the lm_head maps d-dim hidden states back to a distribution over tokens. At convergence, the optimal output projection is CLOSE to (a scaled version of) the embedding — pointing each token's hidden state back toward its own embedding. Tying the weights USES THIS STRUCTURE directly: fewer parameters, faster convergence, better at small model sizes.",
                "Empirical evidence: GPT-2 (all sizes), BERT, T5 — all use tied embeddings. The original GPT-2 paper reports that TYING was necessary for convergence on small models at small data; untying hurt on 124M, neutral on 1.5B.",
                "Why some large models UNTIE: at 100B+ scale, the dataset is large enough that the untied `lm_head` can learn its own specialized projection — the 'embedding ≈ lm_head' prior is too restrictive at scale. PaLM (540B) untied; GPT-4 (inferred) untied; DeepSeek-V3 untied. LLaMA-3 stays TIED even at 70B — so the choice is not universal.",
                "Variations in 2024–2025: LLaMA-3 and most Qwen / Mistral variants still tie. DeepSeek-V3, Grok, and inferred OpenAI models untie. The memory saving is small at 100B scale (a few % of total) but the quality delta is usually < 0.1 pp — small enough that engineering convenience (independent gradient flow through the output head) wins for big labs.",
                "Implementation detail: when tied, the training optimizer must handle the fact that `wte` receives gradients BOTH from embedding lookups (sparse) AND from the lm_head (dense). AdamW accumulates both into the shared tensor — subtle but usually correct. Untied heads have cleaner separate gradient accumulation.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'tying saves HALF the parameters'. It saves "
                "EXACTLY `vocab_size · d_model` — the embedding / "
                "lm_head pair is two of the ~10 largest tensors in a "
                "transformer. At LLaMA-3-8B (8B total params), the "
                "embedding is 128k · 4096 ≈ 524M — tying saves 524M, "
                "or ~6.5% of the model. Not half."
            ),
            "references": [
                "https://arxiv.org/abs/1608.05859",
                "https://arxiv.org/abs/1611.01462",
                "https://github.com/openai/gpt-2",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb06-q07",
            "notebook": _NB_FILENAME,
            "section": "DeepNorm / Sub-LN / QK-norm — Frontier on Training Stability",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["deepnorm", "sub-ln", "qk-norm", "training-stability", "frontier"],
            "question": (
                "At 100B+ params even pre-norm + residual models "
                "become unstable during training. Explain the "
                "2022–2025 frontier on this: DeepNorm's residual "
                "scaling, Sub-LN's placement, and Gemma-2's QK-norm. "
                "What specific failure mode does each address?"
            ),
            "answer_key_points": [
                "Problem at 100B+: even pre-norm transformers hit 'loss spikes' during training — sudden divergences caused by activation-magnitude drift in specific layers (usually mid-depth) that cascade through the residual stream. Gradient clipping doesn't help because the issue is in the FORWARD pass, not the backward.",
                "DeepNorm (Wang et al. 2022): keeps post-norm but SCALES the residual branch: `y = Norm(α · x + Sublayer(x))`. Sets α based on depth (`α = (2N)^(1/4)` for encoder, different for decoder) to keep the residual stream variance stable at init AND at convergence. Enables 1000-layer transformers to train stably. Used in: Microsoft's Kosmos, BLOOM's 176B, earlier BaiChuan.",
                "Sub-LN (Scalable-LN, 2024): a VARIATIONAL placement of LayerNorm INSIDE the sublayer's computation (before and/or after the matmul) — the 'sub' refers to applying LN at the SUB-component level. Addresses the training instability of very deep pre-norm models by bounding the activation magnitude per-component rather than per-block.",
                "QK-norm (Henry et al. 2020; Gemma-2 2024; Qwen-2.5 2024): applies RMSNorm to Q and K BEFORE the attention dot product — `attn = softmax(norm(Q) · norm(K)ᵀ / √d)·V`. Addresses the specific failure mode where one head's Q·K logits drift to extreme magnitudes mid-training, saturating softmax and killing that head. √d scaling is necessary but not SUFFICIENT at 100B+; QK-norm is the 2024 default fix.",
                "Why all three exist: they address DIFFERENT failure modes at scale. DeepNorm: residual-stream variance drift over depth. Sub-LN: per-component activation-magnitude control. QK-norm: per-head attention-logit saturation. A modern 100B+ model often combines pre-norm + QK-norm (Gemma-2 does this) or DeepNorm + QK-norm for maximum stability.",
                "2024 frontier status: pre-norm + RMSNorm + QK-norm is the production default for 10B–100B open models (Gemma-2, Qwen-2.5, DeepSeek-V3). DeepNorm sees selective use for extreme-depth experiments. Sub-LN is still early — appears in some 2024 papers but no flagship model ships it yet.",
                "What's on the horizon (late 2025): muP (Yang et al., μ-parameterization) is a complementary approach — scales initializations and learning rates per-layer to make training invariant to width. OpenReLU-stability analyses and residual-reparameterization tricks from 2024–2025 papers are converging on the same fix space. Expect the 2026 models to inherit 2-3 of these stability primitives as 'the new normal' — the way RMSNorm became normal in 2023.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Lumping DeepNorm, Sub-LN, and QK-norm together as 'new "
                "norms'. They fix DIFFERENT failure modes and are "
                "compositional, not alternatives. DeepNorm rescales "
                "residuals; Sub-LN changes placement inside the "
                "sublayer; QK-norm applies norm to attention logits. A "
                "modern 100B+ model may use all three simultaneously."
            ),
            "references": [
                "https://arxiv.org/abs/2203.00555",
                "https://arxiv.org/abs/2010.04245",
                "https://arxiv.org/abs/2408.00118",
            ],
            "added_in": added_in,
        },
    ]



# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_norm(records: list[dict]) -> list[dict]:
    """Block inserted at the end of the 'Layer Normalization' section.

    Contents: q01 (LayerNorm vs RMSNorm derivation), q02 (pre-norm vs
    post-norm), q04 (parameter-count formula derivation), whiteboard-A
    (implement RMSNorm from scratch and verify numerically against
    `mlx.nn.RMSNorm`), 📐-1 (LayerNorm vs RMSNorm forward latency at
    production-ish shape).
    """
    q01, q02, _q03, q04 = records[0], records[1], records[2], records[3]

    # --- Whiteboard A — implement RMSNorm from scratch + verify vs nn.RMSNorm
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title=(
            "RMSNorm from scratch — verify numerically against mlx.nn.RMSNorm"
        ),
        prompt=(
            "Implement `rmsnorm(x, weight, eps=1e-6)` that computes "
            "`weight · x / sqrt(mean(x², axis=-1) + eps)` directly in "
            "MLX. Then verify your implementation matches the fused "
            "`mlx.nn.RMSNorm` to within 1e-5 float32 tolerance on "
            "random inputs at shape (2, 8, 512). Also confirm "
            "`mx.fast.rms_norm` (when available) matches."
        ),
        constraints=[
            "Use MLX throughout — no numpy, torch, jax. The input `x` has shape "
            "(..., d) with d the last axis (the normalized axis).",
            "Compute RMS along the LAST axis only, keeping dims for broadcast. "
            "`mx.mean(x * x, axis=-1, keepdims=True)` plus `mx.sqrt` plus `+ eps` "
            "inside the sqrt (not after).",
            "Apply the learned `weight` (shape `(d,)`) AS A MULTIPLIER to the "
            "normalized tensor — no bias, no mean subtraction, no shift.",
            "Compare your output to `mlx.nn.RMSNorm(d, eps=1e-6)(x)` with the "
            "fused module's `.weight` initialized to `mx.ones((d,))` so both "
            "compute the same thing. Assert `max |diff| < 1e-5` at float32.",
            "Use `mx.eval` on both outputs before taking the numeric diff.",
        ],
        complexity=(
            "Compute: O(B · T · d) — one reduction plus one sqrt plus one "
            "elementwise multiply over the hidden dim. Memory: O(B · T · d) "
            "for x and the output; O(d) for weight. One reduction pass, "
            "vs LayerNorm's two (mean AND variance)."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "import mlx.nn as _nn\n"
            "\n"
            "def rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:\n"
            "    \"\"\"RMSNorm: weight · x / sqrt(mean(x², axis=-1) + eps).\n"
            "\n"
            "    x shape (..., d). weight shape (d,). Returns same shape as x.\n"
            "    No mean subtraction, no learned bias — just variance-style\n"
            "    normalization via the root-mean-square of the hidden dim.\n"
            "    \"\"\"\n"
            "    # One reduction over the last axis; keepdims=True for broadcast.\n"
            "    _rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)\n"
            "    return weight * (x / _rms)\n"
            "\n"
            "# Random input at a production-ish shape. Names are underscore-\n"
            "# prefixed so they don't collide with the notebook's existing\n"
            "# x, weight, norm globals from earlier cells.\n"
            "_B, _T, _d = 2, 8, 512\n"
            "mx.random.seed(0)\n"
            "_x = mx.random.normal(shape=(_B, _T, _d))\n"
            "_w = mx.ones((_d,))\n"
            "mx.eval(_x, _w)\n"
            "\n"
            "# Our reference implementation.\n"
            "_out_ours = rmsnorm(_x, _w, eps=1e-6)\n"
            "\n"
            "# MLX's built-in module — same math, different accumulation order.\n"
            "_mod = _nn.RMSNorm(_d, eps=1e-6)\n"
            "# Force weight=ones so both implementations compute identical math.\n"
            "_mod.weight = _w\n"
            "_out_builtin = _mod(_x)\n"
            "mx.eval(_out_ours, _out_builtin)\n"
            "\n"
            "# Numeric agreement: same formula, same dtype ⇒ bit-close.\n"
            "_diff = float(mx.max(mx.abs(_out_ours - _out_builtin)).item())\n"
            "assert _diff < 1e-5, (\n"
            "    f\"rmsnorm disagrees with mlx.nn.RMSNorm by {_diff:.4e} (>1e-5)\"\n"
            ")\n"
            "\n"
            "# Sanity: output has the expected shape AND normalization property.\n"
            "# After RMSNorm (weight = ones), each row should have RMS ≈ 1.\n"
            "assert _out_ours.shape == (_B, _T, _d)\n"
            "_rms_after = mx.sqrt(mx.mean(_out_ours * _out_ours, axis=-1))\n"
            "mx.eval(_rms_after)\n"
            "_rms_err = float(mx.max(mx.abs(_rms_after - 1.0)).item())\n"
            "# After normalization (and with weight=1), each row should have RMS ~ 1.\n"
            "# Tolerance reflects the `eps` added inside the sqrt.\n"
            "assert _rms_err < 1e-3, f\"post-norm RMS deviates from 1: {_rms_err:.4e}\"\n"
            "\n"
            "# Optional: also match mx.fast.rms_norm if available on this MLX build.\n"
            "_fast_ok = True\n"
            "if hasattr(mx, \"fast\") and hasattr(mx.fast, \"rms_norm\"):\n"
            "    _out_fast = mx.fast.rms_norm(_x, _w, eps=1e-6)\n"
            "    mx.eval(_out_fast)\n"
            "    _fast_diff = float(mx.max(mx.abs(_out_ours - _out_fast)).item())\n"
            "    assert _fast_diff < 1e-5, (\n"
            "        f\"mx.fast.rms_norm disagrees with ours by {_fast_diff:.4e}\"\n"
            "    )\n"
            "else:\n"
            "    _fast_ok = False\n"
            "\n"
            "print(f\"✅ rmsnorm output shape: {_out_ours.shape}\")\n"
            "print(f\"✅ max |ours - nn.RMSNorm| = {_diff:.4e}  (< 1e-5)\")\n"
            "print(f\"✅ post-norm RMS within {_rms_err:.4e} of 1.0\")\n"
            "if _fast_ok:\n"
            "    print(f\"✅ mx.fast.rms_norm also matches within 1e-5\")\n"
        ),
    )

    # --- 📐-1 Complexity cell — LayerNorm vs RMSNorm forward latency ---
    complexity = T.complexity_analysis_cell(
        op="LayerNorm vs RMSNorm — forward latency & reduction count",
        flops=(
            "RMSNorm: O(B · T · d) elementwise, ONE reduction (mean-of-squares). "
            "LayerNorm: O(B · T · d), TWO reductions (mean, then variance) — "
            "hence 'RMSNorm has fewer ops' story. At d=4096 the arithmetic "
            "difference is ~15% FLOPs"
        ),
        memory=(
            "RMSNorm weights: `d` bf16 params = `2·d` bytes per layer (no bias). "
            "LayerNorm weights: `2·d` params (γ and β) = `4·d` bytes per layer. "
            "Working set for both: O(B · T · d) for x and the output"
        ),
        latency_mlx=(
            "M4 Pro, bf16, (B=1, T=2048, d=4096): nn.RMSNorm ≈ 0.6-1.0 ms/call; "
            "nn.LayerNorm ≈ 0.8-1.3 ms/call. RMSNorm wins ~20-30% on this shape "
            "on Metal. Measured below"
        ),
        scaling=(
            "Both norms are MEMORY-BANDWIDTH-BOUND — the arithmetic is cheap "
            "relative to the HBM traffic of reading B·T·d elements. The ~15% op-"
            "count win translates to a measured ~10-30% wall-clock win via "
            "kernel fusion (one reduction pass vs two) — hence every 2023+ LLM "
            "ships RMSNorm. Without a fused kernel the win disappears."
        ),
    )

    bench_src = (
        "# Benchmark: LayerNorm vs RMSNorm forward latency at LLM-scale shapes\n"
        "# Measures both at (B=1, T=2048, d=4096) bf16 with 3 warmups + mx.eval\n"
        "# inside the timed loop. Underscore-prefixed names avoid colliding\n"
        "# with the notebook's pre-existing x, norm, config globals.\n"
        "import time\n"
        "import mlx.core as mx\n"
        "import mlx.nn as _nn\n"
        "\n"
        "def bench_norm(norm_mod, x: mx.array, n_iter: int = 20, n_warmup: int = 5) -> float:\n"
        "    \"\"\"Return ms_per_call for a single norm forward pass over `x`.\"\"\"\n"
        "    # Warmup — Requirement 5.3.\n"
        "    for _ in range(n_warmup):\n"
        "        _y = norm_mod(x)\n"
        "        mx.eval(_y)\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(n_iter):\n"
        "        _y = norm_mod(x)\n"
        "        mx.eval(_y)\n"
        "    return (time.perf_counter() - t0) / n_iter * 1000.0\n"
        "\n"
        "_B, _T = 1, 2048\n"
        "print(f\"Norm forward latency at B={_B}, T={_T}, bf16 on M4 Pro:\")\n"
        "print(f\"{'d_model':>8} | {'LayerNorm':>12} | {'RMSNorm':>12} | {'speedup':>10}\")\n"
        "print(\"-\" * 52)\n"
        "for _d in (1024, 2048, 4096):\n"
        "    mx.random.seed(0)\n"
        "    _x = mx.random.normal(shape=(_B, _T, _d)).astype(mx.bfloat16)\n"
        "    mx.eval(_x)\n"
        "    _ln = _nn.LayerNorm(_d)\n"
        "    _rms = _nn.RMSNorm(_d)\n"
        "    _ln_ms = bench_norm(_ln, _x)\n"
        "    _rms_ms = bench_norm(_rms, _x)\n"
        "    _speedup = _ln_ms / _rms_ms if _rms_ms > 0 else float('nan')\n"
        "    print(f\"{_d:>8} | {_ln_ms:>10.3f} ms | {_rms_ms:>10.3f} ms | {_speedup:>9.2f}×\")\n"
        "\n"
        "# Expected: RMSNorm is 1.1-1.3× faster at d=4096 on M4 Pro because it has\n"
        "# one fewer reduction pass over the hidden dim AND uses half the weight\n"
        "# bytes (no β). At small d the fixed overhead of op launch dominates and\n"
        "# the gap shrinks.\n"
        "\n"
        "# Final sanity assertion: both norms produce the expected shape and\n"
        "# both outputs are materialized before we read any number from them.\n"
        "mx.random.seed(0)\n"
        "_xs = mx.random.normal(shape=(1, 64, 256)).astype(mx.bfloat16)\n"
        "_y_ln = _nn.LayerNorm(256)(_xs)\n"
        "_y_rms = _nn.RMSNorm(256)(_xs)\n"
        "mx.eval(_y_ln, _y_rms)\n"
        "assert _y_ln.shape == _y_rms.shape == (1, 64, 256)\n"
        "print(\"\\n💡 RMSNorm: ONE reduction (RMS). LayerNorm: TWO (mean, variance).\")\n"
    )
    bench = {"cell_type": "code", "source": bench_src}

    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
    ]





def _block_residual(records: list[dict]) -> list[dict]:
    """Block inserted at the end of the 'Residual Connections' section.

    Contents: q03 (residual connections), q05 (FFN expansion ratio),
    q06 (weight tying), q07 (DeepNorm / Sub-LN / QK-norm frontier),
    whiteboard-B (count the parameters in one transformer block and
    assert against the analytic 12·d² formula), 📐-2 (transformer-
    block memory footprint — weights + peak activation), 🏭 (how
    vLLM/SGLang/TRT-LLM/MLX-LM fuse norm + residual-add kernels),
    🔭 (2024-2026 frontier: DeepNorm, QK-norm, Sub-LN, Post-LN
    resurgence), 🛠️ (norm applied to wrong axis, forgotten residual
    connection, bf16 precision loss in norm statistics).
    """
    q03, q05, q06, q07 = records[2], records[4], records[5], records[6]

    # --- Whiteboard B — count params in a transformer block + assert formula
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title=(
            "Count the parameters in a transformer block and assert "
            "the 12·d² formula"
        ),
        prompt=(
            "Given (d_model=768, n_heads=12, d_head=64, ffn_mult=4, "
            "bias=False, classic GELU FFN — NOT SwiGLU), count the "
            "parameters in ONE transformer block (attention + FFN + "
            "two RMSNorms). Assert the count matches `12·d² + small` "
            "— i.e., within 1% of the `12·d_model²` Karpathy estimate."
        ),
        constraints=[
            "Use `mlx.nn.Linear`, `mlx.nn.RMSNorm`, and a SwiGLU-LESS FFN "
            "(two Linears, GELU activation) so the `12·d²` derivation applies. "
            "The SwiGLU variant has three matrices at 8/3·d — see q05.",
            "Build the block as a small `mlx.nn.Module` subclass with "
            "sub-modules for `.attn_q`, `.attn_k`, `.attn_v`, `.attn_o`, "
            "`.ffn_up`, `.ffn_down`, `.norm1`, `.norm2`. No biases anywhere.",
            "Count parameters by flattening `self.trainable_parameters()` from "
            "`mx.nn.Module` — iterate leaves and `.size` each tensor.",
            "Analytic expectation: attention = 4·d² (Q, K, V, O); FFN = 2 · d · (ffn_mult·d) = 8·d² at ffn_mult=4; "
            "norms = 2·d (two RMSNorms, one weight per norm). Total ≈ 12·d² + 2·d.",
            "Assert the empirical count matches the analytic 12·d² + 2·d formula "
            "EXACTLY — this is a param-count sanity check, not a numeric check.",
            "Use `mx.eval` on a forward pass to prove the block is callable.",
        ],
        complexity=(
            "Per block: 12·d² + small (norms + norms-negligible). Backbone "
            "total: `n_layers · 12 · d²`. The Karpathy scaling formula. "
            "Doubling d_model QUADRUPLES per-block params; doubling n_layers "
            "only doubles — one reason scaling d_model first is cheaper at a "
            "given compute budget."
        ),
        solution_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "import mlx.nn as _nn\n"
            "\n"
            "class _MiniBlock(_nn.Module):\n"
            "    \"\"\"Classic (non-SwiGLU) transformer block: Q/K/V/O + 2-Linear FFN + 2 RMSNorms.\"\"\"\n"
            "    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):\n"
            "        super().__init__()\n"
            "        assert d_model % n_heads == 0\n"
            "        self.d_model = d_model\n"
            "        self.n_heads = n_heads\n"
            "        self.d_head = d_model // n_heads\n"
            "        # Attention — four d→d projections, no biases.\n"
            "        self.attn_q = _nn.Linear(d_model, d_model, bias=False)\n"
            "        self.attn_k = _nn.Linear(d_model, d_model, bias=False)\n"
            "        self.attn_v = _nn.Linear(d_model, d_model, bias=False)\n"
            "        self.attn_o = _nn.Linear(d_model, d_model, bias=False)\n"
            "        # FFN — two Linears at d → ffn_mult·d → d, classic GELU (no gating).\n"
            "        _d_ff = ffn_mult * d_model\n"
            "        self.ffn_up = _nn.Linear(d_model, _d_ff, bias=False)\n"
            "        self.ffn_down = _nn.Linear(_d_ff, d_model, bias=False)\n"
            "        # Two RMSNorms (pre-norm placement) — each has one (d,) weight.\n"
            "        self.norm1 = _nn.RMSNorm(d_model)\n"
            "        self.norm2 = _nn.RMSNorm(d_model)\n"
            "\n"
            "    def __call__(self, x: mx.array) -> mx.array:\n"
            "        # Simplified forward (no multi-head reshape for the param-count demo).\n"
            "        _nx = self.norm1(x)\n"
            "        _q = self.attn_q(_nx); _k = self.attn_k(_nx); _v = self.attn_v(_nx)\n"
            "        _scores = (_q @ _k.swapaxes(-2, -1)) / math.sqrt(self.d_model)\n"
            "        _attn_out = self.attn_o(mx.softmax(_scores, axis=-1) @ _v)\n"
            "        x = x + _attn_out\n"
            "        _m = self.norm2(x)\n"
            "        return x + self.ffn_down(_nn.gelu(self.ffn_up(_m)))\n"
            "\n"
            "def count_params(mod: _nn.Module) -> int:\n"
            "    \"\"\"Sum .size across all trainable leaves of an MLX module.\"\"\"\n"
            "    _total = 0\n"
            "    def _walk(obj):\n"
            "        nonlocal _total\n"
            "        if isinstance(obj, mx.array):\n"
            "            _total += int(obj.size)\n"
            "        elif isinstance(obj, dict):\n"
            "            for _v in obj.values():\n"
            "                _walk(_v)\n"
            "        elif isinstance(obj, (list, tuple)):\n"
            "            for _v in obj:\n"
            "                _walk(_v)\n"
            "    _walk(mod.trainable_parameters())\n"
            "    return _total\n"
            "\n"
            "# Build a classic GPT-2-small block: d=768, heads=12, ffn_mult=4.\n"
            "# Names prefixed with underscore to avoid colliding with earlier cells'\n"
            "# block, d_model, n_heads globals.\n"
            "_d_model = 768\n"
            "_n_heads = 12\n"
            "_ffn_mult = 4\n"
            "_blk = _MiniBlock(_d_model, _n_heads, _ffn_mult)\n"
            "\n"
            "# Force a forward pass so all lazy parameter shapes materialize.\n"
            "_x = mx.random.normal(shape=(1, 4, _d_model))\n"
            "_y = _blk(_x)\n"
            "mx.eval(_y)\n"
            "assert _y.shape == _x.shape, f\"block must preserve shape, got {_y.shape}\"\n"
            "\n"
            "# Empirical count via trainable_parameters().\n"
            "_empirical = count_params(_blk)\n"
            "\n"
            "# Analytic formula (no biases): 4·d² (Q,K,V,O) + 2·ffn_mult·d² (up,down) + 2·d (2 RMSNorms)\n"
            "# = (4 + 2·ffn_mult) · d² + 2·d  ⇒  at ffn_mult=4: 12·d² + 2·d.\n"
            "_analytic = (4 + 2 * _ffn_mult) * _d_model * _d_model + 2 * _d_model\n"
            "_karpathy = 12 * _d_model * _d_model  # the 'small terms dropped' estimate\n"
            "\n"
            "# Assertion 1: empirical EXACTLY matches the full analytic formula.\n"
            "assert _empirical == _analytic, (\n"
            "    f\"empirical param count {_empirical:,} != analytic {_analytic:,}\"\n"
            ")\n"
            "\n"
            "# Assertion 2: the 12·d² Karpathy estimate is within 1% of the full count\n"
            "# (dominant term; the 2·d norm-weight terms are the 'small terms' dropped).\n"
            "_rel_err = abs(_empirical - _karpathy) / _empirical\n"
            "assert _rel_err < 0.01, (\n"
            "    f\"12·d² estimate misses the true count by {_rel_err:.3%} (>1%)\"\n"
            ")\n"
            "\n"
            "print(f\"MiniBlock (d_model={_d_model}, n_heads={_n_heads}, ffn_mult={_ffn_mult}):\")\n"
            "print(f\"  empirical params:    {_empirical:>12,}\")\n"
            "print(f\"  analytic (12d²+2d):  {_analytic:>12,}\")\n"
            "print(f\"  Karpathy (12·d²):    {_karpathy:>12,}\")\n"
            "print(f\"  relative error:      {_rel_err:>12.4%}\")\n"
            "print(f\"✅ empirical == analytic (exact match, no biases, no SwiGLU)\")\n"
            "print(f\"✅ Karpathy estimate within 1% of true count\")\n"
        ),
    )

    # --- 📐-2 Complexity cell — transformer block memory footprint ---
    complexity = T.complexity_analysis_cell(
        op="Transformer block memory — weights (12·d²) + peak activations",
        flops=(
            "Forward per token per layer: attention ≈ 4·d² (Q,K,V,O projections) + "
            "2·T·d (attn compute); FFN ≈ 16·d² (up+down at 4× expansion with 2 "
            "ops/param). Per-token: ~24·d² FLOPs/layer. Per-layer at B=1, T=1, "
            "d=4096: ~400 MFLOP"
        ),
        memory=(
            "Weights per block (no biases, classic GPT): `12·d² + 2·d` params. "
            "At d=4096, bf16: 12·4096² · 2 bytes ≈ 400 MiB/block. "
            "Per-forward peak activation: O(B·T·4d) for the ffn_up intermediate — "
            "dominates over the B·T·d residual stream at 4× expansion"
        ),
        latency_mlx=(
            "M4 Pro, bf16, (B=1, T=128, d=768, n_heads=12): single-block forward "
            "≈ 3-6 ms/call. Measured below across d ∈ {256, 512, 1024}"
        ),
        scaling=(
            "Weights grow as O(d_model²). Doubling d from 4k to 8k QUADRUPLES "
            "weight memory per block (400 MiB → 1.6 GiB). The FFN intermediate "
            "(B·T·4·d) is the peak activation — 4× larger than the residual "
            "stream. This is why MoE / GQA chase d² reductions: attention's d² "
            "contribution is small relative to the FFN's 8·d², and activation-"
            "checkpointing the FFN intermediate is the fastest memory win."
        ),
    )

    mem_bench_src = (
        "# Benchmark: transformer block weight memory scales as O(d²)\n"
        "# Measures parameter count AND peak MLX memory for a single forward\n"
        "# across d_model ∈ {256, 512, 1024} at fixed (B=1, T=128). All names\n"
        "# underscore-prefixed to avoid colliding with earlier-cell globals.\n"
        "import math\n"
        "import time\n"
        "import mlx.core as mx\n"
        "import mlx.nn as _nn\n"
        "\n"
        "class _Blk(_nn.Module):\n"
        "    def __init__(self, d: int, n_heads: int = 8, ffn_mult: int = 4):\n"
        "        super().__init__()\n"
        "        self.aq = _nn.Linear(d, d, bias=False)\n"
        "        self.ak = _nn.Linear(d, d, bias=False)\n"
        "        self.av = _nn.Linear(d, d, bias=False)\n"
        "        self.ao = _nn.Linear(d, d, bias=False)\n"
        "        self.u = _nn.Linear(d, ffn_mult * d, bias=False)\n"
        "        self.dn = _nn.Linear(ffn_mult * d, d, bias=False)\n"
        "        self.n1 = _nn.RMSNorm(d); self.n2 = _nn.RMSNorm(d)\n"
        "        self._d = d\n"
        "    def __call__(self, x):\n"
        "        _nx = self.n1(x)\n"
        "        _q, _k, _v = self.aq(_nx), self.ak(_nx), self.av(_nx)\n"
        "        _s = (_q @ _k.swapaxes(-2, -1)) / math.sqrt(self._d)\n"
        "        _o = self.ao(mx.softmax(_s, axis=-1) @ _v)\n"
        "        x = x + _o\n"
        "        return x + self.dn(_nn.gelu(self.u(self.n2(x))))\n"
        "\n"
        "def _count(mod):\n"
        "    _t = 0\n"
        "    def _w(o):\n"
        "        nonlocal _t\n"
        "        if isinstance(o, mx.array): _t += int(o.size)\n"
        "        elif isinstance(o, dict):\n"
        "            for _v in o.values(): _w(_v)\n"
        "        elif isinstance(o, (list, tuple)):\n"
        "            for _v in o: _w(_v)\n"
        "    _w(mod.trainable_parameters())\n"
        "    return _t\n"
        "\n"
        "_reset = getattr(mx, 'reset_peak_memory', None) or getattr(\n"
        "    getattr(mx, 'metal', None), 'reset_peak_memory', None)\n"
        "_get_peak = getattr(mx, 'get_peak_memory', None) or getattr(\n"
        "    getattr(mx, 'metal', None), 'get_peak_memory', None)\n"
        "\n"
        "print(f\"Transformer block memory at B=1, T=128, bf16:\")\n"
        "print(f\"{'d_model':>8} | {'params':>14} | {'analytic 12d²':>16} | {'peak MiB':>10} | {'ms/call':>10}\")\n"
        "print(\"-\" * 72)\n"
        "for _d in (256, 512, 1024):\n"
        "    mx.random.seed(0)\n"
        "    _blk = _Blk(_d, n_heads=8, ffn_mult=4)\n"
        "    _n = _count(_blk)\n"
        "    _ka = 12 * _d * _d\n"
        "\n"
        "    _x = mx.random.normal(shape=(1, 128, _d)).astype(mx.bfloat16)\n"
        "    mx.eval(_x)\n"
        "    # Warmup — fills caches, allocates params.\n"
        "    for _ in range(3):\n"
        "        _yb = _blk(_x); mx.eval(_yb)\n"
        "\n"
        "    if _reset:\n"
        "        try: _reset()\n"
        "        except Exception: pass\n"
        "    _t0 = time.perf_counter()\n"
        "    for _ in range(5):\n"
        "        _yb = _blk(_x); mx.eval(_yb)\n"
        "    _dt_ms = (time.perf_counter() - _t0) / 5.0 * 1000.0\n"
        "\n"
        "    _peak_mib = (_get_peak() / (1024 * 1024)) if _get_peak else 0.0\n"
        "    print(f\"{_d:>8} | {_n:>14,} | {_ka:>16,} | {_peak_mib:>10.1f} | {_dt_ms:>9.2f}\")\n"
        "\n"
        "# Key observations:\n"
        "# - 'params' == 'analytic 12d²' + 2·d (the small RMSNorm weight term).\n"
        "# - Doubling d quadruples both params (12·d²) AND peak activation memory\n"
        "#   (O(B·T·4·d) from the ffn_up intermediate).\n"
        "# - ms/call grows ~quadratically at small d (d²-bounded matmul), closer to\n"
        "#   linear once compute saturates the GPU at larger d on the M4 Pro.\n"
        "print(\"\\n💡 12·d² is the scaling law; doubling d_model 4×es per-block memory.\")\n"
    )
    mem_bench = {"cell_type": "code", "source": mem_bench_src}

    # --- 🏭 Production cell: norm fusion + residual-add kernel fusion ---
    production = T.production_context_cell(
        concept="Norm + residual-add kernel fusion",
        vllm=(
            "Uses `add_rms_norm` fused kernel — performs `x + sublayer_out` and "
            "the following RMSNorm in ONE kernel launch with ONE HBM pass. Saves "
            "~30% of norm-level wall time on Hopper; avoids materializing the "
            "post-residual `x + sublayer_out` tensor as a separate buffer. vLLM "
            "dispatches through FlashInfer's fused norm implementations"
        ),
        sglang=(
            "FlashInfer-backed fused residual+norm path (same dependency as "
            "vLLM). SGLang additionally pre-computes residual-stream statistics "
            "needed for the NEXT norm inside the current kernel, pipelining "
            "norms across depth for batched prefill. On RadixAttention cache "
            "hits the norm can be reused verbatim from a sibling sequence"
        ),
        trt_llm=(
            "TensorRT-LLM generates a specialized `fused_add_rms_norm` kernel "
            "per (d_model, dtype) at engine-build time via the `GemmPlugin` + "
            "`LayernormPlugin` fusion pattern. Supports FP8 RMSNorm weights on "
            "H100/H200. Fusion happens at engine compile; runtime is just a "
            "single launch per block boundary"
        ),
        mlx_lm=(
            "Routes through `mx.fast.rms_norm` — fused Metal kernel. Residual "
            "add + norm fusion is implicit via MLX's graph scheduler: the "
            "`x + sub(x)` followed by `RMSNorm(...)` sequence compiles to a "
            "single Metal threadgroup when shapes match (d_model ≤ 8192, T·B "
            "fits SRAM). UMA means the fused kernel avoids a round-trip through "
            "system RAM that a discrete-GPU path would pay"
        ),
    )

    # --- 🔭 Frontier cell: DeepNorm, QK-norm, Sub-LN, Post-LN resurgence ---
    frontier = T.frontier_context_cell(
        topic="Normalization & residual placement frontier (2022–2026)",
        papers=[
            (
                "DeepNet: Scaling Transformers to 1,000 Layers (Wang et al.)",
                2022,
                "Post-LN with residual-branch scaling: y = LN(α·x + Sublayer(x)) "
                "with α = (2N)^(1/4) for encoder, different for decoder. "
                "Enables 1,000-layer training; shipped in Kosmos, BLOOM-176B, "
                "and BaiChuan. Re-habilitates post-norm for extreme depth.",
            ),
            (
                "Gemma 2 Technical Report (Google DeepMind)",
                2024,
                "Combines pre-norm + RMSNorm + QK-norm (RMSNorm applied to Q and "
                "K before the attention dot product). Addresses the 100B-scale "
                "failure mode where individual attention heads' logits drift to "
                "extreme magnitudes mid-training and saturate softmax. Became "
                "the 2024 default.",
            ),
            (
                "Qwen 2.5 Technical Report (Alibaba)",
                2024,
                "Pre-norm + RMSNorm + QK-norm (same stack as Gemma 2). 0.5-72B "
                "parameter range all trained with identical norm recipe — "
                "evidence the pattern is shape-agnostic. Demonstrates QK-norm "
                "as robust across model scales.",
            ),
            (
                "Sub-LN: Scalable Layer Normalization for Transformers (2024)",
                2024,
                "Applies LayerNorm WITHIN the sublayer's computation (e.g., "
                "inside Q/K/V projections individually) rather than once per "
                "block. Bounds activation magnitude per-component, reducing "
                "training spikes at 100B+ scale. Still early — no flagship "
                "model ships it yet.",
            ),
            (
                "muP / μ-Transfer (Yang et al.)",
                2024,
                "Complementary to norm-placement work: scales initializations "
                "and LR per-layer so hyperparameter transfer across width "
                "becomes exact. Combined with pre-norm + QK-norm, removes a "
                "large class of 100B+ training failures. Qwen 2.5 and "
                "DeepSeek-V3 ship muP-style LR scaling in their technical "
                "reports.",
            ),
        ],
        current_sota=(
            "Late-2025 production default for 10B–100B open models: pre-norm "
            "+ RMSNorm + QK-norm + muP-style LR scaling + weight tying "
            "(LLaMA-3, Gemma-2) OR untied lm_head (DeepSeek-V3, PaLM-3). "
            "DeepNorm remains selective — used for extreme-depth experiments "
            "(1000+ layers). Sub-LN is a watch-this-space direction; no "
            "flagship model ships it yet but several 2024–2025 papers show "
            "promising training-stability results."
        ),
    )

    # --- 🛠️ Debugging cell ---
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "Three architecture bugs — norm applied to the wrong axis, "
            "forgotten residual connection, bf16 precision loss in norm "
            "statistics"
        ),
        root_causes=[
            "Norm applied to the WRONG AXIS: RMSNorm / LayerNorm must reduce "
            "over the LAST axis (d_model), not over seq_len or batch. Bug: "
            "`mx.mean(x*x, axis=0)` instead of `axis=-1` accidentally "
            "normalizes across batch — every batch item gets the SAME "
            "activations after norm. Symptom: loss plateaus at a high value; "
            "eval perplexity is identical for every sequence. Fix: always "
            "`axis=-1, keepdims=True` for the norm reduction. Diagnostic: "
            "assert `out[0, 0, :]` and `out[1, 0, :]` are NOT bit-identical "
            "after the norm.",
            "Forgotten RESIDUAL CONNECTION: wrote `x = sublayer(norm(x))` "
            "instead of `x = x + sublayer(norm(x))`. Gradient through the "
            "residual identity path vanishes — training stalls after a few "
            "hundred steps as activations collapse. Symptom: loss drops "
            "initially then flatlines at a value well above the tokenizer's "
            "entropy floor; activations at deep layers shrink to ~0. Fix: "
            "every sublayer output goes INTO a `+` with its input. "
            "Diagnostic: print `abs(x).mean()` at each layer during "
            "training — if it's monotonically decreasing with depth, "
            "you're missing a residual.",
            "BF16 PRECISION LOSS in norm statistics: computing "
            "`mean(x*x)` in bf16 loses precision because `x*x` at large "
            "magnitudes overflows bf16's 8-bit mantissa. Symptom: norm "
            "outputs have NaN or Inf sporadically; training spikes. Fix: "
            "upcast to fp32 INSIDE the norm: "
            "`x_fp32 = x.astype(mx.float32); y = rmsnorm(x_fp32).astype(x.dtype)`. "
            "Every production RMSNorm kernel (mx.fast.rms_norm, "
            "torch.nn.functional.rms_norm) does this internally. Diagnostic: "
            "print `mx.max(mx.abs(x*x))` — if it's near bf16.max (~3.4e38) "
            "you're within an order of magnitude of overflow.",
        ],
        diagnostic_code=(
            "import mlx.core as mx\n"
            "import mlx.nn as _nn\n"
            "\n"
            "# All module-level names prefixed with underscore so this cell doesn't\n"
            "# leak x, norm, block over the notebook's pre-existing globals.\n"
            "\n"
            "# -- Symptom 1: norm applied to wrong axis ----------------------\n"
            "# Mix up axis=-1 (correct) with axis=0 (batch, wrong) and show the\n"
            "# output becomes batch-dependent in the WRONG way.\n"
            "_B, _T, _d = 2, 4, 32\n"
            "mx.random.seed(0)\n"
            "_x = mx.random.normal(shape=(_B, _T, _d))\n"
            "mx.eval(_x)\n"
            "\n"
            "def _rms_axis(x, axis: int, eps: float = 1e-6):\n"
            "    _r = mx.sqrt(mx.mean(x * x, axis=axis, keepdims=True) + eps)\n"
            "    return x / _r\n"
            "\n"
            "_y_correct = _rms_axis(_x, axis=-1)  # normalize over d_model\n"
            "_y_wrong = _rms_axis(_x, axis=0)     # normalize over batch (BUG)\n"
            "mx.eval(_y_correct, _y_wrong)\n"
            "\n"
            "# After correct norm, each (batch, seq) row should have RMS ~ 1.\n"
            "_rms_correct = float(\n"
            "    mx.max(mx.abs(mx.sqrt(mx.mean(_y_correct * _y_correct, axis=-1)) - 1.0)).item()\n"
            ")\n"
            "assert _rms_correct < 1e-3, f\"axis=-1 RMS not ~1: {_rms_correct:.4e}\"\n"
            "print(f\"[1] axis=-1 (correct): per-row RMS within {_rms_correct:.4e} of 1.0\")\n"
            "\n"
            "# After WRONG norm (axis=0), the two outputs differ materially from the\n"
            "# correct-axis norm — the bug changes activations batch-wide.\n"
            "_mismatch = float(mx.max(mx.abs(_y_correct - _y_wrong)).item())\n"
            "print(f\"    axis=0 (wrong):    max |y_wrong - y_correct| = {_mismatch:.4f}\")\n"
            "assert _mismatch > 0.1, (\n"
            "    \"axis=0 norm should produce materially different output vs axis=-1\"\n"
            ")\n"
            "print(\"    → symptom: loss plateaus because every batch item gets same activations.\")\n"
            "\n"
            "# -- Symptom 2: forgot residual connection ----------------------\n"
            "# Stack N 'blocks' with and without residuals; measure activation\n"
            "# magnitude at the final layer. Without residuals, signal decays.\n"
            "_d2 = 64\n"
            "_n_layers = 8\n"
            "mx.random.seed(1)\n"
            "_lins = [_nn.Linear(_d2, _d2, bias=False) for _ in range(_n_layers)]\n"
            "_norm = _nn.RMSNorm(_d2)\n"
            "_x2 = mx.random.normal(shape=(1, 16, _d2))\n"
            "mx.eval(_x2)\n"
            "\n"
            "# Without residuals: y = lin(norm(y))\n"
            "_y_no_res = _x2\n"
            "for _lin in _lins:\n"
            "    _y_no_res = _lin(_norm(_y_no_res))\n"
            "mx.eval(_y_no_res)\n"
            "_mag_no_res = float(mx.mean(mx.abs(_y_no_res)).item())\n"
            "\n"
            "# With residuals: y = y + lin(norm(y))\n"
            "_y_with_res = _x2\n"
            "for _lin in _lins:\n"
            "    _y_with_res = _y_with_res + _lin(_norm(_y_with_res))\n"
            "mx.eval(_y_with_res)\n"
            "_mag_with_res = float(mx.mean(mx.abs(_y_with_res)).item())\n"
            "_mag_input = float(mx.mean(mx.abs(_x2)).item())\n"
            "\n"
            "print(f\"[2] input magnitude:                  {_mag_input:.4f}\")\n"
            "print(f\"    after {_n_layers} layers NO residuals:    {_mag_no_res:.4f}  (decayed)\")\n"
            "print(f\"    after {_n_layers} layers WITH residuals:  {_mag_with_res:.4f}  (preserved)\")\n"
            "# With random init, residual-free stack attenuates signal at deep depth; residual stack\n"
            "# preserves / grows it. Assert the magnitude ratio matches the pattern qualitatively.\n"
            "assert _mag_with_res > _mag_no_res, (\n"
            "    \"residual stack should preserve/grow magnitude vs residual-free stack\"\n"
            ")\n"
            "print(\"    → symptom: residual-free training stalls because signal vanishes.\")\n"
            "\n"
            "# -- Symptom 3: bf16 precision loss in norm statistics ----------\n"
            "# Compute RMS of a tensor with magnitudes near bf16's precision wall.\n"
            "# The naive bf16 reduction loses accuracy vs the fp32-upcasted path.\n"
            "mx.random.seed(2)\n"
            "_x3_fp32 = mx.random.normal(shape=(1, 4, 4096)) * 3.0  # moderate magnitude\n"
            "mx.eval(_x3_fp32)\n"
            "_x3_bf16 = _x3_fp32.astype(mx.bfloat16)\n"
            "mx.eval(_x3_bf16)\n"
            "\n"
            "# bf16-native RMS (loses precision in the sum)\n"
            "_rms_bf16 = mx.sqrt(mx.mean(_x3_bf16 * _x3_bf16, axis=-1))\n"
            "# fp32-upcasted RMS (accurate)\n"
            "_rms_upcast = mx.sqrt(\n"
            "    mx.mean(_x3_bf16.astype(mx.float32) * _x3_bf16.astype(mx.float32), axis=-1)\n"
            ")\n"
            "# Ground-truth fp32 RMS\n"
            "_rms_truth = mx.sqrt(mx.mean(_x3_fp32 * _x3_fp32, axis=-1))\n"
            "mx.eval(_rms_bf16, _rms_upcast, _rms_truth)\n"
            "\n"
            "_err_bf16 = float(mx.max(mx.abs(_rms_bf16.astype(mx.float32) - _rms_truth)).item())\n"
            "_err_upcast = float(mx.max(mx.abs(_rms_upcast - _rms_truth)).item())\n"
            "print(f\"[3] RMS error vs fp32 ground truth:\")\n"
            "print(f\"    bf16-native reduction:      {_err_bf16:.6f}\")\n"
            "print(f\"    bf16 input, fp32 reduction: {_err_upcast:.6f}\")\n"
            "# Upcasted path is strictly at least as accurate — often materially more\n"
            "# accurate at d=4096 where the reduction accumulates 4096 terms.\n"
            "assert _err_upcast <= _err_bf16 + 1e-6, (\n"
            "    \"fp32-upcast RMS must be at least as accurate as bf16-native\"\n"
            ")\n"
            "print(\"    → fix: upcast to fp32 INSIDE the norm; downcast at the output.\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q03),
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
    """Transform nb06 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb06 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb06] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    norm_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_NORM)
    )
    residual_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_RESIDUAL)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (residual_end, _block_residual(records), "residual"),
        (norm_end, _block_norm(records), "norm"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb06] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb06] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb06] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb06 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb06] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
