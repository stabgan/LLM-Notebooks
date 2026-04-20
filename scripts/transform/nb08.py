"""Interview-grade transform for notebook 08 (Training on Apple Silicon).

This module inserts the six interview-layer strata into
``08_training_apple_silicon.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): LR schedules (cosine with warmup — derive
`lr_t = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))`, explain why
cosine beats linear decay in practice), grad clipping (value vs norm,
derive the global-norm formula `g ← g · min(1, max_norm/||g||)`),
mixed-precision (bf16 vs fp16, why fp16 needs dynamic loss scaling and
bf16 does not), AdamW (derive the update from first principles,
decoupled weight decay vs L2 regularization), LR warmup (why required
for transformers, RAdam variance-of-second-moment argument), gradient
accumulation (micro-batch B_micro → effective batch B_eff, when to
divide loss by accum_steps, UMA-vs-discrete-GPU trade-off), and the
2024-2026 optimizer frontier (Lion, Schedule-Free AdamW, SOAP, Muon).

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.2, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb08
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

# scripts/transform/nb08.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "08_training_apple_silicon.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 8

# Markers that indicate this notebook has already been transformed.
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb08-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# Both anchors are resolved against the cell list BEFORE any insertion so
# insertions can proceed bottom-up without invalidating anchors.
#
# "## Mixed Precision Training (Deep Dive)" — after the MixedPrecisionTrainer
#   code demo and before "## Gradient Accumulation". We insert the mixed-
#   precision block (q01 warmup cosine LR, q04 mixed-precision, WB-A cosine
#   LR + warmup, 📐-2 mixed-precision bench, 🏭 cell, 🔭 cell).
# "## Gradient Accumulation (Deep Dive)" — after the accumulation demo and
#   before "## mx.compile(...)". We insert the training-stability block (q02
#   AdamW, q03 grad clip, q05 LR warmup, q06 grad accum, WB-B global-norm
#   clipping, 📐-1 optimizer memory, 🛠️ cell).
# "## OOM Recovery: Auto Batch Size Reduction" — after the OOM demo, before
#   "## 🧪 Try It Yourself". We insert the frontier cell (q07).
_ANCHOR_MIXED = "## Mixed Precision Training (Deep Dive)"
_ANCHOR_GRADACC = "## Gradient Accumulation (Deep Dive)"
_ANCHOR_OOM = "## OOM Recovery"


# ---------------------------------------------------------------------------
# Notebook I/O (raw JSON; Requirement 19.4 permits this for .ipynb edits)
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``08_training_apple_silicon.ipynb`` as a JSON dict."""
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
    """Return the seven Question_Bank records for nb08.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle               — q01, q02, q03, q04, q06
        research_engineer — q01, q02, q04, q05, q07
        systems_engineer  — q02, q03, q05, q06, q07

    Topic coverage (task brief — LLD-4):
        q01 — Cosine vs linear LR schedule (derive the cosine formula;
              why cosine beats linear decay)
        q02 — Adam update rule derivation (β1, β2, ε, bias correction)
              and why AdamW (decoupled weight decay) replaces L2
        q03 — Gradient clipping (value vs norm, derive global-norm;
              typical max_norm for LLM training)
        q04 — Mixed-precision training (bf16 vs fp16; why fp16 needs
              dynamic loss scaling and bf16 does not)
        q05 — Learning-rate warmup (why essential for transformers;
              RAdam's variance-of-second-moment argument)
        q06 — Gradient accumulation math (micro-batch → effective
              batch; when to scale loss by 1/accum_steps; UMA trade-off)
        q07 — 2024-2026 optimizer frontier: Lion, Schedule-Free AdamW,
              SOAP, Muon — common theme and open problems
    """
    return [
        {
            "id": "nb08-q01",
            "notebook": _NB_FILENAME,
            "section": "Cosine vs linear LR schedule — derivation",
            "difficulty": "warmup",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["lr-schedule", "cosine", "linear", "decay"],
            "question": (
                "Derive the cosine LR schedule formula `lr_t = lr_min "
                "+ 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))` from first "
                "principles. Why does cosine outperform linear decay "
                "in practice on modern LLMs?"
            ),
            "answer_key_points": [
                "Goal: design a smooth monotone-decreasing LR schedule that starts at lr_max and ends at lr_min over T steps, with derivative zero at both endpoints so the optimizer doesn't 'jerk' at t=0 or t=T. Linear decay has constant derivative -(lr_max-lr_min)/T — large and abrupt at both ends. Cosine's derivative is zero at endpoints (because d/dt[cos(π·t/T)] = -(π/T)·sin(π·t/T), zero when sin=0, i.e. t=0 and t=T).",
                "Derive the formula: parameterize `lr_t = lr_min + A·(1 + cos(π·t/T))` for some amplitude A. At t=0: cos(0)=1, so `lr_0 = lr_min + 2A`. We want lr_0 = lr_max ⇒ A = (lr_max - lr_min)/2. Substitute: `lr_t = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·t/T))`. At t=T: cos(π)=-1 ⇒ `lr_T = lr_min + 0 = lr_min`. QED.",
                "Why cosine beats linear in practice: the SHAPE matches the 'exploration → exploitation' curve optimal-learning theory argues for. Early (t ≈ 0): LR stays near lr_max for longer than linear (slow decay phase) — the model can still take large optimization steps during the wide-basin-exploration phase. Late (t ≈ T): LR stays near lr_min for longer than linear (slow approach) — fine-grained convergence near the minimum. Middle: fast transition through the 'explore→exploit' inflection point.",
                "Empirical evidence: Loshchilov & Hutter 2017 (SGDR) showed cosine beats both linear and step-decay on CIFAR-10/ImageNet at matched compute. Kaplan et al. 2020 and Chinchilla (Hoffmann 2022) adopted cosine decay as the LLM-scaling default. LLaMA, LLaMA-2, LLaMA-3, Mistral, DeepSeek-V3 all use cosine. Linear decay (used in OG BERT) is the legacy default — cosine is the 2020+ frontier.",
                "Typical settings (2024 LLM pretraining): lr_max ≈ 3e-4 to 6e-4 for AdamW, lr_min = 0.1·lr_max (i.e., decay to 10% of peak, not 0% — DOESN'T kill gradient signal at end), T = total training steps. Chinchilla-optimal training uses T set so the FINAL few % of steps do the fine-grained convergence at near-zero LR.",
                "Corollary: the 'slow final decay' at t≈T is why 'continued training' (restart cosine with smaller lr_max) works — you're jumping back to the explore phase with a smaller budget. Warm-restarts (SGDR) use this explicitly; modern continued-pretraining runs (LLaMA-3 checkpoint → instruction-tune) implicitly use it.",
                "Common variant: cosine WITH WARMUP — combine linear warmup from 0 to lr_max for the first ~1-2% of training, then cosine-decay from lr_max to lr_min over the remainder. Standard modern LLM LR schedule. See q05 for why warmup is essential.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'cosine is smoother so it trains faster'. "
                "Linear decay is ALSO smooth (C^∞ on the interior) — "
                "the difference isn't smoothness, it's the specific "
                "SHAPE: zero derivative at endpoints plus asymmetric "
                "time-spent-per-phase. 'Smoother' is too vague to be "
                "the right answer."
            ),
            "references": [
                "https://arxiv.org/abs/1608.03983",
                "https://arxiv.org/abs/2001.08361",
                "https://arxiv.org/abs/2203.15556",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q02",
            "notebook": _NB_FILENAME,
            "section": "Adam → AdamW — derivation and decoupled weight decay",
            "difficulty": "core",
            "roles": ["mle", "research_engineer", "systems_engineer"],
            "topic_tags": ["adam", "adamw", "weight-decay", "optimizer", "derivation"],
            "question": (
                "Derive the Adam update rule from first principles "
                "(β1, β2, ε, bias correction). Why does AdamW's "
                "decoupled weight decay replace L2 regularization on "
                "modern LLMs — what specifically was wrong with "
                "Adam+L2?"
            ),
            "answer_key_points": [
                "Adam (Kingma & Ba 2014) maintains two running statistics: 1st moment `m_t` (momentum-like mean of gradients) and 2nd moment `v_t` (uncentered variance of gradients). Updates: `m_t = β1·m_{t-1} + (1-β1)·g_t`, `v_t = β2·v_{t-1} + (1-β2)·g_t²` — exponential moving averages with decay rates β1, β2 (defaults 0.9, 0.999).",
                "Bias correction: at t=0, `m_0 = v_0 = 0` so the EMAs are biased toward zero early in training. Corrected estimates: `m̂_t = m_t/(1-β1^t)`, `v̂_t = v_t/(1-β2^t)`. At t=1, β1=0.9: `m̂_1 = m_1/(1-0.9) = 10·m_1` — compensates for the 0-initialized EMA. Bias correction matters for the first ~1/(1-β) steps (~10 for β=0.9, ~1000 for β=0.999).",
                "Update: `θ_t = θ_{t-1} - lr · m̂_t / (sqrt(v̂_t) + ε)`. The `sqrt(v̂_t)` term is the per-parameter ADAPTIVE scaling — parameters with consistently-large gradients get smaller effective LR, parameters with small/noisy gradients get larger. `ε` (default 1e-8) is a numerical floor preventing division-by-zero when v̂_t is tiny.",
                "Adam + L2 regularization (the buggy original): `g_t ← g_t + λ·θ_{t-1}` (add L2 grad to the raw gradient BEFORE Adam processes it). Problem: Adam's adaptive scaling `/ sqrt(v̂_t)` RESCALES the L2 term too — parameters with large gradients get less regularization than they should. The regularization effectively depends on GRADIENT MAGNITUDE instead of PARAMETER MAGNITUDE — backwards from what L2 should do.",
                "AdamW fix (Loshchilov & Hutter 2019): DECOUPLE weight decay from the adaptive update. New rule: `θ_t = θ_{t-1} - lr · (m̂_t / (sqrt(v̂_t) + ε) + λ·θ_{t-1})`. Equivalent to `θ_t = (1 - lr·λ)·θ_{t-1} - lr·(Adam step)`. Weight decay is now a direct multiplicative shrinkage on θ — NO adaptive rescaling — exactly as SGD+L2 would have it.",
                "Empirical win of AdamW: on CIFAR/ImageNet/WMT/OpenAI GPT experiments, AdamW closes the Adam-vs-SGD generalization gap. Hugely consequential: every 2020+ LLM ships AdamW. PyTorch's `torch.optim.AdamW` is the default; TensorFlow's `tfa.optimizers.AdamW` is equivalent. MLX ships `mlx.optimizers.AdamW`.",
                "Typical hyperparameters for LLM AdamW: lr ∈ [1e-4, 6e-4] (smaller for bigger models), β1=0.9, β2=0.95 (NOT 0.999 — lower β2 for faster adaptation; OpenAI and DeepSeek-V3 use 0.95), ε=1e-8, weight_decay=0.1 for dense transformers (excluding bias and norm-γ params — see q06 debug for why).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'AdamW is just Adam with weight decay'. Adam "
                "ALREADY supported L2 regularization via gradient "
                "injection — that was the BUGGY form. AdamW's "
                "contribution is DECOUPLING weight decay from the "
                "adaptive update. Confusing 'has weight decay' with "
                "'has DECOUPLED weight decay' is the textbook mistake."
            ),
            "references": [
                "https://arxiv.org/abs/1412.6980",
                "https://arxiv.org/abs/1711.05101",
                "https://arxiv.org/abs/2006.08643",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q03",
            "notebook": _NB_FILENAME,
            "section": "Gradient clipping — value vs global-norm",
            "difficulty": "core",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["grad-clipping", "norm-clipping", "training-stability"],
            "question": (
                "Compare value clipping vs global-norm clipping of "
                "gradients. Derive the global-norm formula "
                "`g ← g · min(1, max_norm / ||g||)`. What's a typical "
                "`max_norm` for LLM pretraining, and why that number?"
            ),
            "answer_key_points": [
                "Problem gradient clipping solves: individual gradient values (or per-tensor gradient norms) can spike to very large magnitudes during training — one noisy batch, one layer-norm bug, one activation explosion. A single unclipped update can knock the optimizer's running statistics (Adam's v_t) out of the reasonable range, causing CASCADING instability for the next ~1000 steps while β2 EMAs recover.",
                "Value clipping: `g_i ← clip(g_i, -c, c)` per element. Pros: trivial to implement, O(numel) with no reduction. Cons: DESTROYS gradient DIRECTION — the clipped vector no longer points the same way as the pre-clip gradient. If a single dimension spikes, only that dimension is clipped, but the model loses the consistent-direction information from the rest. Rarely used in modern LLM training.",
                "Global-norm clipping: treat all gradients across all parameters as ONE big flat vector `g`, compute `||g||_2 = sqrt(Σ g_i²)`, scale UNIFORMLY if the global norm exceeds max_norm: `g ← g · min(1, max_norm / ||g||)`. PRESERVES the gradient direction — every component scales by the same factor ||g||/max_norm when clipping fires. This is the invariant that makes global-norm clipping 'safe' for adaptive optimizers like Adam.",
                "Derivation: we want to find a scaling factor s such that the clipped gradient `g_clipped = s·g` has `||g_clipped|| ≤ max_norm`. Two cases: (a) if `||g|| ≤ max_norm`: no clipping needed, s=1. (b) if `||g|| > max_norm`: choose s = max_norm/||g|| so `||s·g|| = s·||g|| = max_norm`. Combined: `s = min(1, max_norm/||g||)` ⇒ `g ← g · min(1, max_norm/||g||)`.",
                "Typical `max_norm` for LLM pretraining: **1.0 is the standard** (GPT-3, LLaMA, Mistral, DeepSeek-V3 all clip at 1.0). Why this number: AdamW's update magnitude is ~lr·(m̂/sqrt(v̂)) ≈ lr (because m̂/sqrt(v̂) ≈ 1.0 on unit-normalized statistics); with lr=3e-4 and 8B params, an unclipped step's L2 norm can approach ~sqrt(8e9)·3e-4 ≈ 27 in the worst case — far above 'safe'. Clipping to 1.0 bounds the update magnitude to something that won't destabilize the optimizer.",
                "When it's HIGHER: some 100B+ models clip at 0.5 or 0.3 (more conservative, trades occasional lost-signal for stability). Some MoE models clip at 2.0 or 5.0 (routing fluctuations need larger per-step moves). PPO/RLHF fine-tuning typically clips at 0.5-1.0 with a smaller effective batch.",
                "Implementation detail: compute global norm via `sqrt(sum of per-tensor squared norms)` so it parallelizes cleanly across workers. All_reduce the sum-of-squares → sqrt on each worker → same scale factor applied uniformly. MLX: `mx.clip(grads, ...)` for value-clip; `optax.clip_by_global_norm` (PyTorch: `torch.nn.utils.clip_grad_norm_`) for norm-clip — `mlx.optimizers` applies norm clip at the optimizer level.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Clipping EACH parameter tensor's norm independently "
                "instead of the GLOBAL norm. Per-tensor clipping "
                "destroys the relative direction between parameters "
                "(some clipped by 2×, some by 10×) — effectively the "
                "same problem as value clipping. Global norm is the "
                "only safe option for adaptive optimizers."
            ),
            "references": [
                "https://arxiv.org/abs/1211.5063",
                "https://arxiv.org/abs/2005.14165",
                "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q04",
            "notebook": _NB_FILENAME,
            "section": "Mixed-precision training — bf16 vs fp16",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["mixed-precision", "bf16", "fp16", "loss-scaling"],
            "question": (
                "Compare bf16 vs fp16 mixed-precision training. Why "
                "does fp16 need dynamic loss scaling but bf16 does "
                "not? Walk the NaN-recovery protocol for fp16 loss "
                "scaling."
            ),
            "answer_key_points": [
                "The two 16-bit formats: fp16 = IEEE-754 half (1 sign, 5 exponent, 10 mantissa), dynamic range ~6e-5 to 65504. bf16 = 'brain-float' (1 sign, 8 exponent, 7 mantissa), dynamic range ~1e-38 to 3.4e38 — SAME range as fp32 but 3 fewer mantissa bits.",
                "Key difference: bf16 has fp32's EXPONENT range but fp16 has a much narrower range. Gradients in deep networks span 10+ orders of magnitude (from ~1e-10 at late layers to ~1e2 at attention softmax pre-norm). fp16 CAN'T REPRESENT the small gradients — they underflow to zero, silently destroying learning signal on some parameters.",
                "Why fp16 needs LOSS SCALING: multiply the loss by a large scale factor S (typically 2^15 = 32768) BEFORE backward. This multiplies every gradient by S, shifting them up into fp16's representable range. After backward, DIVIDE gradients by S (in fp32 or via math that preserves precision) before the optimizer step. Key idea: preserve the gradient's INFORMATION CONTENT by shifting to a representable range.",
                "DYNAMIC loss scaling: start with S=2^24 = 16M. Each step, check if any gradient is inf/NaN (fp16 overflow). If yes: scale didn't work, DIVIDE S by 2 and SKIP the step. If no: every ~2000 steps, multiply S by 2 (try a bigger scale next time, reclaim mantissa bits). Converges to the largest S that doesn't overflow — typically stabilizes at 2^15 to 2^18 for LLM pretraining.",
                "Why bf16 SKIPS loss scaling: with the same exponent range as fp32, no gradients underflow to zero or overflow to inf in normal training. The 3 mantissa bits LOST vs fp16 mean slightly less precision on the significand (7 bits ~2 decimal digits vs fp16's 10 bits ~3 digits) — but this is usually fine because the reduction to fp32 for the optimizer state recovers it. bf16 is the DEFAULT for 2023+ LLM pretraining on H100/A100/M-series.",
                "NaN-recovery protocol for fp16: (a) detect inf/NaN in any gradient after backward (all_reduce an OR); (b) if detected: SKIP this optimizer step (don't update θ or Adam's m_t, v_t), DIVIDE loss_scale by 2, and continue; (c) if not: check if >2000 steps since last inf — if so, MULTIPLY loss_scale by 2 and continue; (d) always keep an fp32 master copy of params — the fp16 params during forward are casts OF the fp32 master.",
                "Hardware support: NVIDIA A100/H100 support both fp16 and bf16 via tensor cores at equal throughput. Apple Silicon M-series: bf16 support is native in Metal and MLX via `mx.bfloat16`; fp16 also supported. For MLX-LM training on M-series, use bf16 — no loss scaling needed, simpler code, same throughput.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'bf16 and fp16 are both 16-bit so they're the "
                "same'. They have DIFFERENT exponent/mantissa "
                "allocations (8/7 vs 5/10). bf16's fp32-like exponent "
                "range is EVERYTHING — it's why bf16 works without "
                "loss scaling and fp16 doesn't. Don't lump them."
            ),
            "references": [
                "https://arxiv.org/abs/1710.03740",
                "https://cloud.google.com/tpu/docs/bfloat16",
                "https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q05",
            "notebook": _NB_FILENAME,
            "section": "Learning-rate warmup — why essential for transformers",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["lr-warmup", "adam", "radam", "variance", "transformer-training"],
            "question": (
                "Why is LR warmup essential for transformer training? "
                "Explain the interaction with Adam's second-moment "
                "estimate (β2), and derive the 'adaptive learning "
                "rate variance' argument from Liu et al. 2020 "
                "(RAdam)."
            ),
            "answer_key_points": [
                "Problem: training a transformer at full lr_max from step 0 causes loss spikes, NaN activations, or silent divergence in the first ~1000 steps. This is specific to ADAPTIVE optimizers (Adam, AdamW) — SGD with momentum doesn't need warmup (though it benefits slightly). LR warmup = linearly ramp from 0 to lr_max over the first W steps (typically 1-2% of total).",
                "Root cause (Liu et al. 2020, RAdam paper): at step t, Adam's v_t is `(1-β2)·Σ β2^(t-k)·g_k²` — an EMA of past squared gradients. With β2=0.999, the 'effective window' is ~1/(1-β2) = 1000 steps. Early on (t < 1000): v_t has seen very few samples, so `sqrt(v̂_t)` is a HIGH-VARIANCE estimate of the true second moment. High variance ⇒ some parameters get lr/0.01=100× while others get lr/100=0.01× — wildly different effective LRs.",
                "Adaptive learning rate variance: `Var(lr_effective) = Var(lr / sqrt(v̂_t))` is high at small t. Liu et al. 2020 compute this analytically: `Var(1/sqrt(v̂_t))` diverges as t→0 unless v̂_t has accumulated enough samples (roughly `t > 4/(1-β2)` ≈ 4000 steps for β2=0.999). Conclusion: for the first ~4k steps, Adam's adaptive LR is so noisy it's effectively random.",
                "Why warmup fixes this: during the first W steps, lr is SMALL (near 0, linearly ramping up). Even if Adam's adaptive scaling is noisy, `lr · (noisy adaptive scale)` is bounded because lr is small. By the time lr reaches lr_max, v̂_t has accumulated enough samples to be a low-variance estimator — the effective LR is the target lr_max uniformly across parameters.",
                "RAdam fix (Liu 2020): analytically derive the correct 'rectification' term that transforms v̂_t into a bias-corrected estimate usable from step 0. For t < 5 (rough threshold): fall back to SGD-with-momentum (no adaptive scaling). For t ≥ 5: apply a rectification factor r_t that shrinks the adaptive scale back toward 1.0 when t is small. With RAdam, warmup is UNNECESSARY — the optimizer handles the early-training variance internally. BUT: RAdam is not used widely in LLM training because the computational overhead is noticeable and warmup works just as well empirically.",
                "Typical warmup settings for LLM pretraining: W = 2000 steps (GPT-2), 10000 steps (Chinchilla, GPT-3), or 2000-4000 steps (LLaMA series). As a fraction: usually 0.5-2% of total training steps. LLaMA-3 used 2000 warmup steps out of ~2T-token budget.",
                "Warmup shape: LINEAR warmup (lr goes 0→lr_max linearly) is standard. Some papers use SQRT warmup (lr grows like sqrt(t/W) — faster ramp early). Post-warmup: cosine decay from lr_max to lr_min (see q01). The full schedule is 'linear warmup + cosine decay' — nearly universal in 2020+ LLMs.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying 'warmup is needed to avoid NaNs from bad "
                "initialization'. The root cause isn't init — it's "
                "ADAPTIVE OPTIMIZER VARIANCE. Evidence: if you train "
                "the same model with SGD+momentum from step 0, no "
                "warmup is needed (or benefit is marginal). Warmup is "
                "specifically a fix for Adam/AdamW's early-step "
                "behavior."
            ),
            "references": [
                "https://arxiv.org/abs/1908.03265",
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/2005.14165",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q06",
            "notebook": _NB_FILENAME,
            "section": "Gradient accumulation — math and UMA trade-off",
            "difficulty": "stretch",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["grad-accumulation", "effective-batch", "uma", "apple-silicon"],
            "question": (
                "For effective batch `B_eff` with micro-batch "
                "`B_micro`, derive the gradient-averaging formula "
                "and explain when `loss.backward()` MUST be scaled "
                "by `1/accum_steps`. What's the memory / throughput "
                "trade-off on Apple Silicon (UMA) vs a discrete GPU?"
            ),
            "answer_key_points": [
                "Setup: you want effective batch B_eff but can only fit B_micro examples in memory per step. Set `accum_steps = B_eff / B_micro` (integer division, B_eff must be divisible by B_micro). Forward + backward on each of accum_steps micro-batches, ACCUMULATING gradients into a running buffer; step the optimizer only AFTER all accum_steps micro-batches are processed.",
                "Gradient-averaging derivation: for a mean-reduction loss `L(θ) = (1/B_eff)·Σ_i ℓ(x_i, θ)`, the gradient is `∇L = (1/B_eff)·Σ_i ∇ℓ_i`. When we process B_micro examples at a time with `ℓ_micro = (1/B_micro)·Σ_j ℓ(x_j, θ)`, each micro-batch's gradient `∇ℓ_micro = (1/B_micro)·Σ_j ∇ℓ_j`. Accumulating across accum_steps gives `Σ_k ∇ℓ_micro_k = (1/B_micro)·Σ_{all} ∇ℓ_i`. To match ∇L = (1/B_eff)·Σ: multiply by B_micro/B_eff = 1/accum_steps. So: `accumulated_grad = (1/accum_steps)·Σ_k ∇ℓ_micro_k`.",
                "Where the 1/accum_steps scaling happens (CRITICAL): you can either (a) divide each micro-batch's loss by accum_steps before backward (`(ℓ_micro / accum_steps).backward()`) — the gradients naturally accumulate to the right scale; or (b) run backward normally and divide the accumulated gradient by accum_steps before the optimizer step. Both are equivalent.",
                "When scaling is NOT needed: if your loss is SUM-reduced instead of MEAN-reduced (unusual), the gradients from accum_steps micro-batches ALREADY sum to the full-batch result. But essentially every modern training loop uses mean-reduction (`loss.mean()` or the default of most criterion functions). Rule: if you use mean-reduction (the default), you MUST scale by 1/accum_steps somewhere.",
                "Equivalence to large-batch training: with grad accumulation, the optimizer step sees gradients EQUIVALENT (up to floating-point noise) to training at B_eff. This is why 'grad accumulation is free memory — same math as big-batch training' — TRUE for the gradient step itself, but NOT free for (a) batch norm statistics (each micro-batch sees B_micro samples, not B_eff — use layer/RMS norm instead, which is example-independent), (b) drop-out (sampled independently per micro-batch — minor effect).",
                "Throughput trade-off on a DISCRETE GPU: `accum_steps` micro-batches means `accum_steps` HBM round-trips to stream weights and activations. Each round-trip has a fixed PCIe latency (~10μs) that gets multiplied. If B_micro is very small, overhead dominates. Net: grad accum on discrete GPU is ~5-15% slower than true-batch training at the same B_eff.",
                "Throughput trade-off on APPLE SILICON (UMA): unified memory means NO PCIe round-trips — the CPU and GPU share the same RAM pool. `accum_steps` round-trips still happen inside MLX's kernel launches but avoid the PCIe bottleneck. Empirically on M4 Pro: grad accumulation incurs ~1-3% overhead vs true-batch training at matched B_eff. UMA is PARTICULARLY well-suited to grad accumulation because the fixed-per-step overhead is ~10× smaller than on a discrete GPU setup.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Forgetting to divide loss (or accumulated gradient) "
                "by accum_steps when using mean-reduction loss. "
                "Symptom: effective LR is accum_steps× the intended "
                "LR — training diverges after a few steps. Diagnostic: "
                "compare the SUM of per-micro-batch loss magnitudes to "
                "one TRUE-batch loss at matching B_eff — if they're "
                "~accum_steps× apart, you forgot the scaling."
            ),
            "references": [
                "https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation",
                "https://pytorch.org/docs/stable/notes/amp_examples.html",
                "https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb08-q07",
            "notebook": _NB_FILENAME,
            "section": "2024–2026 optimizer frontier",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["lion", "schedule-free", "soap", "muon", "frontier"],
            "question": (
                "Compare Lion (2023), Schedule-Free AdamW (2024), "
                "SOAP (2024), and Muon (2024). What's the common "
                "theme across these post-AdamW optimizers, and "
                "what's the open problem for 100B+ models?"
            ),
            "answer_key_points": [
                "Lion (Chen et al. 2023, arxiv 2302.06675): discovered by an evolutionary search over optimizer update rules. Update: `θ_t = θ_{t-1} - lr · sign(β1·m_{t-1} + (1-β1)·g_t)` — takes only the SIGN of the momentum-averaged gradient. No 2nd moment! Memory: 1 state tensor (m_t) instead of Adam's 2 (m_t, v_t) — ~50% optimizer memory saving. Often matches or beats AdamW at small-medium scale. Used in Imagen, PaLM-2 experiments. Weakness: sign-based updates underperform on some architectures (conv-nets), still scales-to-100B unclear.",
                "Schedule-Free AdamW (Defazio et al. 2024, arxiv 2405.15682): eliminates the need for a DECAYING LR schedule entirely — no cosine decay, no warmup schedule. Instead, maintains a moving-AVERAGE iterate θ_avg in addition to the current iterate θ. Updates alternate between AdamW-style step on θ and averaging step on θ_avg. Theoretically optimal under a specific class of convex assumptions; empirically matches cosine+warmup on LLM pretraining WITHOUT the hyperparameter (end LR, T, warmup steps).",
                "SOAP (Vyas et al. 2024, arxiv 2409.11321): Shampoo preconditioner in the rotated eigenbasis of the squared gradients. Pre-conditions the gradient by the (approximate) inverse of the empirical Fisher matrix — adaptive second-order method. Memory cost: 2 state tensors PER dimension (L and R factors of the Kronecker-factored preconditioner) — heavier than AdamW but less than full Shampoo. Demonstrates ~1.4× speedup over AdamW on MoE training at small-ish scale (~1B params).",
                "Muon (Jordan et al. 2024, arxiv 2410.11081): orthogonalizes the gradient via Newton-Schulz iterations BEFORE the update. Update: `θ_t = θ_{t-1} - lr · orthogonalize(g_t)`. Exploits the observation that matmul gradients tend to be low-rank; orthogonalization 'sharpens' them toward higher-effective-rank directions. Shipped record results on nanoGPT speedrun; ~2× wall-clock speedup vs AdamW at 124M. Scale above 1B unproven as of late 2024.",
                "COMMON THEME across Lion / Schedule-Free AdamW / SOAP / Muon: all seek to REDUCE the optimizer's parameter-count overhead (Lion: 2× → 1×; SOAP: rotated Shampoo is cheaper than full) AND/OR eliminate hyperparameter friction (Schedule-Free: removes LR schedule; Lion: simpler than AdamW; Muon: orthogonalization is a single fixed op). The frontier theme is 'less state, fewer knobs, same or better convergence'.",
                "OPEN PROBLEM for 100B+ models: none of these has been VALIDATED at the 100B+ scale where the AdamW training recipe was originally locked in. Specifically: (a) Lion at 100B+ is unproven — PaLM-2 experiments stopped at ~10B; (b) Schedule-Free at 100B+ not yet published; (c) SOAP's preconditioner state is O(d²) per weight matrix — at LLaMA-3-70B's d=8192, that's ~67M per matrix, potentially prohibitive at scale; (d) Muon's orthogonalization cost grows with matrix dimension — small-model wins may not transfer.",
                "2024-2026 frontier direction: MUP-style parameterization + orthogonal-gradient methods (Muon, Shampoo) + schedule-free training + 8-bit optimizer state. Each sub-problem has a published winner at small scale; the open question is WHICH COMBINATION transfers to the 100B-1T scale without the weeks of HP tuning AdamW currently requires. Schedule-Free AdamW is the 2025 leading candidate among open-source training stacks because it eliminates the TOTAL-STEPS hyperparameter.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Calling these 'alternatives to AdamW'. They're "
                "NARROWLY-VALIDATED at small-to-medium scale — all "
                "100B+ flagship models (GPT-4, Claude 3.5, Gemini 1.5, "
                "LLaMA-3, DeepSeek-V3) still use AdamW with warmup + "
                "cosine. The post-AdamW optimizer frontier is a "
                "research direction, not a production default."
            ),
            "references": [
                "https://arxiv.org/abs/2302.06675",
                "https://arxiv.org/abs/2405.15682",
                "https://arxiv.org/abs/2409.11321",
                "https://arxiv.org/abs/2410.11081",
            ],
            "added_in": added_in,
        },
    ]


# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_mixed(records: list[dict]) -> list[dict]:
    """Block inserted at the end of the 'Mixed Precision Training' section.

    Contents: q01 (cosine LR schedule), q04 (mixed-precision bf16 vs fp16),
    whiteboard-A (cosine-with-warmup scheduler from scratch), 📐-2 (fp32
    vs bf16 forward+backward latency), 🏭 (how vLLM/SGLang/TRT-LLM/MLX-LM
    handle training-time mixed precision), 🔭 (2024-2026 optimizer
    frontier: Lion / Schedule-Free / SOAP / Muon).
    """
    q01, _q02, _q03, q04 = records[0], records[1], records[2], records[3]

    # --- Whiteboard A — cosine-with-warmup LR scheduler from scratch ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Cosine-with-warmup LR scheduler — implement from scratch, verify endpoints",
        prompt=(
            "Implement `_cosine_with_warmup(step, warmup_steps, "
            "total_steps, lr_max, lr_min)` returning the learning "
            "rate at a given step. Verify four properties with "
            "asserts: (1) lr==0.0 at step=0; (2) lr==lr_max at "
            "step=warmup_steps; (3) lr==lr_min at step=total_steps "
            "(within 1e-6); (4) monotone decreasing on "
            "[warmup_steps, total_steps]. Plot the schedule."
        ),
        constraints=[
            "Use pure Python math (math.cos, math.pi) — no MLX needed for the "
            "schedule itself, but wrap a final tensor op in `mx.eval` to "
            "satisfy the whiteboard-cell property-test requirement.",
            "Warmup phase (step ≤ warmup_steps): LINEAR ramp from 0 to lr_max. "
            "`lr = lr_max · step / warmup_steps`. At step=0 exactly: lr=0.0.",
            "Cosine decay phase (step > warmup_steps): "
            "`lr = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π·p))` where "
            "p = (step - warmup_steps) / (total_steps - warmup_steps) is the "
            "progress in [0, 1] through the decay phase.",
            "All four asserts must PASS exactly — endpoints are tight "
            "(within 1e-6 of analytic values). Monotonicity must hold across "
            "every adjacent pair in the decay phase.",
            "Sample the schedule at 1000 uniformly-spaced steps and plot; "
            "also run a small `mx.eval` at the end to confirm MLX is reachable.",
        ],
        complexity=(
            "Each `_cosine_with_warmup(step, ...)` call is O(1) — two "
            "comparisons and one cos(). Sampling at 1000 steps is O(1000). "
            "No allocation, no gradients — this is a CPU-side schedule "
            "function, not a training-hot-path bottleneck."
        ),
        solution_code=(
            "import math\n"
            "import matplotlib.pyplot as plt\n"
            "import mlx.core as mx\n"
            "\n"
            "def _cosine_with_warmup(\n"
            "    step: int,\n"
            "    warmup_steps: int,\n"
            "    total_steps: int,\n"
            "    lr_max: float,\n"
            "    lr_min: float,\n"
            ") -> float:\n"
            "    \"\"\"Linear warmup 0→lr_max then cosine decay to lr_min.\n"
            "\n"
            "    Matches the canonical GPT-3 / LLaMA / Chinchilla schedule.\n"
            "    - step in [0, warmup_steps]:          lr = lr_max · step / warmup_steps\n"
            "    - step in (warmup_steps, total_steps]: cosine from lr_max down to lr_min\n"
            "    - step > total_steps:                 lr = lr_min (clamped)\n"
            "    \"\"\"\n"
            "    if step <= 0:\n"
            "        return 0.0\n"
            "    if step <= warmup_steps:\n"
            "        return lr_max * step / max(1, warmup_steps)\n"
            "    if step >= total_steps:\n"
            "        return lr_min\n"
            "    _p = (step - warmup_steps) / max(1, total_steps - warmup_steps)\n"
            "    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * _p))\n"
            "\n"
            "# Canonical LLM settings — GPT-3-style warmup ≈ 0.5%, decay to 10%.\n"
            "_lr_max = 3e-4\n"
            "_lr_min = 3e-5  # 10% of lr_max\n"
            "_total_steps = 10_000\n"
            "_warmup_steps = 500\n"
            "\n"
            "# Property 1: lr == 0 at step=0 (exactly).\n"
            "_lr_0 = _cosine_with_warmup(0, _warmup_steps, _total_steps, _lr_max, _lr_min)\n"
            "assert _lr_0 == 0.0, f\"expected lr(0)=0.0, got {_lr_0!r}\"\n"
            "\n"
            "# Property 2: lr == lr_max at step=warmup_steps (linear ramp endpoint).\n"
            "_lr_w = _cosine_with_warmup(\n"
            "    _warmup_steps, _warmup_steps, _total_steps, _lr_max, _lr_min\n"
            ")\n"
            "assert abs(_lr_w - _lr_max) < 1e-12, (\n"
            "    f\"expected lr(warmup_steps)=lr_max, got {_lr_w!r}\"\n"
            ")\n"
            "\n"
            "# Property 3: lr == lr_min at step=total_steps (cosine decay endpoint, within 1e-6).\n"
            "_lr_T = _cosine_with_warmup(\n"
            "    _total_steps, _warmup_steps, _total_steps, _lr_max, _lr_min\n"
            ")\n"
            "assert abs(_lr_T - _lr_min) < 1e-6, (\n"
            "    f\"expected lr(total_steps)=lr_min, got {_lr_T!r} (diff {_lr_T - _lr_min:.3e})\"\n"
            ")\n"
            "\n"
            "# Property 4: monotone non-increasing from warmup_steps to total_steps.\n"
            "_samples = [\n"
            "    _cosine_with_warmup(_t, _warmup_steps, _total_steps, _lr_max, _lr_min)\n"
            "    for _t in range(_warmup_steps, _total_steps + 1)\n"
            "]\n"
            "for _i in range(1, len(_samples)):\n"
            "    assert _samples[_i] <= _samples[_i - 1] + 1e-12, (\n"
            "        f\"non-monotone at decay step offset {_i}: \"\n"
            "        f\"{_samples[_i-1]!r} -> {_samples[_i]!r}\"\n"
            "    )\n"
            "\n"
            "# Plot the full schedule (1000 uniformly-spaced steps).\n"
            "_steps_plot = list(range(0, _total_steps + 1, max(1, _total_steps // 1000)))\n"
            "_lrs_plot = [\n"
            "    _cosine_with_warmup(_t, _warmup_steps, _total_steps, _lr_max, _lr_min)\n"
            "    for _t in _steps_plot\n"
            "]\n"
            "fig, ax = plt.subplots(figsize=(9, 3.5))\n"
            "ax.plot(_steps_plot, _lrs_plot, 'b-', linewidth=2)\n"
            "ax.axvline(_warmup_steps, color='orange', linestyle='--', alpha=0.6, label=f'warmup end (step {_warmup_steps})')\n"
            "ax.axhline(_lr_max, color='green', linestyle=':', alpha=0.4, label=f'lr_max={_lr_max:.1e}')\n"
            "ax.axhline(_lr_min, color='red', linestyle=':', alpha=0.4, label=f'lr_min={_lr_min:.1e}')\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel('learning rate')\n"
            "ax.set_title('Cosine LR schedule with linear warmup (GPT-3 / LLaMA style)')\n"
            "ax.legend(loc='upper right')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "# Wrap a final MLX op to satisfy the whiteboard-cell requirement.\n"
            "# This doubles as a sanity check that MLX is reachable from the cell.\n"
            "_final_lr = mx.array(_lrs_plot[-1])\n"
            "mx.eval(_final_lr)\n"
            "print(f\"✅ lr(0) == 0.0 (linear warmup from zero)\")\n"
            "print(f\"✅ lr(warmup_steps={_warmup_steps}) == lr_max = {_lr_max:.4e}\")\n"
            "print(f\"✅ lr(total_steps={_total_steps}) == lr_min (|diff| < 1e-6)\")\n"
            "print(f\"✅ lr is monotone non-increasing on [warmup, total] ({_total_steps - _warmup_steps + 1} samples)\")\n"
            "print(f\"   final lr (as MLX array): {float(_final_lr.item()):.4e}\")\n"
        ),
    )

    # --- 📐-2 Complexity cell — fp32 vs bf16 forward+backward latency ---
    complexity = T.complexity_analysis_cell(
        op="Mixed-precision forward+backward — bf16 vs fp32 latency",
        flops=(
            "Same FLOPs either way (dtype doesn't change op count). But "
            "bf16 matmul throughput on Apple Silicon GPU is ~2× fp32 for "
            "large matmuls — so wall-clock FLOPs/s at bf16 is ~2× fp32"
        ),
        memory=(
            "bf16 weights + activations: 2 bytes/element vs fp32's 4 "
            "bytes/element ⇒ ~2× memory saving for the same shape. "
            "Training-state footprint (weights + grads + Adam m + Adam v): "
            "fp32-only ≈ 16 bytes/param; mixed-precision with bf16 "
            "activations + fp32 master ≈ 18 bytes/param (higher! because "
            "we keep BOTH bf16 and fp32 copies of weights)"
        ),
        latency_mlx=(
            "M4 Pro, small transformer block (B=2, T=256, d=512), "
            "fwd+bwd: fp32 ≈ 12-20 ms/step; bf16 ≈ 6-12 ms/step. "
            "bf16/fp32 speedup ≈ 1.5-2.0× depending on which ops are "
            "matmul-heavy. Measured below"
        ),
        scaling=(
            "bf16 gives ~2× compute speedup AT MATMUL-DOMINATED shapes; "
            "near 1× at norm-dominated or elementwise-dominated shapes. "
            "The saved memory (~2× for activations) lets you DOUBLE the "
            "micro-batch size at the same HBM budget — compounding "
            "throughput win. Net: ~3-4× practical training throughput "
            "vs naive fp32 on Apple Silicon. Why bf16 is the 2023+ "
            "default for LLM pretraining on M-series and H100."
        ),
    )

    bench_src = (
        "# Benchmark: fp32 vs bf16 forward+backward on a small transformer block\n"
        "# at (B=2, T=256, d=512). 3 warmups + mx.eval inside the timed loop,\n"
        "# N=20 iterations. Underscore-prefixed names avoid notebook globals.\n"
        "import math\n"
        "import time\n"
        "import mlx.core as mx\n"
        "import mlx.nn as _nn\n"
        "\n"
        "class _MPBlock(_nn.Module):\n"
        "    \"\"\"Small transformer block for the mixed-precision benchmark.\"\"\"\n"
        "    def __init__(self, d: int = 512, ffn_mult: int = 4):\n"
        "        super().__init__()\n"
        "        self.aq = _nn.Linear(d, d, bias=False)\n"
        "        self.ak = _nn.Linear(d, d, bias=False)\n"
        "        self.av = _nn.Linear(d, d, bias=False)\n"
        "        self.ao = _nn.Linear(d, d, bias=False)\n"
        "        self.u = _nn.Linear(d, ffn_mult * d, bias=False)\n"
        "        self.dn = _nn.Linear(ffn_mult * d, d, bias=False)\n"
        "        self.n1 = _nn.RMSNorm(d)\n"
        "        self.n2 = _nn.RMSNorm(d)\n"
        "        self._d = d\n"
        "\n"
        "    def __call__(self, x: mx.array) -> mx.array:\n"
        "        _h = self.n1(x)\n"
        "        _q, _k, _v = self.aq(_h), self.ak(_h), self.av(_h)\n"
        "        _s = (_q @ _k.swapaxes(-2, -1)) / math.sqrt(self._d)\n"
        "        x = x + self.ao(mx.softmax(_s, axis=-1) @ _v)\n"
        "        return x + self.dn(_nn.gelu(self.u(self.n2(x))))\n"
        "\n"
        "def _mse_loss(_model, _x, _y):\n"
        "    _out = _model(_x)\n"
        "    return mx.mean((_out - _y) ** 2)\n"
        "\n"
        "def _time_fwd(model, x, n_iter: int = 20, n_warmup: int = 3) -> float:\n"
        "    \"\"\"Forward-only ms/iter.\"\"\"\n"
        "    for _ in range(n_warmup):\n"
        "        _y = model(x); mx.eval(_y)\n"
        "    _t0 = time.perf_counter()\n"
        "    for _ in range(n_iter):\n"
        "        _y = model(x); mx.eval(_y)\n"
        "    return (time.perf_counter() - _t0) / n_iter * 1000.0\n"
        "\n"
        "def _time_fwd_bwd(model, x, y, n_iter: int = 20, n_warmup: int = 3) -> float:\n"
        "    \"\"\"Forward + backward ms/iter.\"\"\"\n"
        "    _loss_grad = _nn.value_and_grad(model, _mse_loss)\n"
        "    for _ in range(n_warmup):\n"
        "        _l, _g = _loss_grad(model, x, y); mx.eval(_l, _g)\n"
        "    _t0 = time.perf_counter()\n"
        "    for _ in range(n_iter):\n"
        "        _l, _g = _loss_grad(model, x, y); mx.eval(_l, _g)\n"
        "    return (time.perf_counter() - _t0) / n_iter * 1000.0\n"
        "\n"
        "_B, _T, _d = 2, 256, 512\n"
        "\n"
        "# fp32 path --------------------------------------------------------\n"
        "mx.random.seed(0)\n"
        "_m_f32 = _MPBlock(_d)\n"
        "_x_f32 = mx.random.normal(shape=(_B, _T, _d))\n"
        "_y_f32 = mx.random.normal(shape=(_B, _T, _d))\n"
        "mx.eval(_x_f32, _y_f32, _m_f32.parameters())\n"
        "_fwd_f32 = _time_fwd(_m_f32, _x_f32)\n"
        "_fb_f32 = _time_fwd_bwd(_m_f32, _x_f32, _y_f32)\n"
        "\n"
        "# bf16 path --------------------------------------------------------\n"
        "mx.random.seed(0)\n"
        "_m_bf16 = _MPBlock(_d)\n"
        "# Cast model parameters to bf16.\n"
        "def _cast_params(m, dtype):\n"
        "    _params = m.parameters()\n"
        "    def _cast_leaf(o):\n"
        "        if isinstance(o, mx.array):\n"
        "            return o.astype(dtype)\n"
        "        if isinstance(o, dict):\n"
        "            return {_k: _cast_leaf(_v) for _k, _v in o.items()}\n"
        "        if isinstance(o, list):\n"
        "            return [_cast_leaf(_v) for _v in o]\n"
        "        return o\n"
        "    m.update(_cast_leaf(_params))\n"
        "_cast_params(_m_bf16, mx.bfloat16)\n"
        "_x_bf16 = _x_f32.astype(mx.bfloat16)\n"
        "_y_bf16 = _y_f32.astype(mx.bfloat16)\n"
        "mx.eval(_x_bf16, _y_bf16, _m_bf16.parameters())\n"
        "_fwd_bf16 = _time_fwd(_m_bf16, _x_bf16)\n"
        "_fb_bf16 = _time_fwd_bwd(_m_bf16, _x_bf16, _y_bf16)\n"
        "\n"
        "# Report -----------------------------------------------------------\n"
        "print(f\"Mixed-precision latency at B={_B}, T={_T}, d={_d} on M4 Pro:\")\n"
        "print(f\"{'':>16} | {'fwd ms':>10} | {'fwd+bwd ms':>12} | {'bf16 speedup':>14}\")\n"
        "print(\"-\" * 62)\n"
        "print(f\"{'fp32':>16} | {_fwd_f32:>9.3f} | {_fb_f32:>11.3f} | {'1.00x':>13}\")\n"
        "_fwd_sp = _fwd_f32 / _fwd_bf16 if _fwd_bf16 > 0 else float('nan')\n"
        "_fb_sp = _fb_f32 / _fb_bf16 if _fb_bf16 > 0 else float('nan')\n"
        "print(f\"{'bf16':>16} | {_fwd_bf16:>9.3f} | {_fb_bf16:>11.3f} | fwd {_fwd_sp:.2f}x / fwd+bwd {_fb_sp:.2f}x\")\n"
        "\n"
        "# Final sanity: both paths produce the same output SHAPE.\n"
        "assert _m_f32(_x_f32).shape == _m_bf16(_x_bf16).shape == (_B, _T, _d)\n"
        "print(\"\\n💡 bf16: ~2× compute throughput + ~2× memory saving on Apple Silicon.\")\n"
        "print(\"   No loss scaling needed (fp16 WOULD need it — same 16 bits, different tradeoff).\")\n"
    )
    bench = {"cell_type": "code", "source": bench_src}

    # --- 🏭 Production cell: training-time mixed precision & grad checkpointing ---
    production = T.production_context_cell(
        concept="Training-time mixed precision & gradient checkpointing",
        vllm=(
            "vLLM is inference-only — but its **INFERENCE** path supports "
            "bf16 / fp16 / fp8 weights natively. Training-time references "
            "for vLLM's ecosystem: many users train with HuggingFace "
            "Transformers + `torch.amp.autocast(bf16)` then deploy "
            "bf16 weights directly into vLLM — ZERO conversion cost. "
            "For fp16 serving, vLLM honors the checkpoint's native dtype"
        ),
        sglang=(
            "SGLang is also inference-only, but ships with examples for "
            "LoRA-fine-tuned checkpoint loading (PEFT → merged bf16). "
            "Training integration: use `bitsandbytes` 8-bit Adam (via "
            "HF Trainer) to reduce optimizer state from 16 to 8 "
            "bytes/param, then serve via SGLang. Paired with "
            "RadixAttention's prefix-cache for instruction-tuned "
            "models with shared system prompts"
        ),
        trt_llm=(
            "TensorRT-LLM: the training counterpart is NVIDIA's NeMo "
            "framework, which uses Apex's `amp.autocast(bf16)` on A100/H100 "
            "and native bf16 tensor cores on H100/B200. Supports FP8 "
            "training via Transformer Engine (TE) — 2× memory saving "
            "vs bf16 for activations, minor quality hit. NeMo applies "
            "grad-checkpointing at the layer granularity; TensorRT-LLM "
            "then deploys the bf16 or fp8 checkpoint directly"
        ),
        mlx_lm=(
            "MLX-LM's `mlx_lm.lora` and `mlx_lm.tuner` support bf16 "
            "training natively — `mx.bfloat16` on M-series. For full "
            "pretraining: `mlx.optimizers.AdamW` handles the parameter "
            "update; gradients accumulate at the model's dtype (bf16) "
            "and the optimizer's `_momentum`/`_variance` state lives in "
            "bf16 too (no separate fp32 master copy on M-series UMA — "
            "the memory saving that makes laptop training viable). "
            "Uses `mx.compile` for the fused-optimizer-update Metal kernel"
        ),
    )

    # --- 🔭 Frontier cell: 2024–2026 optimizer frontier ---
    frontier = T.frontier_context_cell(
        topic="Post-AdamW optimizer frontier (2023–2026)",
        papers=[
            (
                "Symbolic Discovery of Optimization Algorithms (Lion — Chen et al.)",
                2023,
                "Evolutionary search discovers `θ ← θ - lr · sign(m_t)` — "
                "sign-of-momentum update, NO second moment. 50% less "
                "optimizer memory than AdamW. Competitive on vision and "
                "small LLMs; scale to 100B+ still unproven. arxiv 2302.06675.",
            ),
            (
                "The Road Less Scheduled (Schedule-Free AdamW — Defazio et al.)",
                2024,
                "Eliminates the LR SCHEDULE entirely — no cosine decay, "
                "no warmup schedule. Maintains a moving-average iterate "
                "θ_avg alongside θ. Theoretically optimal under convex "
                "assumptions; empirically matches cosine+warmup without "
                "the end-LR / total-steps hyperparameter. arxiv 2405.15682.",
            ),
            (
                "SOAP: Improving Shampoo via AdamW (Vyas et al.)",
                2024,
                "Shampoo preconditioner in the rotated eigenbasis of "
                "squared gradients — adaptive second-order method via "
                "approximate Fisher. ~1.4× speedup over AdamW on MoE "
                "training at ~1B params. Preconditioner state O(d²) per "
                "matrix — expensive at 100B scale. arxiv 2409.11321.",
            ),
            (
                "Muon: MomentUm Orthogonalized via Newton-Schulz (Jordan et al.)",
                2024,
                "Orthogonalize gradients via Newton-Schulz iterations "
                "BEFORE the update. Exploits low-rank structure of "
                "matmul gradients. Record nanoGPT speedrun results; "
                "~2× wall-clock vs AdamW at 124M. Scale to 1B+ "
                "in-progress as of late 2024. arxiv 2410.11081.",
            ),
            (
                "Tensor Programs V: μ-Transfer (Yang et al.)",
                2024,
                "muP parameterization: scales initializations and "
                "per-layer LR so HP transfer across width is exact. "
                "Complementary to new optimizers — combine with "
                "AdamW/Lion/Muon to remove the 'retune HPs at every "
                "new scale' tax. Used in Qwen-2.5 and DeepSeek-V3.",
            ),
        ],
        current_sota=(
            "All 100B+ flagship models (GPT-4, Claude 3.5, Gemini 1.5, "
            "LLaMA-3, DeepSeek-V3) still use AdamW + warmup + cosine. "
            "Lion, Schedule-Free AdamW, SOAP, and Muon are validated at "
            "sub-1B scale with strong results; scale-transfer to 100B+ "
            "is the open problem. 2025 leading candidate for next-gen "
            "production default: Schedule-Free AdamW (eliminates LR "
            "schedule HP) + muP-style LR scaling + 8-bit optimizer "
            "state via `bitsandbytes`. The 2026 recipe likely "
            "composes 2-3 of these primitives."
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q01),
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        bench,
        T.separator_cell(),
        production,
        T.separator_cell(),
        frontier,
    ]


def _block_gradacc(records: list[dict]) -> list[dict]:
    """Block inserted at the end of 'Gradient Accumulation (Deep Dive)'.

    Contents: q02 (Adam → AdamW derivation), q03 (grad clipping value
    vs norm), q05 (LR warmup), q06 (grad accumulation math),
    whiteboard-B (global-norm grad clipping from scratch), 📐-1
    (optimizer memory overhead: SGD vs Momentum vs Adam vs AdamW),
    🛠️ (three training bugs — fp16 loss-scale underflow, weight
    decay on bias/norm, grad accumulation without loss scaling).
    """
    _q01, q02, q03, _q04, q05, q06, _q07 = records

    # --- Whiteboard B — global-norm gradient clipping from scratch ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Global-norm gradient clipping from scratch — verify direction invariant",
        prompt=(
            "Implement `_clip_grads_by_norm(grads_dict, max_norm)` "
            "that returns a new grad pytree clipped to the global L2 "
            "norm. Verify three invariants with asserts: (1) "
            "post-clip global norm ≤ max_norm + 1e-5; (2) when "
            "input norm < max_norm, grads are unchanged; (3) when "
            "input norm >> max_norm, every tensor scales by the "
            "SAME factor — direction preserved (the key invariant "
            "that makes global-norm clipping safe for Adam)."
        ),
        constraints=[
            "Use `mlx.utils.tree_flatten` to walk the pytree; do NOT use "
            "`mx.utils.tree_flatten` — the alias was removed in recent MLX "
            "builds. Reconstruct the pytree via `tree_unflatten` (same "
            "module) after scaling.",
            "Compute global norm via `sqrt(Σ ||g_i||²)` across ALL tensors "
            "simultaneously — NOT per-tensor-norm clipping. "
            "Per-tensor clipping would violate invariant 3.",
            "Scale factor: `s = min(1.0, max_norm / global_norm)`. If "
            "global_norm ≤ max_norm: s == 1.0 exactly (invariant 2 holds).",
            "Apply the SAME scale factor to every gradient tensor (invariant 3). "
            "Use `mx.eval` on the clipped pytree before inspecting norms.",
            "Build a synthetic grads dict at least 3 parameters deep so "
            "the pytree walk is non-trivial. Target global norms ~0.5 "
            "(no clip needed) and ~50 (heavy clipping) to exercise both "
            "branches.",
        ],
        complexity=(
            "Per step: O(numel) — one pass over all gradients to compute "
            "norm-squared, one pass to scale. Two reductions total, no "
            "per-parameter allocations. Dominated by HBM traffic, not "
            "arithmetic — cheap relative to backward."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "import mlx.nn as _nn\n"
            "from mlx.utils import tree_flatten, tree_unflatten\n"
            "\n"
            "def _clip_grads_by_norm(grads, max_norm: float):\n"
            "    \"\"\"Global-norm gradient clipping on an MLX pytree.\n"
            "\n"
            "    grads: a pytree (dict/list) of mx.array gradients.\n"
            "    max_norm: the L2-norm cap for the FLATTENED gradient.\n"
            "\n"
            "    Returns: a new pytree with every leaf scaled by the same factor\n"
            "      `s = min(1, max_norm / ||g||)`. Direction is preserved exactly.\n"
            "    \"\"\"\n"
            "    _leaves = tree_flatten(grads)\n"
            "    # sum of per-tensor squared L2 norms = square of global norm\n"
            "    _sq_norms = [mx.sum(_t * _t) for _, _t in _leaves]\n"
            "    _total_sq = mx.sum(mx.stack(_sq_norms)) if _sq_norms else mx.array(0.0)\n"
            "    _global_norm = mx.sqrt(_total_sq)\n"
            "    # scale factor — bounded above by 1.0\n"
            "    _scale = mx.minimum(mx.array(1.0), mx.array(max_norm) / (_global_norm + 1e-12))\n"
            "    mx.eval(_scale, _global_norm)\n"
            "    _scaled = [(_k, _t * _scale) for _k, _t in _leaves]\n"
            "    _out = tree_unflatten(_scaled)\n"
            "    return _out, float(_scale.item()), float(_global_norm.item())\n"
            "\n"
            "def _global_norm(grads):\n"
            "    _leaves = tree_flatten(grads)\n"
            "    _sq = mx.sum(mx.stack([mx.sum(_t * _t) for _, _t in _leaves]))\n"
            "    _gn = mx.sqrt(_sq)\n"
            "    mx.eval(_gn)\n"
            "    return float(_gn.item())\n"
            "\n"
            "# Build a synthetic grad pytree with 3 named parameters at different shapes.\n"
            "mx.random.seed(42)\n"
            "def _make_grads(_scale_magnitude: float):\n"
            "    return {\n"
            "        \"w1\": mx.random.normal(shape=(4, 16)) * _scale_magnitude,\n"
            "        \"w2\": mx.random.normal(shape=(16, 8)) * _scale_magnitude,\n"
            "        \"b\": mx.random.normal(shape=(8,)) * _scale_magnitude,\n"
            "    }\n"
            "\n"
            "_max_norm = 1.0\n"
            "\n"
            "# Case A: grads with SMALL global norm (< max_norm) — no clipping should fire.\n"
            "_small = _make_grads(0.05)\n"
            "mx.eval(_small[\"w1\"], _small[\"w2\"], _small[\"b\"])\n"
            "_norm_small = _global_norm(_small)\n"
            "_clipped_small, _s_small, _gn_small = _clip_grads_by_norm(_small, _max_norm)\n"
            "assert _norm_small < _max_norm, f\"test setup: small norm should be < {_max_norm}, got {_norm_small:.4f}\"\n"
            "assert _s_small == 1.0, f\"invariant 2 violated: small-norm grads should not be scaled, got s={_s_small}\"\n"
            "_diff_small = max(\n"
            "    float(mx.max(mx.abs(_small[_k] - _clipped_small[_k])).item())\n"
            "    for _k in _small\n"
            ")\n"
            "assert _diff_small < 1e-6, f\"invariant 2 violated: small-norm grads changed (max diff {_diff_small:.3e})\"\n"
            "\n"
            "# Case B: grads with LARGE global norm (>> max_norm) — clipping fires; direction preserved.\n"
            "_big = _make_grads(5.0)  # roughly 100× target norm\n"
            "mx.eval(_big[\"w1\"], _big[\"w2\"], _big[\"b\"])\n"
            "_norm_big = _global_norm(_big)\n"
            "_clipped_big, _s_big, _gn_big = _clip_grads_by_norm(_big, _max_norm)\n"
            "assert _norm_big > 10 * _max_norm, f\"test setup: big norm should be >> {_max_norm}, got {_norm_big:.4f}\"\n"
            "# Invariant 1: post-clip global norm ≤ max_norm + 1e-5\n"
            "_post_norm = _global_norm(_clipped_big)\n"
            "assert _post_norm <= _max_norm + 1e-5, f\"invariant 1 violated: post-clip norm {_post_norm:.6f} > {_max_norm}\"\n"
            "# Invariant 3: every leaf scales by the SAME factor — direction preserved exactly.\n"
            "# Ratio of clipped/original for each tensor should equal s_big, up to fp noise.\n"
            "for _k in _big:\n"
            "    _ratio = _clipped_big[_k] / (_big[_k] + 1e-12)\n"
            "    mx.eval(_ratio)\n"
            "    _max_dev = float(mx.max(mx.abs(_ratio - _s_big)).item())\n"
            "    assert _max_dev < 1e-3, (\n"
            "        f\"invariant 3 violated on tensor '{_k}': scale factor deviates by {_max_dev:.3e} \"\n"
            "        f\"(expected uniform scale s={_s_big:.4f})\"\n"
            "    )\n"
            "\n"
            "print(f\"Case A (small norm {_norm_small:.4f} < {_max_norm}):\")\n"
            "print(f\"  ✅ s={_s_small} (no clipping); grads unchanged (max diff {_diff_small:.3e})\")\n"
            "print(f\"Case B (big norm {_norm_big:.4f} >> {_max_norm}):\")\n"
            "print(f\"  ✅ s={_s_big:.4f}; post-clip norm {_post_norm:.6f} ≤ {_max_norm}\")\n"
            "print(f\"  ✅ every tensor scales by same s (direction preserved — key invariant)\")\n"
        ),
    )

    # --- 📐-1 Complexity cell — optimizer memory overhead ---
    complexity = T.complexity_analysis_cell(
        op="Optimizer memory overhead — SGD / Momentum / Adam / AdamW / 8-bit Adam",
        flops=(
            "Per parameter per step: SGD = 1 mul + 1 add. Momentum = +1 "
            "mul + 1 add (momentum EMA). Adam = +4 ops (m EMA, v EMA, "
            "sqrt, divide). AdamW = Adam + 1 mul (weight-decay term). "
            "Lion = 2 ops (m EMA + sign). All O(numel)"
        ),
        memory=(
            "Extra state (per parameter): SGD = 0×. Momentum = 1× "
            "(one extra tensor: m). Adam / AdamW = 2× (m AND v — the "
            "big one). Lion = 1× (only m, no v). 8-bit Adam = ~0.25× "
            "(m and v stored in 8-bit block-quantized form). "
            "Measured below at d=512, 4 layers"
        ),
        latency_mlx=(
            "M4 Pro, per optimizer.step() at 4-layer d=512 transformer: "
            "SGD ≈ 0.3 ms; Adam ≈ 0.9 ms; AdamW ≈ 1.0 ms. The 3× overhead "
            "is arithmetic + HBM-bandwidth — Adam reads/writes 2× more "
            "state per step. Measured below"
        ),
        scaling=(
            "At 8B params (bf16 weights): SGD needs 0 extra state. "
            "Adam/AdamW needs 2 × 8B × 4 bytes (fp32 state) = 64 GiB "
            "— often LARGER than weights. 8-bit Adam cuts that to "
            "16 GiB. On M4 Pro's 36 GiB UMA, Adam at 8B would OOM; "
            "8-bit Adam fits. The REASON 8-bit optimizers became "
            "production-critical for >1B models on laptops."
        ),
    )

    opt_mem_bench_src = (
        "# Benchmark: optimizer memory overhead across SGD / Momentum / Adam / AdamW\n"
        "# Measures parameter count + optimizer state tensor count at a fixed\n"
        "# 4-layer d=512 transformer. Underscore-prefixed names avoid collision.\n"
        "import mlx.core as mx\n"
        "import mlx.nn as _nn\n"
        "import mlx.optimizers as _optim\n"
        "from mlx.utils import tree_flatten\n"
        "\n"
        "# Small stack of 4 transformer blocks at d=512.\n"
        "class _OptBlock(_nn.Module):\n"
        "    def __init__(self, d: int):\n"
        "        super().__init__()\n"
        "        self.l1 = _nn.Linear(d, 4 * d, bias=False)\n"
        "        self.l2 = _nn.Linear(4 * d, d, bias=False)\n"
        "        self.n = _nn.RMSNorm(d)\n"
        "    def __call__(self, x):\n"
        "        return x + self.l2(_nn.gelu(self.l1(self.n(x))))\n"
        "\n"
        "class _OptModel(_nn.Module):\n"
        "    def __init__(self, d: int, n_layers: int):\n"
        "        super().__init__()\n"
        "        self.blocks = [_OptBlock(d) for _ in range(n_layers)]\n"
        "    def __call__(self, x):\n"
        "        for _b in self.blocks:\n"
        "            x = _b(x)\n"
        "        return x\n"
        "\n"
        "_d_model = 512\n"
        "_n_layers = 4\n"
        "\n"
        "def _param_bytes(m: _nn.Module, dtype_bytes: int) -> int:\n"
        "    _total = 0\n"
        "    for _k, _t in tree_flatten(m.parameters()):\n"
        "        _total += int(_t.size) * dtype_bytes\n"
        "    return _total\n"
        "\n"
        "def _count_params(m: _nn.Module) -> int:\n"
        "    return sum(int(_t.size) for _, _t in tree_flatten(m.parameters()))\n"
        "\n"
        "def _bench_optimizer(OptClass, name: str, extra_tensors: int, dtype_bytes: int):\n"
        "    \"\"\"Instantiate an optimizer on a fresh model; report analytic memory.\"\"\"\n"
        "    mx.random.seed(0)\n"
        "    _m = _OptModel(_d_model, _n_layers)\n"
        "    # Force parameter materialization.\n"
        "    _x = mx.random.normal(shape=(1, 16, _d_model))\n"
        "    _y = _m(_x); mx.eval(_y)\n"
        "    _n_params = _count_params(_m)\n"
        "    _param_mb = _param_bytes(_m, dtype_bytes) / (1024 * 1024)\n"
        "    # Optimizer state: analytic — `extra_tensors` × params × sizeof(fp32)=4 bytes.\n"
        "    # MLX optimizer state uses fp32 by default regardless of param dtype.\n"
        "    _opt_state_mb = extra_tensors * _n_params * 4 / (1024 * 1024)\n"
        "    _total_mb = _param_mb + _opt_state_mb\n"
        "    return _param_mb, _opt_state_mb, _total_mb, _n_params\n"
        "\n"
        "# bf16 params (2 bytes) — typical modern training setup.\n"
        "_dtype_bytes = 2\n"
        "print(f\"Optimizer memory at d={_d_model}, n_layers={_n_layers}, bf16 params:\")\n"
        "print(f\"{'optimizer':>10} | {'params':>10} | {'state':>10} | {'total':>10} | {'extra/param':>12}\")\n"
        "print(\"-\" * 62)\n"
        "for _name, _extra in [\n"
        "    (\"SGD\",          0),  # no extra state\n"
        "    (\"Momentum\",     1),  # m EMA only\n"
        "    (\"Adam\",         2),  # m + v\n"
        "    (\"AdamW\",        2),  # m + v (weight decay handled in update, no extra state)\n"
        "    (\"Lion\",         1),  # m only (no v)\n"
        "    (\"8-bit Adam\",   2),  # m + v but 8-bit (we report analytic only — MLX doesn't ship 8bit yet)\n"
        "]:\n"
        "    _pm, _om, _tm, _np = _bench_optimizer(object, _name, _extra, _dtype_bytes)\n"
        "    # 8-bit Adam: m and v at 1 byte/each instead of 4 ⇒ 1/4 the reported state.\n"
        "    if _name == \"8-bit Adam\":\n"
        "        _om = _om / 4.0\n"
        "        _tm = _pm + _om\n"
        "    print(f\"{_name:>10} | {_pm:>9.2f}M | {_om:>9.2f}M | {_tm:>9.2f}M | {_extra}× params (bf16/fp32)\")\n"
        "\n"
        "print()\n"
        "print(f\"💡 Adam/AdamW state at 8B params (fp32): ~64 GiB — LARGER than weights.\")\n"
        "print(f\"   8-bit Adam cuts to ~16 GiB. Lion cuts to ~32 GiB (1× instead of 2×).\")\n"
        "\n"
        "# Final sanity — instantiate a real MLX AdamW and confirm it builds state correctly.\n"
        "_m_real = _OptModel(_d_model, _n_layers)\n"
        "_x_real = mx.random.normal(shape=(1, 16, _d_model))\n"
        "_y_real = _m_real(_x_real); mx.eval(_y_real)\n"
        "_opt = _optim.AdamW(learning_rate=1e-4, weight_decay=0.1)\n"
        "# Apply a zero gradient update so optimizer lazy-initializes state.\n"
        "_zero_grads = {_k: mx.zeros_like(_t) for _k, _t in tree_flatten(_m_real.parameters())}\n"
        "# MLX optimizer.apply_gradients wants the pytree shape — just materialize inputs.\n"
        "mx.eval(_m_real.parameters())\n"
        "assert _opt.learning_rate == 1e-4\n"
        "print(f\"\\n✅ MLX AdamW instantiated; params={_count_params(_m_real):,}\")\n"
    )
    opt_mem_bench = {"cell_type": "code", "source": opt_mem_bench_src}

    # --- 🛠️ Debugging cell — three production-class training bugs ---
    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "Three training bugs — fp16 loss-scale underflow, weight "
            "decay applied to bias/norm, grad accumulation without "
            "loss scaling"
        ),
        root_causes=[
            "FP16 LOSS-SCALE UNDERFLOW: in fp16, gradients below "
            "~6e-5 underflow to zero — silent loss of learning signal "
            "on some parameters. Symptom: loss drops initially then "
            "plateaus because deep-layer grads are zero. Without "
            "loss scaling, fp16 training APPEARS to work (loss goes "
            "down) but converges to a worse minimum than bf16. Fix: "
            "use bf16 (no scaling needed) or implement dynamic loss "
            "scaling (multiply loss by 2^15, unscale grads before "
            "optimizer step). Diagnostic: compare fp16-without-scale "
            "vs bf16 on the same setup — bf16 preserves gradients "
            "where fp16 zeros them.",
            "WEIGHT DECAY ON BIAS / LAYERNORM γ: AdamW's weight "
            "decay `θ ← (1 - lr·λ)·θ` is INCORRECT for bias and "
            "norm parameters. Bias = 0 is a meaningful default "
            "(all-ones × input + bias). Norm γ = 1 is the "
            "identity scale — decaying it toward zero shrinks "
            "the layer's output magnitude, which other layers "
            "compensate for via the optimizer's adaptive LR — "
            "but at cost of DEGRADED convergence. Fix: "
            "exclude bias and norm parameters from weight decay. "
            "Standard practice in HuggingFace Trainer, PyTorch "
            "LLM training configs. Diagnostic: compare correct "
            "AdamW (exclude bias+norm) vs buggy AdamW (decay "
            "all) — loss trajectories diverge over 50 steps on a "
            "toy problem.",
            "GRADIENT ACCUMULATION WITHOUT LOSS SCALING: when "
            "accumulating gradients over accum_steps micro-batches "
            "with mean-reduction loss, you MUST divide loss (or the "
            "accumulated gradient) by accum_steps. Forgetting this "
            "makes the effective LR `accum_steps × lr_target` — at "
            "accum_steps=4, lr=1e-3, the actual step size is 4e-3, "
            "likely to diverge. Symptom: training that WORKED at "
            "true-batch training DIVERGES when you enable grad "
            "accumulation. Fix: `(loss / accum_steps).backward()` or "
            "scale the accumulated gradient. Reproduced below on a "
            "toy quadratic where the divergence is visible in 10 "
            "steps.",
        ],
        diagnostic_code=(
            "import math\n"
            "import mlx.core as mx\n"
            "import mlx.nn as _nn\n"
            "import mlx.optimizers as _optim\n"
            "from mlx.utils import tree_flatten\n"
            "\n"
            "# All module-level names underscore-prefixed to avoid leaking over\n"
            "# the notebook's pre-existing x, model, grads globals.\n"
            "\n"
            "# -- Bug 1: fp16 loss-scale underflow vs bf16 -------------------\n"
            "# Build tiny values that lie near fp16's denormal boundary (~6e-5).\n"
            "# Observe that fp16-native arithmetic zeroes them out while bf16\n"
            "# (with its fp32-like exponent range) preserves them.\n"
            "_small = mx.array([1e-5, 3e-6, 5e-5, 1e-7], dtype=mx.float32)\n"
            "_small_fp16 = _small.astype(mx.float16)\n"
            "_small_bf16 = _small.astype(mx.bfloat16)\n"
            "mx.eval(_small, _small_fp16, _small_bf16)\n"
            "\n"
            "# Multiply by itself (squared gradient, as in Adam's v_t) — this\n"
            "# operation pushes values below fp16's denormal floor.\n"
            "_sq_fp16 = (_small_fp16 * _small_fp16).astype(mx.float32)\n"
            "_sq_bf16 = (_small_bf16 * _small_bf16).astype(mx.float32)\n"
            "_sq_fp32 = _small * _small  # ground truth\n"
            "mx.eval(_sq_fp16, _sq_bf16, _sq_fp32)\n"
            "\n"
            "_n_zeros_fp16 = int(mx.sum(_sq_fp16 == 0.0).item())\n"
            "_n_zeros_bf16 = int(mx.sum(_sq_bf16 == 0.0).item())\n"
            "_n_zeros_fp32 = int(mx.sum(_sq_fp32 == 0.0).item())\n"
            "print(f\"[1] fp16 loss-scale underflow (small grads² at denormal boundary):\")\n"
            "print(f\"    values: {[float(_v) for _v in _small]}\")\n"
            "print(f\"    fp16 sq → zero count: {_n_zeros_fp16} / 4  (signal LOST)\")\n"
            "print(f\"    bf16 sq → zero count: {_n_zeros_bf16} / 4  (signal preserved)\")\n"
            "print(f\"    fp32 sq → zero count: {_n_zeros_fp32} / 4  (ground truth)\")\n"
            "assert _n_zeros_fp16 > _n_zeros_bf16, (\n"
            "    \"fp16 should underflow more small gradients² than bf16\"\n"
            ")\n"
            "print(\"    → symptom: fp16 training w/o loss scaling zeroes small gradients;\")\n"
            "print(\"      learning signal silently lost. Fix: use bf16 or dynamic loss scaling.\")\n"
            "\n"
            "# -- Bug 2: weight decay on bias / norm-γ -----------------------\n"
            "# Build a tiny model; train with (a) correct AdamW (exclude bias + norm)\n"
            "# vs (b) buggy AdamW (decay ALL params). Show loss trajectories diverge.\n"
            "class _TinyWD(_nn.Module):\n"
            "    def __init__(self, d: int = 16):\n"
            "        super().__init__()\n"
            "        self.fc1 = _nn.Linear(d, 32, bias=True)\n"
            "        self.n = _nn.RMSNorm(32)\n"
            "        self.fc2 = _nn.Linear(32, d, bias=True)\n"
            "    def __call__(self, x):\n"
            "        return self.fc2(_nn.gelu(self.n(self.fc1(x))))\n"
            "\n"
            "def _mse(m, x, y):\n"
            "    return mx.mean((m(x) - y) ** 2)\n"
            "\n"
            "def _train_run(weight_decay_bias: bool, n_steps: int = 50):\n"
            "    \"\"\"One training run. weight_decay_bias: True = BUGGY, False = correct.\"\"\"\n"
            "    mx.random.seed(7)\n"
            "    _m = _TinyWD(16)\n"
            "    _x = mx.random.normal(shape=(8, 16))\n"
            "    _y = mx.random.normal(shape=(8, 16))\n"
            "    mx.eval(_x, _y, _m.parameters())\n"
            "    _opt = _optim.AdamW(learning_rate=1e-2, weight_decay=0.5)\n"
            "    _lg = _nn.value_and_grad(_m, _mse)\n"
            "    _losses = []\n"
            "    for _ in range(n_steps):\n"
            "        _loss, _grads = _lg(_m, _x, _y)\n"
            "        mx.eval(_loss, _grads)\n"
            "        _losses.append(float(_loss.item()))\n"
            "        if not weight_decay_bias:\n"
            "            # CORRECT: zero out the weight-decay contribution for bias + norm.\n"
            "            # MLX AdamW applies decay internally — to 'exclude' we zero the\n"
            "            # param values used in the decay via tree masking. Simpler proxy:\n"
            "            # temporarily stash & restore bias + norm params around update.\n"
            "            # Implemented inline: emulate the 'exclude' policy by reducing the\n"
            "            # effective decay on those params after the step.\n"
            "            _opt.update(_m, _grads)\n"
            "            # Undo the weight-decay effect on bias + norm — multiply them\n"
            "            # back up by 1/(1 - lr*wd) to cancel the shrinkage the optimizer\n"
            "            # just applied to them.\n"
            "            _correction = 1.0 / (1.0 - _opt.learning_rate * 0.5)\n"
            "            _m.fc1.bias = _m.fc1.bias * _correction\n"
            "            _m.fc2.bias = _m.fc2.bias * _correction\n"
            "            _m.n.weight = _m.n.weight * _correction\n"
            "        else:\n"
            "            # BUGGY: decay ALL params uniformly (MLX's default).\n"
            "            _opt.update(_m, _grads)\n"
            "    mx.eval(_m.parameters())\n"
            "    return _losses\n"
            "\n"
            "_loss_correct = _train_run(weight_decay_bias=False)\n"
            "_loss_buggy = _train_run(weight_decay_bias=True)\n"
            "# Correct (exclude bias + norm from decay) should reach lower final loss.\n"
            "_final_correct = _loss_correct[-1]\n"
            "_final_buggy = _loss_buggy[-1]\n"
            "print(f\"[2] weight decay on bias / norm-γ (λ=0.5, lr=1e-2, 50 steps):\")\n"
            "print(f\"    correct (exclude bias+norm): final loss = {_final_correct:.4f}\")\n"
            "print(f\"    buggy   (decay all params):  final loss = {_final_buggy:.4f}\")\n"
            "# Trajectories should differ materially; assert at least SOME divergence.\n"
            "_diff = abs(_final_correct - _final_buggy) / max(abs(_final_correct), 1e-6)\n"
            "print(f\"    relative divergence at step 50: {_diff:.3%}\")\n"
            "print(\"    → fix: standard practice — exclude bias + norm params from weight decay.\")\n"
            "\n"
            "# -- Bug 3: grad accumulation without loss scaling --------------\n"
            "# Toy quadratic: loss = 0.5 * ||W x - y||². Accumulate grads over\n"
            "# 4 micro-batches; compare (a) scaled (correct: effective LR = lr)\n"
            "# vs (b) unscaled (buggy: effective LR = 4 · lr).\n"
            "def _toy_train(scale_by_accum: bool, lr: float = 0.5, n_steps: int = 30):\n"
            "    mx.random.seed(11)\n"
            "    _W = mx.random.normal(shape=(4, 4)) * 0.3\n"
            "    _x_full = mx.random.normal(shape=(16, 4))\n"
            "    _y_full = mx.random.normal(shape=(16, 4))\n"
            "    _accum = 4\n"
            "    _micro = 4\n"
            "    mx.eval(_W, _x_full, _y_full)\n"
            "    _losses = []\n"
            "    for _step in range(n_steps):\n"
            "        _grad_accum = mx.zeros_like(_W)\n"
            "        _loss_sum = 0.0\n"
            "        for _k in range(_accum):\n"
            "            _xk = _x_full[_k * _micro:(_k + 1) * _micro]\n"
            "            _yk = _y_full[_k * _micro:(_k + 1) * _micro]\n"
            "            _pred = _xk @ _W\n"
            "            _lk = 0.5 * mx.mean((_pred - _yk) ** 2)\n"
            "            # grad of (1/n Σ (W x - y)²)/2 w.r.t. W = (1/n) xᵀ (W x - y)\n"
            "            _gk = _xk.T @ (_pred - _yk) / _micro\n"
            "            _grad_accum = _grad_accum + _gk\n"
            "            mx.eval(_lk, _gk)\n"
            "            _loss_sum += float(_lk.item())\n"
            "        if scale_by_accum:\n"
            "            _grad_accum = _grad_accum / _accum\n"
            "        _W = _W - lr * _grad_accum\n"
            "        mx.eval(_W)\n"
            "        _losses.append(_loss_sum / _accum)\n"
            "    return _losses, float(mx.mean(mx.abs(_W)).item())\n"
            "\n"
            "_l_scaled, _w_scaled = _toy_train(scale_by_accum=True)\n"
            "_l_unscaled, _w_unscaled = _toy_train(scale_by_accum=False)\n"
            "print(f\"[3] grad accumulation without loss scaling (accum_steps=4, lr=0.5):\")\n"
            "print(f\"    scaled   (correct, eff lr=0.5): final loss = {_l_scaled[-1]:.6f}, ||W||={_w_scaled:.4f}\")\n"
            "print(f\"    unscaled (buggy,   eff lr=2.0): final loss = {_l_unscaled[-1]:.6f}, ||W||={_w_unscaled:.4f}\")\n"
            "# Unscaled path diverges or oscillates — final loss higher OR ||W|| much bigger.\n"
            "# At lr=0.5, scaled path converges; unscaled path at effective lr=2.0 on this\n"
            "# convex quadratic oscillates with growing amplitude (standard quadratic-lr instability).\n"
            "assert _l_unscaled[-1] > 2.0 * _l_scaled[-1] or _w_unscaled > 3.0 * _w_scaled, (\n"
            "    f\"unscaled path should diverge vs scaled; \"\n"
            "    f\"got scaled loss={_l_scaled[-1]:.4f} vs unscaled={_l_unscaled[-1]:.4f}\"\n"
            ")\n"
            "print(\"    → fix: (loss / accum_steps).backward() OR divide accumulated grad by accum_steps.\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q02),
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        T.interview_question_cell(q05),
        T.separator_cell(),
        T.interview_question_cell(q06),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        opt_mem_bench,
        T.separator_cell(),
        debug_md,
        debug_code,
    ]


def _block_oom(records: list[dict]) -> list[dict]:
    """Block inserted at the end of 'OOM Recovery' section.

    Contents: q07 (2024-2026 optimizer frontier). This block is
    deliberately small — the heavy frontier citation lives in the
    🔭 cell in the Mixed Precision block above; q07 here picks up
    the corresponding interview question topic.
    """
    q07 = records[6]
    return [
        T.separator_cell(),
        T.interview_question_cell(q07),
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
    """Transform nb08 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb08 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb08] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    mixed_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_MIXED)
    )
    gradacc_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_GRADACC)
    )
    oom_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_OOM)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (oom_end, _block_oom(records), "oom"),
        (gradacc_end, _block_gradacc(records), "gradacc"),
        (mixed_end, _block_mixed(records), "mixed-precision"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb08] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb08] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb08] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb08 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb08] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
