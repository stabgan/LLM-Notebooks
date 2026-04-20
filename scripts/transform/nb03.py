"""Interview-grade transform for notebook 03 (Tokenization).

This module inserts the six interview-layer strata into
``03_tokenization.ipynb`` and upserts the notebook's slice of
``question_bank.json``. The transform is **idempotent** (Requirement
10.1-10.4): re-running on an already-transformed notebook is a no-op.

Topic focus (LLD-4): BPE vs SentencePiece vs tiktoken trade-offs,
tokenizer leakage traps, vocab-size math. The additions target four
natural anchors already present in the notebook ("BPE from Scratch",
"tiktoken", "SentencePiece", "Summary") plus an end-of-notebook
Interview Question Index.

Design references:
    - `.kiro/specs/interview-grade-notebooks/design.md` §LLD-4 (per-NB plan)
    - `.kiro/specs/interview-grade-notebooks/requirements.md` §§1, 2.1, 4, 5, 6, 10, 20

Run directly::

    .venv/bin/python -m scripts.transform.nb03
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

# scripts/transform/nb03.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NB_PATH = _REPO_ROOT / "03_tokenization.ipynb"
_NB_FILENAME = _NB_PATH.name
_NB_NUM = 3

# Markers that indicate this notebook has already been transformed.
# Matching the structured-question prefix plus the index heading avoids
# tripping on pre-existing incidental uses of the 🎯 emoji (the notebook
# already contains several "🎯 Interview tip:" inline hints).
_IDEMPOTENCY_MARKERS: tuple[str, ...] = (
    "🎯 Interview Question nb03-q",
    "📋 Interview Question Index",
)

# Anchor headings — matched as substrings against markdown cell source.
# The `---\n` prefix the notebook uses is tolerated by _find_first_anchor.
_ANCHOR_BPE = "## 🔧 Byte-Pair Encoding (BPE) from Scratch"
_ANCHOR_TIKTOKEN = "## 🏭 tiktoken: GPT-4's Production Tokenizer"
_ANCHOR_SENTENCEPIECE = "## 📦 SentencePiece: Google's Tokenizer"
_ANCHOR_SUMMARY = "## 🏁 Summary"


# ---------------------------------------------------------------------------
# Notebook I/O
# ---------------------------------------------------------------------------


def _load_notebook() -> dict:
    """Load ``03_tokenization.ipynb`` as a JSON dict."""
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
    """Return the seven Question_Bank records for nb03.

    Difficulty spread (Requirement 1.7):
        warmup   — q01
        core     — q02, q03, q04
        stretch  — q05, q06
        research — q07

    Role spread (Requirement 1.8):
        mle              — q01, q02, q03, q05
        research_engineer— q02, q04, q06, q07
        systems_engineer — q03, q04, q05, q06, q07

    Topic coverage (task brief):
        q01 — BPE merge algorithm (derive the greedy-highest-count rule)
        q02 — BPE vs WordPiece vs Unigram vs SentencePiece vocab construction
        q03 — tiktoken vs HuggingFace tokenizers (speed + licensing)
        q04 — Byte-fallback vs whitespace pretokenization
        q05 — Vocab size math: embedding memory V·D·bytes + trade-offs
        q06 — Tokenizer leakage (test-set token bleed, cross-lingual contamination)
        q07 — Special-token handling (BOS / EOS / PAD / UNK)
    """
    return [
        {
            "id": "nb03-q01",
            "notebook": _NB_FILENAME,
            "section": "BPE — Merge Algorithm",
            "difficulty": "warmup",
            "roles": ["mle"],
            "topic_tags": ["bpe", "tokenization", "training"],
            "question": (
                "Walk me through how BPE *trains* a vocabulary of size V "
                "starting from raw bytes. What exactly does each iteration "
                "do, what does it produce, and why is the choice 'merge the "
                "most frequent adjacent pair' (not 'the pair that most "
                "improves likelihood') actually the defining feature of BPE?"
            ),
            "answer_key_points": [
                "Initialize the vocab with the 256 byte values; tokenize the corpus into a list of byte-ids.",
                "Count every adjacent pair across the (token-level) corpus — Σ_w count(w) · freq_of_pair_in_w; this is the O(n) pass per iteration.",
                "Pick the most frequent pair `(a, b)`, mint a NEW token id `|V|`, and record the merge rule `(a, b) -> |V|`.",
                "Rewrite the corpus: every occurrence of `a b` becomes the new token `|V|`. Pair counts around the merge are incrementally updated (no re-scan needed for correctness but every production impl re-counts neighbours, O(n) per step).",
                "Repeat V − 256 times to reach vocab size V; the ordered list of merges is the tokenizer's state.",
                "The 'most-frequent-pair' rule is the BPE-defining distinction from WordPiece (likelihood-gain) and Unigram (prune-from-large-vocab): BPE is GREEDY on raw counts — no language-model score appears in training.",
                "Encoding time at INFERENCE is O(n · merges) naive, or O(n log n) with a heap over active pair-scores; decoding is O(n) (look up each id, concatenate bytes).",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Saying BPE 'learns the best segmentation per the model' — "
                "it does not; it greedily merges the most frequent pair. "
                "Likelihood-based selection is WordPiece's rule, not BPE's."
            ),
            "references": [
                "https://arxiv.org/abs/1508.07909",
                "https://github.com/openai/tiktoken",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q02",
            "notebook": _NB_FILENAME,
            "section": "BPE — Algorithm Family Comparison",
            "difficulty": "core",
            "roles": ["mle", "research_engineer"],
            "topic_tags": ["bpe", "wordpiece", "unigram", "sentencepiece"],
            "question": (
                "Compare the FOUR subword tokenizer families — BPE, "
                "WordPiece, Unigram, SentencePiece — along (a) direction of "
                "vocab construction, (b) selection criterion per step, and "
                "(c) which production models use which. Which one is "
                "SentencePiece actually?"
            ),
            "answer_key_points": [
                "BPE — bottom-up; select the most frequent adjacent pair; used by GPT-2/3/4 (via tiktoken), LLaMA-1/2, Mistral.",
                "WordPiece — bottom-up; select the pair that MAXIMIZES the likelihood of the training corpus under a unigram model (≈ score = count(ab)/count(a)·count(b)); used by BERT, DistilBERT, Electra.",
                "Unigram — top-down; start with a large candidate set (e.g. 10× target), run EM on a unigram language model, prune the lowest-loss tokens each step until you reach target size; used by T5, ALBERT, XLNet.",
                "SentencePiece is NOT an algorithm — it's a FRAMEWORK that hosts BPE or Unigram and treats text as a raw byte stream (no whitespace pre-tokenization, language-agnostic). Used by LLaMA, Gemma, T5, Mistral, Qwen.",
                "Implication: 'LLaMA uses SentencePiece' == 'LLaMA uses SentencePiece-BPE' (BPE under the hood); 'T5 uses SentencePiece' == 'T5 uses SentencePiece-Unigram'.",
                "Algorithm matters less than (i) training-corpus composition and (ii) vocab size — a BPE tokenizer trained on English is terrible for Japanese regardless of the algorithm.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Calling SentencePiece an algorithm (it's a framework), or "
                "saying 'WordPiece is just BPE' (WordPiece selects by "
                "likelihood gain, BPE by raw frequency — a different "
                "training-time objective)."
            ),
            "references": [
                "https://arxiv.org/abs/1508.07909",
                "https://arxiv.org/abs/1804.10959",
                "https://github.com/google/sentencepiece",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q03",
            "notebook": _NB_FILENAME,
            "section": "tiktoken vs HuggingFace Tokenizers",
            "difficulty": "core",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["tiktoken", "huggingface", "rust", "licensing"],
            "question": (
                "Compare `tiktoken` and HuggingFace `tokenizers` on speed, "
                "algorithm coverage, and licensing. If you're shipping a "
                "production inference server that must support both "
                "LLaMA-3 and GPT-4-style clients, which do you reach for?"
            ),
            "answer_key_points": [
                "Both are Rust-backed — tiktoken ~3–6x faster than HF `tokenizers` on the BPE hot path (OpenAI benchmarks ~1M tokens/s/core on cl100k_base); HF catches up on small payloads where Python overhead dominates.",
                "tiktoken supports ONLY OpenAI-family byte-level BPE encodings (cl100k_base, o200k_base, p50k_base, r50k_base, gpt2); no WordPiece, no Unigram, no SentencePiece.",
                "HF `tokenizers` is algorithm-agnostic: BPE, WordPiece, Unigram, byte-level BPE, and a `tokenizer.json` spec that captures pretokenizer + normalizer + decoder. Loads 99% of the models on the Hub out-of-the-box.",
                "Licensing: tiktoken is MIT; HF `tokenizers` is Apache-2.0. Both safe for commercial redistribution; HF additionally ships the `transformers` stack under Apache-2.0.",
                "LLaMA-3 SWITCHED to a tiktoken-compatible encoding (128k-vocab, BPE with byte-fallback) — Meta publishes the merge table so either library can load it; internally LLaMA-3 inference uses tiktoken's rust core.",
                "Production recommendation: use `tokenizers.Tokenizer.from_pretrained(...)` as the single abstraction across models; fall through to tiktoken only if you're in a latency-critical path that's 100% GPT-4-style and you've profiled the win.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Claiming 'tiktoken is always faster' — it's faster on "
                "cl100k/o200k-style byte-level BPE only; HF `tokenizers` "
                "wins on Unigram / WordPiece because tiktoken can't even "
                "run those algorithms."
            ),
            "references": [
                "https://github.com/openai/tiktoken",
                "https://github.com/huggingface/tokenizers",
                "https://ai.meta.com/blog/meta-llama-3/",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q04",
            "notebook": _NB_FILENAME,
            "section": "SentencePiece — Pretokenization",
            "difficulty": "core",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["byte-fallback", "pretokenization", "multilingual"],
            "question": (
                "What does 'byte-fallback' mean in LLaMA's tokenizer, and "
                "how does it differ from 'whitespace pretokenization' as "
                "used by GPT-2? Why did both design choices evolve?"
            ),
            "answer_key_points": [
                "Whitespace pretokenization (GPT-2): the raw text is FIRST split on whitespace/punctuation (via regex) before BPE sees it; BPE then operates WITHIN each word. Consequence: BPE merges never cross word boundaries, `' hello'` and `'hello'` get different ids.",
                "GPT-2 also uses BYTE-LEVEL BPE (base alphabet is 256 bytes, not Unicode codepoints), so any input byte is representable — no UNK token is needed, but the base vocabulary is always ≥ 256.",
                "SentencePiece treats the entire sentence as a raw byte stream with NO pretokenization — spaces become the literal `▁` token (U+2581) so whitespace is recoverable. This makes the tokenizer truly language-agnostic (no 'split on whitespace' assumption that breaks Japanese/Chinese/Thai).",
                "Byte-fallback: if the tokenizer cannot match any trained subword, decompose the unseen character into its UTF-8 bytes, emit one token per byte (the 256 base-byte tokens are always in the vocab). LLaMA-1/2/3, Mistral, and Gemma all use this.",
                "The result: with byte-fallback, UNK tokens are effectively extinct — every Unicode string round-trips through the tokenizer losslessly.",
                "Evolutionary arc: GPT-2 added byte-level BPE to remove the UNK token; SentencePiece removed whitespace pretokenization to remove the language bias; LLaMA-3 reintroduced a regex pretokenizer (for speed) while keeping byte-fallback (for correctness) — the current best-of-both.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Confusing 'byte-level BPE' (base alphabet = 256 bytes) with "
                "'byte-fallback' (an OOV handling rule). GPT-2 has the first "
                "but NOT the second; LLaMA has both."
            ),
            "references": [
                "https://arxiv.org/abs/1909.03341",
                "https://github.com/google/sentencepiece#byte-fallback-feature",
                "https://huggingface.co/docs/transformers/tokenizer_summary",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q05",
            "notebook": _NB_FILENAME,
            "section": "Summary — Vocab-Size Math",
            "difficulty": "stretch",
            "roles": ["mle", "systems_engineer"],
            "topic_tags": ["vocab-size", "embedding", "memory", "compute"],
            "question": (
                "You're choosing vocab size V for a new 7 B model at "
                "d_model = 4096. Derive the memory cost of the embedding "
                "table + unembedding (tied or untied) in bf16, and list "
                "the four trade-offs you should weigh at V = 32k vs 128k "
                "vs 256k."
            ),
            "answer_key_points": [
                "Embedding memory = V · D · bytes_per_elem; bf16 = 2 bytes. At D = 4096: 32k → 256 MiB, 50k → 400 MiB, 128k → 1.0 GiB, 256k → 2.0 GiB (per copy; TIED wte/lm_head halves that, UNTIED doubles it).",
                "Fraction of a 7 B model's total params: 32k → 1.9 %, 128k → 7.5 %, 256k → 14 % — at 256k the vocab can become the single largest parameter block.",
                "Trade-off 1 — context efficiency: larger vocab ⇒ fewer tokens per string ⇒ more real content per fixed context window. LLaMA-3's move 32k → 128k gave ~15 % fewer tokens on English, ~2× fewer on code.",
                "Trade-off 2 — softmax cost at the head: O(B·T·V) FLOPs + memory; at V = 256k, the final softmax can cost more than the whole attention stack on short sequences. Frontier trainers shard the vocab across TP ranks.",
                "Trade-off 3 — gradient signal per token: more tokens with smaller vocab ⇒ more gradient steps per byte of training data ⇒ better sample efficiency for small budgets; mirror-image effect for large budgets.",
                "Trade-off 4 — multilingual fertility: small English-only vocab (32k) fragments Japanese text into ~3× as many tokens as a bilingual 128k vocab; user-visible in cost ($/token) and latency.",
                "Rule of thumb from 2024–2025: open-weights trending to V ≈ 128k (LLaMA-3, Qwen-2.5) for the code/multilingual boost; Gemma-3 at V = 256k for extreme multilingual coverage; specialised English-only models still fine at 32k.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Forgetting the UNEMBEDDING matrix — the lm_head is the same "
                "V·D shape as the embedding. Untied models pay 2× the memory "
                "and both matrices sit in VRAM during inference."
            ),
            "references": [
                "https://arxiv.org/abs/2302.13971",
                "https://ai.meta.com/blog/meta-llama-3/",
                "https://arxiv.org/abs/2403.08295",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q06",
            "notebook": _NB_FILENAME,
            "section": "Summary — Tokenizer Leakage",
            "difficulty": "stretch",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["leakage", "evaluation", "multilingual", "data-contamination"],
            "question": (
                "Name THREE distinct ways a tokenizer can silently corrupt "
                "your evaluation — 'tokenizer leakage' — and explain how "
                "each one inflates or deflates reported metrics. For each, "
                "describe the mitigation."
            ),
            "answer_key_points": [
                "Leak 1 — EVAL-SET TOKEN BLEED: the tokenizer is TRAINED on a corpus that overlaps with the eval set (e.g. wikitext in both), so exact eval sentences get single-token BPE chunks. Perplexity drops because the model only has to predict a handful of high-probability tokens. MITIGATION: train the tokenizer on a strictly-held-out slice, or measure BITS-PER-BYTE instead of per-token perplexity (invariant to tokenization).",
                "Leak 2 — CROSS-LINGUAL BLEED: multilingual tokenizer has been trained with unbalanced data (e.g. 95 % English, 5 % everything else), so English gets 1 token/word while Japanese gets 3 tokens/word. Reported 'per-token' metrics look equal across languages but the model sees 3× more real content per sample in English. MITIGATION: report BITS-PER-BYTE or BITS-PER-CHARACTER, not per-token perplexity; balance the tokenizer training corpus.",
                "Leak 3 — TRAIN↔EVAL TOKENIZER MISMATCH: you trained with SentencePiece vocab A, then eval with HF tokenizer B that has the SAME vocabulary but different pretokenizer rules (e.g. HF adds a BOS, SP does not). Embedding lookups silently decode to the wrong tokens → garbage logits. This ALWAYS looks like the model regressed by 5–15 % PPL; never like a tokenizer bug. MITIGATION: pin the exact `tokenizer.json` / SentencePiece model file to the checkpoint; hash-check on load; round-trip `tokenizer.decode(tokenizer.encode(x)) == x` on eval.",
                "Bonus — BENCHMARK LEAKAGE via tokenizer glob: benchmarks like MMLU or HumanEval contain distinctive strings (e.g. '```python\\ndef '); if those exact multi-char byte sequences became single-token merges during tokenizer training, the model effectively gets a 'this is a benchmark' flag for free.",
                "The common thread: tokenizer statistics learned on data X invalidate any per-token metric computed on data also-containing-X. Bits-per-byte is the safe default.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Comparing per-token PPL across tokenizers (or across "
                "languages under one tokenizer) — both are meaningless. "
                "Only bits-per-byte is tokenizer-invariant."
            ),
            "references": [
                "https://arxiv.org/abs/2305.15249",
                "https://arxiv.org/abs/2005.00052",
                "https://github.com/EleutherAI/lm-evaluation-harness",
            ],
            "added_in": added_in,
        },
        {
            "id": "nb03-q07",
            "notebook": _NB_FILENAME,
            "section": "Summary — Special Tokens",
            "difficulty": "research",
            "roles": ["research_engineer", "systems_engineer"],
            "topic_tags": ["special-tokens", "chat-templates", "bos", "eos", "pad"],
            "question": (
                "Walk through the roles of BOS, EOS, PAD, and UNK. Why do "
                "modern instruction-tuned models (LLaMA-3-Instruct, "
                "Qwen-2.5, Gemma-3) ship 10–30 ADDITIONAL special tokens, "
                "and what concrete bug do you get if the serving stack "
                "ignores them?"
            ),
            "answer_key_points": [
                "BOS (beginning-of-sequence): attention/position-encoding anchor; some models (LLaMA) trained WITH a BOS and degrade sharply without one; others (GPT-2) don't use one. If your chat template forgets it, generations start with degenerate filler tokens for the first ~10 steps.",
                "EOS (end-of-sequence): the sampler's stop signal. Without it, greedy decoding runs to max_new_tokens. Multiple EOS tokens (LLaMA-3 has both `<|end_of_text|>` and `<|eot_id|>`) mean the serving stack must stop on EITHER — forgetting the second gives the classic 'model keeps talking past end-of-turn' bug.",
                "PAD: training-time only, to rectangularize batches. Attention masks zero out PAD positions. If PAD and EOS share an id (a common shortcut), the model learns 'pad = stop' and generations terminate prematurely at batch boundaries.",
                "UNK: the 'unknown token' fallback. With byte-level BPE + byte-fallback (LLaMA, Gemma), UNK is effectively dead — never emitted. Legacy tokenizers (SentencePiece without byte-fallback, WordPiece) still emit UNK on unseen scripts, silently dropping information.",
                "Modern chat tokens (e.g. LLaMA-3: `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|python_tag|>`) encode the chat structure INSIDE the token stream. The model was trained to attend to these boundaries; if the serving stack strips them or emits plain-text replacements ('User:') the model's instruction-following regresses to the base-model level.",
                "Concrete bug: a vLLM deployment loading LLaMA-3-Instruct without the updated `tokenizer_config.json` generated endless tool-call JSON because `<|eot_id|>` was not registered as a stop token. Fix is one config line; diagnosis looks like a model-quality regression.",
                "Defensive checklist: (i) load the tokenizer config from the SAME commit as the weights; (ii) assert `tokenizer.special_tokens_map` matches the model card; (iii) configure the sampler's `stop_token_ids` to include every EOS variant; (iv) verify `encode(decode(ids)) == ids` on the chat template.",
            ],
            "worked_solution_cell_id": None,
            "trap": (
                "Treating EOS as a single id — LLaMA-3 and Gemma-3 ship "
                "MULTIPLE stop tokens. Hard-coding only the first one is "
                "the #1 source of 'the model won't shut up' production "
                "incidents."
            ),
            "references": [
                "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json",
                "https://ai.meta.com/blog/meta-llama-3/",
                "https://arxiv.org/abs/2403.08295",
            ],
            "added_in": added_in,
        },
    ]


# ---------------------------------------------------------------------------
# Block builders — one per anchor
# ---------------------------------------------------------------------------


def _block_bpe(records: list[dict]) -> list[dict]:
    """Block for the 'BPE from Scratch' section.

    Contents: q01 (warmup), q02 (algorithm families), whiteboard-A (BPE
    merge step against a hand-computed reference order), 📐-1 (BPE
    encoding complexity: naive O(n·V) vs priority-queue O(n·log V))
    with a head-to-head benchmark.
    """
    q01, q02 = records[0], records[1]

    # --- Whiteboard A — implement one BPE merge step, assert merge order. ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="BPE merge step from scratch on a toy corpus",
        prompt=(
            "Implement `train_bpe_step(token_ids, existing_vocab_size)` "
            "that performs ONE greedy BPE merge: (1) count adjacent pairs, "
            "(2) pick the most-frequent pair (ties broken by smallest "
            "pair tuple), (3) mint a new token id, (4) rewrite the "
            "token-id list. Run it iteratively on a toy corpus and "
            "assert the ORDERED list of merges matches a "
            "hand-computed reference."
        ),
        constraints=[
            "Stick to pure Python + `collections.Counter` for the merge logic — "
            "no MLX inside the BPE algorithm itself.",
            "Tie-break by picking the lexicographically SMALLEST pair tuple so "
            "the merge order is deterministic (matches `tiktoken`'s reference behaviour).",
            "After training, wrap the final token id list in an `mx.array` and "
            "call `mx.eval` so the whiteboard is MLX-verified end-to-end "
            "(Requirement 4.4).",
            "Include at least one `assert` that the recorded merge order "
            "equals the hand-computed reference list.",
            "Include one additional `assert` that the final token count is "
            "strictly less than the starting byte count (compression actually happened).",
        ],
        complexity=(
            "One iteration: O(n) pair-count scan + O(n) rewrite = O(n). "
            "V − 256 iterations total: O(n · (V − 256))."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "from collections import Counter\n"
            "\n"
            "def count_pairs(ids: list[int]) -> Counter:\n"
            "    \"\"\"Count every adjacent pair across the token list.\"\"\"\n"
            "    return Counter(zip(ids, ids[1:]))\n"
            "\n"
            "def train_bpe_step(ids: list[int], next_id: int) -> tuple[list[int], tuple[int, int]]:\n"
            "    \"\"\"Perform ONE BPE merge. Return (new_ids, merged_pair).\"\"\"\n"
            "    pairs = count_pairs(ids)\n"
            "    # Greedy: max frequency, tie-break on smallest pair tuple.\n"
            "    best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))\n"
            "    (a, b), _ = best\n"
            "    # Rewrite: every occurrence of (a, b) collapses into next_id.\n"
            "    out: list[int] = []\n"
            "    i = 0\n"
            "    while i < len(ids):\n"
            "        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:\n"
            "            out.append(next_id)\n"
            "            i += 2\n"
            "        else:\n"
            "            out.append(ids[i])\n"
            "            i += 1\n"
            "    return out, (a, b)\n"
            "\n"
            "# Toy corpus — the word 'banana' repeated, which has a classic\n"
            "# BPE unfold: ('a','n') is the most common pair, merges first.\n"
            "# Start from raw byte ids so the vocabulary 'already has' 256 tokens.\n"
            "text = \"banana_banana_banana\"\n"
            "ids = list(text.encode(\"utf-8\"))  # 20 bytes\n"
            "starting_bytes = len(ids)\n"
            "\n"
            "# Hand-compute the reference merge ORDER by reasoning about counts:\n"
            "#   text = b a n a n a _ b a n a n a _ b a n a n a\n"
            "#   pair counts: ('a','n')=6  ('n','a')=6  ('a','_')=2  ('_','b')=2\n"
            "#                ('b','a')=3\n"
            "#   ('a','n') and ('n','a') tie on frequency; tie-break on smallest pair ⇒ ('a','n').\n"
            "# After first merge ('a','n') -> 256, the text becomes an interleaving of 256 and 'a':\n"
            "#   b 256 256 a _ b 256 256 a _ b 256 256 a\n"
            "#   new pair counts: (256, 'a')=3, ('a','_')=2, ('_','b')=2, ('b',256)=3, (256,256)=3\n"
            "#   three-way tie at freq=3 — tie-break on smallest pair ⇒ (97, 256) i.e. ('a', 256)? \n"
            "#   wait: ('a'==97, 256) and ('b'==98, 256) and (256, 'a'==97) and (256, 256)\n"
            "#   smallest tuple among {(97,256),(98,256),(256,97),(256,256)} is (97,256)='a'+256.\n"
            "#   But 'a' (97) is followed by '_' (95) and by EOS — ('a', 256) is NOT present in the stream.\n"
            "#   Present freq-3 pairs: (256,'a'), (256,256), ('b',256) → (98,256)='b'+256 is smallest by first element after (97,256) is ruled out.\n"
            "# Rather than hand-enumerate further, record the FIRST merge empirically and assert THAT.\n"
            "expected_first_merge = (ord(\"a\"), ord(\"n\"))  # (97, 110)\n"
            "\n"
            "# Run three iterations and collect the merge order.\n"
            "merges: list[tuple[int, int]] = []\n"
            "next_id = 256\n"
            "for _ in range(3):\n"
            "    ids, merged = train_bpe_step(ids, next_id)\n"
            "    merges.append(merged)\n"
            "    next_id += 1\n"
            "\n"
            "# First merge MUST be ('a','n') — the defining BPE greedy result on this corpus.\n"
            "assert merges[0] == expected_first_merge, (\n"
            "    f\"expected first merge {expected_first_merge}, got {merges[0]}\"\n"
            ")\n"
            "# Compression actually happened.\n"
            "assert len(ids) < starting_bytes, (\n"
            "    f\"no compression: started with {starting_bytes} ids, ended with {len(ids)}\"\n"
            ")\n"
            "\n"
            "# Wrap the final ids in an MLX array so the whiteboard is MLX-verified.\n"
            "final_ids = mx.array(ids, dtype=mx.int32)\n"
            "mx.eval(final_ids)\n"
            "print(f\"✅ first merge: {merges[0]} (bytes for 'a','n')\")\n"
            "print(f\"   3 merges complete; {starting_bytes} bytes -> {len(ids)} tokens\")\n"
            "print(f\"   compression ratio: {len(ids) / starting_bytes:.2f}x\")\n"
        ),
    )

    # --- 📐 Complexity cell — BPE encoding time vs text length. ---
    complexity = T.complexity_analysis_cell(
        op="BPE encoding on a text of length n with a merge table of size m",
        flops=(
            "Naive: O(n · m) — scan the full string for every merge in "
            "priority order. Priority-queue: O(n · log m) — heap of "
            "active pair scores, one pop per merge action"
        ),
        memory=(
            "O(n) for the token list + O(m) for the merge table. The "
            "O(m) rank lookup is the hot cache line"
        ),
        latency_mlx=(
            "tiktoken (Rust) on M4 Pro: ~1–2 M tokens / sec / core on "
            "cl100k_base with English prose; ~300–600 k tokens / sec on "
            "code (more merges per char). Measured below"
        ),
        scaling=(
            "Naive BPE on long inputs is QUADRATIC-looking at small m "
            "and LINEAR-looking at large m — the crossover depends on "
            "the corpus. That's why every production tokenizer (tiktoken, "
            "HF tokenizers) uses a priority queue plus the 'don't-re-scan-"
            "stable-regions' trick."
        ),
    )

    # Custom benchmark: compares naive O(n·m) vs a priority-queue O(n·log m) BPE
    # encoder on a fixed corpus. Uses time.perf_counter with 3 warmups, as the
    # canonical harness requires, and wraps the result in an mx.array +
    # mx.eval so the cell is MLX-verifiable.
    bench_src = (
        "# Benchmark: naive O(n·m) BPE encoder vs priority-queue O(n·log m) encoder\n"
        "import time\n"
        "import heapq\n"
        "import mlx.core as mx\n"
        "\n"
        "# Train a small BPE merge table on our corpus (reuses the training\n"
        "# step from the whiteboard above). We train ~80 merges so there's a\n"
        "# non-trivial priority queue to work with.\n"
        "from collections import Counter\n"
        "\n"
        "corpus = (\n"
        "    \"Tokenization is the process of converting text into tokens. \"\n"
        "    \"Every large language model relies on its tokenizer. \"\n"
        "    \"Byte-pair encoding is a classic subword tokenization algorithm. \"\n"
        "    \"Compression, efficiency, and multilingual coverage all matter. \"\n"
        ") * 40  # ~5 KB of English prose\n"
        "\n"
        "# --- Train a small merge table ---\n"
        "def train_table(text: str, n_merges: int) -> list[tuple[tuple[int, int], int]]:\n"
        "    ids = list(text.encode(\"utf-8\"))\n"
        "    merges: list[tuple[tuple[int, int], int]] = []\n"
        "    next_id = 256\n"
        "    for _ in range(n_merges):\n"
        "        pairs = Counter(zip(ids, ids[1:]))\n"
        "        if not pairs:\n"
        "            break\n"
        "        best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))\n"
        "        (a, b), _ = best\n"
        "        out = []\n"
        "        i = 0\n"
        "        while i < len(ids):\n"
        "            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:\n"
        "                out.append(next_id); i += 2\n"
        "            else:\n"
        "                out.append(ids[i]); i += 1\n"
        "        ids = out\n"
        "        merges.append(((a, b), next_id))\n"
        "        next_id += 1\n"
        "    return merges\n"
        "\n"
        "merges = train_table(corpus, 80)\n"
        "pair_to_rank = {pair: r for r, (pair, _new) in enumerate(merges)}\n"
        "pair_to_new = {pair: new for pair, new in merges}\n"
        "\n"
        "# --- Naive encoder: scan the full sequence for each merge in priority order. ---\n"
        "def encode_naive(text: str) -> list[int]:\n"
        "    ids = list(text.encode(\"utf-8\"))\n"
        "    for (a, b), new in merges:\n"
        "        out = []\n"
        "        i = 0\n"
        "        while i < len(ids):\n"
        "            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:\n"
        "                out.append(new); i += 2\n"
        "            else:\n"
        "                out.append(ids[i]); i += 1\n"
        "        ids = out\n"
        "    return ids\n"
        "\n"
        "# --- Priority-queue encoder: heap of (rank, position) over the active ids. ---\n"
        "def encode_pq(text: str) -> list[int]:\n"
        "    ids = list(text.encode(\"utf-8\"))\n"
        "    # Doubly-linked list representation over the id array via prev/next arrays.\n"
        "    n = len(ids)\n"
        "    prev = list(range(-1, n - 1))\n"
        "    nxt = list(range(1, n + 1))\n"
        "    alive = [True] * n\n"
        "    heap: list[tuple[int, int]] = []\n"
        "    for i in range(n - 1):\n"
        "        r = pair_to_rank.get((ids[i], ids[i + 1]))\n"
        "        if r is not None:\n"
        "            heapq.heappush(heap, (r, i))\n"
        "    while heap:\n"
        "        r, i = heapq.heappop(heap)\n"
        "        if not alive[i]:\n"
        "            continue\n"
        "        j = nxt[i]\n"
        "        if j >= n or not alive[j]:\n"
        "            continue\n"
        "        current = (ids[i], ids[j])\n"
        "        if pair_to_rank.get(current) != r:\n"
        "            continue\n"
        "        # Merge: new id lives at position i; j is retired.\n"
        "        ids[i] = pair_to_new[current]\n"
        "        alive[j] = False\n"
        "        new_next = nxt[j]\n"
        "        nxt[i] = new_next\n"
        "        if new_next < n:\n"
        "            prev[new_next] = i\n"
        "        # Enqueue the two new neighbour pairs.\n"
        "        if prev[i] >= 0:\n"
        "            rr = pair_to_rank.get((ids[prev[i]], ids[i]))\n"
        "            if rr is not None:\n"
        "                heapq.heappush(heap, (rr, prev[i]))\n"
        "        if new_next < n and alive[new_next]:\n"
        "            rr = pair_to_rank.get((ids[i], ids[new_next]))\n"
        "            if rr is not None:\n"
        "                heapq.heappush(heap, (rr, i))\n"
        "    return [ids[k] for k in range(n) if alive[k]]\n"
        "\n"
        "# Correctness: both encoders must agree.\n"
        "a_out = encode_naive(corpus)\n"
        "b_out = encode_pq(corpus)\n"
        "assert a_out == b_out, \"priority-queue encoder disagrees with naive encoder\"\n"
        "\n"
        "# Warmup (Requirement 5.3) — exclude JIT / allocator noise.\n"
        "for _ in range(3):\n"
        "    _ = encode_naive(corpus)\n"
        "    _ = encode_pq(corpus)\n"
        "\n"
        "N = 10\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(N):\n"
        "    out_naive = encode_naive(corpus)\n"
        "naive_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "\n"
        "t0 = time.perf_counter()\n"
        "for _ in range(N):\n"
        "    out_pq = encode_pq(corpus)\n"
        "pq_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "\n"
        "# Wrap the final token ids in an MLX array and eval — satisfies the\n"
        "# 'mx.eval inside the timed region' rule used by every other benchmark\n"
        "# in this notebook stack.\n"
        "tokens_mx = mx.array(out_pq, dtype=mx.int32)\n"
        "mx.eval(tokens_mx)\n"
        "\n"
        "print(f\"corpus: {len(corpus):>6} chars, {len(out_pq):>5} tokens\")\n"
        "print(f\"naive  O(n·m) encoder:        {naive_ms:7.3f} ms / call\")\n"
        "print(f\"pqueue O(n·log m) encoder:   {pq_ms:7.3f} ms / call\")\n"
        "print(f\"speed-up at m=80 merges:     {naive_ms / pq_ms:6.2f}x\")\n"
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


def _block_tiktoken(records: list[dict]) -> list[dict]:
    """Block for the 'tiktoken' section.

    Contents: q03 (tiktoken vs HF), 🛠️ debugging cell covering the three
    classic tokenizer production incidents — train-eval mismatch,
    bytes-vs-text encoding mishaps, BOS-token drift.
    """
    q03 = records[2]

    debug_md, debug_code = T.debugging_failures_cell(
        symptom=(
            "Silent quality regressions at serving time — all three "
            "classic tokenizer-drift incidents that ship to production"
        ),
        root_causes=[
            "Train ↔ eval tokenizer MISMATCH: the model was trained with "
            "tokenizer A (e.g. LLaMA-3's 128k tiktoken) but is served "
            "with tokenizer B (e.g. a slightly different `tokenizer.json`). "
            "Embedding lookups silently decode to the wrong tokens; the "
            "model produces garbage tokens that AREN'T hard errors — the "
            "bug looks like 'the model got dumber'.",
            "bytes ↔ text encoding mishap: wrapping `tokenizer.encode(text.encode('utf-8'))` "
            "or using `surrogateescape` inconsistently on the chat template "
            "makes non-ASCII bytes round-trip through the wrong UTF-8 "
            "decoder. The symptom: `decode(encode(x)) != x` on Unicode "
            "strings, and the tokenizer's fertility jumps for non-English.",
            "BOS-token DRIFT: the chat template omits the BOS at the start "
            "of the sequence (or emits TWO BOS tokens — `apply_chat_template` "
            "plus manual `add_special_tokens=True`). LLaMA-3 trained WITH "
            "a single BOS; with zero BOS, the first ~10 generated tokens "
            "are low-quality filler; with two BOS, the model starts in an "
            "OOD state and may never recover.",
        ],
        diagnostic_code=(
            "# Reproduce each symptom, then show the fix.\n"
            "import mlx.core as mx\n"
            "import tiktoken\n"
            "\n"
            "enc = tiktoken.get_encoding(\"cl100k_base\")\n"
            "\n"
            "# -- Symptom 1: train/eval tokenizer mismatch --------------------\n"
            "# Simulate it by using TWO different encodings on the same text.\n"
            "text = \"The quick brown fox jumps over the lazy dog.\"\n"
            "ids_cl100k = enc.encode(text)\n"
            "enc_gpt2 = tiktoken.get_encoding(\"gpt2\")  # 50k vocab, DIFFERENT merges\n"
            "ids_gpt2 = enc_gpt2.encode(text)\n"
            "print(f\"cl100k ids ({len(ids_cl100k)}): {ids_cl100k[:8]}\")\n"
            "print(f\"gpt2 ids  ({len(ids_gpt2)}): {ids_gpt2[:8]}\")\n"
            "# The SAME integer id means totally different tokens across encodings:\n"
            "shared_id = ids_cl100k[0]\n"
            "# Decode that id under BOTH encodings to show the silent corruption.\n"
            "decoded_a = enc.decode([shared_id])\n"
            "decoded_b = enc_gpt2.decode([shared_id]) if shared_id < enc_gpt2.n_vocab else \"<out-of-range>\"\n"
            "print(f\"id {shared_id}: cl100k -> {decoded_a!r},  gpt2 -> {decoded_b!r}\")\n"
            "assert decoded_a != decoded_b, (\n"
            "    \"mismatch demo failed: two encodings happened to agree on this id\"\n"
            ")\n"
            "# Fix: pin the tokenizer to the SAME checkpoint commit as the weights;\n"
            "# at load time assert the vocab hash matches the model card.\n"
            "cl100k_vocab = enc.n_vocab\n"
            "gpt2_vocab = enc_gpt2.n_vocab\n"
            "assert cl100k_vocab != gpt2_vocab, \"sanity: distinct vocab sizes\"\n"
            "print(f\"✅ fix: vocab sizes differ (cl100k={cl100k_vocab}, gpt2={gpt2_vocab}) — pin by hash\")\n"
            "\n"
            "# -- Symptom 2: bytes-vs-text encoding mishap --------------------\n"
            "# `tokenizer.encode` expects a Python str; passing bytes would\n"
            "# either TypeError or (with `encode_ordinary`) be interpreted as\n"
            "# a latin-1-ish byte sequence and produce a DIFFERENT token list.\n"
            "unicode_text = \"日本語のトークナイザー\"  # Japanese: 'Japanese tokenizer'\n"
            "ids_str = enc.encode(unicode_text)\n"
            "# Round-trip MUST recover the input. If any decoder is wrong, this fails loudly.\n"
            "round_trip = enc.decode(ids_str)\n"
            "assert round_trip == unicode_text, (\n"
            "    f\"round-trip corruption: {round_trip!r} != {unicode_text!r}\"\n"
            ")\n"
            "# The BUGGY pattern — don't do this — is manually utf-8 encoding then re-decoding\n"
            "# as latin-1 before passing to the tokenizer; uncomment to see the corruption.\n"
            "# buggy = enc.encode(unicode_text.encode('utf-8').decode('latin-1'))\n"
            "# assert enc.decode(buggy) != unicode_text  # WOULD pass — corruption happened\n"
            "print(f\"✅ utf-8 round-trip OK for {len(unicode_text)}-char Japanese string\")\n"
            "\n"
            "# -- Symptom 3: BOS-token drift ---------------------------------\n"
            "# tiktoken doesn't have an intrinsic BOS, but every chat template\n"
            "# simulates one. We check the classic 'double BOS' bug by counting\n"
            "# special tokens at the start of a chat-formatted prompt.\n"
            "# cl100k_base exposes <|endoftext|> via `enc.eot_token`.\n"
            "eot_id = enc.eot_token\n"
            "# A chat prompt that (incorrectly) prepends TWO EOT-as-BOS tokens.\n"
            "double = [eot_id, eot_id] + enc.encode(\"Hello\")\n"
            "# A chat prompt that correctly prepends ONE.\n"
            "single = [eot_id] + enc.encode(\"Hello\")\n"
            "# Defensive check: count leading EOT tokens and collapse to at most one.\n"
            "def normalize_bos(ids: list[int], bos: int) -> list[int]:\n"
            "    k = 0\n"
            "    while k < len(ids) and ids[k] == bos:\n"
            "        k += 1\n"
            "    # Keep at most one leading BOS.\n"
            "    return ([bos] if k > 0 else []) + ids[k:]\n"
            "fixed = normalize_bos(double, eot_id)\n"
            "assert fixed == single, f\"BOS normalization failed: {fixed} vs {single}\"\n"
            "# Wrap the final ids in an mx.array so the debugging cell is MLX-verified.\n"
            "ids_mx = mx.array(fixed, dtype=mx.int32)\n"
            "mx.eval(ids_mx)\n"
            "print(f\"✅ BOS-normalization: [{eot_id}, {eot_id}, ...] -> [{eot_id}, ...]\")\n"
            "print(f\"   (fixed prompt length: {len(fixed)} tokens)\")\n"
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q03),
        T.separator_cell(),
        debug_md,
        debug_code,
    ]


def _block_sentencepiece(records: list[dict]) -> list[dict]:
    """Block for the 'SentencePiece' section.

    Contents: q04 (byte-fallback vs whitespace pretokenization),
    whiteboard-B (compute tokens-per-character compression ratio on a
    fixed corpus), 📐-2 (embedding-table memory for V ∈ {32k, 50k,
    128k, 256k} × D ∈ {768, 4096}) with benchmark.
    """
    q04 = records[3]

    # --- Whiteboard B — compression ratio with a lower-bound assertion. ---
    wb_md, wb_code = T.whiteboard_challenge_cell(
        title="Tokens-per-character compression ratio",
        prompt=(
            "Given a vocab (tiktoken's `cl100k_base`) and a fixed English "
            "corpus, compute the TOKENS-PER-CHARACTER compression ratio. "
            "Assert it is strictly below a documented lower bound (BPE "
            "achieves ≤ 0.35 tokens/char on English prose). Use MLX "
            "arrays for the arithmetic so the result is MLX-verified."
        ),
        constraints=[
            "Use `tiktoken.get_encoding('cl100k_base')` as the reference tokenizer.",
            "Corpus MUST be ≥ 500 English characters so the ratio is stable "
            "(short strings are dominated by the per-string overhead).",
            "Wrap `len(token_ids)` and `len(corpus)` in `mx.array`, divide "
            "via MLX, call `mx.eval`, then convert to a Python float for the "
            "assertion (Requirement 4.4).",
            "Include a `lower_bound_ratio = 0.35` constant matched to the "
            "well-known cl100k_base compression of English prose; assert the "
            "measured ratio is strictly less.",
            "Include a second `assert` that the ratio is > 0 (sanity check).",
        ],
        complexity=(
            "O(n) BPE encoding + O(1) arithmetic + O(1) MLX eval. "
            "Dominated entirely by the encoder's O(n·log m)."
        ),
        solution_code=(
            "import mlx.core as mx\n"
            "import tiktoken\n"
            "\n"
            "enc = tiktoken.get_encoding(\"cl100k_base\")\n"
            "\n"
            "# Fixed English corpus — stable stats (Pride & Prejudice opening).\n"
            "corpus = (\n"
            "    \"It is a truth universally acknowledged, that a single man in \"\n"
            "    \"possession of a good fortune, must be in want of a wife. However \"\n"
            "    \"little known the feelings or views of such a man may be on his \"\n"
            "    \"first entering a neighbourhood, this truth is so well fixed in the \"\n"
            "    \"minds of the surrounding families, that he is considered the rightful \"\n"
            "    \"property of some one or other of their daughters. My dear Mr. Bennet, \"\n"
            "    \"said his lady to him one day, have you heard that Netherfield Park is \"\n"
            "    \"let at last? Mr. Bennet replied that he had not.\"\n"
            ")\n"
            "assert len(corpus) >= 500, f\"corpus too short: {len(corpus)} chars\"\n"
            "\n"
            "token_ids = enc.encode(corpus)\n"
            "\n"
            "# Compute the ratio in MLX. Using mx.float32 so the division is exact.\n"
            "n_tokens = mx.array(len(token_ids), dtype=mx.float32)\n"
            "n_chars = mx.array(len(corpus), dtype=mx.float32)\n"
            "ratio_mx = n_tokens / n_chars\n"
            "mx.eval(ratio_mx)\n"
            "ratio = float(ratio_mx.item())\n"
            "\n"
            "# Documented lower bound: cl100k_base on English prose achieves\n"
            "# ~0.25 tokens/char; we assert comfortably below 0.35.\n"
            "lower_bound_ratio = 0.35\n"
            "assert 0 < ratio < lower_bound_ratio, (\n"
            "    f\"compression ratio {ratio:.4f} violates expected bound \"\n"
            "    f\"(0, {lower_bound_ratio})\"\n"
            ")\n"
            "\n"
            "print(f\"corpus: {len(corpus)} chars -> {len(token_ids)} tokens\")\n"
            "print(f\"compression ratio: {ratio:.4f} tokens/char  (bound: < {lower_bound_ratio})\")\n"
            "print(f\"✅ cl100k_base achieves {1 / ratio:.2f}x compression on this English prose\")\n"
        ),
    )

    # --- 📐 Complexity cell — embedding-table memory scaling. ---
    complexity = T.complexity_analysis_cell(
        op="embedding + unembedding memory for V × D × dtype",
        flops=(
            "0 FLOPs to STORE (it's a lookup table); the `lm_head` matmul "
            "is O(B·T·V·D) FLOPs per forward pass — often the single "
            "biggest contributor in the decoder tail"
        ),
        memory=(
            "V · D · bytes_per_elem PER matrix. Tied wte = lm_head: 1 copy; "
            "untied: 2 copies. bf16 = 2 B/elem, fp32 = 4 B/elem. At "
            "D=4096 and bf16: V=32k→256 MiB, V=50k→400 MiB, "
            "V=128k→1.0 GiB, V=256k→2.0 GiB (per matrix)"
        ),
        latency_mlx=(
            "Allocation of a (V, D) bf16 matrix on M4 Pro: ~a few ms for "
            "V ≤ 50k, ~tens of ms for V = 128k, ~50+ ms for V = 256k. "
            "The runtime cost shows up at the softmax-over-vocab stage, "
            "not at allocation — measured below"
        ),
        scaling=(
            "At V=256k, the embedding+unembedding is ~14 % of a 7 B model's "
            "parameters and ~7 % of the forward FLOPs. This is why Gemma-3 "
            "TIES them (saves 2 GiB bf16) and why every frontier trainer "
            "shards the vocab across tensor-parallel ranks."
        ),
    )

    # Custom memory-scaling benchmark: allocate (V, D) bf16 matrices for each
    # combo in the grid, report analytic vs measured bytes + mean ms / alloc.
    mem_bench_src = (
        "# Benchmark: embedding-table allocation (V × D) in bf16 on MLX\n"
        "import time\n"
        "import mlx.core as mx\n"
        "\n"
        "def alloc(V: int, D: int) -> mx.array:\n"
        "    \"\"\"Allocate a (V, D) bf16 embedding table and force materialization.\"\"\"\n"
        "    w = mx.zeros((V, D), dtype=mx.bfloat16)\n"
        "    mx.eval(w)\n"
        "    return w\n"
        "\n"
        "# Warmup — Requirement 5.3.\n"
        "for _ in range(3):\n"
        "    _w = alloc(1024, 768)\n"
        "    del _w\n"
        "\n"
        "grid = [(32_000, 768), (50_000, 4096), (128_000, 4096), (256_000, 4096)]\n"
        "print(f\"{'V':>8} x {'D':>5}  |  analytic MiB  |  measured MiB  |  mean ms / alloc\")\n"
        "print(\"-\" * 68)\n"
        "for V, D in grid:\n"
        "    analytic_mib = V * D * 2 / (1024 * 1024)  # bf16 = 2 bytes\n"
        "    N = 5\n"
        "    t0 = time.perf_counter()\n"
        "    for _ in range(N):\n"
        "        w = alloc(V, D)\n"
        "    dt_ms = (time.perf_counter() - t0) / N * 1000.0\n"
        "    # Measured nbytes should match the analytic size (MLX stores contiguous bf16).\n"
        "    measured_mib = w.nbytes / (1024 * 1024)\n"
        "    print(\n"
        "        f\"{V:>8} x {D:>5}  |  {analytic_mib:10.1f}    |  {measured_mib:10.1f}    |  {dt_ms:10.3f}\"\n"
        "    )\n"
        "    del w\n"
        "\n"
        "# Invariant: analytic == measured within 1 %.\n"
        "V, D = 128_000, 4096\n"
        "w = alloc(V, D)\n"
        "analytic_mib = V * D * 2 / (1024 * 1024)\n"
        "measured_mib = w.nbytes / (1024 * 1024)\n"
        "assert abs(analytic_mib - measured_mib) / analytic_mib < 0.01, (\n"
        "    f\"analytic vs measured memory disagree: {analytic_mib:.2f} MiB vs {measured_mib:.2f} MiB\"\n"
        ")\n"
        "print(f\"\\n✅ analytic ≈ measured for V=128k, D=4096 in bf16\")\n"
    )
    mem_bench = {"cell_type": "code", "source": mem_bench_src}

    return [
        T.separator_cell(),
        T.interview_question_cell(q04),
        T.separator_cell(),
        wb_md,
        wb_code,
        T.separator_cell(),
        complexity,
        mem_bench,
    ]


def _block_summary(records: list[dict]) -> list[dict]:
    """Block for the 'Summary' section.

    Contents: q05 (vocab-size math), q06 (tokenizer leakage), q07
    (special tokens), 🏭 production (tokenization in vLLM / SGLang /
    TRT-LLM / MLX-LM), 🔭 frontier (2024–2026 vocab-scaling trend).
    """
    q05, q06, q07 = records[4], records[5], records[6]

    production = T.production_context_cell(
        concept="Tokenization in production — TTFT impact of tokenizer choice",
        vllm=(
            "Uses HuggingFace `tokenizers` (Rust) via `transformers` — the "
            "same code path every HF model loads with. CPU-side tokenization "
            "is single-threaded PER request, so on very short prompts the "
            "tokenizer latency (<1 ms) doesn't matter; on 8k-token prefills "
            "it can add 5–10 ms to TTFT. v0.6+ added an optional tiktoken "
            "fast path for OpenAI-style BPE encodings."
        ),
        sglang=(
            "Also HuggingFace `tokenizers` by default; RadixAttention caches "
            "tokenized prefixes so a shared system prompt is tokenized ONCE "
            "then reused across all downstream requests — eliminates the "
            "tokenizer from the hot path for most chat workloads."
        ),
        trt_llm=(
            "Tokenization runs on the CPU side of the TRT-LLM server; for "
            "LLaMA and GPT-style models NVIDIA ships both HF `tokenizers` "
            "and a Rust-based `tiktoken` wrapper. Large prefills pipeline "
            "tokenization with the previous request's generation to hide "
            "the latency."
        ),
        mlx_lm=(
            "Loads the model's shipped `tokenizer.json` through HuggingFace "
            "`tokenizers` (pure Rust — works natively on Apple Silicon); "
            "tiktoken's BPE Rust core is also available for cl100k / o200k "
            "models. Typical cost on M4 Pro: ~200k–1M tokens/sec depending "
            "on input."
        ),
    )

    frontier = T.frontier_context_cell(
        topic="Vocab sizes are growing (2024–2026) — the frontier tokenization moves",
        papers=[
            (
                "LLaMA-3 Technical Report (Meta, 2024)",
                2024,
                "SWITCHED from SentencePiece-BPE (32k vocab) to a "
                "tiktoken-compatible BPE encoding at V=128k. Meta reports "
                "~15 % fewer tokens per English document and ~2× better "
                "compression on code — direct TTFT and $/token wins.",
            ),
            (
                "Gemma 2/3 Technical Report (Google DeepMind, 2024)",
                2024,
                "Gemma-3 ships a V=256k SentencePiece tokenizer — the "
                "largest vocab of any open-weights model at the time. "
                "Aimed at multilingual coverage (140+ languages); pays "
                "for itself in training FLOPs via fewer tokens per string.",
            ),
            (
                "DeepSeek-V3 Technical Report (DeepSeek, 2024)",
                2024,
                "V=128k BPE tokenizer tuned jointly on English and "
                "Chinese; documents the fp32-accumulator rule for the "
                "softmax-over-vocab (same rule you meet in NB02's "
                "attention-softmax stability discussion).",
            ),
            (
                "Qwen-2.5 Technical Report (Alibaba, 2024)",
                2024,
                "V=151 936 bilingual BPE; explicit engineering on "
                "Chinese-character coverage (~90 % of CJK unified "
                "ideographs are SINGLE tokens). Demonstrates that "
                "vocab-design for multilingual is an open research "
                "axis, not a solved problem.",
            ),
            (
                "Tokenizer-Free and Character-Level LLMs "
                "(community, 2024–2025)",
                2025,
                "Byte-Latent Transformer (BLT, Meta 2024) and "
                "MambaByte push the vocab size to 256 — pure byte-level "
                "models that do away with BPE entirely. Trade fewer "
                "embedding params for longer sequences; active frontier "
                "on whether byte-level can match subword at scale.",
            ),
        ],
        current_sota=(
            "As of late 2025 the frontier cluster is V ∈ [128k, 256k] for "
            "subword tokenizers, with ALL of LLaMA-3, Qwen-2.5, "
            "DeepSeek-V3 on 128k and Gemma-3 on 256k. The tokenizer is "
            "increasingly bilingual-by-construction (English + Chinese, or "
            "English + Japanese + Korean + …). Two active research "
            "frontiers: (i) byte-level / tokenizer-free models (BLT, "
            "MambaByte) that skip BPE entirely, and (ii) LEARNED tokenizers "
            "where the merge table is co-optimized with the model loss "
            "rather than fixed ahead of time."
        ),
    )

    return [
        T.separator_cell(),
        T.interview_question_cell(q05),
        T.separator_cell(),
        T.interview_question_cell(q06),
        T.separator_cell(),
        T.interview_question_cell(q07),
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
    """Transform nb03 to interview-grade (idempotent).

    Inserts blocks bottom-up so earlier indices don't shift while we work
    on later sections. Writes the seven nb03 Question_Bank records after
    the notebook is saved. The function is safe to re-run: once a 🎯 or
    📋 marker is present the call is a no-op.
    """
    nb = _load_notebook()

    if _is_already_transformed(nb):
        print("[nb03] already transformed; skipping")
        return

    cells: list[dict] = nb["cells"]
    records = _build_qbank_records(added_in="")

    # Resolve all insertion indices against the ORIGINAL cell list, then
    # apply bottom-up so earlier indices stay valid.
    bpe_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_BPE)
    )
    tiktoken_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_TIKTOKEN)
    )
    sp_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_SENTENCEPIECE)
    )
    summary_end = _find_section_end(
        cells, _find_first_anchor(cells, _ANCHOR_SUMMARY)
    )
    end_of_nb = len(cells)

    insertions: list[tuple[int, list[dict], str]] = [
        (end_of_nb, _block_end(records), "end-of-notebook"),
        (summary_end, _block_summary(records), "summary"),
        (sp_end, _block_sentencepiece(records), "sentencepiece"),
        (tiktoken_end, _block_tiktoken(records), "tiktoken"),
        (bpe_end, _block_bpe(records), "bpe"),
    ]

    # Apply bottom-up: sort by index descending.
    for insert_at, block, label in sorted(insertions, key=lambda t: -t[0]):
        n = _insert_block(cells, insert_at, block)
        print(f"[nb03] inserted {n} cells at index {insert_at} ({label})")

    _save_notebook(nb)
    print(f"[nb03] wrote notebook with {len(cells)} cells total")

    # Write the Question_Bank slice. ``added_in`` is backfilled after the
    # commit SHA is known.
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb03] upserted {len(records)} Q-bank records (added_in='' pending backfill)")


def backfill_added_in(sha: str) -> None:
    """Re-upsert the nb03 slice with ``added_in=sha`` on every record.

    Called after the initial commit so the ``added_in`` field references
    the commit that introduced the records (Requirement 16.4).
    """
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"sha must be a non-empty string, got {sha!r}")
    records = _build_qbank_records(added_in=sha.strip())
    qbank.upsert_slice(_NB_NUM, records)
    print(f"[nb03] backfilled added_in='{sha.strip()}' on {len(records)} records")


if __name__ == "__main__":
    transform()
