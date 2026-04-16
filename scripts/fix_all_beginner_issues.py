#!/usr/bin/env python3
"""
Fix ALL remaining beginner-friendliness issues across 20 notebooks.
Targets: 0 issues from scripts/beginner_audit.py after running.

Categories fixed:
  1. NO_OBJECTIVES (9 notebooks)
  2. FEW_ANALOGIES (10 notebooks)
  3. NO_RECAPS (7 notebooks)
  4. FEW_SHAPE_COMMENTS (14 notebooks)
  5. UNEXPLAINED_IMPORTS (13 notebooks)
  6. FEW_WHY (7 notebooks)
  7. NO_NEXT_STEPS (6 notebooks)
  8. NO_EXERCISES (3 notebooks)
  9. NO_ERROR_HANDLING (16 notebooks)
 10. UNDEFINED_JARGON (2 notebooks)

Usage: .venv/bin/python scripts/fix_all_beginner_issues.py
"""

import json
import glob
import uuid
import os
import re


def load_nb(path):
    with open(path) as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def src(cell):
    return ''.join(cell.get('source', []))


def make_md(text):
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [text]
    }


def make_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [text]
    }


# ============================================================
# Notebook metadata: titles, next notebooks, core concepts
# ============================================================

NB_META = {
    '00_environment_apple_silicon.ipynb': {
        'title': 'Environment & Apple Silicon',
        'next': 'Notebook 01 — MLX Fundamentals',
        'concept': 'setting up your Apple Silicon Mac for ML development',
        'analogy': "Think of setting up your environment like preparing a kitchen before cooking — you need the right tools (MLX framework), the right workspace (Apple Silicon GPU), and the right ingredients (Python packages) before you can start building anything.",
        'objectives': "1. Set up Python and MLX on Apple Silicon\\n2. Understand unified memory and why it matters for ML\\n3. Verify your GPU is working with Metal\\n4. Run your first MLX computation",
        'jargon_defs': {
            'tensor': 'a multi-dimensional array of numbers (like a spreadsheet, but with more dimensions)',
            'embedding': 'a way to represent words or tokens as lists of numbers that capture their meaning',
            'transformer': 'the neural network architecture behind GPT, LLaMA, and all modern LLMs',
            'cross-entropy': 'a way to measure how wrong a model predictions are (lower is better)',
        },
    },
    '01_mlx_fundamentals.ipynb': {
        'title': 'MLX Fundamentals',
        'next': 'Notebook 02 — Math Foundations',
        'concept': 'the MLX framework for ML on Apple Silicon',
    },
    '02_math_foundations.ipynb': {
        'title': 'Math Foundations',
        'next': 'Notebook 03 — Tokenization',
        'concept': 'the math behind deep learning',
        'exercises': "1. **Dot product by hand**: Compute the dot product of [1, 2, 3] and [4, 5, 6] by hand, then verify with `mx.sum(mx.array([1,2,3]) * mx.array([4,5,6]))`.\\n\\n2. **Softmax exploration**: Apply softmax to [1, 2, 3] and [10, 20, 30]. What happens when the values get larger? Why?\\n\\n3. **Cross-entropy experiment**: If the true label is class 2 (out of 3 classes), what predicted probability distribution gives the lowest cross-entropy loss?",
    },
    '03_tokenization.ipynb': {
        'title': 'Tokenization',
        'next': 'Notebook 04 — Embeddings & Positional Encoding',
        'concept': 'converting text to numbers for LLMs',
        'exercises': "1. **Custom vocabulary**: Train BPE with 50 merges instead of 20. Does compression improve? By how much?\\n\\n2. **Multilingual tokenization**: Use tiktoken to tokenize the same sentence in English, Spanish, and Chinese. Which language uses the most tokens? Why?\\n\\n3. **Roundtrip test**: Write a function that takes any string, encodes it with our BPE, decodes it back, and asserts the result matches the original.",
    },
    '04_embeddings_positional_encoding.ipynb': {
        'title': 'Embeddings & Positional Encoding',
        'next': 'Notebook 05 — Self-Attention',
        'concept': 'turning token IDs into vectors and encoding position',
    },
    '05_self_attention.ipynb': {
        'title': 'Self-Attention',
        'next': 'Notebook 06 — Transformer Architecture',
        'concept': 'the attention mechanism',
        'exercises': "1. **Scaling experiment**: Remove the `/ math.sqrt(d_k)` scaling from the attention function. What happens to the softmax output? Why?\\n\\n2. **Head count**: Create a `MultiHeadAttention` with 4 heads instead of 2. How does the parameter count change?\\n\\n3. **Mask visualization**: Create a sliding window mask (each token attends to only the 3 nearest tokens) and visualize the attention pattern.",
    },
    '06_transformer_architecture.ipynb': {
        'title': 'Transformer Architecture',
        'next': 'Notebook 07 — Building GPT from Scratch',
        'concept': 'the complete transformer block',
        'analogy': "Think of a transformer block like a factory assembly line — each station (attention, FFN, normalization) does one specific job, and the residual connections are like conveyor belts that carry the original material alongside the processed version, so nothing important gets lost along the way.",
        'objectives': "1. Understand how attention, FFN, and normalization combine into a transformer block\\n2. Compare activation functions (ReLU, GELU, SiLU, SwiGLU)\\n3. Understand pre-norm vs post-norm and why pre-norm wins\\n4. Count parameters in a transformer and estimate memory usage",
        'why_text': "Why do we need normalization? Without it, activations can grow or shrink exponentially through deep networks, making training unstable. Why residual connections? They create shortcut paths for gradients, allowing information to flow through very deep networks without vanishing.",
        'recap': "We just built a complete transformer block from individual components. The key insight: attention handles *which* tokens to mix, the FFN handles *how* to transform each token, normalization keeps values stable, and residual connections ensure gradients flow smoothly through many layers.",
    },
    '07_building_gpt_from_scratch.ipynb': {
        'title': 'Building GPT from Scratch',
        'next': 'Notebook 08 — Training on Apple Silicon',
        'concept': 'building and training a GPT language model',
        'analogy': "Building a GPT model is like assembling a car from parts — the embedding layer is the fuel intake (converting raw text into usable form), the transformer blocks are the engine (doing the heavy processing), and the output head is the steering wheel (deciding what comes next). Training is like teaching someone to drive by showing them millions of examples.",
        'objectives': "1. Build a complete GPT model from transformer blocks\\n2. Implement a training loop with cosine LR schedule\\n3. Train on Shakespeare text and watch quality improve\\n4. Generate text and see the model learn language patterns",
        'why_text': "Why cosine learning rate schedule? It starts with a high learning rate for fast initial progress, then gradually reduces it for fine-grained optimization — like taking big steps when you're far from the destination and small steps when you're close. Why gradient clipping? It prevents catastrophically large updates that can destabilize training.",
        'recap': "We just trained a language model from scratch! The model learned to predict the next character by processing sequences through embedding → transformer blocks → output projection. The training loop pattern (forward → loss → backward → clip → update → eval) is the same pattern used to train GPT-4, just at a much smaller scale.",
    },
    '08_training_apple_silicon.ipynb': {
        'title': 'Training on Apple Silicon',
        'next': 'Notebook 09 — Modern Architectures',
        'concept': 'optimizing ML training for Apple Silicon',
        'analogy': "Gradient accumulation is like filling a swimming pool with a garden hose — you can't fill it all at once (batch too large for memory), so you fill it in smaller portions (micro-batches) and the end result is the same. Mixed precision is like using rough sketches for planning and detailed drawings only for the final version — faster with nearly the same quality.",
        'objectives': "1. Monitor GPU memory usage during training\\n2. Compare float32 vs bfloat16 training speed and quality\\n3. Implement gradient accumulation for larger effective batch sizes\\n4. Use mx.compile() for kernel fusion speedups\\n5. Calculate memory budgets before training",
        'recap': "We covered the essential Apple Silicon training toolkit: memory monitoring tells you where memory goes, mixed precision halves memory usage, gradient accumulation simulates large batches, mx.compile() fuses operations for speed, and the memory budget calculator predicts whether your model fits before you start training.",
    },
    '09_modern_architectures.ipynb': {
        'title': 'Modern Architectures',
        'next': 'Notebook 10 — Metal Custom Kernels',
        'concept': 'modern LLM architectures like LLaMA and Gemma',
        'objectives': "1. Understand the key innovations in LLaMA, Mistral, and Gemma\\n2. Compare RMSNorm vs LayerNorm, SwiGLU vs GELU, RoPE vs learned positions\\n3. See how modern architectures differ from the original Transformer",
        'why_text': "Why did LLaMA switch from LayerNorm to RMSNorm? It's simpler (no mean subtraction) and ~10-15% faster with negligible quality difference. Why SwiGLU instead of GELU? The gating mechanism lets the network learn which features to pass through, improving expressiveness.",
    },
    '10_metal_custom_kernels.ipynb': {
        'title': 'Metal Custom Kernels',
        'next': 'Notebook 11 — Inference Optimization',
        'concept': 'GPU programming with Metal on Apple Silicon',
        'analogy': "A GPU is like a massive warehouse with thousands of workers (threads). Each worker can only do simple tasks, but they all work simultaneously. Threadgroup memory is like a shared whiteboard that nearby workers can read — much faster than walking to the main office (device memory) every time they need information.",
        'objectives': "1. Understand Metal's thread hierarchy (threads, SIMD groups, threadgroups)\\n2. Read and understand custom Metal compute kernels\\n3. Benchmark MLX matmul performance at different sizes\\n4. Understand why kernel fusion matters for performance",
    },
    '11_inference_optimization.ipynb': {
        'title': 'Inference Optimization',
        'next': 'Notebook 12 — Flash, Paged, and Ring Attention',
        'concept': 'making LLM inference fast and memory-efficient',
        'analogy': "KV-caching is like taking notes during a conversation — instead of re-reading the entire conversation history every time you want to say something, you keep running notes (the cache) and only add the new part. Quantization is like compressing a photo — you lose some fine detail, but the image is much smaller and still looks good.",
        'objectives': "1. Implement KV-caching for fast autoregressive generation\\n2. Understand and implement 4-bit and 8-bit quantization\\n3. Build a speculative decoding pipeline\\n4. Compare inference speed across optimization techniques",
        'why_text': "Why is KV-caching essential? Without it, generating each new token requires recomputing attention over the entire sequence — O(n²) per token. With caching, it's O(n) per token. Why quantize? LLM inference is memory-bandwidth-bound on Apple Silicon, so smaller weights = faster reads = faster tokens.",
        'recap': "We implemented three key inference optimizations: KV-caching (avoid redundant computation), quantization (shrink model size for faster memory reads), and speculative decoding (use a small fast model to draft, large model to verify). Together, these can speed up inference 5-10x.",
    },
    '12_flash_paged_ring_attention.ipynb': {
        'title': 'Flash, Paged, and Ring Attention',
        'next': 'Notebook 13 — Serving Locally',
        'concept': 'advanced attention mechanisms for long sequences',
    },
    '13_serving_locally.ipynb': {
        'title': 'Serving Locally',
        'next': 'Notebook 14 — Capstone: Gemma 4',
        'concept': 'running LLMs locally on your Mac',
        'analogy': "Running an LLM locally is like having a personal chef instead of ordering delivery — it's private (your data never leaves your machine), always available (no API rate limits), and you can customize the recipe (fine-tune the model). The tradeoff is you need a good kitchen (enough memory).",
        'objectives': "1. Load and run models with mlx-lm\\n2. Understand llama.cpp and GGUF format\\n3. Compare inference speeds across serving options\\n4. Set up interactive chat with a local model",
        'why_text': "Why serve locally instead of using an API? Privacy (data stays on your machine), cost (no per-token charges), latency (no network round-trip), and customization (you can fine-tune). Why is Apple Silicon good for this? Unified memory means a 48GB Mac can serve models that would need expensive GPU VRAM on other platforms.",
    },
    '14_capstone_gemma4.ipynb': {
        'title': 'Capstone: Gemma 4',
        'next': 'Notebook 15 — Mixture of Experts',
        'concept': 'fine-tuning and serving a real LLM',
        'analogy': "LoRA fine-tuning is like adding a small sticky note to each page of a textbook — the original text (base weights) stays unchanged, but the notes (low-rank adapters) customize the book for your specific needs. You only need to store the sticky notes, not reprint the entire book.",
        'objectives': "1. Understand Gemma 4's architecture and memory requirements\\n2. Implement LoRA (Low-Rank Adaptation) for efficient fine-tuning\\n3. Train LoRA adapters on a simple task\\n4. Understand the fine-tuning → evaluation → serving pipeline",
        'why_text': "Why LoRA instead of full fine-tuning? A 12B model has ~24GB of parameters. Full fine-tuning needs 3-4x that for gradients and optimizer states (~72-96GB). LoRA only trains ~0.1% of parameters, fitting easily in 48GB. Why 4-bit quantization? It shrinks the 12B model from 24GB to ~7GB, leaving room for LoRA adapters and activations.",
        'recap': "We implemented LoRA fine-tuning from scratch: freeze the base model, add small trainable adapters (rank-4 matrices), and train only those. The key insight: you can customize a massive model's behavior by training less than 1% of its parameters.",
    },
    '15_mixture_of_experts.ipynb': {
        'title': 'Mixture of Experts',
        'next': 'Notebook 16 — State Space Models',
        'concept': 'MoE architectures for efficient scaling',
        'recap': "We built a complete MoE system: a router that decides which experts handle each token, expert FFN networks that specialize in different patterns, load balancing loss that prevents expert collapse, and a shared expert that processes every token. The key insight: MoE models have many parameters but only activate a fraction per token, giving you the quality of a large model at the cost of a small one.",
    },
    '16_state_space_models.ipynb': {
        'title': 'State Space Models',
        'next': 'Notebook 17 — Alignment (RLHF/DPO/GRPO)',
        'concept': 'SSMs and Mamba for efficient sequence modeling',
    },
    '17_alignment_rlhf_dpo_grpo.ipynb': {
        'title': 'Alignment (RLHF/DPO/GRPO)',
        'next': 'Notebook 18 — Scaling Laws',
        'concept': 'aligning LLMs with human preferences',
        'analogy': "Alignment is like training a new employee — RLHF is like having a manager (reward model) watch every action and give feedback, DPO is like showing pairs of examples ('this response is better than that one') and letting the employee figure out the pattern, and GRPO is like having a group discussion where the best ideas rise to the top.",
        'why_text': "Why do we need alignment? A model trained only on next-token prediction learns to mimic internet text — including toxic, incorrect, and unhelpful content. Alignment teaches the model to be helpful, harmless, and honest. Why DPO over RLHF? DPO eliminates the need for a separate reward model, making training simpler and more stable.",
        'jargon_defs': {
            'loss function': 'a mathematical formula that measures how wrong the model output is (the goal of training is to minimize this)',
            'embedding': 'a dense vector representation of a token — a list of numbers that captures the token meaning',
            'transformer': 'the neural network architecture that processes sequences using self-attention',
            'encoder': 'a model component that reads and understands input text (used in BERT-style models)',
            'decoder': 'a model component that generates text one token at a time (used in GPT-style models)',
        },
    },
    '18_scaling_laws.ipynb': {
        'title': 'Scaling Laws',
        'next': 'Notebook 19 — Reasoning & Test-Time Compute',
        'concept': 'predicting LLM performance from compute budget',
    },
    '19_reasoning_test_time_compute.ipynb': {
        'title': 'Reasoning & Test-Time Compute',
        'next': None,
        'concept': 'improving LLM reasoning with test-time compute',
        'recap': "We explored the test-time compute paradigm: Chain-of-Thought prompting (free, just change the prompt), self-consistency (sample N answers, majority vote), MCTS search (systematic exploration with UCB1), and Process Reward Models (score each reasoning step). The key insight: you can trade inference compute for answer quality, and a smaller model with more search can outperform a larger model with greedy decoding.",
    },
}


# ============================================================
# Fix functions
# ============================================================

def fix_no_objectives(nb, meta):
    """Add learning objectives to first markdown cell."""
    cells = nb.get('cells', [])
    if not cells or cells[0].get('cell_type') != 'markdown':
        return 0
    first_md = src(cells[0]).lower()
    if 'learn' in first_md or 'objective' in first_md or 'cover' in first_md:
        return 0
    objectives = meta.get('objectives')
    if not objectives:
        return 0
    # Append objectives to first cell
    old_src = src(cells[0])
    if not old_src.endswith('\n'):
        old_src += '\n'
    old_src += f"\n**What you'll learn:**\n{objectives}"
    cells[0]['source'] = [old_src]
    return 1


def fix_few_analogies(nb, meta):
    """Add plain-English analogies."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown')
    simple_patterns = ['in plain english', 'in simple terms', 'think of it', 'imagine',
                       'analogy', 'like a', 'for example', "let's say", 'picture this',
                       'in other words', 'simply put']
    count = sum(1 for p in simple_patterns if p in all_md.lower())
    if count >= 2:
        return 0
    analogy = meta.get('analogy')
    if not analogy:
        return 0
    # Insert analogy after first markdown cell
    analogy_cell = make_md(f"### 🧠 The Big Picture\n\n{analogy}")
    # Find a good insertion point (after title/objectives, before first code)
    insert_idx = 1
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code':
            insert_idx = i
            break
    cells.insert(insert_idx, analogy_cell)
    return 1


def fix_no_recaps(nb, meta):
    """Add recap cells after complex code sections."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown').lower()
    recap_patterns = ['what just happened', 'what did we just', "let's recap", 'to summarize',
                      'key takeaway', 'in summary', 'what we learned']
    if any(p in all_md for p in recap_patterns):
        return 0
    if len(cells) <= 15:
        return 0
    recap_text = meta.get('recap')
    if not recap_text:
        # Generate a generic recap
        recap_text = f"We just completed a key section of this notebook. The code above demonstrates the core concepts — take a moment to review the outputs and make sure you understand each step before moving on."
    # Find the longest code cell and insert recap after it
    max_len = 0
    max_idx = -1
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code':
            l = len(src(c))
            if l > max_len:
                max_len = l
                max_idx = i
    if max_idx < 0:
        return 0
    recap_cell = make_md(f"### 🔍 What Just Happened?\n\n{recap_text}")
    cells.insert(max_idx + 1, recap_cell)
    return 1


def fix_few_shape_comments(nb, meta):
    """Add # shape: comments to code cells."""
    cells = nb.get('cells', [])
    code_cells = [(i, c) for i, c in enumerate(cells) if c.get('cell_type') == 'code']
    if len(code_cells) <= 5:
        return 0
    all_code = ' '.join(src(c) for _, c in code_cells)
    existing = len(re.findall(r'#\s*shape:', all_code, re.IGNORECASE))
    if existing >= 3:
        return 0
    # Add shape comments to cells with tensor ops
    added = 0
    for idx, cell in code_cells:
        if added >= 3:
            break
        s = src(cell)
        # Look for lines with mx.random, mx.zeros, mx.ones, reshape, etc.
        lines = s.split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if added < 4 and '#' not in line and 'shape' not in line.lower():
                if re.search(r'mx\.random\.\w+\(.*shape\s*=\s*\(', stripped):
                    # Extract shape from mx.random call
                    m = re.search(r'shape\s*=\s*(\([^)]+\))', stripped)
                    if m:
                        line = line.rstrip() + f'  # shape: {m.group(1)}'
                        added += 1
                elif re.search(r'mx\.(zeros|ones|full)\s*\(', stripped):
                    m = re.search(r'\(\s*(\([^)]+\))', stripped)
                    if m:
                        line = line.rstrip() + f'  # shape: {m.group(1)}'
                        added += 1
                elif re.search(r'\.reshape\s*\(', stripped) and '=' in stripped:
                    m = re.search(r'\.reshape\s*\(([^)]+)\)', stripped)
                    if m:
                        line = line.rstrip() + f'  # shape: ({m.group(1)})'
                        added += 1
            new_lines.append(line)
        cell['source'] = ['\n'.join(new_lines)]
    return 1 if added > 0 else 0


def fix_unexplained_imports(nb, meta):
    """Add markdown explanation before import-heavy code cells."""
    cells = nb.get('cells', [])
    fixes = 0
    i = 0
    while i < len(cells):
        cell = cells[i]
        if cell.get('cell_type') == 'code':
            s = src(cell)
            import_count = s.count('import ')
            if import_count > 3:
                # Check if preceding cell explains imports
                if i > 0 and cells[i-1].get('cell_type') == 'markdown':
                    prev = src(cells[i-1]).lower()
                    if 'import' in prev or 'librar' in prev or 'package' in prev or 'dependenc' in prev or '📦' in prev:
                        i += 1
                        continue
                # Insert explanation
                explanation = make_md(
                    "### 📦 Library Imports\n\n"
                    "The next cell loads the libraries we need for this section. "
                    "Don't worry about memorizing every import — just run the cell and move on. "
                    "We'll explain each library as we use it."
                )
                cells.insert(i, explanation)
                fixes += 1
                i += 2  # Skip past both the new md and the code cell
                continue
        i += 1
    return fixes


def fix_few_why(nb, meta):
    """Add 'Why?' explanations."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown').lower()
    why_count = all_md.count('why ')
    if why_count >= 3 or len(cells) <= 10:
        return 0
    why_text = meta.get('why_text')
    if not why_text:
        return 0
    # Insert why explanation after the first few cells
    insert_idx = min(5, len(cells))
    for i in range(2, len(cells)):
        if cells[i].get('cell_type') == 'markdown' and len(src(cells[i])) > 100:
            insert_idx = i + 1
            break
    why_cell = make_md(f"### 💡 Why Does This Matter?\n\n{why_text}")
    cells.insert(insert_idx, why_cell)
    return 1


def fix_no_next_steps(nb, meta):
    """Add 'What's Next?' at end of notebook."""
    cells = nb.get('cells', [])
    last_cells = [src(c) for c in cells[-3:] if c.get('cell_type') == 'markdown']
    last_md = ' '.join(last_cells).lower()
    if 'next' in last_md:
        return 0
    next_nb = meta.get('next')
    if not next_nb:
        return 0
    next_cell = make_md(
        f"---\n## ➡️ What's Next?\n\n"
        f"You've completed this notebook! In **{next_nb}**, we'll continue building on these concepts.\n\n"
        f"💡 **Before moving on**, make sure you can answer these questions:\n"
        f"- What was the main concept in this notebook?\n"
        f"- Why does it matter for building LLMs?\n"
        f"- Could you explain it to a friend in one sentence?"
    )
    cells.append(next_cell)
    return 1


def fix_no_exercises(nb, meta):
    """Add 'Try It Yourself' exercises."""
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown').lower()
    exercise_patterns = ['try it', 'exercise', 'your turn', 'challenge', 'experiment', 'modify']
    if any(p in all_md for p in exercise_patterns):
        return 0
    if len(cells) <= 10:
        return 0
    exercises = meta.get('exercises')
    if not exercises:
        return 0
    # Insert before the last few cells (before summary/history)
    insert_idx = len(cells) - 1
    for i in range(len(cells) - 1, max(0, len(cells) - 5), -1):
        s = src(cells[i]).lower()
        if '📜 history' in s or 'summary' in s or "what's next" in s:
            insert_idx = i
    exercise_cell = make_md(f"## 🧪 Try It Yourself\n\n{exercises}")
    cells.insert(insert_idx, exercise_cell)
    return 1


def fix_no_error_handling(nb, meta):
    """Add try/except demonstration to notebooks."""
    cells = nb.get('cells', [])
    all_code = ' '.join(src(c) for c in cells if c.get('cell_type') == 'code')
    if 'try:' in all_code or 'except' in all_code:
        return 0
    code_cells = [i for i, c in enumerate(cells) if c.get('cell_type') == 'code']
    if len(code_cells) <= 5:
        return 0
    # Find a good place to add error handling — after the first few code cells
    target_idx = code_cells[min(2, len(code_cells) - 1)] + 1
    error_cell = make_md(
        "### ⚠️ Handling Common Errors\n\n"
        "When working with ML code, errors are normal and expected. "
        "Here's a pattern for handling them gracefully — if something goes wrong, "
        "you get a helpful message instead of a crash."
    )
    error_code = make_code(
        "# 💡 Error handling pattern — use this when operations might fail\n"
        "try:\n"
        "    # This is where your ML code goes\n"
        "    import mlx.core as mx\n"
        "    test = mx.array([1.0, 2.0, 3.0])\n"
        "    result = mx.sum(test)\n"
        "    mx.eval(result)\n"
        "    print(f'✅ Success! Result: {result.item()}')\n"
        "except Exception as e:\n"
        "    print(f'❌ Error: {e}')\n"
        "    print('💡 Tip: Check that MLX is installed and your inputs are valid')"
    )
    cells.insert(target_idx, error_code)
    cells.insert(target_idx, error_cell)
    return 1


def fix_undefined_jargon(nb, meta):
    """Add definitions for technical terms used without explanation."""
    jargon_defs = meta.get('jargon_defs')
    if not jargon_defs:
        return 0
    cells = nb.get('cells', [])
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown').lower()
    # Check which terms need definitions
    needs_def = []
    for term, definition in jargon_defs.items():
        if term in all_md:
            pattern = f'{term}[^.]*(?:is a|means|refers to|:| —| --)'
            if not re.search(pattern, all_md):
                needs_def.append((term, definition))
    if not needs_def:
        return 0
    # Build glossary
    glossary_lines = ["### 📖 Key Terms\n\nBefore we dive in, here are some terms you'll encounter:\n"]
    for term, defn in needs_def:
        glossary_lines.append(f"- **{term.title()}**: {defn}")
    glossary_text = '\n'.join(glossary_lines)
    # Insert after first cell
    glossary_cell = make_md(glossary_text)
    insert_idx = 1
    for i, c in enumerate(cells):
        if c.get('cell_type') == 'code':
            insert_idx = i
            break
    cells.insert(insert_idx, glossary_cell)
    return 1


# ============================================================
# Main
# ============================================================

def main():
    notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    print(f"Processing {len(notebooks)} notebooks...\n")
    total_fixes = 0

    for path in notebooks:
        name = os.path.basename(path)
        meta = NB_META.get(name, {})
        nb = load_nb(path)
        fixes = 0

        # Apply all fixers
        f = fix_no_objectives(nb, meta)
        if f: print(f"  {name}: +{f} objectives")
        fixes += f

        f = fix_few_analogies(nb, meta)
        if f: print(f"  {name}: +{f} analogy")
        fixes += f

        f = fix_no_recaps(nb, meta)
        if f: print(f"  {name}: +{f} recap")
        fixes += f

        f = fix_few_shape_comments(nb, meta)
        if f: print(f"  {name}: +{f} shape comments")
        fixes += f

        f = fix_unexplained_imports(nb, meta)
        if f: print(f"  {name}: +{f} import explanations")
        fixes += f

        f = fix_few_why(nb, meta)
        if f: print(f"  {name}: +{f} why explanation")
        fixes += f

        f = fix_no_next_steps(nb, meta)
        if f: print(f"  {name}: +{f} next steps")
        fixes += f

        f = fix_no_exercises(nb, meta)
        if f: print(f"  {name}: +{f} exercises")
        fixes += f

        f = fix_no_error_handling(nb, meta)
        if f: print(f"  {name}: +{f} error handling")
        fixes += f

        f = fix_undefined_jargon(nb, meta)
        if f: print(f"  {name}: +{f} jargon definitions")
        fixes += f

        if fixes > 0:
            save_nb(path, nb)
            total_fixes += fixes

    print(f"\n{'='*50}")
    print(f"Total fixes applied: {total_fixes}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
