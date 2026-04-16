# Enhancement Plan: Deep-Dive LLM Learning Series
## For AI Lab Interview Readiness (April 2026)

## Current State Assessment

### What Exists (15 notebooks + scaffolding)

| # | Notebook | Depth | Quality | Needs |
|---|---------|-------|---------|-------|
| 00 | Environment & Apple Silicon | ⭐⭐⭐ | Good | Minor updates |
| 01 | MLX Fundamentals | ⭐⭐⭐ | Good | Minor updates |
| 02 | Math Foundations | ⭐⭐⭐ | Good | Add backprop math |
| 03 | Tokenization | ⭐⭐⭐ | Good | OK as-is |
| 04 | Embeddings & PE | ⭐⭐⭐ | Good | Add p-RoPE, ALiBi |
| 05 | Self-Attention | ⭐⭐⭐⭐ | Enhanced | Add linear attention comparison |
| 06 | Transformer Architecture | ⭐⭐ | Shallow | NEEDS DEEP REWORK |
| 07 | Building GPT | ⭐⭐ | Shallow | NEEDS DEEP REWORK |
| 08 | Training on Apple Silicon | ⭐⭐ | Shallow | NEEDS DEEP REWORK |
| 09 | Modern Architectures | ⭐⭐⭐ | Enhanced | Add DeepSeek, Phi |
| 10 | Metal Custom Kernels | ⭐⭐ | Shallow | OK for scope |
| 11 | Inference Optimization | ⭐⭐ | Shallow | NEEDS DEEP REWORK |
| 12 | Flash/Paged/Ring Attention | ⭐ | Very Shallow | NEEDS COMPLETE REWRITE |
| 13 | Serving Locally | ⭐⭐ | Shallow | Add benchmarks |
| 14 | Capstone Gemma 4 | ⭐⭐ | Shallow | NEEDS DEEP REWORK |

### Critical Gaps for 2026 AI Lab Interviews

#### MISSING ENTIRELY:
1. **Mixture of Experts (MoE)** — Mixtral, DeepSeek-V3, Gemma 4 MoE all use this
2. **State Space Models** — Mamba, Mamba-2, RWKV-6, Griffin (Google's hybrid)
3. **RLHF / DPO / GRPO** — How models are aligned post-training
4. **Scaling Laws** — Chinchilla, compute-optimal training, emergent abilities
5. **Reasoning & Chain-of-Thought** — o1-style reasoning, DeepSeek-R1, test-time compute
6. **Distributed Training** — Tensor/pipeline/data parallelism, FSDP concepts

#### NEEDS DEEP REWORK:
1. **Flash Attention** — Need full online softmax algorithm with math derivation
2. **Paged Attention** — Need actual block manager implementation
3. **Speculative Decoding** — Need full draft/target implementation
4. **Quantization** — Need GPTQ, AWQ, GGUF format details

---

## Proposed Enhanced Notebook Structure

### Phase 1: Enrich Existing Notebooks (Priority: HIGH)

#### Notebook 06: Transformer Architecture — DEEP REWORK
- Add: Complete backpropagation through a transformer block
- Add: Why pre-norm > post-norm (gradient flow analysis)
- Add: Why SwiGLU > GELU > ReLU (activation comparison with plots)
- Add: Dropout, weight initialization strategies
- Add: Parameter counting formulas for any config

#### Notebook 07: Building GPT — DEEP REWORK
- Add: Full training with tiny_shakespeare (download and use)
- Add: Learning rate warmup + cosine decay implementation
- Add: Gradient clipping implementation
- Add: Checkpoint saving/loading
- Add: Evaluation loop with perplexity
- Add: Generation quality samples at different training stages

#### Notebook 08: Training — DEEP REWORK
- Add: Full gradient accumulation implementation (not just explanation)
- Add: Mixed precision training loop (actual bfloat16 training)
- Add: Memory profiling during training with plots
- Add: OOM recovery implementation
- Add: NaN detection and recovery

#### Notebook 11: Inference Optimization — DEEP REWORK
- Add: Full KV-cache with prefill/decode phases
- Add: Actual quantization with perplexity comparison
- Add: Speculative decoding full implementation
- Add: Continuous batching concept

#### Notebook 12: Flash/Paged/Ring — COMPLETE REWRITE
- Add: Full online softmax algorithm derivation
- Add: Tiled attention implementation step by step
- Add: Memory analysis with actual numbers
- Add: Paged attention block manager
- Add: Ring attention simulation

### Phase 2: New Notebooks (Priority: CRITICAL)

#### NEW Notebook 15: Mixture of Experts (MoE)
- Router mechanisms: Top-K, expert choice, hash routing
- Load balancing loss
- Shared expert (Gemma 4, DeepSeek-V3)
- Auxiliary loss for balanced routing
- MoE vs Dense: when to use which
- Implement: Mixtral-style MoE block
- Implement: Gemma 4 MoE with shared expert
- Memory analysis: why MoE is efficient

#### NEW Notebook 16: Beyond Attention — State Space Models
- Why attention is O(n²) and why that matters
- Linear attention (Katharopoulos et al., 2020)
- S4 (Structured State Spaces, Gu et al., 2021)
- Mamba (Gu & Dao, 2023) — selective state spaces
- Mamba-2 (2024) — connection to attention
- RWKV-6 (2024) — linear RNN with attention-like quality
- Griffin (Google, 2024) — hybrid attention + recurrence
- Jamba (AI21, 2024) — hybrid Mamba + attention
- Implement: Simple SSM from scratch
- Implement: Mamba-style selective scan
- Compare: Attention vs SSM on sequence modeling tasks

#### NEW Notebook 17: Alignment — RLHF, DPO, GRPO
- Why alignment matters (base model → assistant)
- Supervised Fine-Tuning (SFT)
- RLHF: reward model + PPO (2022, InstructGPT)
- DPO: Direct Preference Optimization (2023) — simpler than RLHF
- GRPO: Group Relative Policy Optimization (2025, DeepSeek)
- Constitutional AI (Anthropic)
- Implement: Simple reward model
- Implement: DPO training loop
- Compare: RLHF vs DPO vs GRPO

#### NEW Notebook 18: Scaling Laws & Emergent Abilities
- Kaplan scaling laws (2020)
- Chinchilla scaling (2022) — compute-optimal training
- Emergent abilities debate
- Compute budgeting: how to choose model size vs data size
- Training compute estimation
- Implement: Scaling law prediction

#### NEW Notebook 19: Reasoning & Test-Time Compute
- Chain-of-Thought prompting (Wei et al., 2022)
- Self-consistency (Wang et al., 2022)
- Tree-of-Thought (Yao et al., 2023)
- o1-style reasoning (OpenAI, 2024) — test-time compute scaling
- DeepSeek-R1 (2025) — RL for reasoning
- Process reward models
- Implement: CoT prompting pipeline
- Implement: Simple MCTS for reasoning

### Phase 3: Enrich with Chronological Context (Priority: MEDIUM)

For EVERY major component, add a "History & Alternatives" section:

- **Attention**: Bahdanau (2014) → Luong → Transformer → Flash → GQA → Gemma 4
- **Normalization**: BatchNorm → LayerNorm → RMSNorm → DeepNorm
- **Activation**: ReLU → GELU → SiLU → SwiGLU → GeGLU
- **Position Encoding**: Sinusoidal → Learned → RoPE → ALiBi → p-RoPE
- **Architecture**: Encoder-Decoder → Decoder-only → MoE → Hybrid (Mamba+Attention)
- **Training**: SGD → Adam → AdamW → Lion → Schedule-Free
- **Alignment**: RLHF → DPO → KTO → GRPO

---

## Execution Priority

1. **IMMEDIATE**: Complete current enhancements (notebooks 05, 09) ✅ DONE
2. **HIGH**: Deep rework notebooks 06, 07, 08, 11, 12
3. **CRITICAL**: Create new notebooks 15 (MoE), 16 (SSMs), 17 (Alignment)
4. **IMPORTANT**: Create notebooks 18 (Scaling), 19 (Reasoning)
5. **MEDIUM**: Enrich all notebooks with chronological context

## Implementation Notes

- ALL code in MLX (no PyTorch)
- Every concept: math derivation → step-by-step code → visualization → comparison
- Memory budget: stay under 20GB during any notebook execution
- Each notebook should be self-contained but reference prerequisites
- Interview tips (🎯) at every major concept
