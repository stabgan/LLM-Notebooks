# Implementation Plan: LLM Deep-Dive Enhancement

## Overview

Transform the 15-notebook LLM learning series into a 20-notebook deep-dive curriculum. Implementation uses Python with MLX exclusively on Apple Silicon. All notebook creation and editing uses MCP Jupyter notebook tools (`mcp_jupyter_editor_ipynb_*`). Non-notebook files (`.py`, `.metal`) use `fsWrite`. Every concept follows: math derivation → step-by-step code → visualization → comparison. Peak memory stays under 20GB per notebook.

## Tasks

### Tier 1: New Notebooks (15–19)

- [x] 1. Create Notebook 15: Mixture of Experts
  - [x] 1.1 Create `15_mixture_of_experts.ipynb` with intro, math derivation of Top-K routing (softmax over gating logits, top-k selection), and MoE motivation cells
    - Use MCP Jupyter tools to create notebook and insert markdown + code cells
    - Include 💡🎯⚡⚠️ emoji markers throughout
    - _Requirements: 1.1, 12.3, 12.4_

  - [x] 1.2 Implement MoERouter with Top-K, Expert Choice, and Hash routing in MLX
    - Implement `MoEConfig` dataclass and `MoERouter(nn.Module)` with `route()` returning indices `[batch, seq, k]` and weights `[batch, seq, k]` summing to 1.0
    - Implement Expert Choice and Hash routing variants with side-by-side comparison cells
    - Add shape assertions after every tensor operation
    - _Requirements: 1.2, 1.3, 12.1, 12.6_

  - [x] 1.3 Write property test for router weight normalization
    - **Property 3: Router Weight Normalization** — routing weights sum to 1.0 (±1e-6) for random inputs
    - **Validates: Requirement 1.2**

  - [x] 1.4 Implement ExpertFFN and MoEBlock with load balancing loss and shared expert
    - Implement `ExpertFFN(nn.Module)` and `MoEBlock(nn.Module)` with forward pass routing tokens through `num_active` experts
    - Implement load balancing auxiliary loss: `num_experts × Σ(f_i × p_i)`
    - Implement shared expert (Gemma 4 / DeepSeek-V3 style) that processes all tokens
    - Add OOM recovery: reduce batch size or switch to sequential expert evaluation if memory exceeds 20GB
    - _Requirements: 1.4, 1.5, 1.6, 1.9, 13.1, 13.2_

  - [x] 1.5 Write property tests for MoE block
    - **Property 1: Shape Preservation** — MoE_Block output shape equals input shape `[batch, seq, d_model]`
    - **Property 4: Expert Count Invariant** — each token routed to exactly `num_active` experts
    - **Property 5: Load Balance Loss Formula** — loss equals `num_experts × Σ(f_i × p_i)` and is non-negative
    - **Validates: Requirements 1.4, 1.5, 1.9**

  - [x] 1.6 Add memory analysis and visualization cells comparing MoE vs dense models
    - Compute total params vs active params per token
    - Create bar chart visualization
    - _Requirements: 1.7_

  - [x] 1.7 Add 📜 History & Alternatives section tracing Dense → Switch Transformer (2021) → Mixtral (2023) → DeepSeek-V3 (2025) → Gemma 4 (2025)
    - Chronological timeline with year, team, key contribution
    - _Requirements: 1.8, 11.1, 11.2, 11.5_

- [x] 2. Create Notebook 16: State Space Models
  - [x] 2.1 Create `16_state_space_models.ipynb` with intro explaining O(n²) attention limitation and SSM motivation
    - Math derivation of continuous-time SSM: dx/dt = Ax + Bu, y = Cx
    - Use MCP Jupyter tools for all notebook operations
    - _Requirements: 2.1, 12.3_

  - [x] 2.2 Implement SimpleSSM with Zero-Order Hold discretization in MLX
    - Implement `SSMConfig` dataclass and `SimpleSSM` with `discretize()` producing A_bar = exp(Δ×A) and B_bar = Δ×B
    - Implement sequential scan recurrence: h[t] = A_bar × h[t-1] + B_bar × x[t], y[t] = C[t] @ h[t]
    - Add shape assertions and numerical stability checks
    - _Requirements: 2.2, 12.1, 12.6_

  - [x] 2.3 Write property tests for SSM discretization and causality
    - **Property 6: SSM Discretization Correctness** — A_bar = exp(delta × A), B_bar = delta × B for valid inputs
    - **Property 7: SSM Causality** — output at position t depends only on inputs 0..t
    - **Validates: Requirements 2.2, 2.4**

  - [x] 2.4 Implement SelectiveSSM (Mamba-style) with input-dependent Δ, B, C parameters
    - Implement `SelectiveSSM(nn.Module)` where Δ, B, C are projected from input
    - Implement both sequential and parallel scan, verify equivalence within 1e-5
    - _Requirements: 2.3, 2.5_

  - [x] 2.5 Write property tests for scan equivalence and state boundedness
    - **Property 8: Sequential vs Parallel Scan Equivalence** — identical outputs within 1e-5
    - **Property 9: SSM State Boundedness** — hidden state norm remains bounded for stable A
    - **Validates: Requirements 2.5, 2.6**

  - [x] 2.6 Implement full MambaBlock with input projection, conv1d, selective SSM, SiLU gating, and output projection
    - Implement `MambaBlock(nn.Module)` following Mamba architecture
    - Verify memory usage is O(batch × d_inner × d_state), independent of sequence length
    - _Requirements: 2.7, 2.6_

  - [x] 2.7 Add attention vs SSM comparison cells with benchmark plots (complexity, memory, quality)
    - Side-by-side benchmarks at various sequence lengths
    - Visualization of memory scaling: O(n²) vs O(n)
    - _Requirements: 2.8_

  - [x] 2.8 Add 📜 History & Alternatives: Linear Attention → S4 (2021) → Mamba (2023) → Mamba-2 (2024) → RWKV-6 (2024) → Griffin (2024) → Jamba (2024)
    - _Requirements: 2.9, 11.1, 11.2, 11.6_

- [x] 3. Create Notebook 17: Alignment (RLHF / DPO / GRPO)
  - [x] 3.1 Create `17_alignment_rlhf_dpo_grpo.ipynb` with intro and math derivations for SFT → RLHF → DPO → KTO → GRPO progression
    - Derive each transition mathematically: RLHF objective, DPO reparameterization, GRPO group normalization
    - _Requirements: 3.1, 12.3_

  - [x] 3.2 Implement RewardModel in MLX with linear head on base model's last hidden state
    - Implement `RewardModelConfig`, `RewardModel(nn.Module)` mapping (prompt, response) → scalar reward
    - _Requirements: 3.2_

  - [x] 3.3 Implement DPOTrainer with log-prob computation, DPO loss, and training loop
    - Implement `DPOConfig`, `DPOTrainer` with `compute_log_probs()`, `dpo_loss()`, `train_step()`
    - Loss = -log(σ(β × (log_ratio_chosen - log_ratio_rejected)))
    - Keep reference model frozen (no gradient updates)
    - Add NaN/divergence recovery: reload checkpoint, reduce LR by 10x
    - _Requirements: 3.3, 3.4, 3.5, 13.4_

  - [x] 3.4 Write property tests for DPO
    - **Property 10: DPO Loss Correctness** — loss matches formula and is non-negative
    - **Property 11: Reference Model Frozen** — reference params unchanged after training steps
    - **Validates: Requirements 3.3, 3.4, 3.5**

  - [x] 3.5 Implement GRPOTrainer with group sampling, reward normalization, and policy gradient
    - Implement `GRPOTrainer` with `sample_group()`, `compute_group_rewards()`, `grpo_loss()`
    - Group size ≥ 2, normalize rewards to mean ≈ 0, std ≈ 1
    - _Requirements: 3.6_

  - [x] 3.6 Write property test for GRPO reward normalization
    - **Property 12: GRPO Reward Normalization** — normalized rewards have mean ≈ 0, std ≈ 1
    - **Validates: Requirement 3.6**

  - [x] 3.7 Add comparison cells: RLHF vs DPO vs GRPO on complexity, data requirements, stability
    - Include Constitutional AI (Anthropic) concept explanation
    - _Requirements: 3.7, 3.8_

  - [x] 3.8 Add 📜 History & Alternatives: RLHF (2022) → DPO (2023) → KTO (2024) → GRPO (2025) → Constitutional AI
    - _Requirements: 3.9, 11.1, 11.2, 11.7_

- [x] 4. Checkpoint — Verify Tier 1 notebooks 15-17
  - Ensure all cells execute without error, memory stays under 20GB
  - Verify shape assertions pass, no NaN/Inf in outputs
  - Ask the user if questions arise.

- [x] 5. Create Notebook 18: Scaling Laws & Compute-Optimal Training
  - [x] 5.1 Create `18_scaling_laws.ipynb` with intro and math derivation of Chinchilla scaling law L(N,D) = A/N^α + B/D^β + E
    - Derive power-law relationships, explain calibrated constants
    - _Requirements: 4.1, 12.3_

  - [x] 5.2 Implement ScalingLawPredictor with loss prediction and optimal allocation
    - Implement `ScalingLawParams`, `ComputeBudget`, `ScalingLawPredictor` with `predict_loss()`, `compute_optimal_allocation()`, `estimate_training_flops()`
    - Optimal allocation: 6 × optimal_N × optimal_D within 10% of C
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [x] 5.3 Write property tests for scaling laws
    - **Property 13: Scaling Law Monotonicity** — loss decreases as N or D increases
    - **Property 14: Compute Budget Conservation** — |6 × N × D - C| / C ≤ 0.10
    - **Property 28: Scaling Law Formula** — L(N,D) = A/N^α + B/D^β + E
    - **Validates: Requirements 4.1, 4.2, 4.3**

  - [x] 5.4 Add log-log visualization cells and compute budget calculator
    - Log-log plots: loss vs model size, loss vs data size
    - Interactive compute budget calculator: input FLOPs → optimal N, D, estimated loss
    - Compare Kaplan (2020) vs Chinchilla (2022) predictions
    - _Requirements: 4.4, 4.5, 4.7_

  - [x] 5.5 Add emergent abilities debate discussion and 📜 History section
    - Discuss supporting and opposing evidence for emergent abilities
    - History: Kaplan Scaling Laws (2020) → Chinchilla (2022) → Emergent Abilities Debate
    - _Requirements: 4.6, 4.8, 11.1, 11.2_

- [x] 6. Create Notebook 19: Reasoning & Test-Time Compute
  - [x] 6.1 Create `19_reasoning_test_time_compute.ipynb` with intro on reasoning and test-time compute scaling
    - Motivate why test-time compute matters, overview of approaches
    - _Requirements: 12.3_

  - [x] 6.2 Implement CoT prompting pipeline with self-consistency (majority voting)
    - Implement `ReasoningConfig`, `CoTPromptPipeline` with `generate_with_cot()` and `self_consistency()`
    - Self-consistency with N ≥ 3 samples, majority voting for final answer
    - _Requirements: 5.1, 5.2_

  - [x] 6.3 Implement MCTSReasoner with selection (UCB1), expansion, evaluation, backpropagation
    - Implement `ReasoningNode`, `MCTSReasoner` with `expand()`, `evaluate()`, `search()`
    - Implement UCB1 formula for exploration-exploitation balance
    - _Requirements: 5.3, 5.4_

  - [x] 6.4 Implement ProcessRewardModel scoring individual reasoning steps in [0, 1]
    - Implement `ProcessRewardModel` with `score_step()` and `score_trajectory()`
    - Explain process vs outcome reward models
    - _Requirements: 5.5, 5.6_

  - [x] 6.5 Write property test for process reward model bounds
    - **Property 15: Process Reward Model Bounds** — scores in [0, 1] for any input
    - **Validates: Requirement 5.5**

  - [x] 6.6 Add 📜 History & Alternatives: CoT (2022) → Self-Consistency (2022) → ToT (2023) → o1 (2024) → DeepSeek-R1 (2025)
    - _Requirements: 5.7, 11.1, 11.2_

- [x] 7. Checkpoint — Verify all Tier 1 notebooks (15-19) complete
  - Run all 5 new notebooks end-to-end, verify no errors
  - Ensure all tests pass, ask the user if questions arise.

### Tier 2: Deep Reworks

- [x] 8. Deep rework Notebook 06: Transformer Architecture
  - [x] 8.1 Add backpropagation walkthrough cells showing gradient flow through attention, FFN, residual connections, and normalization
    - Use MCP Jupyter tools to insert new cells into existing `06_transformer_architecture.ipynb`
    - Step-by-step gradient derivation with code verification
    - _Requirements: 6.1, 12.3_

  - [x] 8.2 Implement ActivationComparison: compute and plot ReLU, GELU, SiLU, SwiGLU, GeGLU on same input
    - Side-by-side plots with derivative visualizations
    - _Requirements: 6.2_

  - [x] 8.3 Implement NormalizationComparison: compute and plot LayerNorm, RMSNorm, DeepNorm on same input
    - Include gradient flow analysis: pre-norm vs post-norm showing pre-norm has lower CV of gradient norms
    - _Requirements: 6.3, 6.4_

  - [x] 8.4 Write property test for gradient stability
    - **Property 16: Pre-Norm Gradient Stability** — pre-norm has lower CV of gradient norms than post-norm for depth ≥ 4
    - **Validates: Requirement 6.4**

  - [x] 8.5 Implement ParameterCounter with per-component breakdown and memory estimation
    - Count: attention, FFN, normalization, embedding, total
    - Memory estimate: total params × bytes-per-element for any dtype
    - _Requirements: 6.5, 6.6_

  - [x] 8.6 Write property test for parameter counting consistency
    - **Property 17: Parameter Counting Consistency** — sum of components equals total, memory = params × bytes
    - **Validates: Requirements 6.5, 6.6**

  - [x] 8.7 Add weight initialization strategies (Xavier, Kaiming) and 📜 History section for activations and normalizations
    - History: ReLU → GELU → SiLU → SwiGLU → GeGLU; BatchNorm → LayerNorm → RMSNorm → DeepNorm
    - _Requirements: 6.7, 6.8, 11.1, 11.2_

- [x] 9. Deep rework Notebook 07: Building GPT
  - [x] 9.1 Add dataset download and preparation cells for tiny_shakespeare with train/val split
    - Use MCP Jupyter tools to insert into existing `07_building_gpt_from_scratch.ipynb`
    - _Requirements: 7.1_

  - [x] 9.2 Implement full training pipeline with cosine LR schedule + linear warmup and gradient clipping
    - lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × step / max_steps)) after warmup
    - Gradient clipping to prevent exploding gradients
    - _Requirements: 7.2, 7.3_

  - [x] 9.3 Write property tests for LR schedule and gradient clipping
    - **Property 18: Cosine LR Schedule** — LR matches formula for any step after warmup
    - **Property 19: Gradient Clipping Bound** — global gradient norm ≤ clip threshold after clipping
    - **Validates: Requirements 7.2, 7.3**

  - [x] 9.4 Implement CheckpointManager with save/load and NaN recovery
    - Save model weights, optimizer state, training step to disk
    - Load and resume training producing identical results
    - NaN/Inf detection: reload last checkpoint, reduce LR
    - _Requirements: 7.4, 7.5, 7.8_

  - [x] 9.5 Write property test for checkpoint determinism
    - **Property 20: Checkpoint Determinism** — save/load/resume produces identical results to uninterrupted training
    - **Validates: Requirements 7.4, 7.5**

  - [x] 9.6 Add evaluation loop (validation loss, perplexity) and generation samples at steps 0, 100, 500, 1000
    - Show quality progression from random to coherent text
    - _Requirements: 7.6, 7.7_

  - [x] 9.7 Add 📜 History & Alternatives: GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) → GPT-4 evolution
    - _Requirements: 7.9, 11.1, 11.2_

- [x] 10. Deep rework Notebook 08: Training on Apple Silicon
  - [x] 10.1 Implement working GradientAccumulator with actual micro-batch accumulation loop
    - Use MCP Jupyter tools to insert into existing `08_training_apple_silicon.ipynb`
    - batch_size B/K with accum_steps K produces equivalent gradients to batch_size B
    - _Requirements: 8.1_

  - [x] 10.2 Write property test for gradient accumulation equivalence
    - **Property 21: Gradient Accumulation Equivalence** — accumulated gradients match large-batch gradients
    - **Validates: Requirement 8.1**

  - [x] 10.3 Implement MixedPrecisionTrainer comparing float32 vs bfloat16 on speed, memory, loss stability
    - Actual bfloat16 training step with benchmarks
    - _Requirements: 8.2_

  - [x] 10.4 Implement MemoryProfiler with per-phase tracking and timeline plot
    - Record memory at each training step phase
    - Produce matplotlib timeline visualization
    - _Requirements: 8.3_

  - [x] 10.5 Implement MemoryBudgetCalculator with stacked bar chart (params, gradients, optimizer, activations, KV-cache)
    - For any TransformerConfig, batch size, sequence length
    - _Requirements: 8.4_

  - [x] 10.6 Write property test for memory budget calculator consistency
    - **Property 22: Memory Budget Calculator Consistency** — sum of components equals total
    - **Validates: Requirement 8.4**

  - [x] 10.7 Add OOM recovery (auto batch size reduction) and mx.compile() benchmarks
    - Automatic batch size reduction on OOM without data loss
    - mx.compile() speedup comparison on actual training steps
    - _Requirements: 8.5, 8.6_

  - [x] 10.8 Add 📜 History & Alternatives: SGD → Adam → AdamW → Lion → Schedule-Free
    - _Requirements: 8.7, 11.1, 11.2_

- [x] 11. Checkpoint — Verify Tier 2 reworks for notebooks 06, 07, 08
  - Ensure all reworked cells integrate with existing notebook content
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Deep rework Notebook 11: Inference Optimization
  - [x] 12.1 Implement full KVCacheManager with prefill and decode phases in MLX
    - Use MCP Jupyter tools to insert into existing `11_inference_optimization.ipynb`
    - `prefill()` processes full prompt, `decode_step()` generates one token using cached KV
    - Verify cached generation produces identical logits to full recomputation
    - _Requirements: 9.1, 9.2_

  - [x] 12.2 Write property test for KV-cache equivalence
    - **Property 23: KV-Cache Equivalence** — cached logits identical to full recomputation
    - **Validates: Requirement 9.2**

  - [x] 12.3 Implement Quantizer with 4-bit and 8-bit quantization, configurable group size, and perplexity comparison
    - Implement `quantize_4bit()`, `quantize_8bit()`, `dequantize()`
    - Measure and compare perplexity: original vs 8-bit vs 4-bit
    - _Requirements: 9.3, 9.4, 9.5_

  - [x] 12.4 Write property test for quantization error bound
    - **Property 24: Quantization Error Bound** — max error per group ≤ (max_val - min_val) / (2^N - 1)
    - **Validates: Requirement 9.4**

  - [x] 12.5 Implement SpeculativeDecoder with draft/verify pipeline
    - Small draft model generates candidates, large target model verifies
    - Accepted tokens identical to target-only generation
    - Report acceptance rate
    - _Requirements: 9.6, 9.7_

  - [x] 12.6 Write property test for speculative decoding correctness
    - **Property 25: Speculative Decoding Correctness** — accepted tokens match target-only generation
    - **Validates: Requirement 9.7**

  - [x] 12.7 Add GPTQ, AWQ, GGUF format explanations and 📜 History: FP32 → FP16 → INT8 → INT4 → GPTQ → AWQ → GGUF
    - _Requirements: 9.8, 9.9, 11.1, 11.2_

- [x] 13. Deep rework Notebook 12: Flash, Paged, and Ring Attention (complete rewrite)
  - [x] 13.1 Rewrite `12_flash_paged_ring_attention.ipynb` with online softmax derivation and implementation
    - Use MCP Jupyter tools to replace/rewrite cells in existing notebook
    - Full math derivation: running max, running sum, incremental softmax
    - Implement `OnlineSoftmax` in MLX, verify matches standard softmax within 1e-5
    - _Requirements: 10.1, 10.2_

  - [x] 13.2 Write property test for online softmax equivalence
    - **Property 26: Online Softmax Equivalence** — matches standard softmax within 1e-5
    - **Validates: Requirement 10.2**

  - [x] 13.3 Implement TiledFlashAttention processing Q, K, V in blocks without materializing O(n²) matrix
    - Step-by-step tiled attention with online softmax updates
    - Verify output matches standard attention softmax(QK^T/√d)V within 1e-5
    - Include memory analysis: O(n × d + block_size²) vs O(n² + n × d)
    - _Requirements: 10.3, 10.4, 10.5_

  - [x] 13.4 Write property test for flash attention equivalence
    - **Property 27: Flash Attention Equivalence** — matches standard attention within 1e-5
    - **Validates: Requirement 10.4**

  - [x] 13.5 Implement PagedAttention block manager: allocate, free, append, read KV blocks
    - Fixed-size block allocation and management
    - _Requirements: 10.6_

  - [x] 13.6 Implement RingAttention simulation across virtual devices
    - Simulate passing KV blocks around a ring of devices
    - _Requirements: 10.7_

  - [x] 13.7 Add benchmark cells: standard vs flash attention at seq lengths 512, 1024, 2048, 4096, 8192, 16384
    - Timing and memory comparison plots
    - _Requirements: 10.8_

  - [x] 13.8 Add 📜 History: Flash Attention 1 (2022) → Flash Attention 2 (2023) → FA3, Paged Attention (2023), Ring Attention (2024)
    - _Requirements: 10.9, 11.1, 11.2_

- [x] 14. Checkpoint — Verify all Tier 2 reworks complete (06, 07, 08, 11, 12)
  - Run all reworked notebooks end-to-end
  - Ensure all tests pass, ask the user if questions arise.

### Tier 3: Chronological Enrichment

- [-] 15. Add 📜 History & Alternatives sections to remaining notebooks
  - [x] 15.1 Add history section to Notebook 00 (Environment & Apple Silicon)
    - Cover Apple Silicon evolution: M1 (2020) → M1 Pro/Max (2021) → M2 (2022) → M3 (2023) → M4 (2024)
    - Use MCP Jupyter tools to append cells
    - _Requirements: 11.1, 11.2_

  - [x] 15.2 Add history section to Notebook 01 (MLX Fundamentals)
    - Cover ML frameworks: Theano → TensorFlow → PyTorch → JAX → MLX (2023)
    - _Requirements: 11.1, 11.2_

  - [x] 15.3 Add history section to Notebook 02 (Math Foundations)
    - Cover key mathematical developments relevant to deep learning
    - _Requirements: 11.1, 11.2_

  - [x] 15.4 Add history section to Notebook 03 (Tokenization)
    - Cover: Word-level → BPE (2016) → WordPiece → Unigram → SentencePiece → tiktoken
    - _Requirements: 11.1, 11.2_

  - [x] 15.5 Add history section to Notebook 04 (Embeddings & Positional Encoding)
    - Cover: Sinusoidal → Learned → RoPE (2021) → ALiBi (2021) → p-RoPE (2024)
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 15.6 Add history section to Notebook 05 (Self-Attention)
    - Cover: Bahdanau (2014) → Luong (2015) → Transformer (2017) → Flash Attention (2022) → GQA → MQA
    - _Requirements: 11.1, 11.2, 11.4_

  - [~] 15.7 Add history section to Notebook 09 (Modern Architectures)
    - Cover architecture evolution: Transformer → GPT → BERT → T5 → LLaMA → Mistral → Gemma
    - _Requirements: 11.1, 11.2_

  - [~] 15.8 Add history section to Notebook 10 (Metal Custom Kernels)
    - Cover GPU compute evolution: CUDA → OpenCL → Metal → Metal 3 → custom kernel ecosystem
    - _Requirements: 11.1, 11.2_

  - [~] 15.9 Add history section to Notebook 13 (Serving Locally)
    - Cover local inference: llama.cpp (2023) → MLX (2023) → Ollama → LM Studio → vLLM
    - _Requirements: 11.1, 11.2_

  - [~] 15.10 Add history section to Notebook 14 (Capstone Gemma 4)
    - Cover Gemma evolution: Gemma 1 → Gemma 2 → Gemma 3 → Gemma 4 and Google's open model strategy
    - _Requirements: 11.1, 11.2_

- [~] 16. Final checkpoint — Full series validation
  - Verify all 20 notebooks have 📜 History & Alternatives sections
  - Verify consistent emoji markers (💡⚡🎯⚠️) across all notebooks
  - Verify MLX-only code (no PyTorch/TensorFlow)
  - Verify peak memory under 20GB per notebook
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 11.1, 12.1, 12.2, 12.3, 12.4_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All notebook operations MUST use MCP Jupyter tools (`mcp_jupyter_editor_ipynb_*`) — never `fsWrite` for `.ipynb` files
- Non-notebook files (`.py`, `.metal`) use `fsWrite`
- The venv at `.venv/` has Python 3.13.13 — use it for all test execution
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation between tiers
- Property tests validate the 28 correctness properties from the design document
- All code uses MLX exclusively — no PyTorch or TensorFlow
