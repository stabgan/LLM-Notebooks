# Requirements Document

## Introduction

This document defines the requirements for enhancing the LLM learning series from 15 notebooks to 20 notebooks targeting AI lab interview readiness. The enhancement adds five new notebooks covering Mixture of Experts, State Space Models, alignment techniques, scaling laws, and reasoning. It deeply reworks five existing notebooks (06, 07, 08, 11, 12) to interview-grade depth. It enriches all 20 notebooks with chronological "History & Alternatives" sections. All implementations use MLX on Apple Silicon with peak memory under 20GB and follow a consistent pedagogical mentor/coach style.

## Glossary

- **Notebook**: A Jupyter notebook (.ipynb) file in the learning series
- **MLX**: Apple's machine learning framework optimized for Apple Silicon unified memory
- **MoE_Block**: A Mixture of Experts layer that routes each token to a subset of specialized expert FFNs
- **Router**: A gating mechanism that computes routing weights assigning tokens to experts
- **Expert_FFN**: A feed-forward neural network serving as one expert within an MoE layer
- **SSM**: State Space Model — a sequence model that uses linear recurrence as an alternative to attention
- **Selective_Scan**: Mamba's input-dependent SSM computation where parameters Δ, B, C depend on the input
- **DPO_Trainer**: A training component implementing Direct Preference Optimization on preference pairs
- **GRPO_Trainer**: A training component implementing Group Relative Policy Optimization using group-normalized rewards
- **Reward_Model**: A neural network that maps (prompt, response) pairs to scalar quality scores
- **Scaling_Law_Predictor**: A component that predicts cross-entropy loss from model size N and data size D using power-law formulas
- **MCTS_Reasoner**: A Monte Carlo Tree Search component that explores reasoning trajectories
- **Process_Reward_Model**: A model that scores intermediate reasoning steps rather than final answers
- **Flash_Attention**: An IO-aware tiled attention algorithm that computes exact attention without materializing the O(n²) attention matrix in HBM
- **Online_Softmax**: An algorithm that computes softmax incrementally over blocks using running max and sum statistics
- **Paged_Attention**: A virtual-memory-based KV-cache management system using fixed-size blocks
- **Ring_Attention**: A distributed attention algorithm that passes KV blocks around a ring of devices
- **KV_Cache_Manager**: A component managing key-value caches for autoregressive inference with prefill and decode phases
- **Quantizer**: A component that reduces weight precision (4-bit, 8-bit) with group-wise scaling
- **Speculative_Decoder**: A draft-verify generation accelerator using a small draft model and large target model
- **Training_Pipeline**: A complete training loop with learning rate scheduling, gradient clipping, checkpointing, and evaluation
- **Memory_Profiler**: A component that tracks and visualizes memory usage during training steps
- **Gradient_Accumulator**: A component that accumulates gradients across multiple micro-batches before applying an optimizer step
- **History_Section**: A chronological "📜 History & Alternatives" markdown section tracing the evolution of a concept
- **Pedagogical_Pattern**: The teaching sequence: math derivation → step-by-step code → visualization → comparison

## Requirements

### Requirement 1: Mixture of Experts Notebook

**User Story:** As a learner preparing for AI lab interviews, I want a notebook that teaches MoE architecture from first principles through modern implementations, so that I can understand and explain expert routing, load balancing, and shared experts in systems like Mixtral, DeepSeek-V3, and Gemma 4.

#### Acceptance Criteria

1. WHEN a learner opens Notebook 15, THE Notebook SHALL present a complete math derivation of Top-K routing followed by a step-by-step MLX implementation of the Router
2. WHEN the Router processes an input tensor of shape [batch, seq, d_model], THE Router SHALL return routing indices of shape [batch, seq, k] and routing weights of shape [batch, seq, k] where weights sum to 1.0 (±1e-6) along the last axis for each token
3. THE MoE_Block SHALL implement Top-K, Expert Choice, and Hash routing mechanisms with side-by-side comparison
4. WHEN the Router assigns tokens to experts, THE MoE_Block SHALL process each token through exactly num_active experts (plus the shared expert when enabled)
5. THE MoE_Block SHALL implement a load balancing auxiliary loss that equals num_experts × Σ(f_i × p_i) where f_i is the fraction of tokens routed to expert i and p_i is the mean routing probability for expert i
6. WHEN the MoE_Block includes a shared expert, THE MoE_Block SHALL add the shared expert output to the weighted combination of routed expert outputs for every token
7. THE Notebook SHALL include a memory analysis comparing total parameters versus active parameters per token for MoE versus dense models
8. THE Notebook SHALL trace the chronological evolution from Dense FFN → Switch Transformer (2021) → Mixtral (2023) → DeepSeek-V3 (2025) → Gemma 4 (2025)
9. WHEN the MoE_Block forward pass receives input of shape [batch, seq, d_model], THE MoE_Block SHALL produce output of shape [batch, seq, d_model]

### Requirement 2: State Space Models Notebook

**User Story:** As a learner, I want a notebook that teaches State Space Models as an alternative to attention, so that I can understand the O(n) complexity advantage, implement selective scan, and compare SSMs with transformers.

#### Acceptance Criteria

1. THE Notebook SHALL explain why attention is O(n²) in sequence length and motivate SSMs as an O(n) alternative
2. WHEN the SimpleSSM discretizes continuous parameters A and B using step size delta, THE SimpleSSM SHALL produce discrete parameters A_bar and B_bar using Zero-Order Hold discretization
3. WHEN the Selective_Scan processes an input sequence, THE Selective_Scan SHALL compute input-dependent parameters Δ, B, and C from the input tensor
4. WHEN the Selective_Scan computes output at position t, THE Selective_Scan SHALL depend only on inputs at positions 0 through t (causality)
5. THE Notebook SHALL implement both sequential scan and parallel scan and verify they produce identical outputs within floating-point tolerance of 1e-5
6. WHILE the SSM processes sequences of increasing length, THE SSM SHALL maintain memory usage proportional to batch × d_inner × d_state, independent of sequence length
7. THE MambaBlock SHALL implement input projection, 1D convolution, selective SSM, SiLU gating, and output projection as described in the Mamba architecture
8. THE Notebook SHALL compare attention versus SSM on computational complexity, memory usage, and output quality with benchmark plots
9. THE Notebook SHALL cover the evolution: Linear Attention → S4 (2021) → Mamba (2023) → Mamba-2 (2024) → RWKV-6 (2024) → Griffin (2024) → Jamba (2024)

### Requirement 3: Alignment Techniques Notebook

**User Story:** As a learner, I want a notebook that teaches how base language models become helpful assistants through alignment, so that I can implement reward modeling, DPO, and GRPO and explain the tradeoffs between alignment approaches.

#### Acceptance Criteria

1. THE Notebook SHALL explain the progression from SFT → RLHF → DPO → KTO → GRPO with mathematical derivations for each transition
2. WHEN the Reward_Model receives a (prompt, response) pair, THE Reward_Model SHALL produce a scalar reward score using a linear head on top of the base model's last hidden state
3. WHEN the DPO_Trainer computes loss for a preference pair, THE DPO_Trainer SHALL compute loss = -log(σ(β × (log_ratio_chosen - log_ratio_rejected))) where log_ratio = log_π_policy - log_π_reference
4. THE DPO_Trainer SHALL compute loss values that are non-negative for all valid inputs
5. WHILE the DPO_Trainer trains the policy model, THE DPO_Trainer SHALL keep the reference model parameters frozen with no gradient updates
6. WHEN the GRPO_Trainer processes a prompt, THE GRPO_Trainer SHALL sample a group of at least 2 responses, compute rewards, normalize rewards to mean ≈ 0 and std ≈ 1 within the group, and compute policy gradient weighted by normalized rewards
7. THE Notebook SHALL compare RLHF versus DPO versus GRPO on implementation complexity, data requirements, and training stability
8. THE Notebook SHALL explain Constitutional AI (Anthropic) as a concept
9. THE Notebook SHALL trace the evolution: RLHF (2022) → DPO (2023) → KTO (2024) → GRPO (2025) → Constitutional AI

### Requirement 4: Scaling Laws Notebook

**User Story:** As a learner, I want a notebook that teaches scaling laws and compute-optimal training, so that I can predict model performance, calculate optimal model-data allocation for a compute budget, and discuss the emergent abilities debate.

#### Acceptance Criteria

1. THE Scaling_Law_Predictor SHALL implement the Chinchilla scaling law formula L(N, D) = A/N^α + B/D^β + E with calibrated constants
2. WHEN the Scaling_Law_Predictor predicts loss, THE Scaling_Law_Predictor SHALL produce loss values that decrease monotonically as either N or D increases while the other is held constant
3. WHEN the Scaling_Law_Predictor computes optimal allocation for a compute budget C, THE Scaling_Law_Predictor SHALL return optimal_N and optimal_D such that 6 × optimal_N × optimal_D is within 10% of C
4. THE Notebook SHALL visualize power-law relationships using log-log plots showing loss versus model size and loss versus data size
5. THE Notebook SHALL implement a compute budget calculator that takes total FLOPs as input and returns optimal model size, optimal token count, and estimated loss
6. THE Notebook SHALL discuss the emergent abilities debate with references to both supporting and opposing evidence
7. THE Notebook SHALL compare Kaplan (2020) versus Chinchilla (2022) scaling law predictions and explain why Chinchilla recommends more data relative to model size
8. THE Notebook SHALL trace the evolution: Kaplan Scaling Laws (2020) → Chinchilla (2022) → Emergent Abilities Debate

### Requirement 5: Reasoning and Test-Time Compute Notebook

**User Story:** As a learner, I want a notebook that teaches how models reason and how test-time compute scaling improves answers, so that I can implement Chain-of-Thought prompting, self-consistency, and MCTS-style reasoning search.

#### Acceptance Criteria

1. THE Notebook SHALL implement a Chain-of-Thought prompting pipeline that generates both reasoning steps and a final answer from a language model
2. WHEN the CoT pipeline applies self-consistency with N ≥ 3 samples, THE CoT pipeline SHALL produce accuracy greater than or equal to single-sample accuracy through majority voting
3. THE MCTS_Reasoner SHALL implement selection (UCB1), expansion, evaluation, and backpropagation phases for reasoning tree search
4. WHEN the MCTS_Reasoner runs with increasing budget, THE MCTS_Reasoner SHALL select the highest-reward trajectory with increasing probability
5. THE Process_Reward_Model SHALL score individual reasoning steps given the reasoning context, returning a score in [0, 1]
6. THE Notebook SHALL explain the difference between process reward models (scoring steps) and outcome reward models (scoring final answers)
7. THE Notebook SHALL cover the evolution: CoT (2022) → Self-Consistency (2022) → Tree-of-Thought (2023) → o1 (2024) → DeepSeek-R1 (2025)

### Requirement 6: Transformer Architecture Deep Rework

**User Story:** As a learner, I want the transformer architecture notebook deeply reworked with backpropagation walkthroughs, activation comparisons, and normalization analysis, so that I can explain transformer internals at interview depth.

#### Acceptance Criteria

1. THE Notebook SHALL include a complete backpropagation walkthrough through a single transformer block showing gradient flow through attention, FFN, residual connections, and normalization
2. WHEN the ActivationComparison compares activation functions, THE ActivationComparison SHALL compute and plot outputs for ReLU, GELU, SiLU, SwiGLU, and GeGLU on the same input tensor
3. WHEN the NormalizationComparison compares normalization methods, THE NormalizationComparison SHALL compute and plot outputs for LayerNorm, RMSNorm, and DeepNorm on the same input tensor
4. THE Notebook SHALL include gradient flow analysis comparing pre-norm versus post-norm transformer stacks, showing that pre-norm produces lower coefficient of variation in gradient norms across layers
5. THE ParameterCounter SHALL compute per-component parameter counts (attention, FFN, normalization, embedding, total) for any TransformerConfig
6. THE ParameterCounter SHALL estimate memory usage for any TransformerConfig and dtype combination
7. THE Notebook SHALL cover weight initialization strategies including Xavier and Kaiming initialization with explanations of when each is appropriate
8. THE Notebook SHALL trace the evolution of activations (ReLU → GELU → SiLU → SwiGLU → GeGLU) and normalizations (BatchNorm → LayerNorm → RMSNorm → DeepNorm)

### Requirement 7: Building GPT Deep Rework

**User Story:** As a learner, I want the GPT building notebook deeply reworked with a full training pipeline on real data, so that I can train a small GPT model end-to-end with proper learning rate scheduling, checkpointing, and evaluation.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL download and prepare the tiny_shakespeare dataset, splitting it into training and validation sets
2. WHEN the Training_Pipeline computes the learning rate at a given step, THE Training_Pipeline SHALL follow a cosine schedule: lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × step / max_steps)) after a linear warmup phase
3. THE Training_Pipeline SHALL implement gradient clipping to prevent exploding gradients
4. THE CheckpointManager SHALL save model weights, optimizer state, and training step to disk and load them to resume training
5. WHEN a checkpoint is loaded and training resumes, THE Training_Pipeline SHALL produce results identical to uninterrupted training within floating-point tolerance
6. THE Training_Pipeline SHALL evaluate the model on the validation set and report validation loss and perplexity
7. THE Notebook SHALL display generation samples at training steps 0, 100, 500, and 1000 to show quality progression
8. IF the Training_Pipeline detects NaN or Inf in the loss, THEN THE Training_Pipeline SHALL load the last valid checkpoint and reduce the learning rate
9. THE Notebook SHALL trace the evolution: GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) → GPT-4 architecture evolution

### Requirement 8: Training on Apple Silicon Deep Rework

**User Story:** As a learner, I want the Apple Silicon training notebook deeply reworked with working implementations of gradient accumulation, mixed precision, and memory profiling, so that I can optimize real training workloads on Apple Silicon hardware.

#### Acceptance Criteria

1. WHEN the Gradient_Accumulator trains with batch_size B/K and accum_steps K, THE Gradient_Accumulator SHALL produce gradients equivalent to training with batch_size B and accum_steps 1 within floating-point tolerance
2. THE MixedPrecisionTrainer SHALL implement a training step using bfloat16 and compare speed, memory usage, and loss stability against float32 training
3. THE Memory_Profiler SHALL record memory usage at each phase of a training step and produce a timeline plot
4. THE Notebook SHALL implement a MemoryBudgetCalculator that computes memory breakdown (parameters, gradients, optimizer states, activations, KV-cache) as a stacked bar chart for any TransformerConfig, batch size, and sequence length
5. IF an out-of-memory error occurs during training, THEN THE Memory_Profiler SHALL automatically reduce the batch size and continue training without data loss
6. THE Notebook SHALL demonstrate mx.compile() with actual training step benchmarks showing speedup over non-compiled execution
7. THE Notebook SHALL trace the evolution of optimizers: SGD → Adam → AdamW → Lion → Schedule-Free

### Requirement 9: Inference Optimization Deep Rework

**User Story:** As a learner, I want the inference optimization notebook deeply reworked with full KV-cache, quantization, and speculative decoding implementations, so that I can optimize model inference and explain these techniques at interview depth.

#### Acceptance Criteria

1. THE KV_Cache_Manager SHALL implement prefill (processing the full prompt) and decode (generating one token at a time using cached keys and values) phases
2. WHEN the KV_Cache_Manager generates tokens, THE KV_Cache_Manager SHALL produce logits identical to full recomputation without cache
3. THE Quantizer SHALL implement 4-bit and 8-bit weight quantization with configurable group size
4. WHEN the Quantizer quantizes weights to N bits with group size G, THE Quantizer SHALL produce per-group maximum absolute error no greater than (max_val - min_val) / (2^N - 1)
5. THE Quantizer SHALL measure and compare perplexity between the original model and quantized variants
6. THE Speculative_Decoder SHALL implement draft generation using a small model and verification using a large target model
7. WHEN the Speculative_Decoder accepts tokens, THE Speculative_Decoder SHALL produce tokens identical to those the target model would generate alone
8. THE Notebook SHALL explain GPTQ, AWQ, and GGUF quantization formats
9. THE Notebook SHALL trace the evolution: FP32 → FP16 → INT8 → INT4 → GPTQ → AWQ → GGUF

### Requirement 10: Flash, Paged, and Ring Attention Rewrite

**User Story:** As a learner, I want the attention optimization notebook completely rewritten with full algorithm derivations and implementations, so that I can implement and explain Flash Attention, Paged Attention, and Ring Attention from first principles.

#### Acceptance Criteria

1. THE Notebook SHALL derive the Online_Softmax algorithm showing how softmax can be computed incrementally over blocks using running max and sum statistics
2. WHEN the Online_Softmax processes attention score blocks, THE Online_Softmax SHALL produce results identical to standard two-pass softmax within floating-point tolerance of 1e-5
3. THE Flash_Attention SHALL implement tiled attention that processes Q, K, V in blocks without materializing the full O(n²) attention matrix in HBM
4. WHEN the Flash_Attention computes attention, THE Flash_Attention SHALL produce output identical to standard attention softmax(QK^T/√d)V within tolerance of 1e-5
5. THE Notebook SHALL include memory analysis showing Flash_Attention peak memory is O(n × d + block_size²) versus O(n² + n × d) for standard attention
6. THE Paged_Attention SHALL implement a block manager that allocates, frees, and reads fixed-size KV-cache blocks
7. THE Ring_Attention SHALL simulate distributed attention across virtual devices by passing KV blocks around a ring
8. THE Notebook SHALL benchmark standard versus Flash_Attention at sequence lengths 512, 1024, 2048, 4096, 8192, and 16384
9. THE Notebook SHALL trace the evolution: Flash Attention 1 (2022) → Flash Attention 2 (2023) → Flash Attention 3, Paged Attention (2023), Ring Attention (2024)

### Requirement 11: Chronological Enrichment

**User Story:** As a learner, I want every notebook enriched with a "History & Alternatives" section, so that I can trace the evolution of each concept from its origin to the 2025 state of the art and discuss historical context in interviews.

#### Acceptance Criteria

1. THE Notebook series SHALL include a "📜 History & Alternatives" section in each of the 20 notebooks
2. WHEN a History_Section is presented, THE History_Section SHALL list innovations in chronological order with year, author/team, and key contribution
3. THE History_Section for Notebook 04 SHALL cover: Sinusoidal → Learned → RoPE (2021) → ALiBi (2021) → p-RoPE (2024)
4. THE History_Section for Notebook 05 SHALL cover: Bahdanau (2014) → Luong (2015) → Transformer (2017) → Flash Attention (2022) → GQA → MQA
5. THE History_Section for Notebook 15 SHALL cover: Dense → Switch Transformer (2021) → Mixtral (2023) → DeepSeek-V3 (2025) → Gemma 4 (2025)
6. THE History_Section for Notebook 16 SHALL cover: RNN → LSTM → S4 (2021) → Mamba (2023) → Mamba-2 (2024) → Griffin (2024) → Jamba (2024)
7. THE History_Section for Notebook 17 SHALL cover: RLHF (2022) → DPO (2023) → KTO (2024) → GRPO (2025) → Constitutional AI

### Requirement 12: Cross-Cutting Quality Standards

**User Story:** As a learner, I want consistent quality standards across all notebooks including MLX-only code, memory safety, numerical stability, and a consistent pedagogical style, so that I have a reliable and cohesive learning experience.

#### Acceptance Criteria

1. THE Notebook series SHALL use MLX exclusively for all neural network implementations, with no PyTorch or TensorFlow code
2. WHILE any single Notebook executes, THE Notebook SHALL keep peak memory usage under 20GB as measured by mx.metal.get_active_memory()
3. THE Notebook series SHALL follow the Pedagogical_Pattern (math derivation → step-by-step code → visualization → comparison) for every major concept
4. THE Notebook series SHALL use emoji markers consistently: 💡 for insights, ⚡ for performance tips, 🎯 for interview tips, ⚠️ for pitfalls
5. WHEN any tensor operation receives inputs in the range [-1000, 1000], THE operation SHALL produce outputs free of NaN and Inf values
6. THE Notebook series SHALL validate tensor shapes with assertions after every major tensor operation
7. WHEN any transformer-like block (MoE_Block, MambaBlock, attention variant) receives input of shape [batch, seq, d_model], THE block SHALL produce output of shape [batch, seq, d_model]

### Requirement 13: Error Handling and Recovery

**User Story:** As a learner, I want robust error handling in all notebooks, so that training and inference recover gracefully from common failures like OOM, NaN, and expert collapse.

#### Acceptance Criteria

1. IF the MoE_Block forward pass exceeds the 20GB memory budget, THEN THE MoE_Block SHALL reduce batch size automatically or switch to sequential expert evaluation
2. IF one or more experts receive less than 1% of tokens consistently during MoE training, THEN THE Notebook SHALL increase the load_balance_weight and log per-expert utilization
3. IF the SSM hidden state produces NaN values, THEN THE Notebook SHALL clip delta values, reset the hidden state to zeros, and reduce the learning rate
4. IF DPO training loss increases or becomes NaN, THEN THE DPO_Trainer SHALL reload from the last checkpoint and reduce the learning rate by a factor of 10
5. IF Flash_Attention output differs from standard attention by more than 1e-5, THEN THE Notebook SHALL fall back to standard attention and log the discrepancy for debugging
