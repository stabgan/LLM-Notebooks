# Tasks: LLM Learning Notebook

## Task 1: Project Scaffolding & Shared Utilities

- [x] 1.1 Create directory structure: `utils/`, `data/`, `models/`, `metal_kernels/`
- [x] 1.2 Create `utils/__init__.py` with package imports
- [x] 1.3 Create `utils/checks.py` with `validate_environment()` function that checks Python version (3.13), MLX availability, Metal GPU status, and memory size, returning a dict of results
- [x] 1.4 Create `utils/viz.py` with plotting helpers: `plot_attention_heatmap()`, `plot_loss_curve()`, `plot_token_probabilities()`, `plot_embeddings_2d()` using matplotlib
- [x] 1.5 Create `utils/benchmark.py` with `time_function()`, `memory_snapshot()` (using `mx.metal.get_active_memory()`), and `estimate_model_memory()` helpers
- [x] 1.6 Create `utils/data.py` with `TextDataset` class that loads text files, tokenizes them, and yields `(input_ids, target_ids)` batches as MLX arrays
- [x] 1.7 Create `requirements.txt` pinning all dependencies: mlx>=0.22.0, mlx-lm>=0.20.0, numpy>=1.26.0, tiktoken>=0.7.0, sentencepiece>=0.2.0, matplotlib>=3.9.0, seaborn>=0.13.0, huggingface_hub>=0.25.0, transformers>=4.45.0, datasets>=3.0.0, llama-cpp-python>=0.3.0, einops>=0.8.0, hypothesis>=6.100.0

## Task 2: Notebook 00 — Environment & Apple Silicon Deep Dive

- [x] 2.1 Create `00_environment_apple_silicon.ipynb` with title cell, roadmap overview, and environment validation cell using `utils.checks.validate_environment()`
- [x] 2.2 Add cells for system interrogation: chip info via `sysctl`, CPU core counts (performance + efficiency), total memory, GPU info via `system_profiler`
- [x] 2.3 Add cells explaining Unified Memory Architecture with ASCII diagrams comparing traditional (NVIDIA PCIe) vs Apple Silicon (UMA), with markdown explaining zero-copy and bandwidth advantages
- [x] 2.4 Add cells for Metal GPU capabilities: query `system_profiler SPDisplaysDataType -json`, explain Metal 3 features, SIMD groups, threadgroups, compute shaders
- [x] 2.5 Add memory bandwidth calculation cells: compute theoretical tok/s for various model sizes (GPT-2 through 70B) at different quantization levels, with a formatted table showing fits/doesn't-fit and expected performance
- [x] 2.6 Add tips cells with 💡⚡🎯⚠️ markers covering: interview talking points about UMA vs discrete GPU, why memory bandwidth matters more than TFLOPS for inference, common misconceptions

## Task 3: Notebook 01 — MLX Fundamentals

- [x] 3.1 Create `01_mlx_fundamentals.ipynb` with environment validation cell, MLX vs PyTorch comparison table in markdown
- [x] 3.2 Add cells for MLX array creation: from lists, from NumPy (zero-copy), random arrays, key dtypes (float32, float16, bfloat16) with byte size comparisons
- [x] 3.3 Add cells for lazy evaluation: demonstrate operations are recorded not executed, `mx.eval()` trigger, benchmark lazy vs forced-eager with timing comparison, explain kernel fusion benefits
- [x] 3.4 Add cells for automatic differentiation: `mx.grad` on simple functions (x², sin), second derivatives, `mx.grad` on MSE loss function, gradient visualization
- [x] 3.5 Add cells for `mlx.nn` modules: Linear layer (inspect parameters, forward pass with shape annotations), Embedding, LayerNorm, RMSNorm, activation functions (ReLU, GELU, SiLU)
- [x] 3.6 Add cells for MLX training loop pattern: define TinyMLP, `nn.value_and_grad()`, `optimizer.update()`, `mx.eval()` for lazy evaluation, train on y=sum(x) with loss logging

## Task 4: Notebook 02 — Math Foundations

- [x] 4.1 Create `02_math_foundations.ipynb` with environment validation cell and overview of math needed for transformers
- [x] 4.2 Add cells for dot product: element-wise multiply then sum, `@` operator, cosine similarity, explain connection to attention ("how much should token A attend to token B?")
- [x] 4.3 Add cells for matrix multiplication: batch of token embeddings through linear layer with shape annotations at every step, GPU benchmark for various matrix sizes with GFLOPS calculation
- [x] 4.4 Add cells for softmax: manual implementation with numerical stability trick (subtract max), comparison with `mx.softmax`, matplotlib visualization of logits vs probabilities
- [x] 4.5 Add cells for temperature scaling: show effect of T=0.1, 0.5, 1.0, 2.0, 5.0 on probability distributions with side-by-side bar charts, implement top-k and top-p (nucleus) sampling
- [x] 4.6 Add cells for cross-entropy loss: manual implementation using log-softmax, good vs bad prediction comparison, perplexity explanation (exp(loss) = effective vocabulary confusion), GPT-4 level perplexity reference

## Task 5: Notebook 03 — Tokenization

- [x] 5.1 Create `03_tokenization.ipynb` with environment validation cell and tokenization evolution overview (character → word → BPE → byte-level BPE)
- [x] 5.2 Add cells for character-level tokenization: build vocab from text, encode/decode functions, demonstrate inefficiency (1 char = 1 token) vs BPE
- [x] 5.3 Add cells for BPE from scratch: implement `train_bpe()` showing iterative merge steps with print output at each merge, compression ratio calculation, encode/decode with learned merges
- [x] 5.4 Add cells for tiktoken integration: load GPT-4 tokenizer (cl100k_base), encode various texts, show token IDs and decoded tokens, calculate fertility (tokens/word)
- [x] 5.5 Add cells for SentencePiece: explain unigram model vs BPE, demonstrate with pretrained model, compare tokenization of same text across character/BPE/tiktoken/SentencePiece

## Task 6: Notebook 04 — Embeddings & Positional Encoding

- [x] 6.1 Create `04_embeddings_positional_encoding.ipynb` with environment validation cell and explanation of why position information is needed
- [x] 6.2 Add cells for token embeddings: `nn.Embedding` lookup table, visualize embedding vectors, show that similar tokens have similar embeddings (cosine similarity matrix)
- [x] 6.3 Add cells for sinusoidal positional encoding: implement the sin/cos formula step by step across multiple cells, visualize the encoding matrix as a heatmap, show how different positions have unique patterns
- [x] 6.4 Add cells for RoPE: explain the 2D rotation intuition, implement `precompute_freqs_cis()` and `apply_rope()` step by step, verify norm preservation property, visualize rotation effect on embedding pairs
- [x] 6.5 Add comparison cells: sinusoidal vs RoPE vs learned embeddings, explain why RoPE is used in LLaMA/Gemma (relative position, extrapolation)

## Task 7: Notebook 05 — Self-Attention from Scratch

- [x] 7.1 Create `05_self_attention.ipynb` with environment validation cell and intuitive explanation of attention ("which words should I focus on?")
- [x] 7.2 Add cells for single-head attention: create Q, K, V projection matrices, compute Q=xW_q, K=xW_k, V=xW_v with shape annotations, compute scores=Q@K^T, scale by sqrt(d_k), apply softmax, compute output=weights@V
- [x] 7.3 Add cells for attention visualization: matplotlib heatmap of attention weights for a sample sentence, show which tokens attend to which
- [x] 7.4 Add cells for causal masking: create lower-triangular mask, apply mask before softmax (set masked positions to -inf), verify no future information leakage
- [x] 7.5 Add cells for multi-head attention: split d_model into n_heads, parallel attention computations, concatenate heads, output projection, implement as `MultiHeadAttention(nn.Module)` class
- [x] 7.6 Add cells for Grouped Query Attention (GQA): explain KV head sharing (used in LLaMA 3, Gemma), implement GQA variant, compare memory usage vs standard MHA

## Task 8: Notebook 06 — Transformer Architecture

- [x] 8.1 Create `06_transformer_architecture.ipynb` with environment validation cell and block diagram of a transformer layer in markdown
- [x] 8.2 Add cells for feed-forward network: implement FFN with SiLU activation (SwiGLU variant used in modern LLMs), show shape transformations (d_model → d_ff → d_model)
- [x] 8.3 Add cells for layer normalization: implement LayerNorm from scratch, implement RMSNorm from scratch, compare both, explain pre-norm vs post-norm placement
- [x] 8.4 Add cells for residual connections: explain gradient flow benefits, implement x + sublayer(norm(x)) pattern, visualize gradient magnitudes with and without residuals
- [x] 8.5 Add cells assembling a complete `TransformerBlock(nn.Module)`: attention → add & norm → FFN → add & norm, with shape annotations at every step
- [x] 8.6 Add cells stacking N blocks into a `TransformerStack`: demonstrate that output shape is preserved through all layers, count parameters per block and total

## Task 9: Notebook 07 — Building a GPT from Scratch

- [x] 9.1 Create `07_building_gpt_from_scratch.ipynb` with environment validation cell and GPT architecture overview
- [x] 9.2 Implement complete `GPTModel(nn.Module)` class: token embedding + positional encoding + N transformer blocks + final LayerNorm + output projection to vocab_size, with `__call__` and `generate` methods
- [x] 9.3 Add cells for model inspection: count parameters, print layer shapes, estimate memory usage for different configs (tiny/small/medium)
- [x] 9.4 Add cells for data preparation: load a text dataset (tiny_shakespeare or similar), tokenize with character-level tokenizer, create train/val splits, implement batching
- [x] 9.5 Add cells for training loop: loss function (cross-entropy), `nn.value_and_grad`, AdamW optimizer, cosine LR schedule with warmup, training loop with loss/perplexity logging every N steps, matplotlib loss curve plot
- [x] 9.6 Add cells for text generation: implement `generate()` with temperature, top-k, top-p sampling, generate text from prompts, show how quality improves during training (samples at step 0, 500, 1000, etc.)

## Task 10: Notebook 08 — Training on Apple Silicon

- [x] 10.1 Create `08_training_apple_silicon.ipynb` with environment validation cell and Apple Silicon training considerations overview
- [x] 10.2 Add cells for memory monitoring: `mx.metal.get_active_memory()`, `mx.metal.get_peak_memory()`, `mx.metal.get_cache_memory()`, build `MemoryMonitor` class that tracks usage over time with matplotlib plots
- [x] 10.3 Add cells for mixed precision training: compare float32 vs float16 vs bfloat16 training speed and memory, demonstrate `mx.array` dtype casting, explain when bfloat16 is preferred (better dynamic range)
- [x] 10.4 Add cells for gradient accumulation: implement accumulation over N micro-batches, show effective batch size = micro_batch * accum_steps, demonstrate memory savings
- [x] 10.5 Add cells for `mx.compile()` JIT compilation: wrap training step in `mx.compile`, benchmark compiled vs uncompiled, explain when compilation helps
- [x] 10.6 Add cells for `MemoryBudget` calculator: input model config, compute weights/optimizer/activations/KV-cache memory, display breakdown as stacked bar chart, warn if exceeding 45GB

## Task 11: Notebook 09 — Modern Architectures

- [x] 11.1 Create `09_modern_architectures.ipynb` with environment validation cell and architecture evolution timeline
- [x] 11.2 Add cells for LLaMA architecture: implement key differences from vanilla transformer — RMSNorm (pre-norm), SwiGLU FFN, RoPE, GQA — with side-by-side code comparison
- [x] 11.3 Add cells for Mistral architecture: implement sliding window attention, explain sparse attention patterns, compare memory usage vs full attention
- [x] 11.4 Add cells for Gemma architecture: explain Gemma 1 → 2 → 4 evolution, key innovations (multi-query attention, different FFN variants), implement distinguishing components
- [x] 11.5 Add comparison table and cells: parameter counts, context lengths, attention mechanisms, normalization choices, FFN variants across LLaMA/Mistral/Gemma families

## Task 12: Notebook 10 — Metal Shading Language & Custom Kernels

- [x] 12.1 Create `10_metal_custom_kernels.ipynb` with environment validation cell and Metal compute shader overview (threadgroups, SIMD groups, shared memory)
- [x] 12.2 Create `metal_kernels/softmax.metal` implementing a fused softmax kernel with shared memory for max reduction, and add notebook cells that explain the kernel line by line and invoke it
- [x] 12.3 Create `metal_kernels/rmsnorm.metal` implementing RMS normalization kernel, and add notebook cells comparing performance against `mx.fast.rms_norm`
- [x] 12.4 Create `metal_kernels/matmul_tiled.metal` implementing tiled matrix multiplication with threadgroup shared memory, and add notebook cells explaining tiling strategy and benchmarking against `mx.matmul`
- [x] 12.5 Add cells explaining Metal memory model: device vs threadgroup vs thread memory, memory coalescing, bank conflicts, and how MLX uses Metal under the hood

## Task 13: Notebook 11 — Inference Optimization

- [x] 13.1 Create `11_inference_optimization.ipynb` with environment validation cell and inference bottleneck analysis (memory-bandwidth-bound)
- [x] 13.2 Add cells for KV-cache implementation: build `KVCache` class from scratch, demonstrate prefill vs decode phases, show memory growth over sequence length, benchmark with vs without cache
- [x] 13.3 Add cells for quantization: implement 4-bit and 8-bit group-wise quantization from scratch, quantize a trained model, compare perplexity before/after, measure memory reduction and speedup
- [x] 13.4 Add cells for MLX built-in quantization: use `mlx.core.quantize` and `nn.QuantizedLinear`, compare with manual implementation
- [x] 13.5 Add cells for speculative decoding: explain draft-verify paradigm, implement with a small draft model and larger target model, show acceptance rate and speedup

## Task 14: Notebook 12 — Flash, Paged, and Ring Attention

- [x] 14.1 Create `12_flash_paged_ring_attention.ipynb` with environment validation cell and standard attention memory analysis (O(n²) problem)
- [x] 14.2 Add cells for Flash Attention: explain tiled computation and online softmax algorithm, implement a simplified version showing the tiling concept, compare memory usage vs standard attention, reference `mx.fast.scaled_dot_product_attention`
- [x] 14.3 Add cells for Paged Attention: explain virtual memory analogy for KV-cache, implement block-based KV-cache manager, show how it reduces memory fragmentation
- [x] 14.4 Add cells for Ring Attention: explain distributed sequence parallelism concept, implement a simulation showing how attention is computed in rings across virtual "devices", explain relevance for very long contexts

## Task 15: Notebook 13 — Serving Locally

- [x] 15.1 Create `13_serving_locally.ipynb` with environment validation cell and local serving options overview
- [x] 15.2 Add cells for mlx-lm: load a quantized model from Hugging Face using `mlx_lm.load()`, generate text with `mlx_lm.generate()`, benchmark tokens/sec for different model sizes (3B, 7B, 12B)
- [x] 15.3 Add cells for llama.cpp Metal backend: load a GGUF model with `llama-cpp-python`, generate text, benchmark and compare performance with mlx-lm
- [x] 15.4 Add cells for interactive chat: implement a simple chat loop with conversation history, system prompt, and streaming output
- [x] 15.5 Add cells comparing serving options: mlx-lm vs llama.cpp vs direct MLX model, table of tok/s, memory usage, ease of use, model format support

## Task 16: Notebook 14 — Capstone: Fine-tune & Serve Gemma 4

- [x] 16.1 Create `14_capstone_gemma4.ipynb` with environment validation cell, Gemma 4 architecture overview, and memory budget calculation for 12B 4-bit model
- [x] 16.2 Add cells for loading Gemma 4: download 4-bit quantized model via Hugging Face, load with mlx-lm, verify it fits in memory, run baseline generation
- [x] 16.3 Add cells for LoRA fine-tuning setup: explain LoRA (Low-Rank Adaptation), configure LoRA rank and target modules, calculate trainable parameter count vs total
- [x] 16.4 Add cells for fine-tuning execution: prepare a custom dataset, run fine-tuning loop with memory monitoring, plot training loss curve, save LoRA adapter weights
- [x] 16.5 Add cells for evaluation: compare base vs fine-tuned model outputs on test prompts, measure perplexity on held-out data
- [x] 16.6 Add cells for serving the fine-tuned model: load base model + LoRA adapter, interactive generation, final performance benchmarks (tok/s, memory)

## Task 17: Integration & Validation

- [x] 17.1 Add inline shape assertions after every tensor operation in notebooks 04-09 (e.g., `assert output.shape == (batch, seq_len, d_model)`)
- [x] 17.2 Add softmax validation cells in notebook 02 and 05: verify no NaN, values in [0,1], rows sum to 1.0, test with extreme logits (±1000)
- [x] 17.3 Add tokenizer roundtrip validation cell in notebook 03: encode then decode multiple test strings, assert equality
- [x] 17.4 Add RoPE norm preservation validation cell in notebook 04: compute L2 norm before and after RoPE, assert within tolerance
- [x] 17.5 Add causal mask validation cell in notebook 05: verify `attention_weights[i][j] == 0` for all `j > i`
- [x] 17.6 Add memory budget validation in notebooks 07, 08, 14: check `mx.metal.get_active_memory()` stays under 45GB after key operations
- [x] 17.7 Add OOM error handling in notebook 08 training loop: try/except for memory errors with automatic batch size reduction and gradient accumulation adjustment
- [x] 17.8 Add NaN/Inf detection in notebook 07 and 08 training loops: check `math.isnan(loss)` each step, implement recovery (reduce LR, enable grad clipping, reload checkpoint)
