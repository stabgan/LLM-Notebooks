# Requirements Document: LLM Learning Notebook

## Requirement 1: Notebook Series Structure

### 1.1 The project MUST produce 15 Jupyter notebooks named `{nn}_{snake_case}.ipynb` where nn ranges from 00 to 14, covering: environment/Apple Silicon, MLX fundamentals, math foundations, tokenization, embeddings/positional encoding, self-attention, transformer architecture, building GPT, training on Apple Silicon, modern architectures, Metal custom kernels, inference optimization, flash/paged/ring attention, serving locally, and capstone Gemma 4.

### 1.2 Each notebook MUST begin with an environment validation code cell that checks required imports, library versions, and Metal GPU availability before proceeding.

### 1.3 The project MUST include a `utils/` directory containing shared utility modules: `viz.py` (plotting helpers), `benchmark.py` (timing and memory measurement), `data.py` (dataset loaders), and `checks.py` (environment validation).

### 1.4 Every code cell MUST be preceded by a markdown cell containing a plain English explanation of what the code does and why.

### 1.5 Tips and tricks MUST use consistent emoji markers throughout all notebooks: 💡 for insights, ⚡ for performance tips, 🎯 for interview tips, ⚠️ for common pitfalls.

## Requirement 2: Content Coverage by Phase

### 2.1 Notebook 00 (Environment & Apple Silicon) MUST cover unified memory architecture diagrams, Metal GPU capabilities, memory bandwidth calculations with model size estimates, and system interrogation code that reports chip, cores, memory, and Metal support.

### 2.2 Notebook 01 (MLX Fundamentals) MUST cover MLX array creation and dtypes, lazy evaluation with benchmarks comparing lazy vs eager, automatic differentiation with `mx.grad`, `mlx.nn` modules (Linear, Embedding, LayerNorm, RMSNorm), and the MLX training loop pattern using `nn.value_and_grad`.

### 2.3 Notebook 02 (Math Foundations) MUST implement in MLX with step-by-step breakdowns: dot product and cosine similarity, matrix multiplication with shape annotations and GPU benchmarks, softmax with numerical stability trick, temperature scaling and sampling strategies (top-k, top-p), and cross-entropy loss with perplexity explanation.

### 2.4 Notebook 03 (Tokenization) MUST implement: character-level tokenization, BPE from scratch showing iterative merge steps, integration with tiktoken (GPT-4 cl100k_base), and SentencePiece tokenizer usage.

### 2.5 Notebook 04 (Embeddings & Positional Encoding) MUST implement: token embedding lookup tables, sinusoidal positional encoding with visualization, and Rotary Position Embeddings (RoPE) with the 2D rotation math broken into steps.

### 2.6 Notebook 05 (Self-Attention) MUST implement from scratch in MLX: single-head attention with Q/K/V projections, scaled dot-product attention with shape annotations at every step, causal masking, and multi-head attention with head splitting and concatenation.

### 2.7 Notebook 06 (Transformer Architecture) MUST build a complete transformer block by assembling: multi-head attention, feed-forward network (with SiLU/GELU activation), residual connections, and layer normalization (both LayerNorm and RMSNorm variants).

### 2.8 Notebook 07 (Building GPT) MUST implement a complete GPT model class in MLX with token embedding, positional encoding, N transformer blocks, and output projection to vocabulary, plus a `generate()` method with temperature and top-k/top-p sampling, trained on a text dataset with visible loss curves.

### 2.9 Notebook 08 (Training on Apple Silicon) MUST cover: memory monitoring with `mx.metal.get_active_memory()`, mixed precision training (float16/bfloat16), gradient accumulation for effective larger batch sizes, cosine learning rate schedule with warmup, and performance metrics (tokens/sec, memory usage).

### 2.10 Notebook 09 (Modern Architectures) MUST explain architectural differences between LLaMA (RMSNorm, SwiGLU, RoPE, GQA), Mistral (sliding window attention), and Gemma (including evolution to Gemma 4), with code showing key differentiating components.

### 2.11 Notebook 10 (Metal & Custom Kernels) MUST include: Metal Shading Language basics (threadgroups, SIMD groups, shared memory), at least 3 custom .metal kernel files (softmax, RMSNorm, and tiled matmul), kernel invocation from Python, and performance comparison against MLX built-in operations.

### 2.12 Notebook 11 (Inference Optimization) MUST implement: KV-cache for autoregressive generation with memory tracking, 4-bit and 8-bit post-training quantization with perplexity comparison, and speculative decoding concept with a draft-verify example.

### 2.13 Notebook 12 (Flash/Paged/Ring Attention) MUST explain the concepts of Flash Attention (tiled computation, reduced memory), Paged Attention (virtual memory for KV-cache), and Ring Attention (distributed sequence parallelism), with Metal implementations where feasible and conceptual diagrams where not.

### 2.14 Notebook 13 (Serving Locally) MUST demonstrate: loading and running models with `mlx-lm`, running models with `llama.cpp` Metal backend, benchmarking inference speed (tokens/sec) for different model sizes, and interactive chat-style generation.

### 2.15 Notebook 14 (Capstone) MUST fine-tune a Gemma 4 model (12B or smaller, 4-bit quantized) on a custom dataset using MLX, evaluate the fine-tuned model, and serve it locally for interactive generation, all within the 48GB memory budget.

## Requirement 3: Correctness Properties

### 3.1 All softmax implementations MUST produce valid probability distributions: no NaN or Inf values, all values in [0, 1], and row sums equal to 1.0 within floating-point tolerance (±1e-5 for float16, ±1e-7 for float32), including for extreme input logits up to ±1000.

### 3.2 Causal attention masks MUST prevent information flow from future positions: for all positions i and j where j > i, `attention_weights[i][j]` MUST equal 0.0.

### 3.3 All tokenizer implementations MUST be lossless: `decode(encode(s)) == s` for all strings in the training corpus.

### 3.4 RoPE implementation MUST preserve vector L2 norms: `||RoPE(x, pos)|| == ||x||` within floating-point tolerance for all input vectors and positions.

### 3.5 Quantization error MUST be bounded: for all weight tensors quantized to N bits with group size G, the maximum absolute error per group MUST NOT exceed `(max_val - min_val) / (2^N - 1)`.

### 3.6 KV-cache generation MUST produce identical logits to full recomputation without cache for the same input sequence.

## Requirement 4: Platform & Framework Constraints

### 4.1 All runnable code MUST use MLX as the primary ML framework. PyTorch MAY appear only in explicitly labeled comparison cells.

### 4.2 No CUDA code MUST appear in any executable cell. CUDA concepts MAY be referenced in markdown explanations for comparison purposes only.

### 4.3 All dependencies MUST be installable via pip in a Python 3.13 virtual environment on macOS Apple Silicon.

### 4.4 All visualization cells MUST use matplotlib (and optionally seaborn) for plots, charts, and diagrams.

### 4.5 All training and inference operations MUST stay within the 48GB unified memory budget, with peak usage not exceeding 45GB (leaving 3GB headroom for macOS).

## Requirement 5: Error Handling & Robustness

### 5.1 Training code MUST detect out-of-memory conditions and automatically reduce batch size (with gradient accumulation to maintain effective batch size), displaying a clear memory breakdown.

### 5.2 Training loops MUST check for NaN/Inf in loss values and implement automatic recovery: reduce learning rate by 10x, enable gradient clipping (max_norm=1.0), and restart from the last valid checkpoint.

### 5.3 Notebook 00 MUST check `mx.metal.is_available()` and display a clear warning with diagnostic information if Metal GPU is not available, providing a CPU fallback path.

### 5.4 Each notebook MUST handle import failures gracefully with clear error messages and pip install instructions for missing dependencies.
