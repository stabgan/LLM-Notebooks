#!/usr/bin/env python3
"""
Add learning objectives to notebook intros that lack them,
and 'Try It Yourself' exercise cells before the History section.
"""

import json
import glob
import uuid
import os

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
    return {"cell_type": "markdown", "id": str(uuid.uuid4())[:8], "metadata": {}, "source": [text]}

# Exercises for each notebook
EXERCISES = {
    '00': (
        "## 🧪 Try It Yourself\n\n"
        "Before moving on, try these quick exercises to make sure you understand the hardware:\n\n"
        "1. **Memory calculation**: If you have a 13B parameter model in 4-bit quantization, how many GB does it need? Will it fit on your machine? (Hint: 13B x 0.5 bytes = ?)\n\n"
        "2. **Throughput estimate**: Using the formula `tokens/sec = bandwidth / model_size`, estimate how fast a 3B float16 model would run on your chip.\n\n"
        "3. **Compare**: Look up the specs of an RTX 4090 (24GB VRAM, ~1 TB/s bandwidth). For which model sizes does your Mac win? For which does the 4090 win?"
    ),
    '06': (
        "## 🧪 Try It Yourself\n\n"
        "Test your understanding of the transformer block:\n\n"
        "1. **Residual connections**: Comment out the residual connection in the transformer block (the `+ x` part). Run a forward pass. What happens to the output? Why?\n\n"
        "2. **Pre-norm vs post-norm**: Move the RMSNorm to AFTER the attention/FFN instead of before. Does the output change? (Hint: yes, and it matters for training stability.)\n\n"
        "3. **Parameter counting**: How many parameters does a single transformer block have with d_model=4096, n_heads=32, d_ff=11008? Calculate by hand, then verify with code."
    ),
    '07': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with your GPT model:\n\n"
        "1. **Temperature**: Generate text with temperature=0.1, 0.5, 1.0, and 2.0. How does the output change? Why?\n\n"
        "2. **Model size**: Try doubling d_model from 64 to 128. Does the loss decrease faster? How much more memory does it use?\n\n"
        "3. **Training data**: Replace the training text with something different (a poem, code, etc.). How does the generated output change after training?"
    ),
    '08': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with Apple Silicon training:\n\n"
        "1. **Memory monitoring**: Add `mx.metal.get_active_memory()` calls before and after each training step. Plot the memory usage over 100 steps.\n\n"
        "2. **Batch size experiment**: Try batch sizes of 1, 4, 16, 64. Which gives the best tokens/sec? Why does very large batch size not always help?\n\n"
        "3. **Precision comparison**: Train the same model for 100 steps in float32 and bfloat16. Compare final loss and training time."
    ),
    '09': (
        "## 🧪 Try It Yourself\n\n"
        "Test your understanding of modern architectures:\n\n"
        "1. **Sliding window**: Change the window size from 4 to 8 to 16. How does the attention pattern change? At what window size does it look like full attention?\n\n"
        "2. **Soft-capping**: Try different cap values (10, 50, 100, 1000). What happens when the cap is very small? Very large?\n\n"
        "3. **Architecture quiz**: Without looking, list the 5 key differences between a vanilla transformer and LLaMA. Check your answer against the comparison table."
    ),
    '10': (
        "## 🧪 Try It Yourself\n\n"
        "Explore Metal GPU performance:\n\n"
        "1. **Matrix size experiment**: Run the matmul benchmark with sizes [64, 128, 256, 512, 1024, 2048, 4096]. At what size does GFLOPS plateau?\n\n"
        "2. **Memory bandwidth**: Calculate the theoretical memory bandwidth utilization for each matrix size. (Hint: bytes_read = 2 * N * N * 4 for float32)\n\n"
        "3. **Softmax benchmark**: Time `mx.softmax()` for different row lengths (128, 512, 2048, 8192). Is it compute-bound or memory-bound?"
    ),
    '11': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with inference optimization:\n\n"
        "1. **Quantization quality**: Quantize a random matrix to 8-bit and 4-bit. Compute the mean squared error for each. How much worse is 4-bit?\n\n"
        "2. **Group size**: Try group sizes of 16, 32, 64, 128 for 4-bit quantization. How does the error change? Why do smaller groups give better accuracy?\n\n"
        "3. **KV-cache size**: Calculate the KV-cache memory for a 7B model with 32 layers, 32 heads, d_head=128, at context lengths of 2K, 8K, 32K, 128K."
    ),
    '12': (
        "## 🧪 Try It Yourself\n\n"
        "Deepen your understanding of attention optimization:\n\n"
        "1. **Online softmax by hand**: Compute softmax([3, 1, 4, 1, 5]) using the online algorithm. Show the running max and sum at each step.\n\n"
        "2. **Memory calculation**: For seq_len=8192 and d_model=4096, calculate the memory needed for the full attention matrix in float16. Then calculate Flash Attention's memory.\n\n"
        "3. **Block size**: In tiled Flash Attention, what happens if you use a very small block size (e.g., 4)? Very large (e.g., 1024)? What's the tradeoff?"
    ),
    '13': (
        "## 🧪 Try It Yourself\n\n"
        "Try local model serving:\n\n"
        "1. **Install Ollama**: Run `brew install ollama`, then `ollama run qwen2.5:0.5b`. Chat with it! How fast does it respond?\n\n"
        "2. **Compare sizes**: If you have enough memory, try a 3B and 7B model. Can you feel the speed difference? The quality difference?\n\n"
        "3. **Memory check**: While a model is running, open Activity Monitor and check memory usage. Does it match the expected model size?"
    ),
    '14': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with LoRA:\n\n"
        "1. **Rank experiment**: Create LoRALinear layers with rank=1, 4, 8, 16, 32. How does the number of trainable parameters change?\n\n"
        "2. **Alpha scaling**: With rank=8, try alpha values of 1, 8, 16, 32. What does alpha control? (Hint: it scales the LoRA contribution.)\n\n"
        "3. **Memory budget**: For a 12B model, calculate the memory needed for LoRA adapters targeting all Q, K, V projections with rank=16."
    ),
    '15': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with Mixture of Experts:\n\n"
        "1. **Expert count**: Create MoE blocks with 4, 8, 16, 32 experts (top-k=2). How does the total parameter count change? The active parameter count?\n\n"
        "2. **Load balance**: Run the MoE block 100 times with random inputs. Plot the expert utilization histogram. Is it balanced?\n\n"
        "3. **Shared expert**: Compare MoE output with and without the shared expert. Does the shared expert change the output significantly?"
    ),
    '16': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with State Space Models:\n\n"
        "1. **Discretization**: Try different delta values (0.01, 0.1, 1.0). How does the discretized A_bar change? What happens when delta is very large?\n\n"
        "2. **Causality check**: Modify a token in the middle of a sequence. Verify that only later outputs change (not earlier ones).\n\n"
        "3. **Memory comparison**: For seq_len=1024, compare the memory needed for attention (O(n^2)) vs SSM (O(n * d_state)). At what sequence length does SSM win?"
    ),
    '19': (
        "## 🧪 Try It Yourself\n\n"
        "Experiment with reasoning:\n\n"
        "1. **Self-consistency**: Generate 5 answers to a simple math problem. Do they all agree? What if you increase temperature?\n\n"
        "2. **UCB1 exploration**: Plot the UCB1 score for a node with value=0.7 as the visit count goes from 1 to 100. When does exploration stop dominating?\n\n"
        "3. **Process reward**: Score a 3-step reasoning chain where step 2 is wrong. How does the PRM score differ from an outcome-only reward?"
    ),
}

def add_exercises(path):
    """Add exercise cells to notebooks."""
    nb = load_nb(path)
    cells = nb['cells']
    basename = os.path.basename(path)
    nb_num = basename[:2]
    
    # Check if already has exercises
    all_md = ' '.join(src(c) for c in cells if c.get('cell_type') == 'markdown')
    if 'try it yourself' in all_md.lower() or '🧪' in all_md:
        return False
    
    exercise_text = EXERCISES.get(nb_num)
    if not exercise_text:
        return False
    
    # Find history section to insert before
    history_idx = None
    next_idx = None
    for i, c in enumerate(cells):
        s = src(c)
        if '📜 History' in s:
            history_idx = i
        if '➡️ What' in s:
            next_idx = i
    
    # Insert before "What's Next" if it exists, else before History
    insert_idx = next_idx or history_idx or len(cells)
    
    cells.insert(insert_idx, make_md(exercise_text))
    save_nb(path, nb)
    return True

def main():
    notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    
    print("Adding exercises to notebooks...\n")
    
    count = 0
    for path in notebooks:
        if add_exercises(path):
            count += 1
            print(f"  ✅ Added exercises to {os.path.basename(path)}")
        else:
            print(f"  ⏭️  Skipped {os.path.basename(path)} (already has exercises or no template)")
    
    print(f"\nTotal: {count} exercise sections added")

if __name__ == '__main__':
    main()
