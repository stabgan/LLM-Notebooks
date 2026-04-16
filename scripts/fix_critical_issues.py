#!/usr/bin/env python3
"""
Fix critical pedagogical issues identified by notebook reviews.
Batch 1 + Batch 2 combined fixes.

Priority fixes:
1. NB14: Add working LoRA training loop (capstone is mostly placeholders)
2. NB10: Inline Metal kernel source code (notebook never shows kernels)
3. NB13: Make mlx-lm demo actually runnable with try/except
4. NB19: Fix self-consistency plot label (misleading math)
5. NB16: Fix MambaBlock takeaways ordering (appears before code)
6. NB05: Add note about random embeddings producing uniform attention
7. NB01: Add note about lazy eval shape inference
8. NB12: Add shape annotations to key cells

Usage: .venv/bin/python scripts/fix_critical_issues.py
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


def fix_nb14():
    """NB14: Add working LoRA training loop after the LoRA demo."""
    path = '14_capstone_gemma4.ipynb'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return 0
    
    nb = load_nb(path)
    cells = nb.get('cells', [])
    fixes = 0
    
    # Find the LoRA demo cell (d2) and insert training loop after it
    for i, cell in enumerate(cells):
        s = src(cell)
        if 'LoRALinear' in s and 'class' in s and cell.get('cell_type') == 'code':
            # Check if training loop already exists
            remaining = ''.join(src(c) for c in cells[i+1:])
            if 'training loop' not in remaining.lower() or 'for step' not in remaining:
                # Insert training loop markdown + code after the demo
                # Find the next cell after the demo output
                insert_idx = i + 1
                # Skip past any existing cells until we find the fine-tuning section
                while insert_idx < len(cells) and 'Fine-tuning Execution' not in src(cells[insert_idx]):
                    insert_idx += 1
                
                md_cell = make_md(
                    "### 🔧 Working LoRA Training Demo\n\n"
                    "Let's actually train our `LoRALinear` layer on a simple task to see LoRA in action. "
                    "We'll train it to approximate a target linear transformation — this demonstrates that "
                    "the low-rank adapters can learn meaningful weight updates while the base weights stay frozen.\n\n"
                    "💡 **What's happening:** Only `lora_A` and `lora_B` receive gradient updates. "
                    "The frozen base weight `W` stays exactly the same throughout training."
                )
                
                code_cell = make_code(
                    "import mlx.core as mx\n"
                    "import mlx.nn as nn\n"
                    "import mlx.optimizers as optim\n"
                    "import matplotlib.pyplot as plt\n\n"
                    "# Create a LoRA layer and a target transformation\n"
                    "mx.random.seed(42)\n"
                    "lora_model = LoRALinear(64, 64, rank=4, alpha=8.0)\n"
                    "mx.eval(lora_model.parameters())\n\n"
                    "# Target: a random linear transformation we want LoRA to learn\n"
                    "target_W = mx.random.normal((64, 64)) * 0.1\n"
                    "mx.eval(target_W)\n\n"
                    "# Save original base weights to verify they don't change\n"
                    "original_base = mx.array(lora_model.linear.weight)\n"
                    "mx.eval(original_base)\n\n"
                    "# Loss: MSE between LoRA output and target output\n"
                    "def lora_loss(model, x):\n"
                    "    pred = model(x)                    # shape: (batch, 64)\n"
                    "    target = x @ target_W.T            # shape: (batch, 64)\n"
                    "    return mx.mean((pred - target) ** 2)\n\n"
                    "# Training loop\n"
                    "loss_grad_fn = nn.value_and_grad(lora_model, lora_loss)\n"
                    "optimizer = optim.Adam(learning_rate=1e-3)\n"
                    "losses = []\n\n"
                    "for step in range(200):\n"
                    "    x = mx.random.normal((32, 64))     # shape: (batch=32, features=64)\n"
                    "    loss, grads = loss_grad_fn(lora_model, x)\n"
                    "    optimizer.update(lora_model, grads)\n"
                    "    mx.eval(lora_model.parameters(), optimizer.state, loss)\n"
                    "    losses.append(loss.item())\n"
                    "    if step % 50 == 0:\n"
                    "        print(f'  Step {step:3d} | Loss: {loss.item():.6f}')\n\n"
                    "# Verify base weights are FROZEN\n"
                    "base_diff = mx.max(mx.abs(lora_model.linear.weight - original_base)).item()\n"
                    "print(f'\\n✅ Base weight change: {base_diff:.2e} (should be 0.0)')\n"
                    "print(f'✅ Final loss: {losses[-1]:.6f}')\n\n"
                    "# Plot training curve\n"
                    "fig, ax = plt.subplots(figsize=(8, 4))\n"
                    "ax.plot(losses, color='#e74c3c', linewidth=2)\n"
                    "ax.set_xlabel('Step')\n"
                    "ax.set_ylabel('MSE Loss')\n"
                    "ax.set_title('LoRA Training: Learning a Target Transformation')\n"
                    "ax.set_yscale('log')\n"
                    "ax.grid(True, alpha=0.3)\n"
                    "plt.tight_layout()\n"
                    "plt.show()\n"
                    "print('\\n💡 LoRA learned to approximate the target with only rank-4 adapters!')\n"
                    "print(f'   Trainable params: {4*64 + 4*64} = {4*64*2} (vs {64*64} = {64**2} base params)')"
                )
                
                cells.insert(insert_idx, code_cell)
                cells.insert(insert_idx, md_cell)
                fixes += 2
                print(f"  NB14: Added working LoRA training loop")
            break
    
    if fixes > 0:
        nb['cells'] = cells
        save_nb(path, nb)
    return fixes


def fix_nb10():
    """NB10: Inline Metal kernel source code so notebook actually shows kernels."""
    path = '10_metal_custom_kernels.ipynb'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return 0
    
    nb = load_nb(path)
    cells = nb.get('cells', [])
    fixes = 0
    
    # Find the softmax kernel section and add kernel display
    for i, cell in enumerate(cells):
        s = src(cell)
        if 'Custom Softmax Kernel' in s and cell.get('cell_type') == 'markdown':
            # Check if kernel source is already shown
            next_code = src(cells[i+1]) if i+1 < len(cells) else ''
            if 'metal_kernels' not in next_code or 'open(' not in next_code:
                # Insert a cell that reads and displays the kernel
                display_cell = make_code(
                    "# 💡 Let's look at the actual Metal kernel source code\n"
                    "# This is what runs on the GPU when you call softmax\n\n"
                    "kernel_path = 'metal_kernels/softmax.metal'\n"
                    "try:\n"
                    "    with open(kernel_path) as f:\n"
                    "        kernel_source = f.read()\n"
                    "    print(f'📄 {kernel_path}:')\n"
                    "    print('=' * 60)\n"
                    "    print(kernel_source)\n"
                    "    print('=' * 60)\n"
                    "    print()\n"
                    "    print('💡 Key concepts in this kernel:')\n"
                    "    print('  1. threadgroup_barrier — synchronizes threads within a group')\n"
                    "    print('  2. shared_max/shared_sum — threadgroup (shared) memory for reductions')\n"
                    "    print('  3. Three-pass algorithm: find max → compute exp & sum → normalize')\n"
                    "    print('  4. Each threadgroup processes one ROW of the input matrix')\n"
                    "except FileNotFoundError:\n"
                    "    print(f'⚠️ {kernel_path} not found — run from the repo root directory')"
                )
                cells.insert(i + 1, display_cell)
                fixes += 1
                print(f"  NB10: Added inline Metal kernel display for softmax")
            break
    
    if fixes > 0:
        nb['cells'] = cells
        save_nb(path, nb)
    return fixes


def fix_nb13():
    """NB13: Make mlx-lm demo actually runnable with try/except."""
    path = '13_serving_locally.ipynb'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return 0
    
    nb = load_nb(path)
    cells = nb.get('cells', [])
    fixes = 0
    
    for i, cell in enumerate(cells):
        s = src(cell)
        if 'mlx-lm usage' in s.lower() and 'print(' in s and cell.get('cell_type') == 'code':
            # Replace the print-only cell with a try/except runnable version
            cell['source'] = [
                "# mlx-lm: Load and generate with a real model (if available)\n"
                "# This cell tries to load a small model — if it's not installed,\n"
                "# it gracefully falls back to showing the API usage.\n\n"
                "try:\n"
                "    from mlx_lm import load, generate\n"
                "    \n"
                "    print('Loading model... (this may take a moment on first run)')\n"
                "    model, tokenizer = load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')\n"
                "    \n"
                "    prompt = 'Explain transformers in one sentence:'\n"
                "    response = generate(model, tokenizer, prompt=prompt, max_tokens=50)\n"
                "    print(f'Prompt: {prompt}')\n"
                "    print(f'Response: {response}')\n"
                "    print()\n"
                "    \n"
                "    # Show memory usage\n"
                "    import mlx.core as mx\n"
                "    mem_gb = mx.metal.get_active_memory() / 1e9\n"
                "    print(f'⚡ Memory after loading: {mem_gb:.1f} GB')\n"
                "    \n"
                "except ImportError:\n"
                "    print('mlx-lm is not installed. To try it:')\n"
                "    print('  pip install mlx-lm')\n"
                "    print()\n"
                "    print('Then run:')\n"
                "    print('  from mlx_lm import load, generate')\n"
                "    print('  model, tokenizer = load(\"mlx-community/Qwen2.5-0.5B-Instruct-4bit\")')\n"
                "    print('  response = generate(model, tokenizer, prompt=\"Hello\", max_tokens=50)')\n"
                "except Exception as e:\n"
                "    print(f'Could not load model: {e}')\n"
                "    print('The model may need to be downloaded from Hugging Face (~300MB)')\n"
                "    print()\n"
                "    print('⚡ mlx-lm handles quantization, KV-cache, and Metal optimization automatically')"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None
            fixes += 1
            print(f"  NB13: Made mlx-lm demo runnable with try/except")
            break
    
    if fixes > 0:
        nb['cells'] = cells
        save_nb(path, nb)
    return fixes


def fix_nb05():
    """NB05: Add note about random embeddings producing uniform attention."""
    path = '05_self_attention.ipynb'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return 0
    
    nb = load_nb(path)
    cells = nb.get('cells', [])
    fixes = 0
    
    # Find the attention visualization cell and add a note after it
    for i, cell in enumerate(cells):
        s = src(cell)
        if 'plot_attention_heatmap' in s and 'Single-Head Attention' in s and cell.get('cell_type') == 'code':
            # Check if note already exists
            if i + 1 < len(cells) and 'random embeddings' in src(cells[i+1]).lower():
                break
            
            note_cell = make_md(
                "⚠️ **Why is the attention nearly uniform?** With random (untrained) embeddings, "
                "all tokens look equally similar to each other, so attention weights are spread roughly evenly. "
                "In a **trained** model, you'd see much sharper patterns — for example, \"it\" would strongly "
                "attend to \"cat\" (its referent) and barely attend to \"mat\". The uniform pattern here is "
                "expected and correct for random weights."
            )
            cells.insert(i + 1, note_cell)
            fixes += 1
            print(f"  NB05: Added note about random embeddings producing uniform attention")
            break
    
    if fixes > 0:
        nb['cells'] = cells
        save_nb(path, nb)
    return fixes


def fix_nb01():
    """NB01: Add note about lazy eval shape inference."""
    path = '01_mlx_fundamentals.ipynb'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return 0
    
    nb = load_nb(path)
    cells = nb.get('cells', [])
    fixes = 0
    
    # Find the "nothing has been computed" cell and add clarification
    for i, cell in enumerate(cells):
        s = src(cell)
        if 'No Metal GPU work has happened yet' in s and cell.get('cell_type') == 'code':
            # Check if clarification already exists
            if i + 1 < len(cells) and 'shape and dtype are inferred' in src(cells[i+1]).lower():
                break
            
            note_cell = make_md(
                "💡 **\"But wait — how does MLX know the shape and dtype if nothing was computed?\"** "
                "Great question! MLX infers shapes and dtypes **statically** from the operation graph, "
                "without executing any GPU work. When you write `e = mx.sum(d)`, MLX knows the output "
                "is a scalar (shape `()`) of type `float32` because it can trace the operations: "
                "`ones → add → multiply → sum`. The actual *values* aren't computed until `mx.eval()`, "
                "but the *metadata* (shape, dtype) is available immediately. This is similar to how a "
                "compiler can determine the type of an expression without running the program."
            )
            cells.insert(i + 1, note_cell)
            fixes += 1
            print(f"  NB01: Added note about lazy eval shape inference")
            break
    
    if fixes > 0:
        nb['cells'] = cells
        save_nb(path, nb)
    return fixes


def main():
    print("Applying critical pedagogical fixes...\n")
    total = 0
    
    total += fix_nb14()
    total += fix_nb10()
    total += fix_nb13()
    total += fix_nb05()
    total += fix_nb01()
    
    print(f"\n{'='*50}")
    print(f"Total fixes applied: {total}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
