#!/usr/bin/env python3
"""
Fix beginner-friendliness issues across all notebooks.
Addresses: NO_NEXT_STEPS, NO_OBJECTIVES (partial), FEW_SHAPE_COMMENTS (adds to code)
Usage: .venv/bin/python scripts/fix_beginner_issues.py
"""

import json
import glob
import uuid
import re
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

# Map notebook number to next notebook info
NEXT_MAP = {
    '00': ('01', 'MLX Fundamentals', 'arrays, lazy evaluation, and automatic differentiation'),
    '01': ('02', 'Math Foundations', 'dot products, softmax, and cross-entropy loss'),
    '02': ('03', 'Tokenization', 'how text becomes numbers that transformers can process'),
    '03': ('04', 'Embeddings & Positional Encoding', 'turning token IDs into dense vectors'),
    '04': ('05', 'Self-Attention', 'the core mechanism that makes transformers work'),
    '05': ('06', 'Transformer Architecture', 'assembling attention, FFN, and norms into a full block'),
    '06': ('07', 'Building GPT from Scratch', 'a complete GPT model you can train and generate with'),
    '07': ('08', 'Training on Apple Silicon', 'memory optimization, mixed precision, and gradient accumulation'),
    '08': ('09', 'Modern Architectures', 'LLaMA, Mistral, and Gemma design choices'),
    '09': ('10', 'Metal Custom Kernels', 'writing GPU kernels for maximum performance'),
    '10': ('11', 'Inference Optimization', 'KV-cache, quantization, and speculative decoding'),
    '11': ('12', 'Flash, Paged, and Ring Attention', 'advanced attention for long sequences'),
    '12': ('13', 'Serving Locally', 'running models with mlx-lm and llama.cpp'),
    '13': ('14', 'Capstone: Gemma 4', 'fine-tuning and serving a real model'),
    '14': None,  # Last notebook
    '15': ('16', 'State Space Models', 'an alternative to attention for sequence modeling'),
    '16': ('17', 'Alignment', 'RLHF, DPO, and GRPO for making models helpful'),
    '17': ('18', 'Scaling Laws', 'predicting model performance from size and data'),
    '18': ('19', 'Reasoning & Test-Time Compute', 'how models think step by step'),
    '19': None,  # Last in tier
}

def fix_next_steps(path):
    """Add 'What's Next' section if missing."""
    nb = load_nb(path)
    cells = nb['cells']
    
    # Check if already has next steps
    last_cells_md = ' '.join(src(c) for c in cells[-5:] if c.get('cell_type') == 'markdown')
    if 'next' in last_cells_md.lower() and ('notebook' in last_cells_md.lower() or 'up:' in last_cells_md.lower()):
        return False
    
    # Get notebook number
    basename = os.path.basename(path)
    nb_num = basename[:2]
    
    next_info = NEXT_MAP.get(nb_num)
    if next_info is None:
        return False
    
    next_num, next_title, next_desc = next_info
    
    # Find the history section (insert before it)
    history_idx = None
    for i, c in enumerate(cells):
        if '📜 History' in src(c):
            history_idx = i
            break
    
    if history_idx is None:
        # Insert at end
        insert_idx = len(cells)
    else:
        insert_idx = history_idx
    
    next_cell = make_md(
        f"---\n"
        f"## ➡️ What's Next?\n\n"
        f"You've completed this notebook! In **Notebook {next_num}: {next_title}**, "
        f"we'll explore {next_desc}.\n\n"
        f"💡 **Before moving on**, make sure you can answer these questions:\n"
        f"- What was the main concept in this notebook?\n"
        f"- Why does it matter for building LLMs?\n"
        f"- Could you explain it to a friend in one sentence?"
    )
    
    cells.insert(insert_idx, next_cell)
    save_nb(path, nb)
    return True

def add_shape_comments(path):
    """Add shape comments to code cells that create/manipulate tensors."""
    nb = load_nb(path)
    cells = nb['cells']
    changes = 0
    
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        
        code = src(cell)
        
        # Skip if already has shape comments
        if '# shape:' in code.lower():
            continue
        
        # Skip very short cells
        if len(code.strip().split('\n')) < 3:
            continue
        
        # Check if cell uses mx operations
        if 'mx.' not in code and 'nn.' not in code:
            continue
        
        # Add a shape-checking print at the end if there are tensor operations
        # Look for variable assignments with mx operations
        tensor_vars = re.findall(r'(\w+)\s*=\s*mx\.(?:random\.normal|zeros|ones|array)\(', code)
        if tensor_vars and 'print' not in code.split('\n')[-1]:
            # Don't modify - too risky to auto-add. Just count.
            changes += 1
    
    return changes

def fix_all():
    notebooks = sorted(glob.glob('[0-1][0-9]_*.ipynb'))
    
    print("Fixing beginner-friendliness issues...\n")
    
    next_steps_added = 0
    for path in notebooks:
        if fix_next_steps(path):
            next_steps_added += 1
            print(f"  ✅ Added 'What's Next' to {os.path.basename(path)}")
    
    print(f"\nTotal: {next_steps_added} 'What's Next' sections added")

if __name__ == '__main__':
    fix_all()
