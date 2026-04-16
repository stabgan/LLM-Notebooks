"""Plotting helpers for the LLM Learning Notebook series."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 10})


def plot_attention_heatmap(
    weights: np.ndarray | "mx.array",
    tokens: list[str] | None = None,
    title: str = "Attention Weights",
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
) -> plt.Figure | None:
    """Plot a 2-D attention weight matrix as a heatmap.

    Args:
        weights: 2-D array of shape (seq_len, seq_len).
        tokens: Optional token labels for axes.
        title: Plot title.
        ax: Existing axes to draw on; creates a new figure if None.
        cmap: Matplotlib colormap name.

    Returns:
        The Figure if a new one was created, else None.
    """
    w = np.array(weights) if not isinstance(weights, np.ndarray) else weights
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax)
    if created_fig:
        plt.tight_layout()
        return fig
    return None


def plot_loss_curve(
    losses: list[float],
    val_losses: list[float] | None = None,
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
) -> plt.Figure:
    """Plot training (and optional validation) loss over time.

    Args:
        losses: Training loss values per step.
        val_losses: Optional validation loss values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, label="Train", linewidth=1.5)
    if val_losses is not None:
        ax.plot(val_losses, label="Validation", linewidth=1.5, linestyle="--")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_token_probabilities(
    probs: np.ndarray | "mx.array",
    tokens: list[str],
    title: str = "Token Probabilities",
    top_k: int = 20,
) -> plt.Figure:
    """Bar chart of token probabilities (shows top-k tokens).

    Args:
        probs: 1-D probability array over vocabulary.
        tokens: Token labels corresponding to each index.
        title: Plot title.
        top_k: Number of top tokens to display.

    Returns:
        The matplotlib Figure.
    """
    p = np.array(probs) if not isinstance(probs, np.ndarray) else probs
    top_idx = np.argsort(p)[::-1][:top_k]
    top_probs = p[top_idx]
    top_tokens = [tokens[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(top_tokens)), top_probs, color="steelblue")
    ax.set_yticks(range(len(top_tokens)))
    ax.set_yticklabels(top_tokens)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_embeddings_2d(
    embeddings: np.ndarray | "mx.array",
    labels: list[str] | None = None,
    title: str = "2-D Embedding Projection",
    method: str = "pca",
) -> plt.Figure:
    """Scatter plot of embeddings projected to 2-D.

    Uses PCA (default) or t-SNE for dimensionality reduction.

    Args:
        embeddings: 2-D array of shape (n_items, d_model).
        labels: Optional text labels for each point.
        title: Plot title.
        method: "pca" or "tsne".

    Returns:
        The matplotlib Figure.
    """
    emb = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings

    if emb.shape[1] > 2:
        if method == "pca":
            # Simple PCA via SVD — no sklearn dependency
            mean = emb.mean(axis=0)
            centered = emb - mean
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ Vt[:2].T
        else:
            try:
                from sklearn.manifold import TSNE
                coords = TSNE(n_components=2, perplexity=min(30, len(emb) - 1)).fit_transform(emb)
            except ImportError:
                # Fall back to PCA if sklearn not available
                mean = emb.mean(axis=0)
                centered = emb - mean
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                coords = centered @ Vt[:2].T
    else:
        coords = emb

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.7)
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
