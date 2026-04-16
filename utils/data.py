"""Dataset loaders for the LLM Learning Notebook series."""

from __future__ import annotations

from typing import Iterator


class TextDataset:
    """Load a text file, tokenize it, and yield (input_ids, target_ids) batches as MLX arrays.

    Supports character-level tokenization by default.

    Args:
        text: Raw text string to tokenize.
        seq_len: Sequence length for each sample.
        batch_size: Number of sequences per batch.
        tokenizer: Optional external tokenizer with ``encode(str) -> list[int]``
            interface. When *None*, character-level tokenization is used.
    """

    def __init__(
        self,
        text: str,
        seq_len: int = 128,
        batch_size: int = 32,
        tokenizer: object | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.batch_size = batch_size

        if tokenizer is not None:
            self.token_ids: list[int] = tokenizer.encode(text)  # type: ignore[union-attr]
            self.vocab_size: int = max(self.token_ids) + 1
        else:
            # Character-level tokenization
            chars = sorted(set(text))
            self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
            self.itos: dict[int, str] = {i: ch for ch, i in self.stoi.items()}
            self.token_ids = [self.stoi[ch] for ch in text]
            self.vocab_size = len(chars)

    def __len__(self) -> int:
        """Number of complete batches available."""
        n_samples = (len(self.token_ids) - 1) // self.seq_len
        return max(n_samples // self.batch_size, 0)

    def __iter__(self) -> Iterator[tuple]:
        """Yield ``(input_ids, target_ids)`` batches as MLX arrays.

        Each tensor has shape ``(batch_size, seq_len)``.
        ``target_ids`` is ``input_ids`` shifted right by one position.
        """
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise ImportError(
                "mlx is required for TextDataset iteration. "
                "Install it with: pip install mlx"
            ) from exc

        ids = self.token_ids
        n_samples = (len(ids) - 1) // self.seq_len
        n_batches = n_samples // self.batch_size

        for b in range(n_batches):
            batch_x: list[list[int]] = []
            batch_y: list[list[int]] = []
            for s in range(self.batch_size):
                idx = (b * self.batch_size + s) * self.seq_len
                batch_x.append(ids[idx : idx + self.seq_len])
                batch_y.append(ids[idx + 1 : idx + 1 + self.seq_len])
            yield mx.array(batch_x), mx.array(batch_y)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string (character-level only)."""
        if not hasattr(self, "itos"):
            raise RuntimeError("decode() is only available with character-level tokenization")
        return "".join(self.itos[i] for i in ids)
