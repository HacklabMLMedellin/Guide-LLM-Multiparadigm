"""
A minimal in-memory dataset over tokenized text. Given the toy-scale corpus
this project trains on, a full `torch.utils.data.DataLoader` would be
overkill -- random contiguous windows sampled directly from the tokenized
tensor are enough, and keep the training pipeline's `dataset()` stage
trivial to read.
"""

from __future__ import annotations

from typing import Tuple

import torch


class CharDataset:
    def __init__(self, token_ids: list, context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        if len(self.data) <= context_length + 1:
            raise ValueError(
                f"Corpus ({len(self.data)} tokens) must be longer than "
                f"context_length + 1 ({context_length + 1})."
            )

    def get_autoregressive_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.context_length - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + self.context_length] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.context_length + 1] for s in starts])
        return x, y

    def get_diffusion_batch(self, batch_size: int) -> torch.Tensor:
        max_start = len(self.data) - self.context_length
        starts = torch.randint(0, max_start, (batch_size,))
        return torch.stack([self.data[s : s + self.context_length] for s in starts])
