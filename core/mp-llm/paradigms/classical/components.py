"""
Classical paradigm: plain PyTorch implementations of every component
strategy. This is the reference implementation every other paradigm is
benchmarked against.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import (
    AttentionStrategy,
    EmbeddingStrategy,
    EnergyBasedSamplingStrategy,
    FeedForwardStrategy,
    LayerNormStrategy,
    MatMulStrategy,
    OptimizerStrategy,
    OutputProjectionStrategy,
    PositionalEncodingStrategy,
    ProbabilityEstimationStrategy,
    RandomnessStrategy,
    SamplingStrategy,
    SearchStrategy,
)
from core.types import Paradigm


class ClassicalEmbedding(nn.Module, EmbeddingStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids)


class ClassicalPositionalEncoding(nn.Module, PositionalEncodingStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ClassicalLinear(nn.Module, MatMulStrategy):
    """The classical baseline for the "MatMul" component."""

    paradigm = Paradigm.CLASSICAL

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ClassicalAttention(nn.Module, AttentionStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class ClassicalFeedForward(nn.Module, FeedForwardStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalLayerNorm(nn.Module, LayerNormStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ClassicalOutputProjection(nn.Module, OutputProjectionStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ClassicalSampling(nn.Module, SamplingStrategy):
    """Standard softmax-temperature multinomial sampling."""

    paradigm = Paradigm.CLASSICAL

    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class ClassicalProbabilityEstimation(nn.Module, ProbabilityEstimationStrategy):
    paradigm = Paradigm.CLASSICAL

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class ClassicalRandomness(nn.Module, RandomnessStrategy):
    paradigm = Paradigm.CLASSICAL

    def forward(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape)


class ClassicalSearch(nn.Module, SearchStrategy):
    """Greedy argmax search -- the classical baseline."""

    paradigm = Paradigm.CLASSICAL

    def forward(self, candidates: torch.Tensor, energy_fn=None) -> torch.Tensor:
        if energy_fn is not None:
            energies = energy_fn(candidates)
            return candidates[torch.argmin(energies)]
        return candidates[torch.argmax(candidates[..., 0])] if candidates.ndim > 1 else candidates


class ClassicalEnergyBasedSampling(nn.Module, EnergyBasedSamplingStrategy):
    """Classical fallback for energy-based sampling: simple simulated
    annealing over logits via repeated Gumbel-softmax perturbation.
    """

    paradigm = Paradigm.CLASSICAL

    def forward(self, logits: torch.Tensor, steps: int = 10) -> torch.Tensor:
        state = logits.clone()
        for i in range(steps):
            temp = max(1.0 - i / steps, 0.05)
            noise = -torch.log(-torch.log(torch.rand_like(state) + 1e-9) + 1e-9)
            state = (logits + noise * temp)
        return F.softmax(state, dim=-1)


class ClassicalOptimizer(OptimizerStrategy):
    paradigm = Paradigm.CLASSICAL

    def __init__(self, parameters, lr: float = 3e-4):
        self.opt = torch.optim.AdamW(list(parameters), lr=lr)

    def step(self) -> None:
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
