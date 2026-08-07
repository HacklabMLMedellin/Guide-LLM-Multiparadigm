"""
Quantum paradigm: components implemented via PennyLane variational circuits,
through `adapters.quantum_adapter.QuantumCircuitAdapter`.

Per the spec's scheduler example, Quantum is best suited to Probability
Estimation and Randomness Generation (measurement statistics are a natural,
information-theoretic source of both), and can optionally modulate Attention
scores. Quantum simulation here is exact statevector simulation
(`default.qubit`), not real QPU sampling noise.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.quantum_adapter import QuantumCircuitAdapter
from core.interfaces import (
    AttentionStrategy,
    ProbabilityEstimationStrategy,
    RandomnessStrategy,
    SamplingStrategy,
)
from core.types import Paradigm


class QuantumProbabilityEstimation(nn.Module, ProbabilityEstimationStrategy):
    """Maps hidden-state features through a small variational circuit and
    reads out Pauli-Z expectation values, rescaled into a valid probability
    distribution. This is the spec's "Probability Estimation -> Quantum"
    mapping: expectation values of a parameterized circuit are themselves
    genuine probability-like quantities (they come from Born-rule
    measurement statistics), rather than an arbitrary classical nonlinearity
    dressed up as one.
    """

    paradigm = Paradigm.QUANTUM

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuitAdapter(n_qubits, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squash features into valid rotation angles [0, pi] before encoding.
        angles = torch.sigmoid(x[..., : self.n_qubits]) * math.pi
        expvals = self.circuit(angles)  # in [-1, 1]
        probs = (expvals + 1) / 2
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class QuantumRandomness(nn.Module, RandomnessStrategy):
    """Randomness Generation -> Quantum: draws bits from measurement
    statistics of a variational circuit rather than a classical PRNG.
    """

    paradigm = Paradigm.QUANTUM

    def __init__(self, n_qubits: int = 8, n_layers: int = 1):
        super().__init__()
        self.circuit = QuantumCircuitAdapter(n_qubits, n_layers)
        self.n_qubits = n_qubits

    def forward(self, shape: torch.Size) -> torch.Tensor:
        n = 1
        for d in shape:
            n *= d
        n_draws = (n + self.n_qubits - 1) // self.n_qubits
        bits = self.circuit.sample_bits(n_draws).reshape(-1)[:n]
        return bits.reshape(shape).float()


class QuantumSampling(nn.Module, SamplingStrategy):
    """Uses quantum-measurement randomness (instead of a classical PRNG) to
    drive multinomial sampling from a softmax distribution.
    """

    paradigm = Paradigm.QUANTUM

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.randomness = QuantumRandomness(n_qubits=n_qubits, n_layers=1)

    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        u = self.randomness(torch.Size(probs.shape[:-1])).unsqueeze(-1).clamp(0, 0.999999)
        return (cdf < u).sum(dim=-1)


class QuantumAttention(nn.Module, AttentionStrategy):
    """Uses a variational circuit to produce a learned, input-dependent
    per-head gating vector that reweights classical scaled-dot-product
    attention output. A full O(seq^2) attention score matrix computed one
    quantum circuit evaluation at a time would not be tractable even at toy
    scale, so quantum involvement here is scoped to where circuit evaluation
    counts stay small: one evaluation per sequence position, not per
    (query, key) pair.
    """

    paradigm = Paradigm.QUANTUM

    def __init__(self, d_model: int, n_heads: int, n_qubits: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_circuit = QuantumCircuitAdapter(n_qubits, n_layers=2)
        self.n_qubits = n_qubits
        self.gate_proj_in = nn.Linear(d_model, n_qubits)
        self.gate_proj_out = nn.Linear(n_qubits, d_model)

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
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)

        gate_in = torch.sigmoid(self.gate_proj_in(x)) * math.pi  # (b, t, n_qubits)
        gate_q = self.gate_circuit(gate_in)  # (b, t, n_qubits) in [-1, 1]
        gate = torch.sigmoid(self.gate_proj_out(gate_q))  # (b, t, d_model)

        return self.out_proj(out * gate)
