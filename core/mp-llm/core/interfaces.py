"""
Abstract interfaces (SOLID: Interface Segregation + Dependency Inversion).

Every LLM component (Embeddings, Attention, FFN, Sampling, ...) implements a
small, paradigm-agnostic interface. Concrete Classical/Quantum/Photonic/
Thermodynamic implementations live under `paradigms/<paradigm>/components.py`
and are wired in via Dependency Injection (constructors take their backend/
adapter as an argument rather than constructing it themselves).

`AttentionStrategy` is the Strategy-Pattern base referenced in the spec;
the other component interfaces follow the identical shape so every module
is swappable the same way.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from core.types import Paradigm


class ParadigmComponent(abc.ABC):
    """Base interface shared by every swappable LLM component.

    Concrete subclasses are plain nn.Module-compatible objects (most inherit
    torch.nn.Module too, since even quantum/photonic/thermodynamic modules
    are ultimately wired into a torch autograd graph or invoked from one).
    """

    paradigm: Paradigm

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        ...

    def describe(self) -> str:
        """Human-readable one-liner used in benchmark/selection reports."""
        return f"{self.__class__.__name__}({self.paradigm.value})"


class AttentionStrategy(ParadigmComponent):
    """Strategy interface for attention. Concrete: ClassicalAttention,
    QuantumAttention, PhotonicAttention, ThermodynamicAttention.
    """

    @abc.abstractmethod
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        ...


class FeedForwardStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class EmbeddingStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, token_ids: Tensor) -> Tensor:
        ...


class PositionalEncodingStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class LayerNormStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class OutputProjectionStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class SamplingStrategy(ParadigmComponent):
    """Turns logits into a chosen next token (or a full denoising step, for
    the diffusion model). Concrete: ThermodynamicSampling (energy-based),
    QuantumSampling (quantum-randomness multinomial), ClassicalSampling.
    """

    @abc.abstractmethod
    def forward(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        ...


class ProbabilityEstimationStrategy(ParadigmComponent):
    """Estimates a probability / expectation from logits or hidden state.
    Natural fit for Quantum (measurement statistics of a variational circuit
    map naturally onto probability estimates).
    """

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class RandomnessStrategy(ParadigmComponent):
    """Randomness generation. Quantum measurement outcomes are the textbook
    source of information-theoretically justified randomness.
    """

    @abc.abstractmethod
    def forward(self, shape: torch.Size) -> Tensor:
        ...


class SearchStrategy(ParadigmComponent):
    """Discrete search / optimization (e.g. beam-like candidate selection).
    Thermodynamic energy-based search reframes "search" as "find a low
    energy state", which is what Gibbs sampling / simulated annealing does.
    """

    @abc.abstractmethod
    def forward(self, candidates: Tensor, energy_fn=None) -> Tensor:
        ...


class EnergyBasedSamplingStrategy(ParadigmComponent):
    @abc.abstractmethod
    def forward(self, logits: Tensor, steps: int = 10) -> Tensor:
        ...


class MatMulStrategy(ParadigmComponent):
    """Generic linear-algebra building block. Classical: torch.nn.Linear.
    Photonic: a physically-simulated MZI mesh (PhotonTorch) realizing a
    unitary matrix multiply via optical interference.
    """

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...


class OptimizerStrategy(abc.ABC):
    @abc.abstractmethod
    def step(self) -> None:
        ...

    @abc.abstractmethod
    def zero_grad(self) -> None:
        ...


class TokenizerInterface(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str) -> list:
        ...

    @abc.abstractmethod
    def decode(self, ids: list) -> str:
        ...

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        ...


class LLMInterface(abc.ABC):
    """Common interface every top-level model (ClassicalLLM, ThermodynamicLLM,
    QuantumLLM, PhotonicLLM, HybridLLM) exposes, per the spec.
    """

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        ...

    @abc.abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    def save(self, path: str) -> None:
        ...

    @abc.abstractmethod
    def load(self, path: str) -> None:
        ...

    @abc.abstractmethod
    def benchmark(self, *args, **kwargs) -> Dict[str, Any]:
        ...
