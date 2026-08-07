"""
Shared types/enums used across the Multi-Paradigm LLM (MP-LLM) project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Paradigm(str, Enum):
    """The four computational paradigms the architecture can pick from."""

    CLASSICAL = "classical"
    THERMODYNAMIC = "thermodynamic"
    QUANTUM = "quantum"
    PHOTONIC = "photonic"


class ComponentKind(str, Enum):
    """Every modular LLM component the scheduler/factory knows how to build."""

    TOKENIZER = "tokenizer"
    VOCABULARY = "vocabulary"
    EMBEDDING = "embedding"
    POSITIONAL_ENCODING = "positional_encoding"
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    LAYER_NORM = "layer_norm"
    RESIDUAL = "residual"
    ACTIVATION = "activation"
    OUTPUT_PROJECTION = "output_projection"
    SOFTMAX = "softmax"
    SAMPLING = "sampling"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    CHECKPOINTING = "checkpointing"
    TRAINING_LOOP = "training_loop"
    INFERENCE_ENGINE = "inference_engine"
    MATMUL = "matmul"
    PROBABILITY_ESTIMATION = "probability_estimation"
    SEARCH = "search"
    ROUTING = "routing"
    RANDOMNESS = "randomness_generation"
    MEMORY_RETRIEVAL = "memory_retrieval"
    ENERGY_SAMPLING = "energy_based_sampling"


@dataclass
class ParadigmChoice:
    """A single scheduler decision: which paradigm handles a given component,
    and why. This is what lets the system "explain" its own architecture.
    """

    component: ComponentKind
    paradigm: Paradigm
    reason: str
    estimated_speedup: Optional[float] = None


@dataclass
class HardwareFactors:
    """Configurable multipliers converting *simulated* wall-clock time into a
    *theoretical* future-hardware estimate. These are speculative, openly
    configurable placeholders -- not measurements of real quantum, photonic,
    or thermodynamic hardware, since none of that hardware is being executed
    here. Simulation happens on classical CPU/GPU via PennyLane, PhotonTorch,
    and THRML regardless of which "paradigm" is logically selected.
    """

    classical: float = 1.0
    quantum: float = 0.08
    photonic: float = 0.04
    thermodynamic: float = 0.12

    def factor_for(self, paradigm: Paradigm) -> float:
        return {
            Paradigm.CLASSICAL: self.classical,
            Paradigm.QUANTUM: self.quantum,
            Paradigm.PHOTONIC: self.photonic,
            Paradigm.THERMODYNAMIC: self.thermodynamic,
        }[paradigm]


@dataclass
class TimingRecord:
    """One timing observation, in both simulated and hardware-adjusted form."""

    name: str
    paradigm: Paradigm
    simulated_seconds: float
    hardware_factor: float
    estimated_hardware_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.estimated_hardware_seconds = self.simulated_seconds * self.hardware_factor
