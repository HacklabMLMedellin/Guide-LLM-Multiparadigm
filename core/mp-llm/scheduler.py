"""
Strategy Pattern: the paradigm scheduler.

`SchedulingPolicy` is the strategy interface; `DefaultSchedulingPolicy`
implements the example mapping given in the spec (Embedding -> Classical,
Attention -> Classical or Photonic, etc.), each with a plain-English reason
and a rough estimated speedup. `HybridScheduler` is the context object that
holds a policy and can have it swapped out at runtime
(`scheduler.set_policy(...)`) without touching any calling code -- exactly
the point of the Strategy Pattern.

Speedup estimates are illustrative, derived from `HARDWARE_FACTORS`
(simulated_time * factor) once real components have actually been timed;
until then a component gets a static placeholder estimate so the scheduler
can still explain *why* it would pick a paradigm before anything has run.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.types import ComponentKind, HardwareFactors, Paradigm, ParadigmChoice


# Plain-English justification for each (component, paradigm) pairing this
# project actually implements. Kept as data (not scattered string literals)
# so a new policy can reuse or override individual reasons.
_REASONS: Dict[ComponentKind, Dict[Paradigm, str]] = {
    ComponentKind.EMBEDDING: {
        Paradigm.CLASSICAL: "Embedding lookup is a simple indexed memory read; "
        "there is no linear-algebra or sampling advantage another paradigm could "
        "offer here, so classical dense storage is strictly the cheapest option.",
    },
    ComponentKind.POSITIONAL_ENCODING: {
        Paradigm.CLASSICAL: "A fixed or learned per-position vector; a static lookup, "
        "not a computation worth specializing.",
    },
    ComponentKind.ATTENTION: {
        Paradigm.CLASSICAL: "Default, always-available baseline: dense scaled dot-"
        "product attention on classical hardware.",
        Paradigm.PHOTONIC: "Large matrix multiplications dominate this layer. Photonic "
        "interference can theoretically accelerate the linear-algebra portion of "
        "attention (the Q/K/V and output projections) once real photonic hardware "
        "exists, at the cost of today's simulation overhead.",
        Paradigm.QUANTUM: "A small variational circuit can act as a learned, non-"
        "classical gating function over attention output, exploiting the circuit's "
        "expressivity per parameter -- explored here as a research variant rather "
        "than a speed optimization.",
        Paradigm.THERMODYNAMIC: "Frames 'which keys matter' as an Ising model over "
        "key positions and samples a physically-motivated sparse attention pattern "
        "via Gibbs sampling, instead of dense deterministic softmax weighting -- a "
        "discrete selection problem, which is thermodynamic sampling's strength.",
    },
    ComponentKind.FEED_FORWARD: {
        Paradigm.CLASSICAL: "Dense two-layer MLP; the standard, always-available choice.",
        Paradigm.PHOTONIC: "The inner (dimension-preserving) transform is pure linear "
        "algebra and can, in principle, be realized optically the same way attention's "
        "projections can.",
    },
    ComponentKind.MATMUL: {
        Paradigm.CLASSICAL: "General dense GEMM; always available and, on today's real "
        "hardware, almost always fastest.",
        Paradigm.PHOTONIC: "Matrix multiplication is exactly what a passive "
        "interferometer mesh computes physically (light interference realizing a "
        "unitary transform), which is the theoretical basis for expecting a future "
        "photonic accelerator to win here.",
    },
    ComponentKind.LAYER_NORM: {
        Paradigm.CLASSICAL: "Elementwise normalization; a memory-bandwidth-bound op "
        "with no known paradigm-specific advantage.",
    },
    ComponentKind.OUTPUT_PROJECTION: {
        Paradigm.CLASSICAL: "A single large dense projection to vocabulary size; "
        "classical GEMM is the practical choice at any real vocabulary size.",
    },
    ComponentKind.OPTIMIZER: {
        Paradigm.CLASSICAL: "Gradient-based optimization (AdamW) is inherently a "
        "classical, sequential numerical procedure operating on the entire "
        "parameter vector; none of the other three paradigms as used here expose "
        "a native optimizer.",
    },
    ComponentKind.PROBABILITY_ESTIMATION: {
        Paradigm.CLASSICAL: "Softmax normalization; simple and exact.",
        Paradigm.QUANTUM: "Measurement statistics of a parameterized circuit are "
        "themselves genuine Born-rule probabilities, which is a more direct match "
        "for 'estimate a probability' than an arbitrary classical nonlinearity.",
    },
    ComponentKind.SEARCH: {
        Paradigm.CLASSICAL: "Greedy argmax search; deterministic and cheap.",
        Paradigm.THERMODYNAMIC: "Energy-based probabilistic search naturally maps to "
        "thermodynamic sampling: candidates are framed as system states and the "
        "sampler settles toward low-energy (favoured) ones, the same computation "
        "Extropic's thermodynamic hardware targets physically.",
    },
    ComponentKind.ROUTING: {
        Paradigm.CLASSICAL: "Control flow / dispatch logic; inherently a classical, "
        "discrete decision, not a numerical computation to specialize.",
    },
    ComponentKind.RANDOMNESS: {
        Paradigm.CLASSICAL: "A pseudo-random number generator; fast and sufficient "
        "when no genuine entropy source is required.",
        Paradigm.QUANTUM: "Measurement outcomes of a quantum circuit are the textbook "
        "source of randomness that is not merely pseudo-random, since (in a real "
        "quantum device) they derive from fundamentally probabilistic measurement.",
    },
    ComponentKind.MEMORY_RETRIEVAL: {
        Paradigm.CLASSICAL: "Associative/key-value lookup over stored representations; "
        "a memory-access pattern classical hardware is built for.",
    },
    ComponentKind.SAMPLING: {
        Paradigm.CLASSICAL: "Inverse-CDF multinomial sampling from softmax "
        "probabilities; simple and exact.",
        Paradigm.THERMODYNAMIC: "Energy-based probabilistic sampling naturally maps to "
        "thermodynamic optimization: converting logits into biases on an Ising "
        "model and letting the system settle reframes 'sample a token' as "
        "'find a low-energy state', the native operation of thermodynamic hardware.",
        Paradigm.QUANTUM: "Uses genuine quantum-measurement randomness to drive the "
        "same inverse-CDF sampling procedure, rather than a classical PRNG.",
    },
    ComponentKind.ENERGY_SAMPLING: {
        Paradigm.CLASSICAL: "Simulated-annealing-style Gumbel perturbation; a classical "
        "approximation of energy-based sampling.",
        Paradigm.THERMODYNAMIC: "This *is* the native operation of a thermodynamic "
        "sampler: directly draw from the Boltzmann distribution of an energy "
        "function via Gibbs sampling, rather than approximating it.",
    },
}

# Static, illustrative speedup estimates shown before any real benchmark has
# run (once `benchmarking.report` has real TimingRecords for a component,
# prefer those numbers instead -- see `HybridScheduler.explain`).
_STATIC_SPEEDUP_ESTIMATES: Dict[Paradigm, float] = {
    Paradigm.CLASSICAL: 1.0,
    Paradigm.QUANTUM: 3.0,
    Paradigm.PHOTONIC: 23.0,
    Paradigm.THERMODYNAMIC: 8.0,
}


class SchedulingPolicy(abc.ABC):
    """Strategy interface: given a component kind (and light context), decide
    which paradigm should execute it, with a reason.
    """

    @abc.abstractmethod
    def choose(self, component: ComponentKind, **context) -> ParadigmChoice:
        ...


class DefaultSchedulingPolicy(SchedulingPolicy):
    """Implements the example strategy table from the spec:

        Embedding             -> Classical
        Attention              -> Classical or Photonic
        Matrix Multiplication  -> Photonic
        Sampling                -> Thermodynamic
        Optimization            -> Classical
        Probability Estimation  -> Quantum
        Search                  -> Thermodynamic
        Routing                 -> Classical
        Normalization           -> Classical
        Randomness Generation   -> Quantum
        Memory Retrieval        -> Classical
        Energy-based Sampling   -> Thermodynamic

    `prefer_photonic_attention`/`prefer_quantum_attention` let a caller pick
    which of Attention's two spec-sanctioned options ("Classical or
    Photonic") to use; both remain fully implemented and swappable.
    """

    _DEFAULT_MAP: Dict[ComponentKind, Paradigm] = {
        ComponentKind.EMBEDDING: Paradigm.CLASSICAL,
        ComponentKind.POSITIONAL_ENCODING: Paradigm.CLASSICAL,
        ComponentKind.ATTENTION: Paradigm.CLASSICAL,
        ComponentKind.FEED_FORWARD: Paradigm.CLASSICAL,
        ComponentKind.MATMUL: Paradigm.PHOTONIC,
        ComponentKind.LAYER_NORM: Paradigm.CLASSICAL,
        ComponentKind.OUTPUT_PROJECTION: Paradigm.CLASSICAL,
        ComponentKind.SAMPLING: Paradigm.THERMODYNAMIC,
        ComponentKind.OPTIMIZER: Paradigm.CLASSICAL,
        ComponentKind.PROBABILITY_ESTIMATION: Paradigm.QUANTUM,
        ComponentKind.SEARCH: Paradigm.THERMODYNAMIC,
        ComponentKind.ROUTING: Paradigm.CLASSICAL,
        ComponentKind.RANDOMNESS: Paradigm.QUANTUM,
        ComponentKind.MEMORY_RETRIEVAL: Paradigm.CLASSICAL,
        ComponentKind.ENERGY_SAMPLING: Paradigm.THERMODYNAMIC,
    }

    def __init__(
        self,
        prefer_photonic_attention: bool = False,
        prefer_quantum_attention: bool = False,
    ):
        self.prefer_photonic_attention = prefer_photonic_attention
        self.prefer_quantum_attention = prefer_quantum_attention

    def choose(self, component: ComponentKind, **context) -> ParadigmChoice:
        paradigm = self._DEFAULT_MAP.get(component, Paradigm.CLASSICAL)

        if component == ComponentKind.ATTENTION:
            if self.prefer_photonic_attention:
                paradigm = Paradigm.PHOTONIC
            elif self.prefer_quantum_attention:
                paradigm = Paradigm.QUANTUM

        reason = _REASONS.get(component, {}).get(
            paradigm, f"Default choice for {component.value}."
        )
        speedup = _STATIC_SPEEDUP_ESTIMATES.get(paradigm)
        return ParadigmChoice(
            component=component, paradigm=paradigm, reason=reason, estimated_speedup=speedup
        )


class SingleParadigmPolicy(SchedulingPolicy):
    """Forces one specific paradigm everywhere. Used to build the spec's
    single-paradigm `ClassicalLLM` / `ThermodynamicLLM` / `QuantumLLM` /
    `PhotonicLLM` classes from the exact same model-building code as
    `HybridLLM`. Components that paradigm has no implementation for (e.g.
    Quantum has no Embedding) are requested anyway -- `HybridScheduler.build`
    is responsible for catching that and falling back to Classical, which
    keeps this policy honest about what it *wants* rather than silently
    pre-filtering to only the components a paradigm happens to support.
    """

    def __init__(self, paradigm: Paradigm):
        self.paradigm = paradigm

    def choose(self, component: ComponentKind, **context) -> ParadigmChoice:
        reason = _REASONS.get(component, {}).get(
            self.paradigm,
            f"SingleParadigmPolicy({self.paradigm.value}): requested for every "
            f"component to build a single-paradigm model.",
        )
        speedup = _STATIC_SPEEDUP_ESTIMATES.get(self.paradigm)
        return ParadigmChoice(
            component=component, paradigm=self.paradigm, reason=reason, estimated_speedup=speedup
        )


class AllClassicalPolicy(SchedulingPolicy):
    """A trivial alternative strategy: forces every component classical.
    Useful as a fast, always-differentiable baseline for benchmarking and as
    a demonstration that the scheduler is genuinely swappable."""

    def choose(self, component: ComponentKind, **context) -> ParadigmChoice:
        return ParadigmChoice(
            component=component,
            paradigm=Paradigm.CLASSICAL,
            reason="AllClassicalPolicy: every component forced classical for a "
            "fast, fully-differentiable baseline run.",
            estimated_speedup=1.0,
        )


class HybridScheduler:
    """Context object (Strategy Pattern) holding a replaceable
    `SchedulingPolicy`. `HybridLLM` asks this scheduler which paradigm to use
    for each component while building its layers.
    """

    def __init__(self, policy: Optional[SchedulingPolicy] = None):
        self.policy = policy or DefaultSchedulingPolicy()
        self.decisions: List[ParadigmChoice] = []

    def set_policy(self, policy: SchedulingPolicy) -> None:
        self.policy = policy
        self.decisions.clear()

    def decide(self, component: ComponentKind, **context) -> ParadigmChoice:
        choice = self.policy.choose(component, **context)
        self.decisions.append(choice)
        return choice

    def build(self, component: ComponentKind, factory_fn, **kwargs):
        """Ask the policy for a paradigm, try to build the component with it,
        and transparently fall back to Classical (recording *why*) if that
        paradigm has no implementation for this component. `factory_fn` is
        one of `ModuleFactory.create_*`, called as `factory_fn(paradigm,
        **kwargs)`.
        """
        from factory import UnsupportedParadigmError  # local import: avoids a
        # factory<->scheduler import cycle, since factory.py has no need to
        # import scheduler.py.

        choice = self.decide(component, **kwargs)
        try:
            module = factory_fn(choice.paradigm, **kwargs)
            return module, choice
        except UnsupportedParadigmError:
            fallback_reason = (
                f"{choice.reason} [Fallback: {choice.paradigm.value} has no "
                f"implementation for {component.value} in this codebase; "
                f"used classical instead.]"
            )
            fallback_choice = ParadigmChoice(
                component=component,
                paradigm=Paradigm.CLASSICAL,
                reason=fallback_reason,
                estimated_speedup=1.0,
            )
            self.decisions[-1] = fallback_choice
            module = factory_fn(Paradigm.CLASSICAL, **kwargs)
            return module, fallback_choice

    def explain(self) -> str:
        lines = ["Paradigm selection report:", ""]
        for d in self.decisions:
            lines.append(f"{d.component.value}")
            lines.append(f"  -> {d.paradigm.value}")
            lines.append(f"  Reason: {d.reason}")
            if d.estimated_speedup is not None:
                lines.append(f"  Estimated speedup: {d.estimated_speedup:.1f}x")
            lines.append("")
        return "\n".join(lines)
