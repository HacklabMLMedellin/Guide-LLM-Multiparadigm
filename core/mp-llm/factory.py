"""
Factory Pattern: `ModuleFactory.create_<component>(paradigm, ...)` builds the
concrete paradigm-specific implementation of a component. This is the single
place in the codebase that imports from all four `paradigms/*` packages, so
everything else (models, scheduler, pipelines) only ever depends on the
abstract interfaces in `core.interfaces` -- Dependency Inversion in practice.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from core.types import ComponentKind, Paradigm
from paradigms.classical import components as classical
from paradigms.photonic import components as photonic
from paradigms.quantum import components as quantum
from paradigms.thermodynamic import components as thermodynamic


class UnsupportedParadigmError(ValueError):
    pass


def _filtered_call(ctor: Callable, /, **kwargs) -> Any:
    """Call `ctor` with only the subset of `kwargs` it actually accepts.

    Every `create_*` factory method below accepts one flat superset of
    kwargs (mesh_size, n_qubits, beta, dropout, ...) from callers like
    `HybridScheduler.build`, which does not know in advance which paradigm
    will end up selected and therefore cannot pre-filter. Each concrete
    paradigm implementation only wants the handful of kwargs relevant to it
    (e.g. `PhotonicLinear` wants `mesh_size`, `QuantumCircuitAdapter` wants
    `n_qubits`/`n_layers`); this keeps every component constructor's
    signature honest (no blanket `**kwargs` swallowing typos) while still
    letting the factory stay paradigm-agnostic about *which* kwargs matter.
    """
    sig = inspect.signature(ctor)
    accepted = {
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return ctor(**filtered)


class ModuleFactory:
    """Every `create_*` method takes the desired paradigm plus a flat set of
    keyword arguments (only the ones relevant to the chosen paradigm are
    actually used -- see `_filtered_call`), and returns a ready-to-use
    `nn.Module`/strategy instance. Not every (component, paradigm) pair is
    implemented -- only the ones the spec's example scheduler table calls
    for, or that are needed for a minimally complete Transformer/diffusion
    stack. Unimplemented combinations raise `UnsupportedParadigmError` so
    the scheduler can fall back cleanly instead of silently degrading.
    """

    @staticmethod
    def create_embedding(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalEmbedding, **kwargs)
        raise UnsupportedParadigmError(f"embedding: {paradigm}")

    @staticmethod
    def create_positional_encoding(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalPositionalEncoding, **kwargs)
        raise UnsupportedParadigmError(f"positional_encoding: {paradigm}")

    @staticmethod
    def create_attention(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalAttention, **kwargs)
        if paradigm == Paradigm.PHOTONIC:
            return _filtered_call(photonic.PhotonicAttention, **kwargs)
        if paradigm == Paradigm.QUANTUM:
            return _filtered_call(quantum.QuantumAttention, **kwargs)
        if paradigm == Paradigm.THERMODYNAMIC:
            return _filtered_call(thermodynamic.ThermodynamicAttention, **kwargs)
        raise UnsupportedParadigmError(f"attention: {paradigm}")

    @staticmethod
    def create_feed_forward(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalFeedForward, **kwargs)
        if paradigm == Paradigm.PHOTONIC:
            return _filtered_call(photonic.PhotonicFeedForward, **kwargs)
        raise UnsupportedParadigmError(f"feed_forward: {paradigm}")

    @staticmethod
    def create_matmul(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            kwargs = {**kwargs, "in_features": kwargs.get("dim"), "out_features": kwargs.get("dim")}
            return _filtered_call(classical.ClassicalLinear, **kwargs)
        if paradigm == Paradigm.PHOTONIC:
            return _filtered_call(photonic.PhotonicLinear, **kwargs)
        raise UnsupportedParadigmError(f"matmul: {paradigm}")

    @staticmethod
    def create_layer_norm(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalLayerNorm, **kwargs)
        raise UnsupportedParadigmError(f"layer_norm: {paradigm}")

    @staticmethod
    def create_output_projection(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalOutputProjection, **kwargs)
        raise UnsupportedParadigmError(f"output_projection: {paradigm}")

    @staticmethod
    def create_sampling(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalSampling, **kwargs)
        if paradigm == Paradigm.QUANTUM:
            return _filtered_call(quantum.QuantumSampling, **kwargs)
        if paradigm == Paradigm.THERMODYNAMIC:
            return _filtered_call(thermodynamic.ThermodynamicSampling, **kwargs)
        raise UnsupportedParadigmError(f"sampling: {paradigm}")

    @staticmethod
    def create_probability_estimation(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalProbabilityEstimation, **kwargs)
        if paradigm == Paradigm.QUANTUM:
            return _filtered_call(quantum.QuantumProbabilityEstimation, **kwargs)
        raise UnsupportedParadigmError(f"probability_estimation: {paradigm}")

    @staticmethod
    def create_randomness(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalRandomness, **kwargs)
        if paradigm == Paradigm.QUANTUM:
            return _filtered_call(quantum.QuantumRandomness, **kwargs)
        raise UnsupportedParadigmError(f"randomness_generation: {paradigm}")

    @staticmethod
    def create_search(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalSearch, **kwargs)
        if paradigm == Paradigm.THERMODYNAMIC:
            return _filtered_call(thermodynamic.ThermodynamicSearch, **kwargs)
        raise UnsupportedParadigmError(f"search: {paradigm}")

    @staticmethod
    def create_energy_based_sampling(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalEnergyBasedSampling, **kwargs)
        if paradigm == Paradigm.THERMODYNAMIC:
            return _filtered_call(thermodynamic.ThermodynamicEnergyBasedSampling, **kwargs)
        raise UnsupportedParadigmError(f"energy_based_sampling: {paradigm}")

    @staticmethod
    def create_optimizer(paradigm: Paradigm, **kwargs) -> Any:
        if paradigm == Paradigm.CLASSICAL:
            return _filtered_call(classical.ClassicalOptimizer, **kwargs)
        raise UnsupportedParadigmError(f"optimizer: {paradigm}")
