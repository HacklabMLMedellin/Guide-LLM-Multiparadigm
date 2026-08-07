"""
Benchmarking decorators.

`timer` is the plain wall-clock decorator from the spec. The four paradigm
decorators (`classical_timer`, `quantum_timer`, `photonic_timer`,
`thermodynamic_timer`) additionally record a *theoretical* hardware-adjusted
time using a configurable `hardware_factor`:

    estimated_hardware_time = simulated_time * hardware_factor

This conversion is explicitly a speculative placeholder (per the spec):
quantum/photonic/thermodynamic code here runs as a *simulation* on ordinary
classical hardware (PennyLane's `default.qubit`, PhotonTorch's differential-
equation solver, THRML's JAX Gibbs sampler), so the "hardware seconds" number
is not a measurement -- it is `simulated_seconds * factor`, where `factor` is
meant to be replaced once real hardware benchmarks exist.

All decorators funnel through `TIMING_LOG`, a process-global list of
`TimingRecord`s that `benchmarking/report.py` aggregates into the benchmark
report.
"""

from __future__ import annotations

import functools
from time import perf_counter
from typing import Callable, List

from core.types import HardwareFactors, Paradigm, TimingRecord

# Global, mutable, easy-to-swap-out hardware factor table (Strategy-friendly:
# replace this object, or mutate its fields, as better hardware benchmarks
# become available).
HARDWARE_FACTORS = HardwareFactors()

TIMING_LOG: List[TimingRecord] = []


def reset_timing_log() -> None:
    TIMING_LOG.clear()


def timer(func: Callable) -> Callable:
    """Plain wall-clock timing decorator, exactly as specified."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__}: {end - start:.6f} sec")
        return result

    return wrapper


def _paradigm_timer(paradigm: Paradigm) -> Callable:
    """Factory for the four `@<paradigm>_timer` decorators."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            simulated = perf_counter() - start
            factor = HARDWARE_FACTORS.factor_for(paradigm)
            record = TimingRecord(
                name=func.__qualname__,
                paradigm=paradigm,
                simulated_seconds=simulated,
                hardware_factor=factor,
            )
            TIMING_LOG.append(record)
            return result

        return wrapper

    return decorator


classical_timer = _paradigm_timer(Paradigm.CLASSICAL)
quantum_timer = _paradigm_timer(Paradigm.QUANTUM)
photonic_timer = _paradigm_timer(Paradigm.PHOTONIC)
thermodynamic_timer = _paradigm_timer(Paradigm.THERMODYNAMIC)


def timer_for(paradigm: Paradigm) -> Callable:
    """Look up the right decorator for a paradigm chosen at runtime (used by
    the factory, which doesn't know at import time which paradigm a given
    component instance will use).
    """
    return {
        Paradigm.CLASSICAL: classical_timer,
        Paradigm.QUANTUM: quantum_timer,
        Paradigm.PHOTONIC: photonic_timer,
        Paradigm.THERMODYNAMIC: thermodynamic_timer,
    }[paradigm]
