"""
Builds all five LLM variants (Classical, Quantum, Photonic, Thermodynamic,
Hybrid) at matching hyperparameters, runs each through
`ParadigmLLM.benchmark()`, and writes one combined comparison report -- the
"comprehensive benchmark report comparing paradigms" the spec's Final
Deliverables section asks for: simulated performance, estimated hardware
performance, memory usage, parameter counts, and the paradigm-selection
rationale for every component of every variant.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from benchmarking.report import report_to_markdown
from benchmarking.timers import reset_timing_log
from config.config import MPLLMConfig
from models.paradigm_llms import ClassicalLLM, HybridLLM, PhotonicLLM, QuantumLLM, ThermodynamicLLM

VARIANTS = {
    "ClassicalLLM": ClassicalLLM,
    "QuantumLLM": QuantumLLM,
    "PhotonicLLM": PhotonicLLM,
    "ThermodynamicLLM": ThermodynamicLLM,
    "HybridLLM": HybridLLM,
}


def run_comparison(
    model_kind: str = "transformer",
    out_dir: str = "outputs/benchmarks",
    vocab_size: int = 30,
    d_model: int = 16,
    n_heads: int = 2,
    n_layers: int = 1,
    d_ff: int = 32,
    context_length: int = 12,
    mesh_size: int = 4,
    n_qubits: int = 4,
    max_spins: int = 5,
    diffusion_timesteps: int = 4,
    n_forward: int = 3,
    batch_size: int = 2,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    combined: Dict[str, Any] = {"model_kind": model_kind, "variants": {}}

    for name, cls in VARIANTS.items():
        cfg = MPLLMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            context_length=context_length,
            photonic_mesh_size=mesh_size,
            quantum_n_qubits=n_qubits,
            thermodynamic_max_spins=max_spins,
            diffusion_timesteps=diffusion_timesteps,
            model_kind=model_kind,
        )
        reset_timing_log()
        llm = cls(cfg)
        report = llm.benchmark(n_forward=n_forward, seq_len=context_length, batch_size=batch_size)
        combined["variants"][name] = report
        print(f"[{name}] params={report['parameter_count']} " f"timing_keys={list(report['timing_summary'].keys())}")

    with open(os.path.join(out_dir, f"comparison_{model_kind}.json"), "w") as f:
        json.dump(combined, f, indent=2)

    md = _comparison_to_markdown(combined)
    with open(os.path.join(out_dir, f"comparison_{model_kind}.md"), "w") as f:
        f.write(md)

    print(f"\nComparison report written to {out_dir}/comparison_{model_kind}.{{json,md}}")
    return combined


def _comparison_to_markdown(combined: Dict[str, Any]) -> str:
    lines = [f"# Paradigm Comparison: {combined['model_kind']}", ""]
    lines.append("| Variant | Params | Total sim time (s) | Total est. hw time (s) |")
    lines.append("|---|---|---|---|")
    for name, report in combined["variants"].items():
        sim_total = sum(g["simulated_seconds"] for g in report["timing_summary"].values())
        hw_total = sum(g["estimated_hardware_seconds"] for g in report["timing_summary"].values())
        lines.append(f"| {name} | {report['parameter_count']:,} | {sim_total:.6f} | {hw_total:.6f} |")
    lines.append("")

    for name, report in combined["variants"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(report_to_markdown(report))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_comparison()
