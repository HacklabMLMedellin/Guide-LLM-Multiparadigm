"""
Benchmark reporting: turns `benchmarking.timers.TIMING_LOG` (populated by the
`@classical_timer` / `@quantum_timer` / `@photonic_timer` /
`@thermodynamic_timer` decorators as the model actually runs) into the
comparison the spec asks for: simulated runtime vs. estimated hardware
runtime vs. speedup, plus parameter count and paradigm-selection rationale,
exported as both JSON and Markdown.
"""

from __future__ import annotations

import json
import os
import platform
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import torch

from benchmarking.timers import TIMING_LOG, HARDWARE_FACTORS
from core.types import Paradigm, ParadigmChoice


def _process_memory_mb() -> float:
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if platform.system() == "Linux" else rss / (1024 * 1024)
    except Exception:
        return -1.0


def aggregate_timing_log() -> Dict[str, Any]:
    """Group all recorded timings by (function name, paradigm), summing
    simulated and hardware-adjusted time and counting calls.
    """
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"paradigm": None, "calls": 0, "simulated_seconds": 0.0, "estimated_hardware_seconds": 0.0}
    )
    for rec in TIMING_LOG:
        key = f"{rec.name} [{rec.paradigm.value}]"
        g = grouped[key]
        g["paradigm"] = rec.paradigm.value
        g["calls"] += 1
        g["simulated_seconds"] += rec.simulated_seconds
        g["estimated_hardware_seconds"] += rec.estimated_hardware_seconds
    for g in grouped.values():
        if g["simulated_seconds"] > 0:
            g["speedup_vs_simulated"] = g["simulated_seconds"] / max(g["estimated_hardware_seconds"], 1e-12)
        else:
            g["speedup_vs_simulated"] = None
    return dict(grouped)


def build_benchmark_report(
    model_name: str,
    model: Optional[torch.nn.Module] = None,
    paradigm_choices: Optional[List[ParadigmChoice]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "model_name": model_name,
        "hardware_factors": asdict(HARDWARE_FACTORS),
        "timing_summary": aggregate_timing_log(),
        "process_peak_memory_mb": _process_memory_mb(),
    }
    if model is not None:
        report["parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["total_parameter_count"] = sum(p.numel() for p in model.parameters())
    if paradigm_choices is not None:
        report["paradigm_selection"] = [
            {
                "component": c.component.value,
                "paradigm": c.paradigm.value,
                "reason": c.reason,
                "estimated_speedup": c.estimated_speedup,
            }
            for c in paradigm_choices
        ]
    if extra:
        report.update(extra)
    return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Benchmark Report: {report.get('model_name', 'model')}", ""]

    if "parameter_count" in report:
        lines.append(f"**Trainable parameters:** {report['parameter_count']:,}")
        lines.append(f"**Total parameters:** {report['total_parameter_count']:,}")
    lines.append(f"**Process peak memory:** {report.get('process_peak_memory_mb', -1):.1f} MB")
    lines.append("")

    lines.append("## Hardware time-conversion factors (configurable, theoretical)")
    lines.append("")
    lines.append("| Paradigm | Factor (real_time = simulated_time x factor) |")
    lines.append("|---|---|")
    for k, v in report.get("hardware_factors", {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Timing summary (simulated vs. estimated future hardware)")
    lines.append("")
    lines.append("| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |")
    lines.append("|---|---|---|---|---|")
    for name, g in report.get("timing_summary", {}).items():
        speedup = g["speedup_vs_simulated"]
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
        lines.append(
            f"| {name} | {g['calls']} | {g['simulated_seconds']:.6f} | "
            f"{g['estimated_hardware_seconds']:.6f} | {speedup_str} |"
        )
    lines.append("")

    if "paradigm_selection" in report:
        lines.append("## Paradigm selection rationale")
        lines.append("")
        for c in report["paradigm_selection"]:
            lines.append(f"**{c['component']} -> {c['paradigm']}**")
            lines.append("")
            lines.append(f"> {c['reason']}")
            if c["estimated_speedup"] is not None:
                lines.append(f">\n> Estimated speedup: {c['estimated_speedup']:.1f}x")
            lines.append("")

    return "\n".join(lines)


def save_report(report: Dict[str, Any], out_dir: str, basename: str = "benchmark_report") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{basename}.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, f"{basename}.md"), "w") as f:
        f.write(report_to_markdown(report))
