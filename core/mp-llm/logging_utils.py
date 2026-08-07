"""
Logging: a small structured logger that prints human-readable lines to the
console and appends the same events as JSONL to a log file, covering every
field the spec asks to be logged (loss, perplexity, accuracy, execution
time, GPU/RAM usage, paradigm selection, benchmark summary).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import torch


def _gpu_usage_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def _ram_usage_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return -1.0


class MPLLMLogger:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _emit(self, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", time.time())
        event.setdefault("gpu_usage_mb", _gpu_usage_mb())
        event.setdefault("ram_usage_mb", _ram_usage_mb())
        print(
            " | ".join(f"{k}={v}" for k, v in event.items() if k not in ("timestamp",))
        )
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_step(
        self,
        step: int,
        loss: float,
        perplexity: Optional[float] = None,
        accuracy: Optional[float] = None,
        execution_time: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = {"event": "train_step", "step": step, "loss": round(loss, 5)}
        if perplexity is not None:
            event["perplexity"] = round(perplexity, 5)
        if accuracy is not None:
            event["accuracy"] = round(accuracy, 5)
        if execution_time is not None:
            event["execution_time_s"] = round(execution_time, 5)
        if extra:
            event.update(extra)
        self._emit(event)

    def log_paradigm_selection(self, component: str, paradigm: str, reason: str) -> None:
        self._emit(
            {
                "event": "paradigm_selection",
                "component": component,
                "paradigm": paradigm,
                "reason": reason,
            }
        )

    def log_benchmark_summary(self, summary: Dict[str, Any]) -> None:
        self._emit({"event": "benchmark_summary", **summary})

    def log_message(self, message: str) -> None:
        self._emit({"event": "message", "message": message})
