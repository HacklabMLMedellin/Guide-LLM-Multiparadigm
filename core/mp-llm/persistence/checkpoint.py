"""
Model persistence: save/load checkpoints, resume training, save tokenizer +
configuration + benchmark results alongside the weights so a run directory
is fully self-describing and reproducible.

Layout of a saved run directory:

    run_dir/
      config.json          <- MPLLMConfig
      tokenizer.json        <- Vocabulary
      model.pt               <- model.state_dict()
      optimizer.pt            <- optimizer state (for resuming training)
      training_state.json      <- epoch/step counters
      benchmark_report.json/.md  <- see benchmarking/report.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    run_dir: str,
    model: torch.nn.Module,
    config,
    tokenizer=None,
    optimizer=None,
    training_state: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    config.save(os.path.join(run_dir, "config.json"))

    if tokenizer is not None:
        tokenizer.save(os.path.join(run_dir, "tokenizer.json"))

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    if optimizer is not None and hasattr(optimizer, "opt"):
        torch.save(optimizer.opt.state_dict(), os.path.join(run_dir, "optimizer.pt"))

    if training_state is not None:
        with open(os.path.join(run_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)


def load_model_state(run_dir: str, model: torch.nn.Module, strict: bool = True) -> None:
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state, strict=strict)


def load_optimizer_state(run_dir: str, optimizer) -> bool:
    path = os.path.join(run_dir, "optimizer.pt")
    if not os.path.exists(path) or not hasattr(optimizer, "opt"):
        return False
    state = torch.load(path, map_location="cpu")
    optimizer.opt.load_state_dict(state)
    return True


def load_training_state(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "training_state.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
