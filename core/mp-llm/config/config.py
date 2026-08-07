"""
Configuration management. A single dataclass holds every parameter the spec
asks to be configurable (vocab size, embedding/hidden size, layers, heads,
context length, optimizer, learning rate, selected paradigm policy,
scheduler policy) plus the paradigm-specific knobs (mesh size, qubit count,
thermodynamic beta) needed to actually construct the paradigm components.
Saved/loaded as JSON for simplicity and zero extra dependencies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class MPLLMConfig:
    # --- architecture ---
    vocab_size: int = 0  # filled in once the tokenizer is built
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    context_length: int = 64
    dropout: float = 0.0

    # --- training ---
    optimizer: str = "adamw"
    learning_rate: float = 3e-3
    batch_size: int = 8
    epochs: int = 1
    steps_per_epoch: int = 20

    # --- paradigm scheduling ---
    # one of: "hybrid" (DefaultSchedulingPolicy), "classical", "quantum",
    # "photonic", "thermodynamic", "all_classical"
    scheduler_policy: str = "hybrid"
    prefer_photonic_attention: bool = True
    prefer_quantum_attention: bool = False

    # --- paradigm-specific knobs (kept small: real simulators, not real
    #     hardware -- see README for why these stay toy-scale) ---
    photonic_mesh_size: int = 4
    quantum_n_qubits: int = 4
    quantum_n_layers: int = 2
    thermodynamic_beta: float = 1.0
    thermodynamic_max_spins: int = 8

    # --- diffusion-model-specific ---
    diffusion_timesteps: int = 8
    mask_token_id: Optional[int] = None

    # --- misc ---
    seed: int = 0
    model_kind: str = "transformer"  # "transformer" | "diffusion"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "MPLLMConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
