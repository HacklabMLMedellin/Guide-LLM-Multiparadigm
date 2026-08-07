"""
`ParadigmLLM` implements the spec's common interface (forward / train_step /
inference / save / load / benchmark) around either a `TransformerLM` or a
`DiffusionLM`. The five named classes the spec asks for --
`ClassicalLLM`, `ThermodynamicLLM`, `QuantumLLM`, `PhotonicLLM`,
`HybridLLM` -- are thin factory functions/subclasses that differ only in
which `SchedulingPolicy` they hand to the shared model classes, so there is
exactly one real implementation of "how a transformer/diffusion model is
built out of components" (SOLID: Open/Closed -- new paradigms extend the
scheduler+factory without touching model code; Single Responsibility --
`ParadigmLLM` only handles the *training-loop-facing* interface, not model
architecture).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarking.report import build_benchmark_report, save_report
from benchmarking.timers import reset_timing_log
from config.config import MPLLMConfig
from core.interfaces import LLMInterface
from core.types import Paradigm
from factory import ModuleFactory
from models.diffusion_llm import DiffusionLM
from models.transformer_llm import TransformerLM
from persistence import checkpoint as ckpt
from scheduler import (
    AllClassicalPolicy,
    DefaultSchedulingPolicy,
    HybridScheduler,
    SingleParadigmPolicy,
)


def _build_core_model(config: MPLLMConfig, scheduler: HybridScheduler) -> nn.Module:
    if config.model_kind == "diffusion":
        return DiffusionLM(config, scheduler)
    return TransformerLM(config, scheduler)


class ParadigmLLM(nn.Module, LLMInterface):
    def __init__(self, config: MPLLMConfig, scheduler: HybridScheduler, name: str, tokenizer=None):
        super().__init__()
        self.config = config
        self.scheduler = scheduler
        self.name = name
        self.tokenizer = tokenizer
        self.model = _build_core_model(config, scheduler)
        self.optimizer = ModuleFactory.create_optimizer(
            Paradigm.CLASSICAL, parameters=self.model.parameters(), lr=config.learning_rate
        )
        self.global_step = 0

    # -- LLMInterface -----------------------------------------------------
    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        start = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad()

        if self.config.model_kind == "diffusion":
            clean_ids = batch["input_ids"]
            b = clean_ids.shape[0]
            timesteps = torch.randint(0, self.config.diffusion_timesteps, (b,))
            noisy_ids, mask = self.model.corrupt(clean_ids, timesteps)
            logits = self.model(noisy_ids, timesteps)
            if mask.any():
                loss = F.cross_entropy(logits[mask], clean_ids[mask])
                with torch.no_grad():
                    acc = (logits[mask].argmax(dim=-1) == clean_ids[mask]).float().mean().item()
            else:
                loss = logits.sum() * 0.0
                acc = 0.0
        else:
            input_ids = batch["input_ids"]
            targets = batch["targets"]
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == targets).float().mean().item()

        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        elapsed = time.perf_counter() - start
        loss_val = loss.item()
        return {
            "loss": loss_val,
            "perplexity": float(torch.exp(torch.tensor(min(loss_val, 20.0)))),
            "accuracy": acc,
            "execution_time": elapsed,
            "step": self.global_step,
        }

    def inference(self, *args, **kwargs) -> Any:
        if self.config.model_kind == "diffusion":
            return self.model.generate(*args, **kwargs)
        return self.model.generate(*args, **kwargs)

    def save(self, path: str) -> None:
        ckpt.save_checkpoint(
            path,
            self.model,
            self.config,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            training_state={"global_step": self.global_step, "name": self.name},
        )

    def load(self, path: str) -> None:
        ckpt.load_model_state(path, self.model)
        ckpt.load_optimizer_state(path, self.optimizer)
        state = ckpt.load_training_state(path)
        self.global_step = state.get("global_step", 0)

    def benchmark(self, n_forward: int = 3, seq_len: int = 8, batch_size: int = 2) -> Dict[str, Any]:
        reset_timing_log()
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_forward):
                if self.config.model_kind == "diffusion":
                    x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
                    t = torch.randint(0, self.config.diffusion_timesteps, (batch_size,))
                    self.model(x, t)
                else:
                    x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
                    self.model(x)
        report = build_benchmark_report(
            model_name=self.name,
            model=self.model,
            paradigm_choices=self.model.all_choices(),
        )
        return report

    # -- convenience --------------------------------------------------------
    def save_benchmark_report(self, out_dir: str, **benchmark_kwargs) -> Dict[str, Any]:
        report = self.benchmark(**benchmark_kwargs)
        save_report(report, out_dir, basename=f"{self.name}_benchmark")
        return report


# ---------------------------------------------------------------------------
# The five spec-required named classes. Each simply pins a `SchedulingPolicy`
# and delegates everything else to `ParadigmLLM` + the shared model classes.
# ---------------------------------------------------------------------------


class ClassicalLLM(ParadigmLLM):
    """Every component forced classical -- the fast, always-differentiable
    single-paradigm baseline."""

    def __init__(self, config: MPLLMConfig, tokenizer=None):
        super().__init__(config, HybridScheduler(AllClassicalPolicy()), name="ClassicalLLM", tokenizer=tokenizer)


class ThermodynamicLLM(ParadigmLLM):
    """Thermodynamic requested everywhere; components THRML has no
    implementation for (embedding, attention, FFN, norm, ...) transparently
    fall back to classical (see `HybridScheduler.build`), while Sampling /
    Search / Energy-based Sampling genuinely run on THRML's Gibbs sampler."""

    def __init__(self, config: MPLLMConfig, tokenizer=None):
        super().__init__(
            config,
            HybridScheduler(SingleParadigmPolicy(Paradigm.THERMODYNAMIC)),
            name="ThermodynamicLLM",
            tokenizer=tokenizer,
        )


class QuantumLLM(ParadigmLLM):
    """Quantum requested everywhere; genuinely runs PennyLane circuits for
    Attention (gating), Probability Estimation, Randomness, and Sampling,
    falling back to classical elsewhere."""

    def __init__(self, config: MPLLMConfig, tokenizer=None):
        super().__init__(
            config,
            HybridScheduler(SingleParadigmPolicy(Paradigm.QUANTUM)),
            name="QuantumLLM",
            tokenizer=tokenizer,
        )


class PhotonicLLM(ParadigmLLM):
    """Photonic requested everywhere; genuinely runs the PhotonTorch MZI
    mesh for Attention and the FFN's inner mix, falling back to classical
    elsewhere."""

    def __init__(self, config: MPLLMConfig, tokenizer=None):
        super().__init__(
            config,
            HybridScheduler(SingleParadigmPolicy(Paradigm.PHOTONIC)),
            name="PhotonicLLM",
            tokenizer=tokenizer,
        )


class HybridLLM(ParadigmLLM):
    """Dynamically combines all four paradigms via `DefaultSchedulingPolicy`
    -- this is the model the spec's example strategy table describes."""

    def __init__(
        self,
        config: MPLLMConfig,
        tokenizer=None,
        prefer_photonic_attention: bool = True,
        prefer_quantum_attention: bool = False,
    ):
        policy = DefaultSchedulingPolicy(
            prefer_photonic_attention=prefer_photonic_attention,
            prefer_quantum_attention=prefer_quantum_attention,
        )
        super().__init__(config, HybridScheduler(policy), name="HybridLLM", tokenizer=tokenizer)
