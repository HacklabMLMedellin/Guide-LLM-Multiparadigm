"""
Model 1: a nanoGPT-style autoregressive decoder-only Transformer, built
entirely from scheduler-selected, factory-constructed components. This one
class *is* the "HybridLLM" (and, with a `SingleParadigmPolicy`, also the
`ClassicalLLM`/`ThermodynamicLLM`/`QuantumLLM`/`PhotonicLLM`) for the
autoregressive architecture -- see `models/paradigm_llms.py` for the thin
wrappers that name these five variants per the spec while sharing this one
implementation (SOLID: no duplicated model code across the five classes).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarking.timers import timer_for
from config.config import MPLLMConfig
from core.types import ComponentKind, ParadigmChoice
from factory import ModuleFactory
from scheduler import HybridScheduler


class TransformerBlock(nn.Module):
    """One decoder block: scheduler-chosen Attention + FeedForward, each
    wrapped with a residual connection and a scheduler-chosen LayerNorm
    (pre-norm, as in GPT-2/LLaMA-style models).
    """

    def __init__(self, scheduler: HybridScheduler, config: MPLLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.ln1, self.ln1_choice = scheduler.build(
            ComponentKind.LAYER_NORM, ModuleFactory.create_layer_norm, d_model=config.d_model
        )
        self.attn, self.attn_choice = scheduler.build(
            ComponentKind.ATTENTION,
            ModuleFactory.create_attention,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            mesh_size=config.photonic_mesh_size,
            n_qubits=config.quantum_n_qubits,
        )
        self.ln2, self.ln2_choice = scheduler.build(
            ComponentKind.LAYER_NORM, ModuleFactory.create_layer_norm, d_model=config.d_model
        )
        self.ffn, self.ffn_choice = scheduler.build(
            ComponentKind.FEED_FORWARD,
            ModuleFactory.create_feed_forward,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            mesh_size=config.photonic_mesh_size,
        )

        self._attn_timed = timer_for(self.attn_choice.paradigm)(self._run_attn)
        self._ffn_timed = timer_for(self.ffn_choice.paradigm)(self._run_ffn)

    def _run_attn(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.attn(x, mask)

    def _run_ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self._attn_timed(self.ln1(x), mask)
        x = x + self._ffn_timed(self.ln2(x))
        return x

    def choices(self) -> List[ParadigmChoice]:
        return [self.ln1_choice, self.attn_choice, self.ln2_choice, self.ffn_choice]


class TransformerLM(nn.Module):
    """The full autoregressive Transformer, per the spec's "Model 1"
    (nanoGPT / GPT-2 / LLaMA-style decoder). `scheduler` decides the
    paradigm for every swappable component as the model is built; pass a
    `HybridScheduler(DefaultSchedulingPolicy())` for the true multi-paradigm
    HybridLLM, or `HybridScheduler(SingleParadigmPolicy(paradigm))` /
    `HybridScheduler(AllClassicalPolicy())` to build a single-paradigm
    variant with identical code.
    """

    def __init__(self, config: MPLLMConfig, scheduler: Optional[HybridScheduler] = None):
        super().__init__()
        self.config = config
        self.scheduler = scheduler or HybridScheduler()

        self.embedding, self.embedding_choice = self.scheduler.build(
            ComponentKind.EMBEDDING,
            ModuleFactory.create_embedding,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
        self.pos_encoding, self.pos_choice = self.scheduler.build(
            ComponentKind.POSITIONAL_ENCODING,
            ModuleFactory.create_positional_encoding,
            d_model=config.d_model,
            max_len=config.context_length,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.scheduler, config, i) for i in range(config.n_layers)]
        )
        self.ln_f, self.ln_f_choice = self.scheduler.build(
            ComponentKind.LAYER_NORM, ModuleFactory.create_layer_norm, d_model=config.d_model
        )
        self.output_proj, self.output_choice = self.scheduler.build(
            ComponentKind.OUTPUT_PROJECTION,
            ModuleFactory.create_output_projection,
            d_model=config.d_model,
            vocab_size=config.vocab_size,
        )
        self.sampling, self.sampling_choice = self.scheduler.build(
            ComponentKind.SAMPLING,
            ModuleFactory.create_sampling,
            vocab_size=config.vocab_size,
            temperature=1.0,
            beta=config.thermodynamic_beta,
            max_spins=config.thermodynamic_max_spins,
            n_qubits=config.quantum_n_qubits,
        )

        self._embedding_timed = timer_for(self.embedding_choice.paradigm)(self._run_embedding)
        self._sampling_timed = timer_for(self.sampling_choice.paradigm)(self._run_sampling)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def _run_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    def _run_sampling(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return self.sampling(logits, temperature)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, t = token_ids.shape
        x = self._embedding_timed(token_ids)
        x = self.pos_encoding(x)
        mask = self.causal_mask[:, :, :t, :t]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            cond = token_ids[:, -self.config.context_length :]
            logits = self.forward(cond)
            next_logits = logits[:, -1, :]
            next_token = self._sampling_timed(next_logits, temperature)
            if next_token.ndim == 0:
                next_token = next_token.unsqueeze(0)
            token_ids = torch.cat([token_ids, next_token.unsqueeze(-1)], dim=1)
        return token_ids

    def all_choices(self) -> List[ParadigmChoice]:
        choices = [
            self.embedding_choice,
            self.pos_choice,
            self.ln_f_choice,
            self.output_choice,
            self.sampling_choice,
        ]
        for block in self.blocks:
            choices.extend(block.choices())
        return choices

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
