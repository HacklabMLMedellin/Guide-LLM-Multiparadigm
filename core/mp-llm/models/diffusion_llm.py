"""
Model 2: a masked diffusion language model (per the spec's "Diffusion-LM /
masked diffusion transformer" option). Training randomly masks a fraction of
tokens (fraction driven by a sampled diffusion timestep) and the model
learns to reconstruct them from bidirectional context -- unlike Model 1,
there is no causal mask, since every position may condition on any other.

Generation starts from an all-masked sequence and iteratively "denoises" it
over `diffusion_timesteps` steps, each step filling in the model's
most-confident masked positions (a standard, simple masked-diffusion
decoding schedule), using the scheduler-selected `SamplingStrategy` exactly
like Model 1 does -- the same Strategy Pattern component is reused, this
time called once per position per step instead of once per generated token,
which is why the diffusion timestep count is kept small at this toy scale.

Shares `TransformerBlock` with `models.transformer_llm` (SOLID: no
duplicated attention/FFN/norm wiring between the two architectures).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarking.timers import timer_for
from config.config import MPLLMConfig
from core.types import ComponentKind, ParadigmChoice
from factory import ModuleFactory
from models.transformer_llm import TransformerBlock
from scheduler import HybridScheduler


class TimestepEmbedding(nn.Module):
    """Classical sinusoidal timestep embedding (diffusion models universally
    use a classical scalar-conditioning mechanism here; no paradigm in this
    project offers a natural alternative for "embed an integer step index").
    """

    def __init__(self, d_model: int, max_timesteps: int):
        super().__init__()
        self.embed = nn.Embedding(max_timesteps, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embed(t)


class DiffusionLM(nn.Module):
    """Bidirectional masked-diffusion Transformer sharing its block
    architecture, scheduler, and factory with `TransformerLM`. `mask_token_id`
    is appended to the vocabulary automatically (id == vocab_size) so the
    caller's tokenizer does not need to reserve one.
    """

    def __init__(self, config: MPLLMConfig, scheduler: Optional[HybridScheduler] = None):
        super().__init__()
        self.config = config
        self.scheduler = scheduler or HybridScheduler()
        self.mask_token_id = config.mask_token_id or config.vocab_size
        total_vocab = self.mask_token_id + 1  # + the mask token itself

        self.embedding, self.embedding_choice = self.scheduler.build(
            ComponentKind.EMBEDDING,
            ModuleFactory.create_embedding,
            vocab_size=total_vocab,
            d_model=config.d_model,
        )
        self.pos_encoding, self.pos_choice = self.scheduler.build(
            ComponentKind.POSITIONAL_ENCODING,
            ModuleFactory.create_positional_encoding,
            d_model=config.d_model,
            max_len=config.context_length,
        )
        self.time_embed = TimestepEmbedding(config.d_model, config.diffusion_timesteps)

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
            vocab_size=total_vocab,
        )
        self.sampling, self.sampling_choice = self.scheduler.build(
            ComponentKind.SAMPLING,
            ModuleFactory.create_sampling,
            vocab_size=total_vocab,
            beta=config.thermodynamic_beta,
            max_spins=config.thermodynamic_max_spins,
            n_qubits=config.quantum_n_qubits,
        )
        self.prob_estimation, self.prob_choice = self.scheduler.build(
            ComponentKind.PROBABILITY_ESTIMATION,
            ModuleFactory.create_probability_estimation,
            n_qubits=config.quantum_n_qubits,
            n_layers=config.quantum_n_layers,
        )

        self._embedding_timed = timer_for(self.embedding_choice.paradigm)(self._run_embedding)
        self._sampling_timed = timer_for(self.sampling_choice.paradigm)(self._run_sampling)

    def _run_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    def _run_sampling(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return self.sampling(logits, temperature)

    def forward(self, noisy_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = self._embedding_timed(noisy_ids)
        x = self.pos_encoding(x)
        x = x + self.time_embed(timesteps).unsqueeze(1)
        for block in self.blocks:
            x = block(x, mask=None)  # bidirectional: no causal mask
        x = self.ln_f(x)
        return self.output_proj(x)

    def corrupt(self, clean_ids: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Randomly replace tokens with the mask token; fraction masked
        grows with the timestep (t=0 -> mostly clean, t=T-1 -> mostly/fully
        masked), the standard masked-diffusion forward process.
        """
        b, t_len = clean_ids.shape
        frac = (timesteps.float() + 1) / self.config.diffusion_timesteps
        rand = torch.rand(b, t_len)
        mask = rand < frac.unsqueeze(1)
        noisy = torch.where(mask, torch.full_like(clean_ids, self.mask_token_id), clean_ids)
        return noisy, mask

    @torch.no_grad()
    def generate(self, seq_len: int, batch_size: int = 1, temperature: float = 1.0) -> torch.Tensor:
        """Iterative denoising from an all-masked sequence."""
        self.eval()
        ids = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long)
        n_steps = self.config.diffusion_timesteps
        for step in range(n_steps):
            t = torch.full((batch_size,), n_steps - 1 - step, dtype=torch.long)
            logits = self.forward(ids, t)
            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            is_masked = ids == self.mask_token_id
            # Unmask a growing fraction of the still-masked positions each
            # step, prioritizing the model's most confident predictions.
            unmask_frac = 1.0 / (n_steps - step)
            for bi in range(batch_size):
                masked_positions = is_masked[bi].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                k = max(1, int(len(masked_positions) * unmask_frac))
                conf_here = confidence[bi, masked_positions]
                top_k = masked_positions[torch.topk(conf_here, min(k, len(masked_positions))).indices]
                for pos in top_k:
                    tok = self._sampling_timed(logits[bi, pos : pos + 1, :], temperature)
                    ids[bi, pos] = tok.reshape(())
        return ids

    def all_choices(self) -> List[ParadigmChoice]:
        choices = [
            self.embedding_choice,
            self.pos_choice,
            self.ln_f_choice,
            self.output_choice,
            self.sampling_choice,
            self.prob_choice,
        ]
        for block in self.blocks:
            choices.extend(block.choices())
        return choices

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
