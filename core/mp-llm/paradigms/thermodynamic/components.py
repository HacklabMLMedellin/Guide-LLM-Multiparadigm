"""
Thermodynamic paradigm: components implemented via real block-Gibbs sampling
from an Ising energy-based model, through
`adapters.thermodynamic_adapter.ThermodynamicSamplerAdapter`.

Per the spec's scheduler rationale, this paradigm owns Sampling, Search, and
Energy-based Sampling: all three are naturally "find/prefer low-energy
states of a system", which is exactly what THRML's Gibbs sampler computes
-- and exactly the computational primitive Extropic's thermodynamic hardware
is built to accelerate physically instead of simulating on a GPU.

These modules are *not* differentiable end-to-end the way the classical/
quantum/photonic ones are (THRML samples via JAX/discrete spin flips, not
torch autograd), which is itself an honest, reportable property of this
paradigm: it is excellent at stochastic sampling/search over discrete
choices, but -- as implemented here -- sits outside the backprop graph, so
it is used at the *output* of the network (turning logits into a token; or
as a post-hoc search / self-consistency step) rather than inside a
gradient-trained hidden layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.thermodynamic_adapter import ThermodynamicSamplerAdapter
from core.interfaces import (
    AttentionStrategy,
    EnergyBasedSamplingStrategy,
    SamplingStrategy,
    SearchStrategy,
)
from core.types import Paradigm


def _logits_to_spin_count(vocab_size: int, max_spins: int = 12) -> int:
    """We can't give one Ising spin per vocab entry for any real vocab size
    (block Gibbs over thousands of fully-connected spins is not toy-scale).
    Instead we use `max_spins` spins to encode a binary-search-style
    reduction: repeatedly halve the candidate token range based on sampled
    spin values. `max_spins` should satisfy 2**max_spins >= vocab_size for
    the reduction to reach a single token; smaller vocabularies need fewer.
    """
    import math

    return min(max_spins, max(1, math.ceil(math.log2(max(vocab_size, 2)))))


class ThermodynamicAttention(nn.Module, AttentionStrategy):
    """Attention -> Thermodynamic: uses real Ising Gibbs sampling to decide,
    per forward pass, which key positions the whole layer should attend to
    at all -- a physically-motivated stochastic sparsification, rather than
    softmax's fully-dense deterministic weighting. Each key position is one
    spin, biased by how much query/key affinity it attracts on average;
    THRML then samples a binary "attend / don't attend" configuration, which
    masks the (still classically computed) scaled-dot-product scores before
    the final softmax. Query/Key/Value projections stay classical -- the
    paradigm is applied to the *selection* decision (a discrete, sampling-
    shaped problem, exactly what Gibbs sampling is for), not the linear
    algebra (which is Photonic's role -- see `PhotonicAttention`).

    Only tractable for small sequence lengths (one spin per key position, so
    a real block-Gibbs sample is drawn every forward call); sequences longer
    than `max_keys` are coarsened into `max_keys` buckets first.
    """

    paradigm = Paradigm.THERMODYNAMIC

    def __init__(self, d_model: int, n_heads: int, beta: float = 1.0, max_keys: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_keys = max_keys
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.key_selector = ThermodynamicSamplerAdapter(n_spins=max_keys, beta=beta)

    def _sample_key_mask(self, scores: torch.Tensor, t: int) -> torch.Tensor:
        # Average affinity each key attracts, over batch/heads/queries ->
        # (t,); this is what the Ising biases are built from.
        importance = scores.mean(dim=(0, 1, 2)).detach()
        n_buckets = min(self.max_keys, t)
        bucket_size = (t + n_buckets - 1) // n_buckets
        padded = F.pad(importance, (0, bucket_size * n_buckets - t), value=0.0)
        bucket_importance = padded.view(n_buckets, bucket_size).mean(dim=-1)

        biases = torch.zeros(self.max_keys)
        biases[:n_buckets] = bucket_importance - bucket_importance.mean()
        sample = self.key_selector.sample(biases=biases, n_samples=1, n_warmup=10, steps_per_sample=1)
        bucket_active = sample[0, :n_buckets]
        if bucket_active.sum() == 0:
            # Degenerate draw (can happen with weak biases): keep the single
            # highest-importance bucket rather than masking every key.
            bucket_active = torch.zeros(n_buckets)
            bucket_active[torch.argmax(bucket_importance)] = 1.0
        key_active = bucket_active.repeat_interleave(bucket_size)[:t]
        return key_active.bool()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))

        key_active = self._sample_key_mask(att, t)  # (t,), True = attend
        if not key_active.all():
            att = att.masked_fill(~key_active.view(1, 1, 1, t), float("-inf"))

        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class ThermodynamicSampling(nn.Module, SamplingStrategy):
    """Energy-based alternative to softmax sampling: converts logits into
    Ising biases and runs real Gibbs sampling, then decodes the resulting
    spin configuration back into a token index via a binary-reduction
    scheme (each spin halves the remaining candidate range, weighted by the
    logit mass in each half -- so high-probability tokens are, in
    expectation, still favoured, but the *mechanism* selecting them is
    physical energy minimization rather than inverse-CDF sampling).
    """

    paradigm = Paradigm.THERMODYNAMIC

    def __init__(self, vocab_size: int, beta: float = 1.0, max_spins: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_spins = _logits_to_spin_count(vocab_size, max_spins)
        self.sampler = ThermodynamicSamplerAdapter(n_spins=self.n_spins, beta=beta)

    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        b = logits.shape[0]
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1).detach()
        out_tokens = torch.zeros(b, dtype=torch.long)
        for bi in range(b):
            lo, hi = 0, self.vocab_size
            p = probs[bi]
            for _ in range(self.n_spins):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                mass_low = p[lo:mid].sum()
                mass_high = p[mid:hi].sum()
                bias = torch.tensor(
                    [float(mass_low - mass_high)] * self.n_spins
                )
                sample = self.sampler.sample(
                    biases=bias, n_samples=1, n_warmup=8, steps_per_sample=1
                )
                spin_up = sample[0, 0].item() > 0.5
                if spin_up:
                    hi = mid
                else:
                    lo = mid
            out_tokens[bi] = lo
        return out_tokens


class ThermodynamicSearch(nn.Module, SearchStrategy):
    """Search -> Thermodynamic: given a bank of candidates and an energy
    function scoring them, runs simulated-annealing-flavoured Gibbs
    sampling (via decreasing beta, i.e. rising temperature -> falling, i.e.
    annealing towards low energy) to pick a low-energy (favoured) candidate,
    rather than an exhaustive or purely greedy classical search.
    """

    paradigm = Paradigm.THERMODYNAMIC

    def __init__(self, max_candidates: int = 16, beta: float = 1.5):
        super().__init__()
        self.max_candidates = max_candidates
        self.sampler = ThermodynamicSamplerAdapter(n_spins=max_candidates, beta=beta)

    def forward(self, candidates: torch.Tensor, energy_fn=None) -> torch.Tensor:
        n = min(candidates.shape[0], self.max_candidates)
        cand = candidates[:n]
        if energy_fn is not None:
            energies = energy_fn(cand).detach().float()
        else:
            energies = -cand.float().mean(dim=tuple(range(1, cand.ndim))) if cand.ndim > 1 else -cand.float()
        # Lower energy = more favoured -> higher bias for the sampler.
        biases = torch.zeros(self.max_candidates)
        biases[:n] = -energies
        sample = self.sampler.sample(biases=biases, n_samples=1, n_warmup=15, steps_per_sample=2)
        active = sample[0, :n]
        if active.sum() == 0:
            # No spin landed "up" (can happen with weak biases); fall back
            # to the lowest-energy candidate directly.
            idx = torch.argmin(energies)
        else:
            idx = torch.argmin(energies + (1 - active) * 1e6)
        return cand[idx]


class ThermodynamicEnergyBasedSampling(nn.Module, EnergyBasedSamplingStrategy):
    """Energy-based Sampling -> Thermodynamic: directly exposes the Ising
    Gibbs sampler as a distribution-shaping step over logits, returning a
    resampled probability vector reflecting the physical sampler's
    visitation frequencies rather than the raw softmax.
    """

    paradigm = Paradigm.THERMODYNAMIC

    def __init__(self, max_bins: int = 16, beta: float = 1.0):
        super().__init__()
        self.max_bins = max_bins
        self.sampler = ThermodynamicSamplerAdapter(n_spins=max_bins, beta=beta)

    def forward(self, logits: torch.Tensor, steps: int = 10) -> torch.Tensor:
        vocab = logits.shape[-1]
        bins = min(self.max_bins, vocab)
        # Coarsen the vocabulary into `bins` buckets, bias each bucket by
        # its total logit mass, sample visitation frequency, then spread
        # that back out uniformly within each bucket as a probability.
        bucket_size = (vocab + bins - 1) // bins
        padded = F.pad(logits, (0, bucket_size * bins - vocab), value=float("-inf"))
        bucket_logits = padded.view(*logits.shape[:-1], bins, bucket_size)
        bucket_mass = torch.logsumexp(bucket_logits, dim=-1)

        flat_mass = bucket_mass.reshape(-1, bins)
        out = torch.zeros_like(flat_mass)
        for i in range(flat_mass.shape[0]):
            biases = torch.zeros(self.max_bins)
            biases[:bins] = flat_mass[i] - flat_mass[i].mean()
            samples = self.sampler.sample(
                biases=biases, n_samples=max(steps, 1), n_warmup=10, steps_per_sample=1
            )
            freq = samples[:, :bins].mean(dim=0)
            out[i] = freq / freq.sum().clamp_min(1e-8)

        bucket_probs = out.reshape(*logits.shape[:-1], bins)
        token_probs = (bucket_probs.unsqueeze(-1) / bucket_size).expand(
            *bucket_probs.shape, bucket_size
        ).reshape(*logits.shape[:-1], bins * bucket_size)[..., :vocab]
        return token_probs / token_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
