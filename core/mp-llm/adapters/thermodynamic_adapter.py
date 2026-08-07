"""
Adapter Pattern: wraps THRML (Extropic's JAX-based thermodynamic/energy-based
sampling library) behind a plain interface so the rest of the codebase never
touches THRML/JAX directly.

THRML samples from an Ising energy-based model,

    E(s) = -beta * ( sum_i b_i s_i + sum_(i,j) J_ij s_i s_j )

via block Gibbs sampling -- the same computational structure Extropic's
thermodynamic hardware is designed to accelerate natively (their sampling
units settle into low-energy states physically, instead of via a simulated
Markov chain). Here it runs as a Gibbs-sampling simulation on CPU/GPU via
JAX; we treat `beta` (inverse temperature) as the practical knob and expose
it as the primary interface, since it is what actually changes sampling
behaviour (high beta -> sharper / more greedy, low beta -> more exploratory
-- directly analogous to softmax temperature, which is why this maps onto
LLM *sampling* and *search* so naturally).
"""

from __future__ import annotations

from typing import Optional

import jax
import numpy as np
import torch
from jax import numpy as jnp

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


class ThermodynamicSamplerAdapter:
    """A small fully-connected (or ring-connected) Ising energy-based model
    used as a stochastic sampler / search mechanism over `n_spins` binary
    units. Bias vector is set per-call from (e.g.) token logits, so this can
    act as an energy-based alternative to softmax sampling: high-logit
    tokens get a large positive bias, i.e. a low-energy (favoured) state.
    """

    def __init__(self, n_spins: int, beta: float = 1.0, seed: int = 0):
        self.n_spins = n_spins
        self.beta = beta
        self._key = jax.random.PRNGKey(seed)

        self.nodes = [SpinNode() for _ in range(n_spins)]
        # Ring graph: sparse enough to 2-colour (required for block Gibbs),
        # dense enough to let neighbouring spins interact.
        self.edges = [(self.nodes[i], self.nodes[(i + 1) % n_spins]) for i in range(n_spins)]
        self.free_blocks = [Block(self.nodes[::2]), Block(self.nodes[1::2])]
        self.observe_block = [Block(self.nodes)]

    def _next_key(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def sample(
        self,
        biases: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        n_warmup: int = 20,
        steps_per_sample: int = 2,
    ) -> torch.Tensor:
        """Run block Gibbs sampling and return `n_samples` spin configurations.

        Args:
            biases: (n_spins,) tensor, e.g. derived from token logits.
            weights: (n_spins,) tensor of ring-edge couplings (defaults to a
                mild ferromagnetic coupling encouraging locally-consistent
                picks).
            n_samples: number of samples to draw.
        Returns:
            (n_samples, n_spins) float tensor in {0., 1.} (spin "up"/"down").
        """
        biases_j = jnp.array(biases.detach().cpu().numpy(), dtype=jnp.float32)
        if weights is None:
            weights_j = jnp.full((self.n_spins,), 0.1, dtype=jnp.float32)
        else:
            weights_j = jnp.array(weights.detach().cpu().numpy(), dtype=jnp.float32)

        model = IsingEBM(
            self.nodes, self.edges, biases_j, weights_j, jnp.array(self.beta)
        )
        program = IsingSamplingProgram(model, self.free_blocks, clamped_blocks=[])
        init_state = hinton_init(self._next_key(), model, self.free_blocks, ())
        schedule = SamplingSchedule(
            n_warmup=n_warmup, n_samples=n_samples, steps_per_sample=steps_per_sample
        )
        samples = sample_states(
            self._next_key(), program, schedule, init_state, [], self.observe_block
        )
        arr = np.asarray(samples[0])  # (n_samples, n_spins) bool
        return torch.tensor(arr, dtype=torch.float32)
