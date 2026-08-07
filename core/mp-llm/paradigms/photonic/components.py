"""
Photonic paradigm: components implemented via a real, physically-simulated
mesh of trainable Mach-Zehnder interferometers (PhotonTorch's Clements
architecture), through `adapters.photonic_adapter.PhotonicMeshAdapter`.

Per the spec's scheduler rationale ("large matrix multiplications dominate
this layer; photonic interference can theoretically accelerate linear
algebra"), photonic involvement here is scoped to the *linear* sub-steps of
attention/FFN (the projections), while nonlinearities (softmax, GELU) stay
classical -- an optical mesh performs a unitary linear transform, it has no
native notion of a nonlinearity.

Because a physical interferometer mesh only realizes a *unitary* (energy-
preserving, square) transform of a bounded, non-negative optical power, two
practical adaptations are used everywhere below:

  1. Block application: a mesh of size `mesh_size` (kept small since a real
     time/frequency-domain photonic simulation does not scale like a dense
     GEMM) is applied independently to `dim // mesh_size` chunks of the
     feature vector -- physically analogous to tiling a large weight matrix
     across several smaller physical photonic chips.
  2. Differential (push-pull) encoding: since detected optical power is
     always >= 0, a signed activation x is split into (relu(x), relu(-x)),
     each half sent through the *same* mesh, and the outputs subtracted --
     the standard balanced-photodetection trick real photonic accelerators
     (e.g. MIT/Lightmatter-style designs) use to represent signed weights.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters.photonic_adapter import PhotonicMeshAdapter
from core.interfaces import AttentionStrategy, FeedForwardStrategy, MatMulStrategy
from core.types import Paradigm


class PhotonicLinear(nn.Module, MatMulStrategy):
    """A square, dim-preserving linear layer realized via a real photonic
    MZI mesh, block-applied across the feature dimension with differential
    encoding for signed values, followed by a small classical affine
    "electronic readout" stage (real photonic accelerators are hybrid
    opto-electronic systems; gain/offset calibration at the detector is
    standard, not a way of secretly doing the linear algebra classically --
    the mesh's trainable phase shifters are still what perform the
    dimension-mixing transform).
    """

    paradigm = Paradigm.PHOTONIC

    def __init__(self, dim: int, mesh_size: int = 4):
        super().__init__()
        if dim % mesh_size != 0:
            raise ValueError(f"dim={dim} must be divisible by mesh_size={mesh_size}")
        self.dim = dim
        self.mesh_size = mesh_size
        self.n_chunks = dim // mesh_size
        self.mesh = PhotonicMeshAdapter(n_ports=mesh_size)
        # Register the mesh's physical phase-shifter parameters as this
        # module's parameters so optimizers/checkpointing see them normally.
        for i, p in enumerate(self.mesh.parameters()):
            self.register_parameter(f"mesh_phase_{i}", p)
        self.readout_scale = nn.Parameter(torch.ones(dim))
        self.readout_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        flat = x.reshape(-1, self.dim)  # (N, dim)
        chunks = flat.reshape(-1, self.mesh_size)  # (N * n_chunks, mesh_size)

        pos = F.relu(chunks)
        neg = F.relu(-chunks)
        out_pos = self.mesh.transform(pos)
        out_neg = self.mesh.transform(neg)
        out = (out_pos - out_neg).reshape(-1, self.dim)

        out = out * self.readout_scale + self.readout_bias
        return out.reshape(orig_shape)


class PhotonicAttention(nn.Module, AttentionStrategy):
    """Scaled-dot-product attention whose Q/K/V and output projections run
    through `PhotonicLinear` instead of `nn.Linear`; softmax stays classical
    (it is not a linear-algebra operation and has no natural optical
    analogue in a passive interferometer mesh).
    """

    paradigm = Paradigm.PHOTONIC

    def __init__(self, d_model: int, n_heads: int, mesh_size: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = PhotonicLinear(d_model, mesh_size)
        self.k_proj = PhotonicLinear(d_model, mesh_size)
        self.v_proj = PhotonicLinear(d_model, mesh_size)
        self.out_proj = PhotonicLinear(d_model, mesh_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class PhotonicFeedForward(nn.Module, FeedForwardStrategy):
    """FFN with its (dimension-preserving) inner transform realized
    photonically; the up/down projections that change dimensionality stay
    classical since a passive unitary mesh cannot change vector dimension,
    only the square "mixing" sub-step can.
    """

    paradigm = Paradigm.PHOTONIC

    def __init__(self, d_model: int, d_ff: int, mesh_size: int = 4):
        super().__init__()
        if d_ff % mesh_size != 0:
            d_ff = (d_ff // mesh_size) * mesh_size or mesh_size
        self.up = nn.Linear(d_model, d_ff)
        self.mix = PhotonicLinear(d_ff, mesh_size)
        self.act = nn.GELU()
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.mix(h)
        h = self.act(h)
        return self.down(h)
