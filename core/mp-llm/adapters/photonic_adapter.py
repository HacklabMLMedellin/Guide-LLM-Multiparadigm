"""
Adapter Pattern: wraps PhotonTorch behind a plain, paradigm-agnostic
`PhotonicMeshAdapter` so the rest of the codebase never touches PhotonTorch's
API directly.

PhotonTorch (last released 2019, v0.4.1) predates several PyTorch APIs it
depends on internally. Rather than pin an incompatible old torch (which
would fragment the whole project's environment), this module applies three
small, well-scoped compatibility shims the first time it is imported:

  1. Named tensors (`Tensor.names` / `.rename` / `.align_to` / `.refine_names`)
     were removed from modern PyTorch. PhotonTorch's `Network._handle_source`
     uses them purely for axis bookkeeping (t, w, s, b / c, t, w, s, b), never
     to reorder physics -- so a minimal polyfill that tracks names via a
     plain Python attribute and performs a real `permute` in `align_to`
     is behaviourally equivalent to the removed API for this use case.

  2. `torch.solve` was removed in favour of `torch.linalg.solve` (with the
     argument order reversed). We restore the old call signature as a thin
     wrapper.

  3. PhotonTorch's `Environment` defaults to `grad=False` and wraps its
     simulation in `torch.no_grad()` unless told otherwise. We always
     construct it with `grad=True` so gradients reach the trainable MZI
     phase-shifter parameters -- without this, the "photonic weights" could
     never be trained.

These are documented here (not silently monkey-patched elsewhere) because
they are genuine version-compatibility fixes, not shortcuts around the
physics: the forward pass is a real time-/frequency-domain simulation of
light interfering through a mesh of trainable Mach-Zehnder interferometers
(the Clements architecture, the standard universal-unitary photonic layout).
"""

from __future__ import annotations

import torch

_SHIMMED = False


def _apply_compat_shims() -> None:
    global _SHIMMED
    if _SHIMMED:
        return

    def _rename(self, *names, **kwargs):
        if len(names) == 1 and names[0] is None:
            self._pt_names = tuple([None] * self.dim())
        else:
            self._pt_names = tuple(names)
        return self

    def _names_prop(self):
        return getattr(self, "_pt_names", tuple([None] * self.dim()))

    def _align_to(self, *target_names):
        current = self.names
        if current == tuple(target_names):
            return self
        perm = [current.index(n) for n in target_names]
        out = self.permute(*perm).contiguous()
        out._pt_names = tuple(target_names)
        return out

    if not hasattr(torch.Tensor, "_pt_names_patched"):
        torch.Tensor.rename = _rename
        torch.Tensor.names = property(_names_prop)
        torch.Tensor.refine_names = _rename
        torch.Tensor.align_to = _align_to
        torch.Tensor._pt_names_patched = True

    if not hasattr(torch, "_pt_solve_patched"):
        def _torch_solve(B, A):
            x = torch.linalg.solve(A, B)
            return x, A

        torch.solve = _torch_solve
        torch._pt_solve_patched = True

    _SHIMMED = True


_apply_compat_shims()

import photontorch as pt  # noqa: E402  (must follow the shim)


class PhotonicMeshAdapter:
    """Wraps a PhotonTorch ClementsNxN unitary mesh: a physically-simulated
    grid of trainable Mach-Zehnder interferometers that performs linear
    algebra via optical interference instead of MAC units.

    Only supports small N (a handful of ports) -- real time/frequency-domain
    photonic circuit simulation does not scale the way a dense matmul does,
    which is itself one of the honest data points this project's
    benchmarking is meant to surface.
    """

    def __init__(self, n_ports: int, wavelength: float = 1.55e-6):
        if n_ports < 2:
            raise ValueError("PhotonicMeshAdapter needs at least 2 ports")
        self.n_ports = n_ports
        self.wavelength = wavelength

        base = pt.ClementsNxN(N=n_ports)
        terms = [pt.Source(name=f"src{i}") for i in range(n_ports)] + [
            pt.Detector(name=f"det{i}") for i in range(n_ports)
        ]
        self.network = base.terminate(terms)
        self.env = pt.Environment(
            wl=wavelength, freqdomain=True, num_t=1, grad=True
        )

    def parameters(self):
        return self.network.parameters()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the photonic mesh to a batch of real-valued vectors.

        Args:
            x: (batch, n_ports) tensor of non-negative input amplitudes
               (this mesh, run in this mode, transforms optical *power*;
               see PhotonicLinear in paradigms/photonic for how signed
               activations are handled around this).
        Returns:
            (batch, n_ports) tensor of output optical powers after
            interference through the mesh.
        """
        batch = x.shape[0]
        # PhotonTorch source layout is (t, w, s, b): 1 timestep, 1 wavelength,
        # n_ports sources, batch.
        src = x.t().unsqueeze(0).unsqueeze(0).contiguous()
        with self.env:
            self.network.initialize()
            out = self.network(src)  # (t=1, w=1, n_ports, batch)
        return out[0, 0].t()  # (batch, n_ports)
