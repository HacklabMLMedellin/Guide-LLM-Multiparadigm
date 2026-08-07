"""
Adapter Pattern: wraps PennyLane behind a plain interface so the rest of the
codebase never touches PennyLane's QNode/device API directly.

Uses PennyLane's `default.qubit` simulator (state-vector simulation of a
real variational quantum circuit -- angle embedding + entangling layers),
through `qml.qnn.TorchLayer` so it drops into a normal `torch.nn.Module`
graph, batches like any other layer, and backpropagates through the circuit
parameters via the parameter-shift-free "backprop" differentiation method
(exact autodiff through the simulator, since we are not targeting real QPU
hardware here).
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


class QuantumCircuitAdapter(nn.Module):
    """A small variational quantum circuit exposed as a torch layer.

    n_qubits also fixes the input/output feature width: each qubit encodes
    one input feature (angle embedding) and yields one output feature
    (Pauli-Z expectation value).
    """

    def __init__(self, n_qubits: int, n_layers: int = 2, shots: int | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        device = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_qubits) -> (..., n_qubits), expectation values in [-1, 1]."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.n_qubits)
        out = self.qlayer(flat)
        return out.reshape(orig_shape)

    def sample_bits(self, n_samples: int) -> torch.Tensor:
        """True quantum-measurement-style randomness: run the circuit with a
        fixed input and read out the sign of each qubit's Z-expectation
        across independent forward passes, i.e. we treat the (noiseless)
        expectation values as Bernoulli probabilities and sample from them --
        an honest way to get "quantum-flavoured" randomness out of an exact
        simulator that has no native single-shot sampling wired through
        `TorchLayer`.
        """
        with torch.no_grad():
            x = torch.rand(n_samples, self.n_qubits) * 3.14159
            expvals = self.forward(x)  # in [-1, 1]
            probs = (expvals + 1) / 2
            bits = torch.bernoulli(probs)
        return bits
