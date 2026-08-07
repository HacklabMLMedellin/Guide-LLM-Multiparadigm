"""
A tiny, built-in, offline toy corpus. The point of this project is to prove
out a *real* multi-paradigm architecture end-to-end, including the
Quantum/Photonic/Thermodynamic simulators -- not to train a good language
model. Keeping the corpus small and self-contained means training actually
finishes at toy scale without needing any external dataset download.
"""

TOY_CORPUS = """
the multi paradigm model learns from a tiny corpus.
classical computing runs on ordinary transistors and dense matrix math.
quantum computing uses qubits, superposition, and entanglement.
photonic computing sends light through waveguides and interferometers.
thermodynamic computing samples low energy states directly from physics.
the hybrid scheduler picks the best paradigm for each part of the model.
attention lets a token look at every other token in the sequence.
sampling turns a probability distribution into a single chosen token.
a diffusion model learns to denoise a sequence back into clean text.
an autoregressive model predicts the next token one step at a time.
""".strip()
