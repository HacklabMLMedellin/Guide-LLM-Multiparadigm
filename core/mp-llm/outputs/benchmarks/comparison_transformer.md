# Paradigm Comparison: transformer

| Variant | Params | Total sim time (s) | Total est. hw time (s) |
|---|---|---|---|
| ClassicalLLM | 3,246 | 0.002844 | 0.002844 |
| QuantumLLM | 3,406 | 0.020801 | 0.002180 |
| PhotonicLLM | 2,440 | 0.950445 | 0.038136 |
| ThermodynamicLLM | 3,246 | 3.468204 | 0.416809 |
| HybridLLM | 2,358 | 0.708692 | 0.029066 |

## ClassicalLLM

# Benchmark Report: ClassicalLLM

**Trainable parameters:** 3,246
**Total parameters:** 3,246
**Process peak memory:** 905.1 MB

## Hardware time-conversion factors (configurable, theoretical)

| Paradigm | Factor (real_time = simulated_time x factor) |
|---|---|
| classical | 1.0 |
| quantum | 0.08 |
| photonic | 0.04 |
| thermodynamic | 0.12 |

## Timing summary (simulated vs. estimated future hardware)

| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |
|---|---|---|---|---|
| TransformerLM._run_embedding [classical] | 2 | 0.000287 | 0.000287 | 1.00x |
| TransformerBlock._run_attn [classical] | 2 | 0.001623 | 0.001623 | 1.00x |
| TransformerBlock._run_ffn [classical] | 2 | 0.000934 | 0.000934 | 1.00x |

## Paradigm selection rationale

**embedding -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**positional_encoding -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**output_projection -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**sampling -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**attention -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x

**feed_forward -> classical**

> AllClassicalPolicy: every component forced classical for a fast, fully-differentiable baseline run.
>
> Estimated speedup: 1.0x


## QuantumLLM

# Benchmark Report: QuantumLLM

**Trainable parameters:** 3,406
**Total parameters:** 3,406
**Process peak memory:** 906.9 MB

## Hardware time-conversion factors (configurable, theoretical)

| Paradigm | Factor (real_time = simulated_time x factor) |
|---|---|
| classical | 1.0 |
| quantum | 0.08 |
| photonic | 0.04 |
| thermodynamic | 0.12 |

## Timing summary (simulated vs. estimated future hardware)

| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |
|---|---|---|---|---|
| TransformerLM._run_embedding [classical] | 2 | 0.000096 | 0.000096 | 1.00x |
| TransformerBlock._run_attn [quantum] | 2 | 0.020240 | 0.001619 | 12.50x |
| TransformerBlock._run_ffn [classical] | 2 | 0.000466 | 0.000466 | 1.00x |

## Paradigm selection rationale

**embedding -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for embedding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**positional_encoding -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for positional_encoding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**output_projection -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for output_projection in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**sampling -> quantum**

> Uses genuine quantum-measurement randomness to drive the same inverse-CDF sampling procedure, rather than a classical PRNG.
>
> Estimated speedup: 3.0x

**layer_norm -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**attention -> quantum**

> A small variational circuit can act as a learned, non-classical gating function over attention output, exploiting the circuit's expressivity per parameter -- explored here as a research variant rather than a speed optimization.
>
> Estimated speedup: 3.0x

**layer_norm -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**feed_forward -> classical**

> SingleParadigmPolicy(quantum): requested for every component to build a single-paradigm model. [Fallback: quantum has no implementation for feed_forward in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x


## PhotonicLLM

# Benchmark Report: PhotonicLLM

**Trainable parameters:** 2,440
**Total parameters:** 2,440
**Process peak memory:** 922.9 MB

## Hardware time-conversion factors (configurable, theoretical)

| Paradigm | Factor (real_time = simulated_time x factor) |
|---|---|
| classical | 1.0 |
| quantum | 0.08 |
| photonic | 0.04 |
| thermodynamic | 0.12 |

## Timing summary (simulated vs. estimated future hardware)

| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |
|---|---|---|---|---|
| TransformerLM._run_embedding [classical] | 2 | 0.000123 | 0.000123 | 1.00x |
| TransformerBlock._run_attn [photonic] | 2 | 0.762435 | 0.030497 | 25.00x |
| TransformerBlock._run_ffn [photonic] | 2 | 0.187886 | 0.007515 | 25.00x |

## Paradigm selection rationale

**embedding -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for embedding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**positional_encoding -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for positional_encoding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**output_projection -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for output_projection in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**sampling -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for sampling in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**attention -> photonic**

> Large matrix multiplications dominate this layer. Photonic interference can theoretically accelerate the linear-algebra portion of attention (the Q/K/V and output projections) once real photonic hardware exists, at the cost of today's simulation overhead.
>
> Estimated speedup: 23.0x

**layer_norm -> classical**

> SingleParadigmPolicy(photonic): requested for every component to build a single-paradigm model. [Fallback: photonic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**feed_forward -> photonic**

> The inner (dimension-preserving) transform is pure linear algebra and can, in principle, be realized optically the same way attention's projections can.
>
> Estimated speedup: 23.0x


## ThermodynamicLLM

# Benchmark Report: ThermodynamicLLM

**Trainable parameters:** 3,246
**Total parameters:** 3,246
**Process peak memory:** 1067.1 MB

## Hardware time-conversion factors (configurable, theoretical)

| Paradigm | Factor (real_time = simulated_time x factor) |
|---|---|
| classical | 1.0 |
| quantum | 0.08 |
| photonic | 0.04 |
| thermodynamic | 0.12 |

## Timing summary (simulated vs. estimated future hardware)

| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |
|---|---|---|---|---|
| TransformerLM._run_embedding [classical] | 2 | 0.000210 | 0.000210 | 1.00x |
| TransformerBlock._run_attn [thermodynamic] | 2 | 3.467494 | 0.416099 | 8.33x |
| TransformerBlock._run_ffn [classical] | 2 | 0.000500 | 0.000500 | 1.00x |

## Paradigm selection rationale

**embedding -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for embedding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**positional_encoding -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for positional_encoding in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**output_projection -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for output_projection in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**sampling -> thermodynamic**

> Energy-based probabilistic sampling naturally maps to thermodynamic optimization: converting logits into biases on an Ising model and letting the system settle reframes 'sample a token' as 'find a low-energy state', the native operation of thermodynamic hardware.
>
> Estimated speedup: 8.0x

**layer_norm -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**attention -> thermodynamic**

> Frames 'which keys matter' as an Ising model over key positions and samples a physically-motivated sparse attention pattern via Gibbs sampling, instead of dense deterministic softmax weighting -- a discrete selection problem, which is thermodynamic sampling's strength.
>
> Estimated speedup: 8.0x

**layer_norm -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for layer_norm in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x

**feed_forward -> classical**

> SingleParadigmPolicy(thermodynamic): requested for every component to build a single-paradigm model. [Fallback: thermodynamic has no implementation for feed_forward in this codebase; used classical instead.]
>
> Estimated speedup: 1.0x


## HybridLLM

# Benchmark Report: HybridLLM

**Trainable parameters:** 2,358
**Total parameters:** 2,358
**Process peak memory:** 1067.6 MB

## Hardware time-conversion factors (configurable, theoretical)

| Paradigm | Factor (real_time = simulated_time x factor) |
|---|---|
| classical | 1.0 |
| quantum | 0.08 |
| photonic | 0.04 |
| thermodynamic | 0.12 |

## Timing summary (simulated vs. estimated future hardware)

| Function [paradigm] | Calls | Simulated (s) | Est. hardware (s) | Speedup |
|---|---|---|---|---|
| TransformerLM._run_embedding [classical] | 2 | 0.000108 | 0.000108 | 1.00x |
| TransformerBlock._run_attn [photonic] | 2 | 0.707943 | 0.028318 | 25.00x |
| TransformerBlock._run_ffn [classical] | 2 | 0.000640 | 0.000640 | 1.00x |

## Paradigm selection rationale

**embedding -> classical**

> Embedding lookup is a simple indexed memory read; there is no linear-algebra or sampling advantage another paradigm could offer here, so classical dense storage is strictly the cheapest option.
>
> Estimated speedup: 1.0x

**positional_encoding -> classical**

> A fixed or learned per-position vector; a static lookup, not a computation worth specializing.
>
> Estimated speedup: 1.0x

**layer_norm -> classical**

> Elementwise normalization; a memory-bandwidth-bound op with no known paradigm-specific advantage.
>
> Estimated speedup: 1.0x

**output_projection -> classical**

> A single large dense projection to vocabulary size; classical GEMM is the practical choice at any real vocabulary size.
>
> Estimated speedup: 1.0x

**sampling -> thermodynamic**

> Energy-based probabilistic sampling naturally maps to thermodynamic optimization: converting logits into biases on an Ising model and letting the system settle reframes 'sample a token' as 'find a low-energy state', the native operation of thermodynamic hardware.
>
> Estimated speedup: 8.0x

**layer_norm -> classical**

> Elementwise normalization; a memory-bandwidth-bound op with no known paradigm-specific advantage.
>
> Estimated speedup: 1.0x

**attention -> photonic**

> Large matrix multiplications dominate this layer. Photonic interference can theoretically accelerate the linear-algebra portion of attention (the Q/K/V and output projections) once real photonic hardware exists, at the cost of today's simulation overhead.
>
> Estimated speedup: 23.0x

**layer_norm -> classical**

> Elementwise normalization; a memory-bandwidth-bound op with no known paradigm-specific advantage.
>
> Estimated speedup: 1.0x

**feed_forward -> classical**

> Dense two-layer MLP; the standard, always-available choice.
>
> Estimated speedup: 1.0x

