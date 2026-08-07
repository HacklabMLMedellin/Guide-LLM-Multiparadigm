# Benchmark Report: HybridLLM

**Trainable parameters:** 13,000
**Total parameters:** 13,000
**Process peak memory:** 991.2 MB

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
| TransformerLM._run_embedding [classical] | 23 | 0.006989 | 0.006989 | 1.00x |
| TransformerBlock._run_attn [photonic] | 46 | 28.305928 | 1.132237 | 25.00x |
| TransformerBlock._run_ffn [classical] | 46 | 0.027443 | 0.027443 | 1.00x |

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
