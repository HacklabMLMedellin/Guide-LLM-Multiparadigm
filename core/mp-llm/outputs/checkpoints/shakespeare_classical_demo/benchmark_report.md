# Benchmark Report: ClassicalLLM

**Trainable parameters:** 108,026
**Total parameters:** 108,026
**Process peak memory:** 920.5 MB

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
| TransformerLM._run_embedding [classical] | 153 | 0.018610 | 0.018610 | 1.00x |
| TransformerBlock._run_attn [classical] | 459 | 2.214451 | 2.214451 | 1.00x |
| TransformerBlock._run_ffn [classical] | 459 | 0.725850 | 0.725850 | 1.00x |

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
