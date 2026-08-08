# Multi-Paradigm LLM (MP-LLM)

A research-oriented LLM architecture that builds every component (embedding,
attention, FFN, sampling, ...) from one of four **real** computational
paradigms -- Classical (PyTorch), Quantum (PennyLane), Photonic
(PhotonTorch), Thermodynamic (THRML) -- selected by a swappable scheduler,
and benchmarks them against each other.

Two complete models are implemented: an autoregressive Transformer
(`TransformerLM`, nanoGPT-style) and a masked Diffusion LM (`DiffusionLM`).
Both can be built under five policies -- `ClassicalLLM`, `QuantumLLM`,
`PhotonicLLM`, `ThermodynamicLLM`, `HybridLLM` -- sharing one
implementation each, via the same Strategy/Factory-built component stack.

## Read this first: what "real" means here

Every paradigm library is genuinely installed, imported, and exercised --
nothing is mocked or hand-waved. Concretely:

- **Classical**: standard `torch.nn` layers.
- **Quantum**: real PennyLane `default.qubit` state-vector circuits
  (`AngleEmbedding` + `BasicEntanglerLayers`), wired into `torch.autograd`
  via `qml.qnn.TorchLayer`. Backprop through the circuit is exact autodiff
  through the simulator, not parameter-shift on real QPU hardware.
- **Photonic**: a real PhotonTorch `ClementsNxN` mesh -- light physically
  interfering through trainable Mach-Zehnder interferometers (frequency-
  domain steady-state simulation), not a matrix multiply dressed up to look
  photonic. Getting this working against modern PyTorch required three
  documented compatibility fixes in `adapters/photonic_adapter.py`
  (PhotonTorch was last released in 2019): a polyfill for the removed
  named-tensor API, a `torch.solve` -> `torch.linalg.solve` shim, and
  discovering that `photontorch.Environment` defaults to `grad=False` and
  silently wraps its simulation in `torch.no_grad()` unless told otherwise.
- **Thermodynamic**: real block-Gibbs sampling from an Ising energy-based
  model via THRML (Extropic's JAX library) -- confirmed to actually skew
  sampled spins toward high-bias states, not just return noise.

**None of this is a measurement of real quantum/photonic/thermodynamic
hardware.** All four "paradigms" execute as software simulation on ordinary
CPU today (PennyLane's state-vector simulator, PhotonTorch's PDE solver,
THRML's JAX Gibbs sampler all run on classical silicon). The
`@quantum_timer` / `@photonic_timer` / `@thermodynamic_timer` decorators
record real wall-clock *simulation* time, then multiply by a configurable,
openly speculative `hardware_factor` to produce an "estimated hardware
time" -- a placeholder for once real hardware benchmarks exist, exactly as
the project spec requests. See `benchmarking/timers.py`.

**Everything runs at toy scale on purpose.** A real photonic circuit
simulation or block-Gibbs sampler does not scale like a dense GEMM;
`d_model`, qubit counts, mesh sizes, and spin counts are all kept small
(single digits to low tens) so the whole stack -- including PhotonTorch and
THRML -- actually finishes in seconds rather than minutes. This is itself
one of the honest results the benchmark reports below surface: e.g. a
photonic attention layer at toy scale runs ~500x *slower* in simulation
than the classical baseline, while its *theoretical* hardware-adjusted time
(after applying `hardware_factor`) is ~25x faster -- both numbers are
worth reporting, and neither should be mistaken for the other.

## Architecture

```
core/interfaces.py        Abstract Strategy interfaces every component implements
core/types.py                Paradigm enum, ComponentKind enum, timing/choice dataclasses

adapters/                   Adapter Pattern: wrap PennyLane / PhotonTorch / THRML
  quantum_adapter.py
  photonic_adapter.py        (+ the 3 PyTorch-compat shims, documented inline)
  thermodynamic_adapter.py

paradigms/<paradigm>/components.py   Strategy Pattern: concrete component implementations
  classical/    -- every component (reference implementation)
  quantum/      -- Attention (gating), ProbabilityEstimation, Randomness, Sampling
  photonic/     -- Attention, FeedForward, generic Linear (all via the MZI mesh)
  thermodynamic/-- Attention (sparse key selection), Sampling, Search,
                    EnergyBasedSampling (all via Ising Gibbs sampling)

factory.py                  Factory Pattern: ModuleFactory.create_<component>(paradigm, ...)
scheduler.py                 Strategy Pattern: HybridScheduler + swappable SchedulingPolicy
                               (DefaultSchedulingPolicy, SingleParadigmPolicy, AllClassicalPolicy)
                               + automatic classical-fallback + human-readable rationale

models/
  transformer_llm.py          Model 1: autoregressive Transformer (TransformerLM)
  diffusion_llm.py             Model 2: masked Diffusion LM (DiffusionLM)
  paradigm_llms.py              ParadigmLLM (shared LLMInterface) + the 5 named classes:
                                  ClassicalLLM, QuantumLLM, PhotonicLLM, ThermodynamicLLM, HybridLLM
  tokenizer.py                   Character-level Tokenizer + Vocabulary

pipelines/
  training_pipeline.py         Pipeline Pattern: load_dataset -> build_tokenizer -> tokenize ->
                                 build_model -> train -> evaluate -> save_model -> save_metrics
  inference_pipeline.py         Pipeline Pattern: load_model -> load_tokenizer -> encode ->
                                 forward -> sample -> decode -> print

benchmarking/
  timers.py                    @timer, @classical_timer, @quantum_timer, @photonic_timer,
                                 @thermodynamic_timer + configurable HARDWARE_FACTORS
  report.py                     Aggregates timings into JSON/Markdown benchmark reports

persistence/checkpoint.py     save/load checkpoints, tokenizer, config, resume training
config/config.py               MPLLMConfig dataclass (every tunable parameter)
data/                          Tiny built-in toy corpus + in-memory batch sampler
logging_utils.py               Structured logging (loss, perplexity, accuracy, timing, RAM/GPU)
scripts/compare_paradigms.py   Builds all 5 variants and writes a combined comparison report
main.py                        CLI: train / infer / compare
tests/test_smoke.py             End-to-end integration tests (all 5 variants x both architectures)
```

### Design patterns, concretely

- **Strategy**: `AttentionStrategy` / `SamplingStrategy` / etc. in
  `core/interfaces.py`; every paradigm implements the same interface.
  `SchedulingPolicy` (in `scheduler.py`) is also a Strategy -- swap
  `DefaultSchedulingPolicy` for `AllClassicalPolicy` or write your own,
  at runtime, with zero changes to model code.
- **Factory**: `ModuleFactory.create_attention(paradigm, ...)` etc. is the
  only place that imports concrete paradigm implementations; everything
  else depends only on abstract interfaces.
- **Adapter**: `adapters/*_adapter.py` wrap PennyLane/PhotonTorch/THRML
  behind plain, paradigm-agnostic classes.
- **Pipeline**: `TrainingPipeline` / `InferencePipeline` split the workflow
  into named, independently-overridable stages matching the spec's diagram.
- **Dependency Injection**: every component receives its backend as a
  constructor argument (e.g. `PhotonicLinear` builds its own
  `PhotonicMeshAdapter` internally but every *model* class receives its
  `HybridScheduler` from outside); `ParadigmLLM` receives its model class
  and policy rather than hardcoding either.
- **SOLID**: one `TransformerBlock`/`ParadigmLLM` implementation is reused
  by all five named LLM classes and both architectures (no duplication);
  new paradigms extend the scheduler+factory without touching model code
  (Open/Closed); `core/interfaces.py` enforces Interface Segregation.

## Install

```bash
pip install -r requirements.txt --break-system-packages
```

## Quick start

```bash
# Train the true multi-paradigm Hybrid model on the built-in toy corpus
python main.py train --llm hybrid --model transformer \
    --run-dir outputs/checkpoints/hybrid_run \
    --d-model 256 --n-heads 16 --n-layers 16 --steps-per-epoch 5000

# Generate from it
python main.py infer --llm hybrid --run-dir outputs/checkpoints/hybrid_run \
    --prompt "the " --max-new-tokens 30

# Build all 5 variants and write a comparison benchmark report
python main.py compare --model transformer --out-dir outputs/benchmarks
python main.py compare --model diffusion --out-dir outputs/benchmarks
```

`--llm` accepts `classical`, `quantum`, `photonic`, `thermodynamic`,
`hybrid`. `--model` accepts `transformer` or `diffusion`.

Programmatically:

```python
from config.config import MPLLMConfig
from models.paradigm_llms import HybridLLM
from data.toy_corpus import TOY_CORPUS
from pipelines.training_pipeline import TrainingPipeline

cfg = MPLLMConfig(d_model=32, n_heads=4, n_layers=2, steps_per_epoch=20)
pipe = TrainingPipeline(cfg, HybridLLM, TOY_CORPUS, run_dir="outputs/checkpoints/run1")
result = pipe.run()
print(result["llm"].model.scheduler.explain())  # paradigm-selection rationale
```

## Using more/real data

The built-in `TOY_CORPUS` (~500 characters) is deliberately tiny so the
whole stack -- including the Quantum/Photonic/Thermodynamic simulators --
runs fast out of the box. It is not enough data to learn real language
structure; a 500-character corpus is essentially memorized, not learned
from. `data/corpus_loader.py` adds a `tinyshakespeare` option (downloaded
once from `raw.githubusercontent.com` and cached under `data/.cache/`, with
an automatic fallback to the toy corpus if there's no network access):

```bash
python main.py train --llm classical --model transformer \
    --run-dir outputs/checkpoints/shakespeare_run \
    --corpus tinyshakespeare --max-chars 20000 \
    --d-model 64 --n-heads 4 --n-layers 3 --d-ff 128 --context-length 48 \
    --batch-size 16 --steps-per-epoch 150
```

This genuinely helps: on a 20K-character slice with a slightly bigger model
and 150 steps, next-character accuracy went from ~4% (random guess over the
58-character vocabulary) to ~36%, and generated text starts producing
real short words and punctuation patterns instead of noise. Three things
matter together, not just corpus size:

- **Model capacity** needs to grow with the corpus (`--d-model`,
  `--n-layers`) or the extra data mostly goes unused.
- **`--steps-per-epoch` / `--epochs`** need to grow too -- more data with
  the same handful of gradient steps just means each step sees a different
  random window, not more learning per step.
- **Paradigm-specific costs scale differently.** Larger `--d-model` slows
  the photonic mesh roughly linearly (more `mesh_size`-wide chunks per
  call). `--max-spins` (thermodynamic) only needs to grow with
  `log2(vocab_size)`, so a bigger corpus barely affects thermodynamic
  sampling cost -- the default (6-8) comfortably covers even
  tinyshakespeare's 58-character vocabulary. More diffusion timesteps or a
  longer `max_new_tokens` at generation time multiply thermodynamic/quantum
  calls directly, regardless of corpus size -- that's the knob to watch for
  `--llm hybrid`/`thermodynamic`, not corpus size.

To add your own corpus, add an entry to `_CORPUS_URLS` in
`data/corpus_loader.py`, or just pass a local file's contents directly to
`TrainingPipeline(cfg, llm_cls, my_text, run_dir=...)` instead of going
through `main.py`.

## Run the tests

```bash
python -m tests.test_smoke
```

(Run as a module -- `python tests/test_smoke.py` directly will fail to
resolve the project's absolute imports.)

## Configuration

All tunables live in `config/config.py::MPLLMConfig` (architecture size,
optimizer/lr, `scheduler_policy`, and the paradigm-specific knobs
`photonic_mesh_size`, `quantum_n_qubits`, `thermodynamic_beta`,
`thermodynamic_max_spins`, `diffusion_timesteps`). Saved/loaded as JSON
alongside every checkpoint.

## Honest limitations

- **THRML components are not end-to-end differentiable** here (JAX-side
  discrete Gibbs sampling doesn't hook into `torch.autograd`); they're used
  at points that don't need gradients through them (token sampling at
  generation time), not inside a trained hidden layer. This is a genuine
  property of the paradigm as implemented, not a bug to hide.
- **Vocabulary is tiny by design** (character-level, built-in toy corpus)
  so `ThermodynamicSampling`'s binary-search-over-spins decoding and
  `PhotonicLinear`'s block-mesh application both stay fast; neither
  approach is described as scaling to a real-sized vocabulary/`d_model`.
- **PhotonTorch (0.4.1, 2019) and current PyTorch (2.13)** are bridged with
  three small, explicitly-documented compatibility shims in
  `adapters/photonic_adapter.py` rather than pinning an old torch globally.
- **Photonic/Quantum "attention"** only replace the *linear* sub-steps
  (projections / a learned gate); softmax and other nonlinearities remain
  classical, since neither a passive interferometer mesh nor a variational
  circuit's expectation values are a drop-in nonlinearity.
- **Thermodynamic sampling is slow at generation time.** `HybridLLM`'s
  default policy picks Thermodynamic for token Sampling, and each sampled
  token runs a binary-search-style reduction (up to `log2(vocab_size)`
  sequential THRML Gibbs-sampling calls, each paying real JAX JIT-
  compilation overhead). Generating even a handful of tokens can take tens
  of seconds at this toy scale -- a genuine, reportable data point about
  this paradigm's current practicality for autoregressive decoding, not a
  bug. `python main.py compare` captures this quantitatively; for fast
  interactive generation, use `--llm classical`.
