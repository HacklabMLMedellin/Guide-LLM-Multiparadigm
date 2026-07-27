# LLM Multi-Paradigm Guide

**Exploring thermodynamic (probabilistic) and quantum computing as alternative substrates for language models**

---

## Table of Contents

- [Overview](#overview)
- [Why These Paradigms?](#why-these-paradigms)
- [Learning Roadmap](#learning-roadmap)
- [Project Architecture](#project-architecture)
- [Tools & Libraries](#tools--libraries)
- [References](#references)
- [Community & Goals](#community--goals)
- [Status](#status)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project explores how a language model could be built on two non-classical computing paradigms — **thermodynamic (energy-based/probabilistic) computing** and **quantum computing** — alongside the standard classical paradigm used today.

The goal is not to claim a working advantage out of the gate, but to **study the mathematical properties of each paradigm** (probability distributions, sampling dynamics, linear algebra over Hilbert spaces, energy landscapes, etc.) and evaluate whether they offer a path toward:

- **Faster inference**
- **Faster Training** 
- **Cheaper**
- **Greater computational and energy efficiency**

for large language models.

No physical thermodynamic or quantum computer is required (or available) to do this work. Instead, this project relies on software libraries that **simulate** these paradigms on classical hardware:

- [**THRML**](https://github.com/extropic-ai/thrml) (Extropic) — a JAX library for building and sampling probabilistic graphical models and energy-based models (EBMs), designed as a prototyping ground for future thermodynamic sampling hardware.
- [**PennyLane**](https://pennylane.ai/quantum-machine-learning) — a cross-platform library for quantum machine learning, quantum computing, and quantum chemistry, with native integration into PyTorch/TensorFlow-style differentiable pipelines.

## Why These Paradigms?

A natural question is: *why not photonic computing? Why not stay purely classical? Why not biological computing?*

The short answer is **mathematical fit**. Language modeling is fundamentally:

1. **Probabilistic** — next-token prediction is a sampling problem over a probability distribution.
2. **Linear-algebra heavy** — embeddings, attention, and projections are all linear (or bilinear) operations over vector spaces.

Thermodynamic computing (via energy-based models and Gibbs/Boltzmann-style sampling) and quantum computing (via unitary evolution over Hilbert spaces and Born-rule sampling) each offer a **native mathematical language** for exactly these two properties — probability and linear algebra — in a way that photonic, purely-classical, or biological substrates don't map onto as directly. That overlap is what makes them worth studying here, rather than as a rejection of those other paradigms.

## Learning Roadmap

The project follows an incremental, "build it to understand it" workflow, moving from a minimal classical baseline to two experimental branches:

```
Foundations
├── Probability & Statistics
├── Linear Algebra
└── Computer Science Fundamentals 

1. nanoGPT (Andrej Karpathy)      →  minimal, from-scratch GPT
2. GPT-2                          →  intermediate-scale classical LLM
3. CS336 – Stanford               →  advanced, full training/systems pipeline
                                        |
                    ┌───────────────────┴───────────────────┐
                    │                                       │
        Quantum LLM Module                     Thermodynamic LLM Module
        (built with PennyLane)                 (built with THRML)

                    └───────────────────┬───────────────────┘
                                        │
                                        │
                                        │
                           Hybrid Adaptive LLM
        (combines Classical, Quantum, and Thermodynamic modules,
         dynamically selecting or combining them whenever each
         paradigm is best suited for the computation.)
```

| Stage | Reference | What it teaches |
|---|---|---|
| Nano | [nanoGPT](https://github.com/karpathy/nanoGPT) | Minimal, readable implementation of GPT training/inference |
| Intermediate | GPT-2 (OpenAI) | Tokenization at scale, larger architectures, sampling strategies |
| Advanced | [CS336 — Language Modeling from Scratch (Stanford)](https://cs336.stanford.edu/) | Walks through the entire pipeline of building an LLM: data collection and cleaning for pretraining, transformer construction, training, and evaluation before deployment |
| Branch A | PennyLane quantum ML | Encoding embeddings/attention as parameterized quantum circuits |
| Branch B | THRML thermodynamic sampling | Replacing (parts of) inference/training with energy-based sampling |

## Tools & Libraries

| Library | Paradigm | What it provides |
|---|---|---|
| [THRML](https://github.com/extropic-ai/thrml) | Thermodynamic / probabilistic | A JAX library for building and sampling probabilistic graphical models, focused on efficient block-Gibbs sampling and energy-based models — built by Extropic as a way to prototype today what will eventually run on dedicated thermodynamic sampling hardware. |
| [PennyLane](https://pennylane.ai/quantum-machine-learning) | Quantum | Differentiable quantum circuits that integrate with standard ML frameworks, used to build hybrid quantum-classical models. |

## References

- nanoGPT — Andrej Karpathy: <https://github.com/karpathy/nanoGPT>
- GPT-2 — OpenAI
- CS336, *Language Modeling from Scratch* (Stanford): <https://cs336.stanford.edu/>
- THRML (Extropic): <https://github.com/extropic-ai/thrml>
- PennyLane — Quantum Machine Learning: <https://pennylane.ai/quantum-machine-learning>
- Laine, T. A. (2025). *Quantum LLMs: Using Quantum Computing to Analyze and Process Semantic Information* — proposes a mapping between LLM embedding spaces and quantum circuits, and reports an experimental estimate of cosine similarity between sentence embeddings computed on real quantum hardware. arXiv:2512.02619.

## Community & Goals

Beyond the technical exploration, a goal of this project is to **learn by building** and eventually contribute the results to a broader community. One target venue is the [**Escuela de Matemáticas Cuánticas 2027**](https://sites.google.com/view/escuelacuantica2027/inicio) (Saarland University × Outsourcing Math Research), a hybrid, Spanish-language program for students in Latin America and Spain. It runs in three stages: a virtual introductory course on quantum information, a project-development phase with a mentor and a small group of participants, and a final in-person workshop in Saarbrücken, Germany, where selected participants present their work to local and visiting researchers.

## Status

🚧 Early-stage, educational research project — expect breaking changes as each paradigm branch matures.

# Project Timeline

| Week(s) | Focus |
|----------|-------|
| **Week 1** | Project planning and roadmap |
| **Week 2** | Computer Science foundations behind **Quantum Computing** and **Thermal Computing** |
| **Week 3** | Linear Algebra foundations |
| **Week 4** | Probability foundations |
| **Weeks 5–6** | Implement and study **nanoGPT** |
| **Weeks 7–10** | Implement and study **GPT-2** |
| **Weeks 10–15** | Build an industrial-scale language model |
| **Weeks 15–20** | Implement **Quantum Language Models** |
| **Weeks 20–25** | Implement **Thermal Language Models** |
| **Weeks 25–30** | Integrate and combine **classical**, **quantum**, and **thermal** computing principles into a hybrid language model |


# Resources

- **Thermal Computer (Simulation):**
  - Extropic Simulator

- **Quantum Computer:**
  - Pennylane

- **Classical Computer:**
  - 32-core Apollo workstation/server

- **Language Model Dataset:**
  - Depends on the selected training tutorial (e.g., TinyStories, FineWeb, The Pile, or Common Crawl derivatives)

## Contributing

Issues and pull requests are welcome, especially around:
- Additional benchmarks comparing classical vs. quantum vs. thermodynamic layers
- Cleaner abstractions in `core/`
- Documentation of the underlying math for each paradigm

## License

GPLv3
