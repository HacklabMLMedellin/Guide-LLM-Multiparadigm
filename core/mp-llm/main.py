"""
CLI entry point.

    python main.py train --llm hybrid --model transformer --run-dir outputs/checkpoints/run1
    python main.py infer --run-dir outputs/checkpoints/run1 --llm hybrid --prompt "the "
    python main.py compare --model transformer --out-dir outputs/benchmarks

See README.md for a full walkthrough.
"""

from __future__ import annotations

import argparse
import os

from config.config import MPLLMConfig
from data.corpus_loader import available_corpora, get_corpus
from models.paradigm_llms import ClassicalLLM, HybridLLM, PhotonicLLM, QuantumLLM, ThermodynamicLLM
from pipelines.inference_pipeline import InferencePipeline
from pipelines.training_pipeline import TrainingPipeline

LLM_REGISTRY = {
    "classical": ClassicalLLM,
    "quantum": QuantumLLM,
    "photonic": PhotonicLLM,
    "thermodynamic": ThermodynamicLLM,
    "hybrid": HybridLLM,
}


def _base_config(args) -> MPLLMConfig:
    return MPLLMConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        context_length=args.context_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.lr,
        photonic_mesh_size=args.mesh_size,
        quantum_n_qubits=args.n_qubits,
        thermodynamic_max_spins=args.max_spins,
        diffusion_timesteps=args.diffusion_timesteps,
        model_kind=args.model,
    )


def cmd_train(args) -> None:
    cfg = _base_config(args)
    llm_cls = LLM_REGISTRY[args.llm]
    corpus = get_corpus(args.corpus, max_chars=args.max_chars)
    print(f"Using corpus '{args.corpus}': {len(corpus):,} characters "
          f"(context_length={cfg.context_length}, need > that to train).")
    pipe = TrainingPipeline(cfg, llm_cls, corpus, run_dir=args.run_dir)
    result = pipe.run()
    print("\nFinal metrics:", result["metrics_history"][-1])
    print("Eval:", result["eval_result"])
    print(f"\nSaved checkpoint + benchmark report to: {args.run_dir}")


def cmd_infer(args) -> None:
    llm_cls = LLM_REGISTRY[args.llm]
    pipe = InferencePipeline(args.run_dir, llm_cls)
    pipe.run(prompt=args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)


def cmd_compare(args) -> None:
    from scripts.compare_paradigms import run_comparison

    run_comparison(
        model_kind=args.model,
        out_dir=args.out_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        context_length=args.context_length,
        mesh_size=args.mesh_size,
        n_qubits=args.n_qubits,
        max_spins=args.max_spins,
        diffusion_timesteps=args.diffusion_timesteps,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-Paradigm LLM (MP-LLM)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--model", choices=["transformer", "diffusion"], default="transformer")
        sp.add_argument("--d-model", dest="d_model", type=int, default=32)
        sp.add_argument("--n-heads", dest="n_heads", type=int, default=4)
        sp.add_argument("--n-layers", dest="n_layers", type=int, default=2)
        sp.add_argument("--d-ff", dest="d_ff", type=int, default=64)
        sp.add_argument("--context-length", dest="context_length", type=int, default=32)
        sp.add_argument("--mesh-size", dest="mesh_size", type=int, default=4)
        sp.add_argument("--n-qubits", dest="n_qubits", type=int, default=4)
        sp.add_argument("--max-spins", dest="max_spins", type=int, default=6)
        sp.add_argument("--diffusion-timesteps", dest="diffusion_timesteps", type=int, default=6)

    train_p = sub.add_parser("train")
    add_common(train_p)
    train_p.add_argument("--llm", choices=list(LLM_REGISTRY.keys()), default="hybrid")
    train_p.add_argument("--run-dir", required=True)
    train_p.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument("--steps-per-epoch", dest="steps_per_epoch", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=3e-3)
    train_p.add_argument(
        "--corpus", choices=available_corpora(), default="tiny",
        help="'tiny' = built-in offline corpus; others download+cache on first use.",
    )
    train_p.add_argument(
        "--max-chars", dest="max_chars", type=int, default=None,
        help="Truncate the corpus to this many characters (useful for a fast first run).",
    )
    train_p.set_defaults(func=cmd_train)

    infer_p = sub.add_parser("infer")
    infer_p.add_argument("--llm", choices=list(LLM_REGISTRY.keys()), default="hybrid")
    infer_p.add_argument("--run-dir", required=True)
    infer_p.add_argument("--prompt", default="the ")
    infer_p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=30)
    infer_p.add_argument("--temperature", type=float, default=0.8)
    infer_p.set_defaults(func=cmd_infer)

    compare_p = sub.add_parser("compare")
    add_common(compare_p)
    compare_p.add_argument("--out-dir", default="outputs/benchmarks")
    compare_p.set_defaults(func=cmd_compare)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/benchmarks", exist_ok=True)
    args.func(args)
