"""
End-to-end smoke tests. Kept deliberately toy-scale (tiny d_model, 1 layer,
few steps) so the Quantum/Photonic/Thermodynamic simulators finish quickly
-- these are integration tests proving the real adapters wire together
correctly, not a performance suite.

Run with:  python -m pytest tests/test_smoke.py -v -s
       or: python -m tests.test_smoke   (falls back to a plain runner)
"""

from __future__ import annotations

import shutil
import tempfile

import torch

from config.config import MPLLMConfig
from data.toy_corpus import TOY_CORPUS
from models.paradigm_llms import ClassicalLLM, HybridLLM, PhotonicLLM, QuantumLLM, ThermodynamicLLM
from pipelines.inference_pipeline import InferencePipeline
from pipelines.training_pipeline import TrainingPipeline


def _tiny_config(**overrides) -> MPLLMConfig:
    base = dict(
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        context_length=10,
        batch_size=2,
        epochs=1,
        steps_per_epoch=2,
        learning_rate=5e-3,
        photonic_mesh_size=4,
        quantum_n_qubits=4,
        thermodynamic_max_spins=4,
        diffusion_timesteps=3,
    )
    base.update(overrides)
    return MPLLMConfig(**base)


def test_all_variants_build_and_forward_transformer():
    for cls in [ClassicalLLM, QuantumLLM, PhotonicLLM, ThermodynamicLLM, HybridLLM]:
        cfg = _tiny_config(vocab_size=20, model_kind="transformer")
        llm = cls(cfg)
        x = torch.randint(0, 20, (1, 5))
        out = llm.model(x)
        assert out.shape == (1, 5, 20), f"{cls.__name__} bad output shape: {out.shape}"


def test_all_variants_build_and_forward_diffusion():
    for cls in [ClassicalLLM, QuantumLLM, PhotonicLLM, ThermodynamicLLM, HybridLLM]:
        cfg = _tiny_config(vocab_size=20, model_kind="diffusion")
        llm = cls(cfg)
        x = torch.randint(0, 20, (1, 5))
        t = torch.zeros(1, dtype=torch.long)
        out = llm.model(x, t)
        assert out.shape == (1, 5, 21), f"{cls.__name__} bad output shape: {out.shape}"  # +1 for mask token


def test_classical_train_step_reduces_reachable():
    cfg = _tiny_config(vocab_size=20, model_kind="transformer")
    llm = ClassicalLLM(cfg)
    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))
    metrics = llm.train_step({"input_ids": x, "targets": y})
    assert "loss" in metrics and metrics["loss"] > 0


def test_hybrid_train_step_uses_multiple_paradigms():
    cfg = _tiny_config(vocab_size=20, model_kind="transformer", prefer_photonic_attention=True)
    llm = HybridLLM(cfg)
    paradigms_used = {c.paradigm.value for c in llm.model.all_choices()}
    assert "classical" in paradigms_used
    assert "photonic" in paradigms_used or "quantum" in paradigms_used
    x = torch.randint(0, 20, (1, 6))
    y = torch.randint(0, 20, (1, 6))
    metrics = llm.train_step({"input_ids": x, "targets": y})
    assert metrics["loss"] > 0


def test_full_pipeline_train_save_load_infer():
    tmp_dir = tempfile.mkdtemp()
    try:
        cfg = _tiny_config(model_kind="transformer", steps_per_epoch=2)
        pipe = TrainingPipeline(cfg, ClassicalLLM, TOY_CORPUS, run_dir=tmp_dir)
        result = pipe.run()
        assert len(result["metrics_history"]) == 2

        infer = InferencePipeline(tmp_dir, ClassicalLLM)
        text = infer.run(prompt="the ", max_new_tokens=5)
        assert isinstance(text, str) and len(text) > 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_benchmark_report_has_expected_fields():
    cfg = _tiny_config(vocab_size=20, model_kind="transformer")
    llm = HybridLLM(cfg)
    report = llm.benchmark(n_forward=1, seq_len=5, batch_size=1)
    assert "parameter_count" in report
    assert "timing_summary" in report
    assert "paradigm_selection" in report
    assert "hardware_factors" in report


if __name__ == "__main__":
    tests = [
        test_all_variants_build_and_forward_transformer,
        test_all_variants_build_and_forward_diffusion,
        test_classical_train_step_reduces_reachable,
        test_hybrid_train_step_uses_multiple_paradigms,
        test_full_pipeline_train_save_load_infer,
        test_benchmark_report_has_expected_fields,
    ]
    for t in tests:
        print(f"running {t.__name__} ...")
        t()
        print(f"  OK")
    print("\nAll smoke tests passed.")
