"""
Pipeline Pattern: the training workflow, split into the exact stages the
spec diagrams:

    train()
      -> load_dataset()
      -> build_tokenizer()
      -> tokenize()
      -> build_model()
      -> train()              (the per-step loop below)
      -> evaluate()
      -> save_model()
      -> save_metrics()

...where the per-step loop is itself:

    train_step()
      -> forward()
      -> loss()
      -> backpropagation()
      -> optimizer()
      -> checkpoint()          (periodic, not every step)

Each stage is a separate method so it can be overridden/reused
independently (e.g. `build_tokenizer` is reused verbatim by
`InferencePipeline`).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Type

import torch

from benchmarking.report import build_benchmark_report, save_report
from config.config import MPLLMConfig
from data.dataset import CharDataset
from logging_utils import MPLLMLogger
from models.tokenizer import CharTokenizer


class TrainingPipeline:
    def __init__(
        self,
        config: MPLLMConfig,
        llm_cls: Type,
        corpus: str,
        run_dir: str,
        logger: Optional[MPLLMLogger] = None,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.llm_cls = llm_cls
        self.corpus = corpus
        self.run_dir = run_dir
        self.logger = logger or MPLLMLogger(os.path.join(run_dir, "train_log.jsonl"))
        self.llm_kwargs = llm_kwargs or {}

        self.tokenizer: Optional[CharTokenizer] = None
        self.dataset: Optional[CharDataset] = None
        self.llm = None
        self.metrics_history: List[Dict[str, Any]] = []

    # -- pipeline stages ----------------------------------------------------
    def build_tokenizer(self) -> CharTokenizer:
        self.tokenizer = CharTokenizer.from_text(self.corpus)
        self.config.vocab_size = self.tokenizer.vocab_size
        return self.tokenizer

    def tokenize(self) -> List[int]:
        return self.tokenizer.encode(self.corpus)

    def load_dataset(self, token_ids: List[int]) -> CharDataset:
        self.dataset = CharDataset(token_ids, self.config.context_length)
        return self.dataset

    def build_model(self):
        self.llm = self.llm_cls(self.config, tokenizer=self.tokenizer, **self.llm_kwargs)
        return self.llm

    def _get_batch(self) -> Dict[str, torch.Tensor]:
        if self.config.model_kind == "diffusion":
            clean = self.dataset.get_diffusion_batch(self.config.batch_size)
            return {"input_ids": clean}
        x, y = self.dataset.get_autoregressive_batch(self.config.batch_size)
        return {"input_ids": x, "targets": y}

    def train(self) -> List[Dict[str, Any]]:
        """The per-step loop: forward -> loss -> backpropagation ->
        optimizer (all inside `llm.train_step`) -> periodic checkpoint.
        """
        step = 0
        for epoch in range(self.config.epochs):
            for _ in range(self.config.steps_per_epoch):
                batch = self._get_batch()
                metrics = self.llm.train_step(batch)
                metrics["epoch"] = epoch
                self.metrics_history.append(metrics)
                self.logger.log_step(
                    step=metrics["step"],
                    loss=metrics["loss"],
                    perplexity=metrics["perplexity"],
                    accuracy=metrics["accuracy"],
                    execution_time=metrics["execution_time"],
                )
                step += 1
                if step % max(self.config.steps_per_epoch, 1) == 0:
                    self.checkpoint()
        return self.metrics_history

    def evaluate(self) -> Dict[str, float]:
        """Toy evaluation: average loss/accuracy over a few fresh batches
        with the model in eval mode (dropout off); no held-out split is
        maintained given the corpus's toy scale.
        """
        self.llm.model.eval()
        losses, accs = [], []
        with torch.no_grad():
            for _ in range(3):
                batch = self._get_batch()
                if self.config.model_kind == "diffusion":
                    clean_ids = batch["input_ids"]
                    b = clean_ids.shape[0]
                    t = torch.randint(0, self.config.diffusion_timesteps, (b,))
                    noisy, mask = self.llm.model.corrupt(clean_ids, t)
                    logits = self.llm.model(noisy, t)
                    if mask.any():
                        import torch.nn.functional as F

                        loss = F.cross_entropy(logits[mask], clean_ids[mask])
                        acc = (logits[mask].argmax(-1) == clean_ids[mask]).float().mean().item()
                    else:
                        continue
                else:
                    logits = self.llm.model(batch["input_ids"])
                    import torch.nn.functional as F

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), batch["targets"].reshape(-1)
                    )
                    acc = (logits.argmax(-1) == batch["targets"]).float().mean().item()
                losses.append(loss.item())
                accs.append(acc)
        result = {
            "eval_loss": sum(losses) / max(len(losses), 1),
            "eval_accuracy": sum(accs) / max(len(accs), 1),
        }
        self.logger.log_message(f"evaluate: {result}")
        return result

    def checkpoint(self) -> None:
        self.llm.save(self.run_dir)

    def save_model(self) -> None:
        self.llm.save(self.run_dir)

    def save_metrics(self, eval_result: Optional[Dict[str, float]] = None) -> None:
        import json

        payload = {"history": self.metrics_history, "final_eval": eval_result}
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)

        report = build_benchmark_report(
            model_name=self.llm.name,
            model=self.llm.model,
            paradigm_choices=self.llm.model.all_choices(),
            extra={"train_metrics_final": self.metrics_history[-1] if self.metrics_history else None},
        )
        save_report(report, self.run_dir, basename="benchmark_report")
        self.logger.log_benchmark_summary(
            {"parameter_count": report.get("parameter_count"), "run_dir": self.run_dir}
        )

    # -- top-level orchestration --------------------------------------------
    def run(self) -> Dict[str, Any]:
        self.build_tokenizer()
        token_ids = self.tokenize()
        self.load_dataset(token_ids)
        self.build_model()
        self.train()
        eval_result = self.evaluate()
        self.save_model()
        self.save_metrics(eval_result)
        return {
            "llm": self.llm,
            "tokenizer": self.tokenizer,
            "metrics_history": self.metrics_history,
            "eval_result": eval_result,
        }
