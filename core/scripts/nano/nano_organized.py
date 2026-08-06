"""
BigramLanguageModel refactored with:
  - Strategy Pattern  : swap tokenization, sampling, and loss algorithms at runtime
  - Pipeline Pattern  : compose training and inference as ordered, reusable stages
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

# ─────────────────────────────────────────────
# 0.  Data loading (unchanged from original)
# ─────────────────────────────────────────────
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# ═══════════════════════════════════════════════════════
#  STRATEGY PATTERN
#  Define a *family* of algorithms, encapsulate each one,
#  and make them interchangeable.
# ═══════════════════════════════════════════════════════


# ── 1. Tokenization Strategies ──────────────────────────
class TokenizationStrategy(ABC):
    """Interface: text → token ids and back."""

    @abstractmethod
    def encode(self, s: str) -> List[int]:
        ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...


class CharTokenizationStrategy(TokenizationStrategy):
    """Original character-level tokenizer."""

    def __init__(self, chars: List[str]):
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> List[int]:
        return [self._stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)


class ByteTokenizationStrategy(TokenizationStrategy):
    """Alternative: raw UTF-8 bytes (vocab = 256)."""

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def decode(self, ids: List[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256


# ── 2. Sampling Strategies ───────────────────────────────
class SamplingStrategy(ABC):
    """Interface: logits (B,C) → next token ids (B,1)."""

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...


class MultinomialSamplingStrategy(SamplingStrategy):
    """Original: softmax then multinomial draw."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class GreedySamplingStrategy(SamplingStrategy):
    """Always pick the highest-probability token (deterministic)."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)


class TopKSamplingStrategy(SamplingStrategy):
    """Restrict sampling to the top-k tokens before drawing."""

    def __init__(self, k: int = 10):
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        top_k_logits, _ = torch.topk(logits, self.k, dim=-1)
        threshold = top_k_logits[:, -1].unsqueeze(-1)
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ── 3. Loss Strategies ───────────────────────────────────
class LossStrategy(ABC):
    """Interface: (logits, targets) → scalar loss."""

    @abstractmethod
    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...


class CrossEntropyLossStrategy(LossStrategy):
    """Original cross-entropy."""

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))


class LabelSmoothingLossStrategy(LossStrategy):
    """Cross-entropy with label smoothing (regularisation)."""

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self._loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, C = logits.shape
        return self._loss_fn(logits.view(B * T, C), targets.view(B * T))


# ═══════════════════════════════════════════════════════
#  MODEL  –  strategies injected via constructor
# ═══════════════════════════════════════════════════════
class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        loss_strategy: LossStrategy = None,
        sampling_strategy: SamplingStrategy = None,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        # defaults keep backwards-compatibility with original code
        self.loss_strategy = loss_strategy or CrossEntropyLossStrategy()
        self.sampling_strategy = sampling_strategy or MultinomialSamplingStrategy()

    # swap strategies at runtime ──────────────────────────
    def set_loss_strategy(self, strategy: LossStrategy):
        self.loss_strategy = strategy

    def set_sampling_strategy(self, strategy: SamplingStrategy):
        self.sampling_strategy = strategy

    # forward ─────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.token_embedding_table(idx)  # (B, T, C)
        loss = (
            self.loss_strategy.compute(logits, targets) if targets is not None else None
        )
        return logits, loss

    # generate ────────────────────────────────────────────
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            idx_next = self.sampling_strategy.sample(logits)  # (B, 1)  ← strategy!
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ═══════════════════════════════════════════════════════
#  PIPELINE PATTERN
#  Chain discrete, reusable stages; each stage receives
#  a shared context dict and may read/write any key.
# ═══════════════════════════════════════════════════════


@dataclass
class PipelineContext:
    """Shared state passed through every pipeline stage."""

    config: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)  # mutable runtime state
    results: Dict[str, Any] = field(default_factory=dict)  # outputs / metrics


class PipelineStage(ABC):
    """Base class for a single pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        ...

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        print(f"  [stage] {self.name}")
        return self.run(ctx)


class Pipeline:
    """Execute a sequence of stages, threading the context through."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self.stages:
            ctx = stage(ctx)
        return ctx

    # convenience: add a stage after construction
    def append(self, stage: PipelineStage) -> "Pipeline":
        self.stages.append(stage)
        return self


# ── Training pipeline stages ─────────────────────────────


class DataPreparationStage(PipelineStage):
    name = "DataPreparation"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        tok: TokenizationStrategy = ctx.config["tokenization_strategy"]
        raw_text: str = ctx.config["raw_text"]
        split_frac: float = ctx.config.get("train_split", 0.9)

        data = torch.tensor(tok.encode(raw_text), dtype=torch.long)
        n = int(split_frac * len(data))
        ctx.state["train_data"] = data[:n]
        ctx.state["val_data"] = data[n:]
        ctx.results["dataset_size"] = len(data)
        return ctx


class ModelInitStage(PipelineStage):
    name = "ModelInit"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        tok = ctx.config["tokenization_strategy"]
        model = BigramLanguageModel(
            vocab_size=tok.vocab_size,
            loss_strategy=ctx.config.get("loss_strategy"),
            sampling_strategy=ctx.config.get("sampling_strategy"),
        )
        ctx.state["model"] = model
        ctx.state["optimizer"] = torch.optim.AdamW(
            model.parameters(),
            lr=ctx.config.get("learning_rate", 1e-3),
        )
        return ctx


class TrainingLoopStage(PipelineStage):
    name = "TrainingLoop"

    def _get_batch(
        self,
        data: torch.Tensor,
        batch_size: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    def run(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.state["model"]
        optimizer = ctx.state["optimizer"]
        train_data = ctx.state["train_data"]
        steps = ctx.config.get("train_steps", 1000)
        batch_size = ctx.config.get("batch_size", 4)
        block_size = ctx.config.get("block_size", 8)

        losses = []
        for step in range(steps):
            xb, yb = self._get_batch(train_data, batch_size, block_size)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (step + 1) % (steps // 5) == 0:
                print(f"      step {step+1:>5}/{steps}  loss={loss.item():.4f}")

        ctx.results["train_losses"] = losses
        ctx.results["final_loss"] = losses[-1]
        return ctx


class EvaluationStage(PipelineStage):
    name = "Evaluation"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.state["model"]
        val_data = ctx.state["val_data"]
        bs = ctx.config.get("batch_size", 4)
        bl = ctx.config.get("block_size", 8)

        model.eval()
        with torch.no_grad():
            ix = torch.randint(len(val_data) - bl, (bs,))
            xb = torch.stack([val_data[i : i + bl] for i in ix])
            yb = torch.stack([val_data[i + 1 : i + bl + 1] for i in ix])
            _, loss = model(xb, yb)
        model.train()

        ctx.results["val_loss"] = loss.item()
        return ctx


# ── Inference pipeline stages ────────────────────────────


class PromptEncodingStage(PipelineStage):
    name = "PromptEncoding"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        tok = ctx.config["tokenization_strategy"]
        prompt = ctx.config.get("prompt", "\n")
        ctx.state["prompt_tensor"] = torch.tensor(
            [tok.encode(prompt)], dtype=torch.long
        )
        return ctx


class GenerationStage(PipelineStage):
    name = "Generation"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        model = ctx.state["model"]
        idx = ctx.state["prompt_tensor"]
        max_tokens = ctx.config.get("max_new_tokens", 200)
        model.eval()
        with torch.no_grad():
            ctx.state["generated_ids"] = model.generate(idx, max_tokens)
        model.train()
        return ctx


class DecodingStage(PipelineStage):
    name = "Decoding"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        tok = ctx.config["tokenization_strategy"]
        ids = ctx.state["generated_ids"][0].tolist()
        ctx.results["generated_text"] = tok.decode(ids)
        return ctx


# ═══════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(1337)

    # ── choose strategies ─────────────────────────────
    tokenization = CharTokenizationStrategy(chars)  # swap → ByteTokenizationStrategy()
    loss_strat = CrossEntropyLossStrategy()  # swap → LabelSmoothingLossStrategy(0.1)
    sample_strat = (
        MultinomialSamplingStrategy()
    )  # swap → GreedySamplingStrategy() / TopKSamplingStrategy(10)

    # ── shared config ─────────────────────────────────
    shared_config = dict(
        raw_text=text,
        tokenization_strategy=tokenization,
        loss_strategy=loss_strat,
        sampling_strategy=sample_strat,
        train_split=0.9,
        learning_rate=1e-3,
        batch_size=32,
        block_size=8,
        train_steps=3000,
        max_new_tokens=300,
    )

    # ════════════════════════════════════════════════
    #  TRAINING PIPELINE
    # ════════════════════════════════════════════════
    print("\n━━━━  Training Pipeline  ━━━━")
    train_ctx = PipelineContext(config=shared_config)
    train_pipeline = Pipeline(
        [
            DataPreparationStage(),
            ModelInitStage(),
            TrainingLoopStage(),
            EvaluationStage(),
        ]
    )
    train_ctx = train_pipeline.run(train_ctx)
    print(f"\n  ✓ final train loss : {train_ctx.results['final_loss']:.4f}")
    print(f"  ✓ val   loss       : {train_ctx.results['val_loss']:.4f}")

    # ════════════════════════════════════════════════
    #  INFERENCE PIPELINE  (reuses trained model)
    # ════════════════════════════════════════════════
    print("\n━━━━  Inference Pipeline (multinomial)  ━━━━")
    infer_ctx = PipelineContext(
        config={**shared_config, "prompt": "ROMEO:\n"},
        state={"model": train_ctx.state["model"]},  # hand off trained model
    )
    infer_pipeline = Pipeline(
        [
            PromptEncodingStage(),
            GenerationStage(),
            DecodingStage(),
        ]
    )
    infer_ctx = infer_pipeline.run(infer_ctx)
    print("\n--- Generated text ---")
    print(infer_ctx.results["generated_text"])

    # ════════════════════════════════════════════════
    #  DEMO: swap sampling strategy at runtime
    # ════════════════════════════════════════════════
    print("\n━━━━  Inference Pipeline (greedy – strategy swapped)  ━━━━")
    model = train_ctx.state["model"]
    model.set_sampling_strategy(GreedySamplingStrategy())  # ← runtime swap!

    greedy_ctx = PipelineContext(
        config={**shared_config, "prompt": "ROMEO:\n", "max_new_tokens": 100},
        state={"model": model},
    )
    greedy_ctx = infer_pipeline.run(greedy_ctx)
    print("\n--- Generated text (greedy) ---")
    print(greedy_ctx.results["generated_text"])

    # ════════════════════════════════════════════════
    #  DEMO: swap sampling strategy again → top-k
    # ════════════════════════════════════════════════
    print("\n━━━━  Inference Pipeline (top-k=5 – strategy swapped again)  ━━━━")
    model.set_sampling_strategy(TopKSamplingStrategy(k=5))  # ← runtime swap!

    topk_ctx = PipelineContext(
        config={**shared_config, "prompt": "ROMEO:\n", "max_new_tokens": 100},
        state={"model": model},
    )
    topk_ctx = infer_pipeline.run(topk_ctx)
    print("\n--- Generated text (top-k=5) ---")
    print(topk_ctx.results["generated_text"])
