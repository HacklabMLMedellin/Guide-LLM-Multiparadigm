from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    batch_size: int = 16
    block_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 200
    learning_rate: float = 1e-3
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Strategy — Attention
# ---------------------------------------------------------------------------


class AttentionStrategy(ABC):
    @abstractmethod
    def build(self, cfg: GPTConfig) -> nn.Module:
        ...


class SingleHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList(
            [SingleHeadAttention(cfg, head_size) for _ in range(cfg.n_head)]
        )
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MultiHeadAttentionStrategy(AttentionStrategy):
    def build(self, cfg: GPTConfig) -> nn.Module:
        return MultiHeadAttentionModule(cfg)


# ---------------------------------------------------------------------------
# Strategy — FeedForward
# ---------------------------------------------------------------------------


class FeedForwardStrategy(ABC):
    @abstractmethod
    def build(self, cfg: GPTConfig) -> nn.Module:
        ...


class ReLUFeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GELUFeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReLUFeedForwardStrategy(FeedForwardStrategy):
    def build(self, cfg: GPTConfig) -> nn.Module:
        return ReLUFeedForward(cfg)


class GELUFeedForwardStrategy(FeedForwardStrategy):
    def build(self, cfg: GPTConfig) -> nn.Module:
        return GELUFeedForward(cfg)


# ---------------------------------------------------------------------------
# Strategy — Optimizer
# ---------------------------------------------------------------------------


class OptimizerStrategy(ABC):
    @abstractmethod
    def build(self, params, cfg: GPTConfig) -> torch.optim.Optimizer:
        ...


class AdamWStrategy(OptimizerStrategy):
    def build(self, params, cfg: GPTConfig) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=cfg.learning_rate)


class SGDStrategy(OptimizerStrategy):
    def build(self, params, cfg: GPTConfig) -> torch.optim.Optimizer:
        return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9)


# ---------------------------------------------------------------------------
# Strategy — Sampling / Generation
# ---------------------------------------------------------------------------


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...


class MultinomialSampling(SamplingStrategy):
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class GreedySampling(SamplingStrategy):
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)


class TopKSampling(SamplingStrategy):
    def __init__(self, k: int = 10):
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = min(self.k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        attn_strategy: AttentionStrategy,
        ff_strategy: FeedForwardStrategy,
    ):
        super().__init__()
        self.sa = attn_strategy.build(cfg)
        self.ffwd = ff_strategy.build(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class GPTModel(nn.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        vocab_size: int,
        attn_strategy: AttentionStrategy,
        ff_strategy: FeedForwardStrategy,
        sample_strategy: SamplingStrategy,
    ):
        super().__init__()
        self.cfg = cfg
        self.sample_strategy = sample_strategy
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(cfg, attn_strategy, ff_strategy)
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.cfg.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            idx_next = self.sample_strategy.sample(logits)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Data Pipeline Stage
# ---------------------------------------------------------------------------


@dataclass
class Tokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        return "".join(self.itos[i] for i in l)


@dataclass
class DataSplit:
    train: torch.Tensor
    val: torch.Tensor


# ---------------------------------------------------------------------------
# Pipeline Steps (callable objects)
# ---------------------------------------------------------------------------


class PipelineStep(ABC):
    @abstractmethod
    def __call__(self, ctx: dict) -> dict:
        ...


class LoadTextStep(PipelineStep):
    def __init__(self, path: str):
        self.path = path

    def __call__(self, ctx: dict) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            ctx["text"] = f.read()
        return ctx


class BuildTokenizerStep(PipelineStep):
    def __call__(self, ctx: dict) -> dict:
        chars = sorted(set(ctx["text"]))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        ctx["tokenizer"] = Tokenizer(stoi, itos)
        ctx["vocab_size"] = len(chars)
        return ctx


class SplitDataStep(PipelineStep):
    def __init__(self, cfg: GPTConfig):
        self.cfg = cfg

    def __call__(self, ctx: dict) -> dict:
        data = torch.tensor(ctx["tokenizer"].encode(ctx["text"]), dtype=torch.long)
        n = int(0.9 * len(data))
        ctx["splits"] = DataSplit(train=data[:n], val=data[n:])
        return ctx


class BuildModelStep(PipelineStep):
    def __init__(
        self,
        cfg: GPTConfig,
        attn_strategy: AttentionStrategy,
        ff_strategy: FeedForwardStrategy,
        sample_strategy: SamplingStrategy,
    ):
        self.cfg = cfg
        self.attn_strategy = attn_strategy
        self.ff_strategy = ff_strategy
        self.sample_strategy = sample_strategy

    def __call__(self, ctx: dict) -> dict:
        model = GPTModel(
            self.cfg,
            ctx["vocab_size"],
            self.attn_strategy,
            self.ff_strategy,
            self.sample_strategy,
        )
        ctx["model"] = model.to(self.cfg.device)
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        return ctx


class TrainStep(PipelineStep):
    def __init__(self, cfg: GPTConfig, opt_strategy: OptimizerStrategy):
        self.cfg = cfg
        self.opt_strategy = opt_strategy

    def _get_batch(
        self, split: str, splits: DataSplit
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = splits.train if split == "train" else splits.val
        ix = torch.randint(len(data) - self.cfg.block_size, (self.cfg.batch_size,))
        x = torch.stack([data[i : i + self.cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.cfg.block_size + 1] for i in ix])
        return x.to(self.cfg.device), y.to(self.cfg.device)

    @torch.no_grad()
    def _estimate_loss(self, model: GPTModel, splits: DataSplit) -> dict[str, float]:
        model.eval()
        result = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                X, Y = self._get_batch(split, splits)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            result[split] = losses.mean().item()
        model.train()
        return result

    def __call__(self, ctx: dict) -> dict:
        model = ctx["model"]
        splits = ctx["splits"]
        optimizer = self.opt_strategy.build(model.parameters(), self.cfg)
        for it in range(self.cfg.max_iters):
            if it % self.cfg.eval_interval == 0 or it == self.cfg.max_iters - 1:
                losses = self._estimate_loss(model, splits)
                print(
                    f"step {it:4d}  train={losses['train']:.4f}  val={losses['val']:.4f}"
                )
            xb, yb = self._get_batch("train", splits)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return ctx


class InferenceStep(PipelineStep):
    def __init__(self, cfg: GPTConfig, prompt: str, max_new_tokens: int):
        self.cfg = cfg
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def __call__(self, ctx: dict) -> dict:
        model = ctx["model"]
        tokenizer = ctx["tokenizer"]
        model.eval()
        context = torch.tensor(
            [tokenizer.encode(self.prompt)], dtype=torch.long, device=self.cfg.device
        )
        generated = model.generate(context, self.max_new_tokens)[0].tolist()
        ctx["generated"] = tokenizer.decode(generated)
        print(ctx["generated"])
        return ctx


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self):
        self._steps: list[PipelineStep] = []

    def pipe(self, step: PipelineStep) -> Pipeline:
        self._steps.append(step)
        return self

    def run(self, ctx: dict | None = None) -> dict:
        ctx = ctx or {}
        for step in self._steps:
            ctx = step(ctx)
        return ctx


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(1337)
    cfg = GPTConfig()

    (
        Pipeline()
        .pipe(LoadTextStep("input.txt"))
        .pipe(BuildTokenizerStep())
        .pipe(SplitDataStep(cfg))
        .pipe(
            BuildModelStep(
                cfg,
                attn_strategy=MultiHeadAttentionStrategy(),
                ff_strategy=GELUFeedForwardStrategy(),
                sample_strategy=TopKSampling(k=10),
            )
        )
        .pipe(TrainStep(cfg, opt_strategy=AdamWStrategy()))
        .pipe(InferenceStep(cfg, prompt="KING:", max_new_tokens=300))
        .run()
    )


if __name__ == "__main__":
    main()
