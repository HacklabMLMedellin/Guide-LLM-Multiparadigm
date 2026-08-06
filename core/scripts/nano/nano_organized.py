import torch
import torch.nn as nn
from torch.nn import functional as F
from abc import ABC, abstractmethod

# ─────────────────────────────────────────────
#  STRATEGY PATTERN  – swappable algorithms
# ─────────────────────────────────────────────


class EncodingStrategy(ABC):
    """Swap how characters are mapped to integers."""

    @abstractmethod
    def encode(self, s: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, l: list[int]) -> str:
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...


class CharLevelEncoding(EncodingStrategy):
    """Original character-level encoding from the notebook."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]) -> str:
        return "".join(self.itos[i] for i in l)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class LossStrategy(ABC):
    """Swap how the training loss is computed."""

    @abstractmethod
    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...


class CrossEntropyLoss(LossStrategy):
    """Standard cross-entropy (original behaviour)."""

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        return F.cross_entropy(logits, targets)


class SamplingStrategy(ABC):
    """Swap how the next token is sampled from logits."""

    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...  # returns (B, 1)


class MultinomialSampling(SamplingStrategy):
    """Original: softmax → multinomial draw."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)  # (B, C)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)


class GreedySampling(SamplingStrategy):
    """Always pick the highest-probability token."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)


# ─────────────────────────────────────────────
#  MODEL  – strategies are injected
# ─────────────────────────────────────────────


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, loss_strategy: LossStrategy):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.loss_strategy = loss_strategy  # injected

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        loss = None
        if targets is not None:
            loss = self.loss_strategy.compute(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens: int, sampling_strategy: SamplingStrategy):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            idx_next = sampling_strategy.sample(logits)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ─────────────────────────────────────────────
#  PIPELINE PATTERN  – training & inference
# ─────────────────────────────────────────────


class Stage(ABC):
    """One step in a pipeline."""

    @abstractmethod
    def run(self, ctx: dict) -> dict:
        ...


class DataLoadStage(Stage):
    """Load text, build vocab, split into train/val tensors."""

    def __init__(self, path: str, encoding_strategy: EncodingStrategy):
        self.path = path
        self.enc = encoding_strategy

    def run(self, ctx: dict) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read()
        print("length of dataset in characters: ", len(text))
        print(text[:1000])
        print("".join(sorted(set(text))))
        print(self.enc.vocab_size)

        data = torch.tensor(self.enc.encode(text), dtype=torch.long)
        print(data.shape, data.dtype)
        print(data[:1000])

        n = int(0.9 * len(data))
        ctx["train_data"] = data[:n]
        ctx["val_data"] = data[n:]
        return ctx


class BatchStage(Stage):
    """Demonstrate context/target pairs and produce one batch."""

    def __init__(self, batch_size: int, block_size: int):
        self.batch_size = batch_size
        self.block_size = block_size

    def _get_batch(self, data: torch.Tensor):
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y

    def run(self, ctx: dict) -> dict:
        train_data = ctx["train_data"]

        # show context/target pairs on a tiny slice
        x = train_data[: self.block_size]
        y = train_data[1 : self.block_size + 1]
        for t in range(self.block_size):
            print(f"when input is {x[:t+1].tolist()} the target: {y[t]}")

        xb, yb = self._get_batch(train_data)
        print("inputs:")
        print(xb.shape)
        print(xb)
        print("targets:")
        print(yb.shape)
        print(yb)
        print("----")
        for b in range(self.batch_size):
            for t in range(self.block_size):
                print(f"when input is {xb[b,:t+1].tolist()} the target: {yb[b,t]}")

        ctx["xb"] = xb
        ctx["yb"] = yb
        return ctx


class ModelInitStage(Stage):
    """Instantiate the model with the chosen loss strategy."""

    def __init__(self, vocab_size: int, loss_strategy: LossStrategy):
        self.vocab_size = vocab_size
        self.loss_strategy = loss_strategy

    def run(self, ctx: dict) -> dict:
        torch.manual_seed(1337)
        model = BigramLanguageModel(self.vocab_size, self.loss_strategy)
        ctx["model"] = model
        return ctx


class ForwardPassStage(Stage):
    """Run one forward pass and print logits shape + loss."""

    def run(self, ctx: dict) -> dict:
        model, xb, yb = ctx["model"], ctx["xb"], ctx["yb"]
        logits, loss = model(xb, yb)
        print(logits.shape)
        print(loss)
        return ctx


class InferenceStage(Stage):
    """Generate text from a prompt using the chosen sampling strategy."""

    def __init__(
        self,
        encoding_strategy: EncodingStrategy,
        sampling_strategy: SamplingStrategy,
        max_new_tokens: int = 200,
    ):
        self.enc = encoding_strategy
        self.sampling_strategy = sampling_strategy
        self.max_new_tokens = max_new_tokens

    def run(self, ctx: dict) -> dict:
        model = ctx["model"]

        # null prompt (zeros) – original demo
        idx = torch.zeros((1, 1), dtype=torch.long)
        output = model.generate(
            idx, max_new_tokens=100, sampling_strategy=self.sampling_strategy
        )
        print(self.enc.decode(output[0].tolist()))

        # "hello" prompt
        prompt = "hello"
        idx = torch.tensor([self.enc.encode(prompt)], dtype=torch.long)
        output = model.generate(
            idx,
            max_new_tokens=self.max_new_tokens,
            sampling_strategy=self.sampling_strategy,
        )
        print(self.enc.decode(output[0].tolist()))
        return ctx


class Pipeline:
    """Run a sequence of stages, threading a shared context dict."""

    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def run(self, ctx: dict | None = None) -> dict:
        if ctx is None:
            ctx = {}
        for stage in self.stages:
            ctx = stage.run(ctx)
        return ctx


# ─────────────────────────────────────────────
#  WIRING  – choose strategies, build pipelines
# ─────────────────────────────────────────────

if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    # ── pick strategies ──────────────────────
    encoding = CharLevelEncoding(raw)  # swap: BPEEncoding, WordLevelEncoding …
    loss = CrossEntropyLoss()  # swap: FocalLoss, NLLLoss …
    sampling = MultinomialSampling()  # swap: GreedySampling …

    # ── training pipeline ────────────────────
    torch.manual_seed(1337)
    training_pipeline = Pipeline(
        [
            DataLoadStage("input.txt", encoding),
            BatchStage(batch_size=4, block_size=8),
            ModelInitStage(encoding.vocab_size, loss),
            ForwardPassStage(),
        ]
    )
    ctx = training_pipeline.run()

    # ── inference pipeline ───────────────────
    inference_pipeline = Pipeline(
        [
            InferenceStage(encoding, sampling, max_new_tokens=200),
        ]
    )
    inference_pipeline.run(ctx)  # reuses model from ctx
