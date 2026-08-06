"""
diffusion_llm.py — A complete Masked Diffusion Language Model (MDLM) in one file.

Based on the architecture described in:
  "How to Build a Diffusion Language Model" (Kuleshov et al., 2026)
  LLaDA: Large Language Diffusion with mAsking

Usage:
  python diffusion_llm.py train          # train on sample text
  python diffusion_llm.py generate       # generate text from saved model
  python diffusion_llm.py chat           # interactive chat loop

The model is saved/loaded as a single .pt file (diffusion_llm.pt).
"""

import os
import sys
import math
import json
import random
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# 1. TOKENIZER  (character-level for zero deps)
# ──────────────────────────────────────────────


class CharTokenizer:
    """Tiny character-level tokenizer. No external libraries needed."""

    PAD_TOKEN = "<pad>"
    MASK_TOKEN = "<mask>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SPECIAL = [PAD_TOKEN, MASK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, text: str | None = None, vocab: dict | None = None):
        if vocab is not None:
            self.token2id = vocab
        else:
            chars = sorted(set(text))
            self.token2id = {tok: i for i, tok in enumerate(self.SPECIAL)}
            offset = len(self.SPECIAL)
            for i, c in enumerate(chars):
                self.token2id[c] = i + offset

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.mask_id = self.token2id[self.MASK_TOKEN]
        self.bos_id = self.token2id[self.BOS_TOKEN]
        self.eos_id = self.token2id[self.EOS_TOKEN]
        self.vocab_size = len(self.token2id)

    def encode(self, text: str) -> list[int]:
        return (
            [self.bos_id]
            + [self.token2id.get(c, self.pad_id) for c in text]
            + [self.eos_id]
        )

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        return "".join(self.id2token.get(i, "?") for i in ids if i not in skip)

    def to_dict(self) -> dict:
        return self.token2id


# ──────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────


class TextDataset(Dataset):
    """Sliding-window character dataset."""

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        ids = tokenizer.encode(text)
        # split into non-overlapping chunks
        self.samples = []
        for i in range(0, len(ids) - seq_len, seq_len // 2):
            chunk = ids[i : i + seq_len]
            if len(chunk) == seq_len:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────
# 3. MODEL
# ──────────────────────────────────────────────


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, L, D)
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # bidirectional self-attention (no causal mask — key feature of diffusion LLMs)
        h, _ = self.attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class MaskedDiffusionLM(nn.Module):
    """
    Bidirectional transformer trained with the MDLM objective.
    Given a partially masked sequence x_t, predicts the original clean tokens x_0.
    """

    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mask_id = mask_id
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = SinusoidalPositionEmbedding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying (embed ↔ head) — common best-practice
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t : (B, L)  — partially masked token ids
        returns logits (B, L, V)
        """
        h = self.pos_emb(self.embed(x_t))
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


# ──────────────────────────────────────────────
# 4. DIFFUSION FORWARD / TRAINING OBJECTIVE
# ──────────────────────────────────────────────


def mask_tokens(x0: torch.Tensor, mask_id: int, mask_rate: float | None = None):
    """
    Forward masking process.
    - mask_rate=None  → sample uniformly from (0, 1) per batch (MDLM training)
    - mask_rate=float → fixed rate (for debugging / generation)
    Returns (x_t, mask_rate, masked_positions).
    """
    B, L = x0.shape
    if mask_rate is None:
        # sample a random masking rate per example, uniform over (0,1)
        rates = torch.rand(B, 1, device=x0.device)  # (B,1)
    else:
        rates = torch.full((B, 1), mask_rate, device=x0.device)

    # bernoulli mask  — each token masked independently with probability = rate
    mask = torch.bernoulli(rates.expand(B, L)).bool()  # (B, L)
    x_t = x0.clone()
    x_t[mask] = mask_id
    return x_t, rates.squeeze(1), mask  # x_t (B,L), rates (B,), mask (B,L)


def compute_loss(model: MaskedDiffusionLM, x0: torch.Tensor) -> torch.Tensor:
    """
    MDLM ELBO loss (Equation 3 in the survey).

    L = E_t[ CE(model(x_t), x0) / E[fraction masked] ]

    In practice we compute:
      - sample a random masking rate t per example
      - mask tokens with that rate
      - cross-entropy only on masked positions
      - normalise by the expected mask rate t

    This is the simplified variational lower-bound on log p(x0).
    """
    B, L = x0.shape
    x_t, rates, mask = mask_tokens(x0, model.mask_id)  # rates (B,)

    # if nothing is masked in a sample, skip it (avoid divide-by-zero)
    num_masked = mask.sum(dim=1).float()  # (B,)
    valid = num_masked > 0
    if not valid.any():
        return torch.tensor(0.0, device=x0.device, requires_grad=True)

    logits = model(x_t)  # (B, L, V)

    # cross-entropy over all positions; we'll zero out unmasked ones
    ce = F.cross_entropy(
        logits.reshape(B * L, -1), x0.reshape(B * L), reduction="none"
    ).reshape(
        B, L
    )  # (B, L)

    # zero loss on unmasked tokens
    ce = ce * mask.float()  # (B, L)

    # sum per example, normalise by rate t (eq. 3 in the paper)
    # rates (B,) — expected fraction masked equals rates
    loss_per = ce.sum(dim=1) / (rates.clamp(min=1e-6) * L)  # (B,)

    return loss_per[valid].mean()


# ──────────────────────────────────────────────
# 5. GENERATION (SAMPLING)
# ──────────────────────────────────────────────


@torch.no_grad()
def generate(
    model: MaskedDiffusionLM,
    tokenizer: CharTokenizer,
    prompt: str = "",
    gen_len: int = 200,
    steps: int = 40,
    temperature: float = 1.0,
    remask_frac: float = 0.0,  # fraction of newly unmasked tokens to re-mask (error correction)
    device: str = "cpu",
) -> str:
    """
    Masked diffusion sampling with optional remasking (error correction).

    Algorithm:
      1. Start with a fully masked sequence of length gen_len.
      2. At each step t (from T→0):
         a. model predicts x0 from x_t
         b. sample tokens from predictions
         c. unmask a fraction of positions (scheduled)
         d. optionally remask a small fraction (error correction)
    """
    model.eval()
    model.to(device)

    # encode prompt; keep it fixed
    prompt_ids = tokenizer.encode(prompt) if prompt else []
    total_len = len(prompt_ids) + gen_len
    total_len = min(total_len, 512)
    gen_start = len(prompt_ids)

    # initial sequence: prompt is fixed, generation part is fully masked
    x_t = torch.full((1, total_len), tokenizer.mask_id, dtype=torch.long, device=device)
    if prompt_ids:
        x_t[0, :gen_start] = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    # track which generation positions are still masked
    gen_mask = torch.ones(1, total_len, dtype=torch.bool, device=device)
    gen_mask[0, :gen_start] = False  # prompt is never masked

    for step in range(steps):
        # ── predict x0 ──────────────────────────────────────────
        logits = model(x_t)  # (1, L, V)
        logits = logits / max(temperature, 1e-6)

        # sample from predictions
        probs = F.softmax(logits, dim=-1)
        x0_hat = torch.multinomial(probs.reshape(-1, model.vocab_size), 1).reshape(
            1, total_len
        )

        # ── decide how many positions to unmask this step ───────
        # linear schedule: unmask ~1/steps fraction each step
        frac_unmasked_so_far = step / steps
        frac_unmasked_target = (step + 1) / steps
        n_masked_now = gen_mask.sum().item()
        n_total_gen = gen_len
        n_to_unmask = max(
            1, round((frac_unmasked_target - frac_unmasked_so_far) * n_total_gen)
        )
        n_to_unmask = min(n_to_unmask, int(n_masked_now))

        # ── unmask the highest-confidence positions ──────────────
        # confidence = probability of the predicted token
        confidence = (
            probs.squeeze(0).gather(1, x0_hat.squeeze(0).unsqueeze(1)).squeeze(1)
        )
        confidence = confidence.masked_fill(~gen_mask.squeeze(0), -1.0)

        if n_to_unmask > 0:
            topk_idx = confidence.topk(n_to_unmask).indices
            x_t[0, topk_idx] = x0_hat[0, topk_idx]
            gen_mask[0, topk_idx] = False

        # ── optional remasking (error correction) ────────────────
        if remask_frac > 0.0 and step < steps - 1:
            # re-mask a small fraction of previously unmasked generation tokens
            unmasked_gen = (~gen_mask[0]) & torch.arange(total_len, device=device).ge(
                gen_start
            )
            unmasked_idx = unmasked_gen.nonzero(as_tuple=True)[0]
            n_remask = max(0, round(remask_frac * len(unmasked_idx)))
            if n_remask > 0:
                perm = torch.randperm(len(unmasked_idx), device=device)[:n_remask]
                rm_idx = unmasked_idx[perm]
                x_t[0, rm_idx] = tokenizer.mask_id
                gen_mask[0, rm_idx] = True

    # fill any remaining masks with best prediction on final pass
    if gen_mask.any():
        logits = model(x_t)
        final_tokens = logits.argmax(dim=-1)
        x_t[0, gen_mask.squeeze(0)] = final_tokens[0, gen_mask.squeeze(0)]

    gen_ids = x_t[0, gen_start:].tolist()
    return tokenizer.decode(gen_ids)


# ──────────────────────────────────────────────
# 6. SAVE / LOAD  (single .pt file)
# ──────────────────────────────────────────────

SAVE_PATH = "diffusion_llm.pt"


def save_model(model: MaskedDiffusionLM, tokenizer: CharTokenizer, cfg: dict):
    """Save everything needed for inference into one .pt file."""
    payload = {
        "cfg": cfg,
        "vocab": tokenizer.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, SAVE_PATH)
    size_mb = os.path.getsize(SAVE_PATH) / 1e6
    print(f"✔  Model saved → {SAVE_PATH}  ({size_mb:.1f} MB)")


def load_model(path: str = SAVE_PATH, device: str = "cpu"):
    """Load model + tokenizer from a single .pt file."""
    payload = torch.load(path, map_location=device)
    cfg = payload["cfg"]
    tokenizer = CharTokenizer(vocab=payload["vocab"])
    model = MaskedDiffusionLM(
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_id,
        **{
            k: cfg[k]
            for k in ("d_model", "n_layers", "n_heads", "d_ff", "max_len", "dropout")
        },
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    print(
        f"✔  Model loaded from {path}  (vocab={tokenizer.vocab_size}, params={sum(p.numel() for p in model.parameters()):,})"
    )
    return model, tokenizer, cfg


# ──────────────────────────────────────────────
# 7. TRAINER
# ──────────────────────────────────────────────

DEFAULT_CFG = dict(
    d_model=256,
    n_layers=6,
    n_heads=8,
    d_ff=1024,
    max_len=256,
    dropout=0.1,
    seq_len=128,
    batch=16,
    lr=3e-4,
    epochs=10,
    grad_clip=1.0,
)

SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog.
In the beginning was the Word, and the Word was with God.
To be or not to be, that is the question.
It was the best of times, it was the worst of times.
Call me Ishmael. Some years ago—never mind how long precisely.
All happy families are alike; each unhappy family is unhappy in its own way.
The sky above the port was the color of television, tuned to a dead channel.
It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife.
Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy lies a small unregarded yellow sun.
Diffusion language models generate text by starting from a noisy draft and iteratively refining it.
Unlike autoregressive models that generate tokens left to right, diffusion models attend to the whole sequence.
Masked diffusion trains a bidirectional transformer to fill in randomly masked tokens.
Generation starts from a fully masked sequence and progressively unmasks positions over many steps.
Remasking allows the model to correct earlier mistakes by revisiting already-committed tokens.
The model learns a variational lower bound on the log-likelihood, equivalent to a randomized BERT objective.
"""


def train(cfg: dict | None = None, text: str | None = None):
    cfg = cfg or DEFAULT_CFG
    text = text or SAMPLE_TEXT

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = CharTokenizer(text)
    dataset = TextDataset(text, tokenizer, seq_len=cfg["seq_len"])
    if len(dataset) == 0:
        print("Text too short; duplicating for training.")
        text = text * 20
        tokenizer = CharTokenizer(text)
        dataset = TextDataset(text, tokenizer, seq_len=cfg["seq_len"])

    loader = DataLoader(dataset, batch_size=cfg["batch"], shuffle=True, drop_last=True)

    model = MaskedDiffusionLM(
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_id,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | vocab={tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_steps = 0.0, 0
        t0 = time.time()
        for batch in loader:
            batch = batch.to(device)
            loss = compute_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            total_steps += 1

        scheduler.step()
        avg_loss = total_loss / max(total_steps, 1)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{cfg['epochs']}  loss={avg_loss:.4f}  ({elapsed:.1f}s)"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, cfg)

        # quick sample every 2 epochs
        if epoch % 2 == 0:
            sample = generate(
                model, tokenizer, prompt="The ", gen_len=80, steps=30, device=device
            )
            print(f"  Sample: «The {sample}»\n")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")
    return model, tokenizer


# ──────────────────────────────────────────────
# 8. CLI
# ──────────────────────────────────────────────


def cmd_train(args):
    text = None
    if args.data and os.path.isfile(args.data):
        with open(args.data, encoding="utf-8") as f:
            text = f.read()
        print(f"Training on '{args.data}'  ({len(text):,} chars)")
    else:
        print("No --data file specified; using built-in sample text.")

    cfg = dict(DEFAULT_CFG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.dmodel:
        cfg["d_model"] = args.dmodel
    if args.layers:
        cfg["n_layers"] = args.layers

    train(cfg, text)


def cmd_generate(args):
    if not os.path.isfile(SAVE_PATH):
        print(
            f"No saved model found at '{SAVE_PATH}'. Run: python diffusion_llm.py train"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_model(SAVE_PATH, device)

    prompt = args.prompt or ""
    out = generate(
        model,
        tokenizer,
        prompt=prompt,
        gen_len=args.length,
        steps=args.steps,
        temperature=args.temp,
        remask_frac=args.remask,
        device=device,
    )
    print(f"\nPrompt : {repr(prompt)}")
    print(f"Output : {repr(out)}")
    print(f"\n{prompt}{out}")


def cmd_chat(args):
    if not os.path.isfile(SAVE_PATH):
        print(
            f"No saved model found at '{SAVE_PATH}'. Run: python diffusion_llm.py train"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_model(SAVE_PATH, device)

    print("\n=== Diffusion LLM Chat ===")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break

        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            gen_len=args.length,
            steps=args.steps,
            temperature=args.temp,
            remask_frac=args.remask,
            device=device,
        )
        print(f"Model: {prompt}{out}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Masked Diffusion Language Model — single file"
    )
    sub = parser.add_subparsers(dest="cmd")

    # train
    t = sub.add_parser("train", help="Train the model")
    t.add_argument("--data", type=str, default="", help="Path to .txt training file")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--dmodel", type=int, default=None, help="Embedding dim")
    t.add_argument(
        "--layers", type=int, default=None, help="Number of transformer layers"
    )

    # generate
    g = sub.add_parser("generate", help="Generate text from saved model")
    g.add_argument("--prompt", type=str, default="", help="Prompt string")
    g.add_argument(
        "--length", type=int, default=200, help="Number of tokens to generate"
    )
    g.add_argument("--steps", type=int, default=40, help="Diffusion steps")
    g.add_argument("--temp", type=float, default=0.9, help="Sampling temperature")
    g.add_argument(
        "--remask", type=float, default=0.05, help="Remasking fraction (0=off)"
    )

    # chat
    c = sub.add_parser("chat", help="Interactive chat")
    c.add_argument("--length", type=int, default=150)
    c.add_argument("--steps", type=int, default=40)
    c.add_argument("--temp", type=float, default=0.9)
    c.add_argument("--remask", type=float, default=0.05)

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        print("\nQuick start:")
        print(
            "  python diffusion_llm.py train                    # train on sample text"
        )
        print(
            "  python diffusion_llm.py train --data mytext.txt  # train on custom text"
        )
        print("  python diffusion_llm.py generate --prompt 'The '")
        print("  python diffusion_llm.py chat")


if __name__ == "__main__":
    main()
# python diffusion_llm.py train --data tinystories.txt --epochs 20
