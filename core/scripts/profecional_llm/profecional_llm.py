import os
import sys
import json
import math
import time
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ─────────────────────────────────────────────
# BPE TOKENIZER (self-contained, no external deps beyond stdlib)
# ─────────────────────────────────────────────


def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    PAD = "<|pad|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    USR = "<|user|>"
    AST = "<|assistant|>"
    SPECIAL = [PAD, BOS, EOS, USR, AST]

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: Dict[str, int] = {}
        self.inv_special: Dict[int, str] = {}

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = _merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        next_id = max(self.vocab) + 1
        for tok in self.SPECIAL:
            self.special[tok] = next_id
            self.inv_special[next_id] = tok
            next_id += 1

    @property
    def vocab_size(self) -> int:
        return max(self.vocab) + 1 + len(self.special)

    @property
    def pad_id(self) -> int:
        return self.special[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.special[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.special[self.EOS]

    @property
    def usr_id(self) -> int:
        return self.special[self.USR]

    @property
    def ast_id(self) -> int:
        return self.special[self.AST]

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        parts = []
        for i in ids:
            if i in self.inv_special:
                parts.append(self.inv_special[i].encode("utf-8"))
            elif i in self.vocab:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        ids = [self.bos_id]
        for msg in messages:
            if msg["role"] == "user":
                ids.append(self.usr_id)
            else:
                ids.append(self.ast_id)
            ids.extend(self.encode(msg["content"].strip()))
            ids.append(self.eos_id)
        return ids

    def save(self, path: str):
        data = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "special": self.special,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.merges = {(k[0], k[1]): v for k, v in data["merges"]}
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self.special = data["special"]
        self.inv_special = {v: k for k, v in self.special.items()}


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
CODE_ALPACA_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"


def download_json(url: str, cache_path: str) -> List[Dict]:
    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(url, cache_path)
    with open(cache_path) as f:
        return json.load(f)


def build_conversational_dataset(
    tokenizer: BPETokenizer, max_seq_len: int
) -> Tuple[List[List[int]], List[List[int]]]:
    alpaca = download_json(ALPACA_URL, "alpaca_cleaned.json")
    code = download_json(CODE_ALPACA_URL, "code_alpaca.json")
    raw = alpaca + code

    sequences: List[List[int]] = []
    for item in raw:
        instruction = item.get("instruction", "").strip()
        context = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue
        user_text = instruction if not context else f"{instruction}\n\n{context}"
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output},
        ]
        ids = tokenizer.encode_chat(messages)
        if 8 <= len(ids) <= max_seq_len:
            sequences.append(ids)

    sequences.sort(key=len)
    split = int(0.9 * len(sequences))
    return sequences[:split], sequences[split:]


def pad_batch(
    seqs: List[List[int]], pad_id: int, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_x, batch_y = [], []
    for ids in seqs:
        ids = ids[: max_len + 1]
        x = ids[:-1]
        y = ids[1:]
        pad_len = max_len - len(x)
        x = x + [pad_id] * pad_len
        y = y + [-100] * pad_len
        batch_x.append(x)
        batch_y.append(y)
    return torch.tensor(batch_x, dtype=torch.long), torch.tensor(
        batch_y, dtype=torch.long
    )


def sample_batch(
    sequences: List[List[int]], batch_size: int, pad_id: int, max_seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(len(sequences), (batch_size,)).tolist()
    batch = [sequences[i] for i in idx]
    return pad_batch(batch, pad_id, max_seq_len)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────


@dataclass
class Config:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    n_experts: int = 4
    n_experts_active: int = 2
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 512
    bpe_vocab_size: int = 4096
    vocab_size: int = 0

    pretrain_epochs: int = 3000
    sft_epochs: int = 1000
    rl_epochs: int = 500
    batch_size: int = 8
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_interval: int = 100
    checkpoint_path: str = "checkpoint.pt"
    tokenizer_path: str = "tokenizer.json"


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def precompute_freqs_cis(
    dim: int, seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seq, n_kv, hd = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, seq, n_kv, n_rep, hd)
        .reshape(bsz, seq, n_kv * n_rep, hd)
    )


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads else cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.cfg = cfg

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        freqs_cis = precompute_freqs_cis(
            self.head_dim, seq_len, self.cfg.rope_theta
        ).to(x.device)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        keys = repeat_kv(xk, self.n_rep).transpose(1, 2)
        values = repeat_kv(xv, self.n_rep).transpose(1, 2)
        xq = xq.transpose(1, 2)

        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        scores = (
            torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim) + mask
        )
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        out = torch.matmul(scores, values)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class Expert(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(2 * (4 * cfg.dim) / 3)
        if cfg.ffn_dim_multiplier:
            hidden = int(cfg.ffn_dim_multiplier * hidden)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_active = cfg.n_experts_active
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        gate_logits = self.gate(x_flat)
        weights, selected = torch.topk(gate_logits, self.n_active, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_out = expert(expert_input)
            expert_weights = weights[mask]
            positions = (selected[mask] == i).float()
            w = (expert_weights * positions).sum(dim=-1, keepdim=True)
            out[mask] += w * expert_out

        return out.view(bsz, seq_len, dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ff_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.moe = MoE(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.tok_embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, step: int, path: str
):
    raw = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path,
    )
    print(f"  checkpoint saved → {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    path: str,
    device: torch.device,
) -> int:
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"  resumed from {path} at step {ckpt['step']}")
    return ckpt["step"]


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────


def get_lr(
    step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        max_lr: float,
        min_lr: float,
    ):
        self.total = total_steps
        self.warmup = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, lr_lambda=self._lambda)

    def _lambda(self, step: int) -> float:
        return (
            get_lr(step, self.total, self.warmup, self.max_lr, self.min_lr)
            / self.max_lr
        )


# ─────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sequences: List[List[int]],
    cfg: Config,
    pad_id: int,
    device: torch.device,
    n_batches: int = 10,
) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(sequences, cfg.batch_size, pad_id, cfg.max_seq_len)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ─────────────────────────────────────────────
# PRETRAINING
# ─────────────────────────────────────────────


def pretrain(
    rank: int,
    world_size: int,
    cfg: Config,
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    tokenizer: BPETokenizer,
):
    is_main = rank == 0

    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = (
        torch.device("cuda", rank)
        if torch.cuda.is_available() and torch.cuda.device_count() > rank
        else torch.device("cpu")
    )

    model = Transformer(cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    scheduler = CosineWithWarmup(
        optimizer, cfg.pretrain_epochs, cfg.warmup_steps, cfg.lr, cfg.min_lr
    )

    start_step = 0
    if is_main and os.path.exists(cfg.checkpoint_path):
        start_step = load_checkpoint(
            model, optimizer, scheduler, cfg.checkpoint_path, device
        )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print(f"  PRETRAINING")
        print(f"  params:  {n_params:,}")
        print(f"  vocab:   {cfg.vocab_size}")
        print(f"  device:  {device}  world_size: {world_size}")
        print(f"  train:   {len(train_seqs):,} sequences")
        print(f"  val:     {len(val_seqs):,} sequences")
        print(f"{'='*60}\n")

    t0 = time.time()
    for step in range(start_step, cfg.pretrain_epochs):
        x, y = sample_batch(
            train_seqs, cfg.batch_size, tokenizer.pad_id, cfg.max_seq_len
        )
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if is_main and step % cfg.log_interval == 0:
            val_loss = evaluate(model, val_seqs, cfg, tokenizer.pad_id, device)
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  step {step:5d}/{cfg.pretrain_epochs} | train {loss.item():.4f} | val {val_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s"
            )
            t0 = time.time()

    if is_main:
        save_checkpoint(
            model, optimizer, scheduler, cfg.pretrain_epochs, cfg.checkpoint_path
        )

    if world_size > 1:
        dist.destroy_process_group()

    return model


# ─────────────────────────────────────────────
# SFT (SUPERVISED FINE-TUNING)
# ─────────────────────────────────────────────


def sft(
    model: nn.Module,
    cfg: Config,
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    tokenizer: BPETokenizer,
    device: torch.device,
):
    print(f"\n{'='*60}")
    print(f"  SFT (SUPERVISED FINE-TUNING)")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.1, weight_decay=0.1)
    scheduler = CosineWithWarmup(
        optimizer, cfg.sft_epochs, cfg.warmup_steps // 2, cfg.lr * 0.1, cfg.min_lr * 0.1
    )

    sft_ckpt = cfg.checkpoint_path.replace(".pt", "_sft.pt")

    t0 = time.time()
    for step in range(cfg.sft_epochs):
        x, y = sample_batch(
            train_seqs, cfg.batch_size, tokenizer.pad_id, cfg.max_seq_len
        )
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % cfg.log_interval == 0:
            val_loss = evaluate(model, val_seqs, cfg, tokenizer.pad_id, device)
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  sft {step:5d}/{cfg.sft_epochs} | train {loss.item():.4f} | val {val_loss:.4f} | lr {lr_now:.2e} | {elapsed:.1f}s"
            )
            t0 = time.time()

    save_checkpoint(model, optimizer, scheduler, cfg.sft_epochs, sft_ckpt)


# ─────────────────────────────────────────────
# REWARD MODEL (for RLHF/RLVR)
# ─────────────────────────────────────────────


class RewardModel(nn.Module):
    def __init__(self, base: Transformer):
        super().__init__()
        self.base = deepcopy(base)
        self.head = nn.Linear(base.cfg.dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base.tok_embeddings(x)
        for layer in self.base.layers:
            h = layer(h)
        h = self.base.norm(h)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)


def train_reward_model(
    rm: RewardModel,
    train_seqs: List[List[int]],
    cfg: Config,
    tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 200,
):
    print(f"\n{'='*60}")
    print(f"  REWARD MODEL TRAINING")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(rm.parameters(), lr=1e-4)
    rm.train()

    for step in range(epochs):
        idx_chosen = torch.randint(len(train_seqs), (cfg.batch_size,))
        idx_rejected = torch.randint(len(train_seqs), (cfg.batch_size,))

        chosen = [train_seqs[i] for i in idx_chosen]
        rejected = [train_seqs[i] for i in idx_rejected]

        xc, _ = pad_batch(chosen, tokenizer.pad_id, cfg.max_seq_len)
        xr, _ = pad_batch(rejected, tokenizer.pad_id, cfg.max_seq_len)
        xc, xr = xc.to(device), xr.to(device)

        r_chosen = rm(xc)
        r_rejected = rm(xr)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(rm.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            print(
                f"  rm step {step:4d} | loss {loss.item():.4f} | r_chosen {r_chosen.mean().item():.3f} | r_rejected {r_rejected.mean().item():.3f}"
            )


# ─────────────────────────────────────────────
# RLHF via REINFORCE (policy gradient)
# ─────────────────────────────────────────────


def rollout(
    policy: Transformer,
    prompt_ids: List[int],
    cfg: Config,
    tokenizer: BPETokenizer,
    device: torch.device,
    max_new: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    policy.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    log_probs = []

    with torch.no_grad():
        for _ in range(max_new):
            inp = ids[:, -cfg.max_seq_len :]
            logits, _ = policy(inp)
            logits_last = logits[:, -1] / 0.9
            probs = F.softmax(logits_last, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            lp = torch.log(probs.gather(1, next_tok) + 1e-8)
            log_probs.append(lp)
            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() == tokenizer.eos_id:
                break

    policy.train()
    generated = ids[0].tolist()
    log_probs_tensor = torch.cat(log_probs, dim=1).squeeze(0)
    return torch.tensor(generated, dtype=torch.long, device=device), log_probs_tensor


def rlhf(
    policy: Transformer,
    ref_policy: Transformer,
    rm: RewardModel,
    train_seqs: List[List[int]],
    cfg: Config,
    tokenizer: BPETokenizer,
    device: torch.device,
):
    print(f"\n{'='*60}")
    print(f"  RLHF (REINFORCE + KL PENALTY)")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    kl_coeff = 0.1

    for step in range(cfg.rl_epochs):
        idx = torch.randint(len(train_seqs), (1,)).item()
        prompt = train_seqs[idx][: cfg.max_seq_len // 2]

        generated, log_probs = rollout(policy, prompt, cfg, tokenizer, device)

        with torch.no_grad():
            gen_tensor = generated.unsqueeze(0)
            if gen_tensor.shape[1] > cfg.max_seq_len:
                gen_tensor = gen_tensor[:, : cfg.max_seq_len]
            reward = rm(gen_tensor).item()

            ref_logits, _ = ref_policy(gen_tensor)
            ref_probs = F.softmax(ref_logits[:, len(prompt) - 1 : -1] / 0.9, dim=-1)
            n = min(len(log_probs), ref_probs.shape[1])
            ref_lp = torch.log(
                ref_probs[:, :n]
                .gather(
                    2,
                    generated[len(prompt) : len(prompt) + n].unsqueeze(0).unsqueeze(-1),
                )
                .squeeze(-1)
                + 1e-8
            )
            kl = (log_probs[:n] - ref_lp.squeeze(0)).mean().item()

        advantage = reward - kl_coeff * kl
        policy_loss = -(log_probs * advantage).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            print(
                f"  rl  step {step:4d} | reward {reward:.4f} | kl {kl:.4f} | loss {policy_loss.item():.4f}"
            )

    rl_ckpt = cfg.checkpoint_path.replace(".pt", "_rl.pt")
    torch.save(policy.state_dict(), rl_ckpt)
    print(f"  RL checkpoint saved → {rl_ckpt}")


# ─────────────────────────────────────────────
# RLVR (RL from Verifiable Rewards)
# ─────────────────────────────────────────────


def verify_reward(response: str, correct_answer: str) -> float:
    response_clean = response.strip().lower().replace(".", "").replace(",", "")
    answer_clean = correct_answer.strip().lower().replace(".", "").replace(",", "")
    if answer_clean in response_clean:
        return 1.0
    words_r = set(response_clean.split())
    words_a = set(answer_clean.split())
    overlap = len(words_r & words_a) / max(1, len(words_a))
    return overlap * 0.5


def rlvr(
    policy: Transformer,
    train_seqs: List[List[int]],
    cfg: Config,
    tokenizer: BPETokenizer,
    device: torch.device,
    qa_pairs: List[Tuple[str, str]],
):
    print(f"\n{'='*60}")
    print(f"  RLVR (RL FROM VERIFIABLE REWARDS)")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6)
    baseline = 0.0
    alpha = 0.9

    for step in range(cfg.rl_epochs):
        q, a = qa_pairs[step % len(qa_pairs)]
        prompt_ids = tokenizer.encode_chat([{"role": "user", "content": q}])
        prompt_ids = prompt_ids[: cfg.max_seq_len // 2]

        generated, log_probs = rollout(policy, prompt_ids, cfg, tokenizer, device)
        response_text = tokenizer.decode(generated[len(prompt_ids) :].tolist())
        reward = verify_reward(response_text, a)

        baseline = alpha * baseline + (1 - alpha) * reward
        advantage = reward - baseline

        policy_loss = -(log_probs * advantage).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            print(
                f"  rlvr step {step:4d} | reward {reward:.3f} | baseline {baseline:.3f} | loss {policy_loss.item():.4f}"
            )


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────


def generate(
    model: nn.Module,
    prompt: str,
    tokenizer: BPETokenizer,
    cfg: Config,
    device: torch.device,
    max_new: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()

    ids = tokenizer.encode_chat([{"role": "user", "content": prompt}])
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new):
            inp = tokens[:, -cfg.max_seq_len :]
            logits, _ = raw(inp)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_id = torch.gather(sorted_idx, -1, torch.multinomial(sorted_probs, 1))
            tokens = torch.cat([tokens, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break

    out = tokens[0].tolist()
    if tokenizer.eos_id in out:
        out = out[: out.index(tokenizer.eos_id)]
    return tokenizer.decode(out[len(ids) :])


# ─────────────────────────────────────────────
# DDP SPAWN WRAPPER
# ─────────────────────────────────────────────


def _ddp_worker(
    rank: int,
    world_size: int,
    cfg: Config,
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    tokenizer: BPETokenizer,
):
    pretrain(rank, world_size, cfg, train_seqs, val_seqs, tokenizer)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()

    print("\n[1/6] Training BPE tokenizer...")
    import urllib.request as _req

    _req.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "shakespeare.txt",
    )
    with open("shakespeare.txt") as f:
        raw_text = f.read()

    tokenizer = BPETokenizer()
    tokenizer.train(raw_text[:80000], vocab_size=cfg.bpe_vocab_size)
    tokenizer.save(cfg.tokenizer_path)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"  vocab_size={cfg.vocab_size}")

    print("\n[2/6] Building conversational dataset (Alpaca + CodeAlpaca)...")
    train_seqs, val_seqs = build_conversational_dataset(tokenizer, cfg.max_seq_len)
    print(f"  train={len(train_seqs):,}  val={len(val_seqs):,}")

    world_size = max(1, torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[3/6] Pretraining (DDP)...")
    if world_size > 1:
        mp.spawn(
            _ddp_worker,
            args=(world_size, cfg, train_seqs, val_seqs, tokenizer),
            nprocs=world_size,
            join=True,
        )
    else:
        pretrain(0, 1, cfg, train_seqs, val_seqs, tokenizer)

    print("\n[4/6] Loading checkpoint for SFT...")
    model = Transformer(cfg).to(device)
    if os.path.exists(cfg.checkpoint_path):
        ckpt = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"  loaded {cfg.checkpoint_path}")

    sft(model, cfg, train_seqs, val_seqs, tokenizer, device)

    print("\n[5/6] RLHF + RLVR...")
    ref_model = deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    rm = RewardModel(model).to(device)
    train_reward_model(rm, train_seqs, cfg, tokenizer, device, epochs=200)

    rlhf(model, ref_model, rm, train_seqs, cfg, tokenizer, device)

    qa_pairs = [
        ("What is the capital of France?", "Paris"),
        ("What is 2 + 2?", "4"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the speed of light?", "299792458"),
        ("What language is Python?", "programming"),
    ]
    rlvr(model, train_seqs, cfg, tokenizer, device, qa_pairs)

    print("\n[6/6] Generation test...")
    prompts = [
        "Explain recursion in simple terms.",
        "Write a Python function to reverse a string.",
        "What are three benefits of exercise?",
    ]
    for p in prompts:
        out = generate(model, p, tokenizer, cfg, device)
        print(f"\n  USER: {p}")
        print(f"  ASST: {out[:300]}")

    print("\nDone.")
