"""
Tokenizer / Vocabulary component.

A character-level tokenizer keeps the toy-scale training loop genuinely
fast (small vocab -> small output-projection and sampling components,
which matters a lot for the Quantum/Photonic/Thermodynamic paradigms whose
simulators do not scale like real hardware would). It is still a real,
persistent, reproducible tokenizer -- not a stub -- and follows the spec's
Tokenizer/Vocabulary component split (vocabulary is exposed separately so a
different Tokenizer implementation could reuse it).
"""

from __future__ import annotations

import json
from typing import Dict, List

from core.interfaces import TokenizerInterface


class Vocabulary:
    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars})

    @classmethod
    def from_json(cls, s: str) -> "Vocabulary":
        data = json.loads(s)
        return cls(data["chars"])

    @classmethod
    def build(cls, text: str) -> "Vocabulary":
        chars = sorted(set(text))
        return cls(chars)


class CharTokenizer(TokenizerInterface):
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    def encode(self, text: str) -> List[int]:
        return [self.vocab.stoi[ch] for ch in text if ch in self.vocab.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.vocab.itos[i] for i in ids)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.vocab.to_json())

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            vocab = Vocabulary.from_json(f.read())
        return cls(vocab)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(Vocabulary.build(text))
