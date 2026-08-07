"""
Pipeline Pattern: the inference workflow, split into the spec's stages:

    inference()
      -> load_model()
      -> load_tokenizer()
      -> encode()
      -> forward()      (happens inside generate(), looped internally)
      -> sample()
      -> decode()
      -> print()
"""

from __future__ import annotations

from typing import Optional, Type

from config.config import MPLLMConfig
from models.tokenizer import CharTokenizer


class InferencePipeline:
    def __init__(self, run_dir: str, llm_cls: Type, llm_kwargs: Optional[dict] = None):
        self.run_dir = run_dir
        self.llm_cls = llm_cls
        self.llm_kwargs = llm_kwargs or {}
        self.config: Optional[MPLLMConfig] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.llm = None

    def load_tokenizer(self) -> CharTokenizer:
        import os

        self.tokenizer = CharTokenizer.load(os.path.join(self.run_dir, "tokenizer.json"))
        return self.tokenizer

    def load_model(self):
        import os

        self.config = MPLLMConfig.load(os.path.join(self.run_dir, "config.json"))
        self.llm = self.llm_cls(self.config, tokenizer=self.tokenizer, **self.llm_kwargs)
        self.llm.load(self.run_dir)
        self.llm.model.eval()
        return self.llm

    def encode(self, text: str):
        return self.tokenizer.encode(text)

    def sample(self, prompt: str, max_new_tokens: int = 20, temperature: float = 1.0):
        import torch

        token_ids = torch.tensor(self.encode(prompt)).unsqueeze(0)
        if self.config.model_kind == "diffusion":
            out = self.llm.inference(
                seq_len=self.config.context_length, batch_size=1, temperature=temperature
            )
        else:
            out = self.llm.inference(token_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        return out

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())

    def run(self, prompt: str = "", max_new_tokens: int = 20, temperature: float = 1.0) -> str:
        self.load_tokenizer()
        self.load_model()
        out_ids = self.sample(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        text = self.decode(out_ids)
        print(text)
        return text
