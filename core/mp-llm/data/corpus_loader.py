"""
Optional larger, real-text corpora -- kept separate from `toy_corpus.py` so
the fast, fully-offline default demo path is untouched. `get_corpus(name)`
downloads once, caches to disk under `data/.cache/`, and falls back to the
built-in toy corpus if the network is unavailable (so training never hard-
fails just because a sandbox/CI environment has no internet access).

Bigger corpus != automatically better toy-scale results here. The rest of
the config still matters:
  - Model capacity (`d_model`, `n_layers`) needs to grow with the corpus, or
    the extra data mostly won't be used.
  - `steps_per_epoch` / `epochs` need to grow too -- more data with the same
    handful of gradient steps just means each step sees a different random
    window, not more learning per step.
  - Paradigm-specific costs scale differently: larger `d_model` slows the
    photonic mesh roughly linearly (more `mesh_size`-wide chunks per call);
    `thermodynamic_max_spins` only grows with log2(vocab_size), so a bigger
    corpus barely affects thermodynamic sampling cost -- but more diffusion
    timesteps or longer generation loops multiply thermodynamic/quantum
    calls directly, regardless of corpus size.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

from data.toy_corpus import TOY_CORPUS

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# name -> source URL. All served from raw.githubusercontent.com.
_CORPUS_URLS = {
    "tinyshakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt"
    ),
}


def available_corpora() -> list:
    return ["tiny"] + sorted(_CORPUS_URLS.keys())


def get_corpus(name: str = "tiny", max_chars: int | None = None) -> str:
    """Returns corpus text. `name="tiny"` (the default) always returns the
    built-in offline `TOY_CORPUS` with no network access. Any other name is
    looked up in `_CORPUS_URLS`, downloaded once, cached under
    `data/.cache/<name>.txt`, and reused from cache thereafter. `max_chars`
    truncates the result (useful for keeping a first run fast; the full
    tinyshakespeare corpus is ~1.1M characters).
    """
    if name == "tiny" or name not in _CORPUS_URLS:
        text = TOY_CORPUS
    else:
        text = _load_or_download(name)

    if max_chars is not None:
        text = text[:max_chars]
    return text


def _load_or_download(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{name}.txt")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    url = _CORPUS_URLS[name]
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(
            f"[corpus_loader] Could not download '{name}' from {url} ({e}); "
            f"falling back to the built-in toy corpus."
        )
        return TOY_CORPUS

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
