from __future__ import annotations

import hashlib
from typing import Dict, Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashEmbeddingProvider:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        seed = _hash_to_seed(text)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        norm = np.linalg.norm(vec) or 1.0
        vec = vec / norm
        self._cache[text] = vec
        return vec


def _hash_to_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)
