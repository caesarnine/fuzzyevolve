from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class SentenceTransformerProvider:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with `pip install sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        if vec.ndim != 1:
            vec = np.asarray(vec).reshape(-1)
        norm = np.linalg.norm(vec) or 1.0
        vec = vec / norm
        self._cache[text] = vec
        return vec
