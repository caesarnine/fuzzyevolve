from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from fuzzyevolve.config import Config
from fuzzyevolve.core.descriptors import default_text_descriptor
from fuzzyevolve.core.embeddings import HashEmbeddingProvider


class SemanticDescriptor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg.semantic_embedding_model and cfg.semantic_embedding_model != "hash":
            raise ValueError(
                "Only 'hash' embeddings are supported in v0.2; "
                "set semantic_embedding_model='hash' or leave it unset."
            )
        self.provider = HashEmbeddingProvider()
        rng = np.random.default_rng(cfg.semantic_projection_seed)
        self.r1 = _random_unit_vector(rng, self.provider.dim)
        self.r2 = _random_unit_vector(rng, self.provider.dim)

    def __call__(self, text: str) -> dict[str, Any]:
        desc = default_text_descriptor(text)
        vec = self.provider.embed(text)
        desc["semantic_x"] = float(np.dot(vec, self.r1))
        desc["semantic_y"] = float(np.dot(vec, self.r2))
        return desc


def build_descriptor_fn(cfg: Config) -> Callable[[str], dict[str, Any]]:
    if cfg.descriptor_mode == "semantic":
        return SemanticDescriptor(cfg)
    return default_text_descriptor


def _random_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm
