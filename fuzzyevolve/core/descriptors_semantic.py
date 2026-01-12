from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import numpy as np

from fuzzyevolve.config import Config
from fuzzyevolve.core.descriptors import default_text_descriptor
from fuzzyevolve.core.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerProvider,
)


class SemanticDescriptor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        model_name = cfg.semantic_embedding_model
        if model_name and model_name != "hash":
            self.provider = SentenceTransformerProvider(model_name)
        else:
            self.provider = HashEmbeddingProvider()
        rng = np.random.default_rng(cfg.semantic_projection_seed)
        self.scale = math.sqrt(self.provider.dim)
        self.r1 = _random_unit_vector(rng, self.provider.dim)
        self.r2 = _random_unit_vector(rng, self.provider.dim)

    def __call__(self, text: str) -> dict[str, Any]:
        desc = default_text_descriptor(text)
        vec = self.provider.embed(text)
        desc["semantic_x"] = float(np.dot(vec, self.r1) * self.scale)
        desc["semantic_y"] = float(np.dot(vec, self.r2) * self.scale)
        return desc


def build_descriptor_fn(cfg: Config) -> Callable[[str], dict[str, Any]]:
    if cfg.descriptor_mode == "semantic":
        return SemanticDescriptor(cfg)
    return default_text_descriptor


def _random_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm
