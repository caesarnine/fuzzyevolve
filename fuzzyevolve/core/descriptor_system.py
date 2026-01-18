from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from fuzzyevolve.config import Config
from fuzzyevolve.core.descriptors import DescriptorSpace, build_descriptor_space
from fuzzyevolve.core.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerProvider,
)


def build_descriptor_system(
    cfg: Config,
) -> tuple[Callable[[str], dict[str, Any]], DescriptorSpace]:
    """
    Returns (describe(text) -> descriptor dict, descriptor_space).

    The descriptor dict is guaranteed to include all axes required by the returned
    space (and typically nothing else).
    """
    if cfg.descriptor.kind == "embedding_2d":
        embedding_cfg = cfg.descriptor.embedding_2d
        space = build_descriptor_space(
            {
                "embed_x": {"bins": embedding_cfg.bins_x},
                "embed_y": {"bins": embedding_cfg.bins_y},
            }
        )
        describe = Embedding2DDescriptor(
            embedding_model=embedding_cfg.embedding_model,
            projection_seed=embedding_cfg.projection_seed,
        )
        return describe, space

    if cfg.descriptor.kind == "length":
        space = build_descriptor_space({"len": {"bins": cfg.descriptor.length_bins}})
        return length_descriptor, space

    raise ValueError(f"Unknown descriptor kind '{cfg.descriptor.kind}'.")


def length_descriptor(text: str) -> dict[str, Any]:
    return {"len": len(text)}


class Embedding2DDescriptor:
    def __init__(self, *, embedding_model: str | None, projection_seed: int) -> None:
        if embedding_model and embedding_model != "hash":
            self.provider = SentenceTransformerProvider(embedding_model)
        else:
            self.provider = HashEmbeddingProvider()
        rng = np.random.default_rng(projection_seed)
        self.scale = math.sqrt(self.provider.dim)
        self.r1 = _random_unit_vector(rng, self.provider.dim)
        self.r2 = _random_unit_vector(rng, self.provider.dim)

    def __call__(self, text: str) -> dict[str, Any]:
        vec = self.provider.embed(text)
        return {
            "embed_x": float(np.dot(vec, self.r1) * self.scale),
            "embed_y": float(np.dot(vec, self.r2) * self.scale),
        }


def _random_unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm
