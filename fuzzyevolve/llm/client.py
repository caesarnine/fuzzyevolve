"""Model selection utilities for PydanticAI-backed LLM calls."""

from __future__ import annotations

import random
from typing import Sequence

from pydantic_ai.settings import ModelSettings

from fuzzyevolve.llm.models import ModelSpec


class ModelEnsemble:
    """Weighted random selection over model specs."""

    def __init__(
        self, llm_ensemble: Sequence[ModelSpec], rng: random.Random | None = None
    ) -> None:
        self.llm_ensemble = list(llm_ensemble)
        if not self.llm_ensemble:
            raise ValueError("LLM ensemble cannot be empty.")
        self.rng = rng or random.Random()

    def pick(self) -> tuple[str, ModelSettings]:
        models, probs, temps = zip(
            *[(e.model, e.p, e.temperature) for e in self.llm_ensemble]
        )
        idx = self.rng.choices(range(len(models)), weights=probs)[0]
        settings: ModelSettings = {"temperature": temps[idx]}
        return models[idx], settings
