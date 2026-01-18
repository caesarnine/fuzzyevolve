from __future__ import annotations

import random
import threading
from typing import Sequence

from pydantic_ai.settings import ModelSettings

from fuzzyevolve.config import ModelSpec


class ModelEnsemble:
    """Weighted random selection over model specs."""

    def __init__(
        self, specs: Sequence[ModelSpec], rng: random.Random | None = None
    ) -> None:
        self.specs = list(specs)
        if not self.specs:
            raise ValueError("Model ensemble cannot be empty.")
        self.rng = rng or random.Random()
        self._rng_lock = threading.Lock()

    def pick(self) -> tuple[str, ModelSettings]:
        models, weights, temps = zip(
            *[(spec.model, spec.weight, spec.temperature) for spec in self.specs]
        )
        with self._rng_lock:
            idx = self.rng.choices(range(len(models)), weights=weights)[0]
        settings: ModelSettings = {"temperature": temps[idx]}
        return models[idx], settings
