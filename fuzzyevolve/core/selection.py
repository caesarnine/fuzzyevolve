from __future__ import annotations

import math
import random
from typing import Any

import trueskill as ts

from fuzzyevolve.core.archive import MapElitesArchive


def optimistic_score(ratings: dict[str, ts.Rating], beta: float) -> float:
    if not ratings:
        return 0.0
    total = 0.0
    for rating in ratings.values():
        total += rating.mu + beta * rating.sigma
    return total / len(ratings)


class ParentSelector:
    def __init__(
        self,
        mode: str,
        beta: float,
        temp: float,
        rng: random.Random | None = None,
    ) -> None:
        self.mode = mode
        self.beta = beta
        self.temp = temp
        self.rng = rng or random.Random()

    def select_parent(self, archive: MapElitesArchive) -> Any:
        if self.mode == "uniform_cell":
            return archive.random_elite()
        if self.mode != "optimistic_cell_softmax":
            raise ValueError(f"Unknown selection mode '{self.mode}'.")

        cells = [(key, bucket) for key, bucket in archive.iter_cells() if bucket]
        if not cells:
            raise ValueError("Cannot select parent from empty archive.")

        cell_scores = [
            max(optimistic_score(elite.ratings, self.beta) for elite in bucket)
            for _, bucket in cells
        ]
        max_score = max(cell_scores)
        scaled = [(score - max_score) / self.temp for score in cell_scores]
        weights = [math.exp(value) for value in scaled]
        selected_index = self.rng.choices(range(len(cells)), weights=weights)[0]
        _, bucket = cells[selected_index]
        return max(bucket, key=lambda elite: optimistic_score(elite.ratings, self.beta))
