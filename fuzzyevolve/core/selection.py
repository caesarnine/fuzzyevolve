from __future__ import annotations

import random

import trueskill as ts

from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool


def optimistic_score(ratings: dict[str, ts.Rating], beta: float) -> float:
    if not ratings:
        return 0.0
    total = 0.0
    for rating in ratings.values():
        total += float(rating.mu) + float(beta) * float(rating.sigma)
    return total / len(ratings)


class MixedParentSelector:
    """Mixture selector: uniform sampling + optimistic tournament."""

    def __init__(
        self,
        *,
        uniform_probability: float,
        tournament_size: int,
        optimistic_beta: float,
        rng: random.Random | None = None,
    ) -> None:
        if not (0.0 <= float(uniform_probability) <= 1.0):
            raise ValueError("uniform_probability must be between 0 and 1.")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be a positive integer.")
        if optimistic_beta < 0:
            raise ValueError("optimistic_beta must be >= 0.")
        self.uniform_probability = float(uniform_probability)
        self.tournament_size = int(tournament_size)
        self.optimistic_beta = float(optimistic_beta)
        self.rng = rng or random.Random()

    def select_parent(self, pool: CrowdedPool) -> Elite:
        if len(pool) <= 0:
            raise ValueError("Cannot select parent from an empty pool.")
        if len(pool) == 1:
            return pool.random_elite()

        if self.rng.random() < self.uniform_probability:
            return pool.random_elite()

        contenders = pool.sample(self.tournament_size)
        if not contenders:
            return pool.random_elite()

        return max(
            contenders,
            key=lambda elite: optimistic_score(elite.ratings, self.optimistic_beta),
        )
