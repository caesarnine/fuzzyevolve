from __future__ import annotations

import random

import trueskill as ts

from fuzzyevolve.core.multiobjective import Scalarizer, nondominated_indices
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool


def optimistic_score(ratings: dict[str, ts.Rating], beta: float) -> float:
    if not ratings:
        return 0.0
    total = 0.0
    for rating in ratings.values():
        total += float(rating.mu) + float(beta) * float(rating.sigma)
    return total / len(ratings)


def ucb_vector(
    ratings: dict[str, ts.Rating],
    *,
    metrics: list[str],
    optimistic_beta: float,
) -> list[float]:
    vec: list[float] = []
    for metric in metrics:
        rating = ratings.get(metric)
        if rating is None:
            vec.append(float("-inf"))
            continue
        vec.append(float(rating.mu) + float(optimistic_beta) * float(rating.sigma))
    return vec


class MixedParentSelector:
    """Mixture selector: uniform sampling + optimistic tournament."""

    def __init__(
        self,
        *,
        uniform_probability: float,
        tournament_size: int,
        optimistic_beta: float,
        rng: random.Random | None = None,
        metrics: list[str] | None = None,
        scalarizer: Scalarizer | None = None,
        pareto: bool = True,
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
        self.scalarizer = scalarizer
        self.pareto = bool(pareto)
        if metrics is None and scalarizer is not None:
            metrics = list(scalarizer.metrics)
        self.metrics = list(metrics) if metrics else []

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

        if not self.metrics:
            return max(
                contenders,
                key=lambda elite: optimistic_score(elite.ratings, self.optimistic_beta),
            )

        if not self.pareto and self.scalarizer is None:
            return max(
                contenders,
                key=lambda elite: optimistic_score(elite.ratings, self.optimistic_beta),
            )

        vectors = [
            ucb_vector(
                elite.ratings,
                metrics=self.metrics,
                optimistic_beta=self.optimistic_beta,
            )
            for elite in contenders
        ]

        chosen = (
            list(nondominated_indices(vectors))
            if self.pareto
            else list(range(len(contenders)))
        )
        if not chosen:
            chosen = list(range(len(contenders)))

        weights = None
        if self.scalarizer is not None:
            weights = self.scalarizer.weights_for(self.metrics)
        scalarized: list[float] = []
        for idx in chosen:
            if weights is None:
                vec = vectors[idx]
                scalarized.append(sum(vec) / len(vec) if vec else float("-inf"))
            else:
                scalarized.append(sum(w * v for w, v in zip(weights, vectors[idx])))

        best = max(scalarized)
        eps = 1e-12
        best_positions = [
            pos for pos, score in enumerate(scalarized) if score >= best - eps
        ]
        pick = chosen[self.rng.choice(best_positions)]
        return contenders[pick]
