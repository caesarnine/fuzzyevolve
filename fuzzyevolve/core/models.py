from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import trueskill as ts

Ratings = dict[str, ts.Rating]


class RatedText(Protocol):
    text: str
    ratings: Ratings
    age: int


@dataclass(slots=True)
class Elite:
    text: str
    embedding: np.ndarray
    ratings: Ratings
    age: int

    def clone(self) -> "Elite":
        ratings = {
            name: ts.Rating(rating.mu, rating.sigma)
            for name, rating in self.ratings.items()
        }
        return Elite(
            text=self.text,
            embedding=self.embedding.copy(),
            ratings=ratings,
            age=self.age,
        )


@dataclass(slots=True)
class Anchor:
    text: str
    ratings: Ratings
    age: int
    label: str = ""


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    text: str
    operator: str = ""
    uncertainty_scale: float = 1.0
    focus: str | None = None
    partner_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IterationSnapshot:
    iteration: int
    best_score: float
    pool_size: int
    best_elite: Elite


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    best_elite: Elite
    best_score: float
