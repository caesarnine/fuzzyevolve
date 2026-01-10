from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import trueskill as ts

Descriptor = dict[str, Any]


@dataclass(slots=True)
class Elite:
    text: str
    descriptor: Descriptor
    ratings: dict[str, ts.Rating]
    age: int
    cell_key: tuple[Any, ...] | None = None

    def clone(self) -> "Elite":
        ratings = {
            name: ts.Rating(rating.mu, rating.sigma)
            for name, rating in self.ratings.items()
        }
        return Elite(
            text=self.text,
            descriptor=dict(self.descriptor),
            ratings=ratings,
            age=self.age,
            cell_key=self.cell_key,
        )


@dataclass(frozen=True, slots=True)
class MutationEvent:
    iteration: int
    island: int
    parent_score: float
    child_score: float
    diff: str


@dataclass(frozen=True, slots=True)
class IterationSnapshot:
    iteration: int
    best_score: float
    empty_cells: int
    best_elite: Elite


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    best_elite: Elite
    best_score: float
