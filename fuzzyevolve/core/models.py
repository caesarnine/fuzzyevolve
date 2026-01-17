from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import trueskill as ts

Descriptor = dict[str, Any]
Ratings = dict[str, ts.Rating]


class RatedText(Protocol):
    text: str
    descriptor: Descriptor
    ratings: Ratings
    age: int


@dataclass(slots=True)
class Elite:
    text: str
    descriptor: Descriptor
    ratings: Ratings
    age: int

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
        )


@dataclass(slots=True)
class Anchor:
    text: str
    descriptor: Descriptor
    ratings: Ratings
    age: int
    label: str = ""


@dataclass(frozen=True, slots=True)
class TextEdit:
    search: str
    replace: str


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    text: str
    edits: tuple[TextEdit, ...] = ()

    @property
    def search_block(self) -> str:
        return "\n\n".join(edit.search for edit in self.edits)

    @property
    def replace_block(self) -> str:
        return "\n\n".join(edit.replace for edit in self.edits)


@dataclass(frozen=True, slots=True)
class MutationEvent:
    iteration: int
    island: int
    parent_score: float
    child_score: float
    search: str
    replace: str


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

