from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Callable

import trueskill as ts

from fuzzyevolve.core.descriptors import DescriptorSpace
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.scoring import score_ratings


class MapElitesArchive:
    def __init__(
        self,
        space: DescriptorSpace,
        elites_per_cell: int,
        rng: random.Random | None = None,
        score_fn: Callable[[dict[str, ts.Rating]], float] = score_ratings,
    ) -> None:
        if elites_per_cell <= 0:
            raise ValueError("elites_per_cell must be a positive integer.")
        self.space = space
        self.elites_per_cell = elites_per_cell
        self._cells: dict[tuple[Any, ...], list[Elite]] = {}
        self.empty_cells = space.total_cells
        self.rng = rng or random.Random()
        self._score = score_fn

    def add(self, elite: Elite) -> None:
        key = self.space.cell_key(elite.descriptor)
        elite.cell_key = key
        if key not in self._cells:
            self._cells[key] = []
            self.empty_cells -= 1
        bucket = self._cells[key]
        bucket.append(elite)
        self._sort_bucket(bucket)
        del bucket[self.elites_per_cell :]

    def resort(self, elite: Elite) -> None:
        if elite.cell_key is None:
            return
        bucket = self._cells.get(elite.cell_key)
        if bucket:
            self._sort_bucket(bucket)

    def random_elite(self) -> Elite:
        if not self._cells:
            raise ValueError("Cannot sample from an empty archive.")
        key = self.rng.choice(list(self._cells.keys()))
        return self.rng.choice(self._cells[key])

    def sample_elites(self, count: int) -> list[Elite]:
        elites = list(self.iter_elites())
        if not elites:
            return []
        return self.rng.sample(elites, k=min(count, len(elites)))

    def sample_one_per_cell(self) -> list[Elite]:
        return [self.rng.choice(bucket) for bucket in self._cells.values() if bucket]

    def iter_elites(self) -> Iterable[Elite]:
        for bucket in self._cells.values():
            yield from bucket

    @property
    def best(self) -> Elite:
        if not self._cells:
            raise ValueError("Archive is empty.")
        return max(self.iter_elites(), key=lambda elite: self._score(elite.ratings))

    def _sort_bucket(self, bucket: list[Elite]) -> None:
        bucket.sort(key=lambda elite: self._score(elite.ratings), reverse=True)
