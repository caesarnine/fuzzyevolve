from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Callable

import trueskill as ts

from fuzzyevolve.core.descriptors import DescriptorSpace
from fuzzyevolve.core.models import Elite

class MapElitesArchive:
    def __init__(
        self,
        space: DescriptorSpace,
        elites_per_cell: int,
        rng: random.Random | None = None,
        score_fn: Callable[[dict[str, ts.Rating]], float] | None = None,
    ) -> None:
        if elites_per_cell <= 0:
            raise ValueError("elites_per_cell must be a positive integer.")
        if score_fn is None:
            raise ValueError("score_fn is required.")
        self.space = space
        self.elites_per_cell = elites_per_cell
        self._cells: dict[tuple[Any, ...], list[Elite]] = {}
        self._text_index: dict[str, Elite] = {}
        self.empty_cells = space.total_cells
        self.rng = rng or random.Random()
        self._score = score_fn

    def contains_text(self, text: str) -> bool:
        return text in self._text_index

    def add(self, elite: Elite) -> None:
        if elite.text in self._text_index:
            return

        key = self.space.cell_key(elite.descriptor)
        if key not in self._cells:
            self._cells[key] = []
            self.empty_cells -= 1

        bucket = self._cells[key]
        bucket.append(elite)
        self._text_index[elite.text] = elite

        self._sort_bucket(bucket)
        if len(bucket) > self.elites_per_cell:
            removed = bucket[self.elites_per_cell :]
            del bucket[self.elites_per_cell :]
            for rem in removed:
                self._text_index.pop(rem.text, None)

    def resort(self, elite: Elite) -> None:
        key = self.space.cell_key(elite.descriptor)
        bucket = self._cells.get(key)
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

    def iter_cells(self) -> Iterable[tuple[tuple[Any, ...], list[Elite]]]:
        return self._cells.items()

    def is_new_cell(self, descriptor: dict[str, Any]) -> bool:
        key = self.space.cell_key(descriptor)
        return key not in self._cells

    @property
    def best(self) -> Elite:
        if not self._cells:
            raise ValueError("Archive is empty.")
        return max(self.iter_elites(), key=lambda elite: self._score(elite.ratings))

    def _sort_bucket(self, bucket: list[Elite]) -> None:
        bucket.sort(key=lambda elite: self._score(elite.ratings), reverse=True)
