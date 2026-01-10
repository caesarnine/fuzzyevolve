"""
This module contains the MixedArchive class for storing elites in the MAP-Elites algorithm.
"""

import random
from typing import Any, Dict, List, Tuple

from fuzzyevolve.evolution.scoring import ts_score


class MixedArchive:
    """A MAP-Elites archive that stores the top-k elites per cell."""

    def __init__(self, axes: Dict[str, Any], k_top: int):
        self.axes, self.k_top = axes, k_top
        self.cell: Dict[Tuple, List[Dict]] = {}
        total = 1
        for spec in axes.values():
            total *= len(spec) if isinstance(spec, list) else (len(spec["bins"]) - 1)
        self.total_cells = total
        self.empty_cells = total

    def _key(self, desc: Dict[str, Any]) -> Tuple:
        """Calculates the cell key for a given descriptor."""
        key = []
        for name, spec in self.axes.items():
            v = desc[name]
            if isinstance(spec, list):
                key.append(v)
            else:
                edges = [float(x) for x in spec["bins"]]
                idx = max(i for i, e in enumerate(edges) if float(v) >= e)
                key.append(idx)
        return tuple(key)

    def _sort_bucket(self, bucket: List[Dict]):
        """Sorts a bucket of elites by their TrueSkill score."""
        bucket.sort(key=lambda e: ts_score(e["rating"]), reverse=True)

    def add(self, desc: Dict[str, Any], elite: Dict):
        """Adds an elite to the archive."""
        key = self._key(desc)
        elite["cell_key"] = key
        if key not in self.cell:
            self.cell[key] = []
            self.empty_cells -= 1
        bucket = self.cell[key]
        bucket.append(elite)
        self._sort_bucket(bucket)
        del bucket[self.k_top :]

    def resort_elite(self, elite: Dict):
        """Resorts the bucket containing the given elite."""
        key = elite["cell_key"]
        if key in self.cell:
            self._sort_bucket(self.cell[key])

    def random_elite(self) -> Dict:
        """Selects a random elite from the archive."""
        key = random.choice(list(self.cell.keys()))
        bucket = self.cell[key]
        return random.choice(bucket)

    @property
    def best(self) -> Dict:
        """Returns the best elite in the archive."""
        return max(
            (e for b in self.cell.values() for e in b),
            key=lambda e: ts_score(e["rating"]),
        )
