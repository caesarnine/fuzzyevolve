"""Tests for the MAP-Elites archive implementation."""

import random

from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.models import Elite


class DummyRating:
    def __init__(self, mu: float, sigma: float = 1.0):
        self.mu = mu
        self.sigma = sigma


def make_elite(text: str, mu: float, desc: dict[str, object]) -> Elite:
    return Elite(
        text=text,
        descriptor=desc,
        ratings={"metric": DummyRating(mu)},
        age=0,
    )


class TestMapElitesArchive:
    def setup_method(self):
        self.axes = {
            "lang": ["txt", "code"],
            "len": {"bins": [0, 100, 500, 1000, 10000]},
        }
        space = build_descriptor_space(self.axes)
        self.archive = MapElitesArchive(space, elites_per_cell=3, rng=random.Random(0))

    def test_initialization(self):
        assert self.archive.elites_per_cell == 3
        assert self.archive.empty_cells == self.archive.space.total_cells
        assert list(self.archive.iter_elites()) == []

    def test_key_calculation(self):
        desc1 = {"lang": "txt", "len": 50}
        key1 = self.archive.space.cell_key(desc1)
        assert key1 == ("txt", 0)

        desc2 = {"lang": "code", "len": 250}
        key2 = self.archive.space.cell_key(desc2)
        assert key2 == ("code", 1)

        desc3 = {"lang": "txt", "len": 999}
        key3 = self.archive.space.cell_key(desc3)
        assert key3 == ("txt", 2)

    def test_add_elite(self):
        desc = {"lang": "txt", "len": 12}
        elite = make_elite("test content", mu=25.0, desc=desc)

        self.archive.add(elite)

        assert self.archive.empty_cells == self.archive.space.total_cells - 1
        assert elite.cell_key == ("txt", 0)
        assert list(self.archive.iter_elites())[0] == elite

    def test_top_k_limit(self):
        desc = {"lang": "txt", "len": 50}
        for i in range(5):
            elite = make_elite(f"test {i}", mu=float(i), desc=desc)
            self.archive.add(elite)

        key = ("txt", 0)
        bucket = list(self.archive._cells[key])
        assert len(bucket) == 3
        scores = [elite.ratings["metric"].mu for elite in bucket]
        assert scores == sorted(scores, reverse=True)

    def test_random_elite(self):
        desc = {"lang": "txt", "len": 50}
        for i in range(3):
            elite = make_elite(f"test {i}", mu=25.0, desc=desc)
            self.archive.add(elite)

        selected = self.archive.random_elite()
        assert selected is not None
        assert selected.text.startswith("test")

    def test_best_property(self):
        for i in range(3):
            for j in range(2):
                desc = {"lang": "txt" if j == 0 else "code", "len": 50}
                elite = make_elite(f"test {i}-{j}", mu=float(i * 10 + j), desc=desc)
                self.archive.add(elite)

        best = self.archive.best
        assert best.ratings["metric"].mu == 21.0
