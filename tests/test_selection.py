"""Tests for parent selection policies."""

import random

import trueskill as ts

from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.selection import ParentSelector


def test_optimistic_selector_prefers_high_score_cell():
    space = build_descriptor_space({"lang": ["txt"], "len": {"bins": [0, 10, 100]}})
    archive = MapElitesArchive(
        space,
        elites_per_cell=1,
        rng=random.Random(0),
        score_fn=lambda ratings: ratings["m"].mu,
    )
    low = Elite(
        text="low",
        descriptor={"lang": "txt", "len": 5},
        ratings={"m": ts.Rating(mu=10.0, sigma=1.0)},
        age=0,
    )
    high = Elite(
        text="high",
        descriptor={"lang": "txt", "len": 50},
        ratings={"m": ts.Rating(mu=50.0, sigma=1.0)},
        age=0,
    )
    archive.add(low)
    archive.add(high)

    selector = ParentSelector(
        mode="optimistic_cell_softmax",
        beta=0.0,
        temp=0.05,
        rng=random.Random(0),
    )
    parent = selector.select_parent(archive)
    assert parent.text == "high"
