"""Tests for parent selection policies (mixture + optimistic tournament)."""

import random

import numpy as np
import trueskill as ts

from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.selection import MixedParentSelector


def _elite(text: str, mu: float) -> Elite:
    return Elite(
        text=text,
        embedding=np.array([1.0], dtype=float),
        ratings={"m": ts.Rating(mu=mu, sigma=1.0)},
        age=0,
    )


def test_optimistic_tournament_prefers_high_score():
    pool = CrowdedPool(
        max_size=10, rng=random.Random(0), score_fn=lambda r: float(r["m"].mu)
    )
    pool.add(_elite("low", 10.0))
    pool.add(_elite("high", 50.0))

    selector = MixedParentSelector(
        uniform_probability=0.0,
        tournament_size=2,
        optimistic_beta=0.0,
        rng=random.Random(0),
    )
    parent = selector.select_parent(pool)
    assert parent.text == "high"
