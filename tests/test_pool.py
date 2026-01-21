"""Tests for fixed-size population crowding (closest-pair elimination)."""

import random

import numpy as np
import trueskill as ts

from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool


def _elite(text: str, *, mu: float, embedding: np.ndarray) -> Elite:
    return Elite(
        text=text,
        embedding=embedding,
        ratings={"m": ts.Rating(mu=mu, sigma=1.0)},
        age=0,
    )


def test_pool_dedupes_by_text():
    pool = CrowdedPool(
        max_size=10, rng=random.Random(0), score_fn=lambda r: float(r["m"].mu)
    )
    pool.add(_elite("dup", mu=10.0, embedding=np.array([1.0, 0.0])))
    pool.add(_elite("dup", mu=99.0, embedding=np.array([0.0, 1.0])))

    members = list(pool.iter_elites())
    assert len(members) == 1
    assert members[0].ratings["m"].mu == 10.0


def test_pool_eliminates_weaker_of_closest_pair():
    pool = CrowdedPool(
        max_size=2, rng=random.Random(0), score_fn=lambda r: float(r["m"].mu)
    )
    e1 = _elite("e1", mu=10.0, embedding=np.array([1.0, 0.0]))
    e2 = _elite("e2", mu=20.0, embedding=np.array([0.999, 0.0447]))
    e3 = _elite("e3", mu=15.0, embedding=np.array([0.0, 1.0]))

    pool.add_many([e1, e2, e3])

    texts = {e.text for e in pool.iter_elites()}
    assert texts == {"e2", "e3"}
    assert pool.best.text == "e2"
