"""Tests for anchor pool behavior."""

import numpy as np
import trueskill as ts

from fuzzyevolve.core.anchors import AnchorPool
from fuzzyevolve.core.models import Elite


def _make_elite(text: str) -> Elite:
    return Elite(
        text=text,
        embedding=np.array([1.0], dtype=float),
        ratings={"m1": ts.Rating()},
        age=0,
    )


def test_anchor_pool_dedupes_by_text():
    pool = AnchorPool(metrics=["m1"])

    seed1 = pool.add_seed("seed", mu=25.0, sigma=0.001)
    seed2 = pool.add_seed("seed", mu=30.0, sigma=0.002)

    assert seed1 is seed2
    assert len(pool._anchors) == 1

    ghost_same = pool.add_ghost(_make_elite("seed"))
    assert ghost_same is seed1
    assert len(pool._anchors) == 1

    ghost1 = pool.add_ghost(_make_elite("ghost"))
    assert ghost1 is not seed1
    assert len(pool._anchors) == 2

    ghost2 = pool.add_ghost(_make_elite("ghost"))
    assert ghost2 is ghost1
    assert len(pool._anchors) == 2
