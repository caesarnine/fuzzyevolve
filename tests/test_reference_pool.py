"""Tests for reference pool anchor handling."""

import trueskill as ts

from fuzzyevolve.core.descriptors import default_text_descriptor
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.reference_pool import ReferencePool


def _make_elite(text: str) -> Elite:
    return Elite(
        text=text,
        descriptor=default_text_descriptor(text),
        ratings={"m1": ts.Rating()},
        age=0,
    )


def test_reference_pool_dedupes_anchors_by_text():
    pool = ReferencePool(metrics=["m1"])

    seed1 = pool.add_seed_anchor(
        "seed",
        descriptor_fn=default_text_descriptor,
        mu=25.0,
        sigma=0.001,
    )
    seed2 = pool.add_seed_anchor(
        "seed",
        descriptor_fn=default_text_descriptor,
        mu=30.0,
        sigma=0.002,
    )

    assert seed1 is seed2
    assert len(pool._anchors) == 1

    ghost_same = pool.add_ghost_anchor(_make_elite("seed"))
    assert ghost_same is seed1
    assert len(pool._anchors) == 1

    ghost1 = pool.add_ghost_anchor(_make_elite("ghost"))
    assert ghost1 is not seed1
    assert len(pool._anchors) == 2

    ghost2 = pool.add_ghost_anchor(_make_elite("ghost"))
    assert ghost2 is ghost1
    assert len(pool._anchors) == 2
