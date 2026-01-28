from __future__ import annotations

import random

import numpy as np
import trueskill as ts

from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.multiobjective import Scalarizer
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.selection import MixedParentSelector


def _elite(text: str, *, m1: float, m2: float) -> Elite:
    return Elite(
        text=text,
        embedding=np.array([1.0, 0.0], dtype=float),
        ratings={
            "m1": ts.Rating(mu=m1, sigma=1.0),
            "m2": ts.Rating(mu=m2, sigma=1.0),
        },
        age=0,
    )


def test_scalarized_pruning_prefers_metric_weighted_specialist():
    a = _elite("a", m1=10.0, m2=0.0)
    b = _elite("b", m1=0.0, m2=9.0)

    scalarizer = Scalarizer(["m1", "m2"], rng=random.Random(0))
    scalarizer.set_weights({"m1": 0.9, "m2": 0.1})
    pool = CrowdedPool(
        max_size=1,
        rng=random.Random(0),
        score_fn=lambda r: float(r["m1"].mu + r["m2"].mu),
        pruning_strategy="closest_pair",
        metrics=["m1", "m2"],
        score_lcb_c=0.0,
        scalarizer=scalarizer,
        pareto=True,
    )
    pool.add_many([a, b])
    assert {e.text for e in pool.iter_elites()} == {"a"}

    scalarizer2 = Scalarizer(["m1", "m2"], rng=random.Random(0))
    scalarizer2.set_weights({"m1": 0.1, "m2": 0.9})
    pool2 = CrowdedPool(
        max_size=1,
        rng=random.Random(0),
        score_fn=lambda r: float(r["m1"].mu + r["m2"].mu),
        pruning_strategy="closest_pair",
        metrics=["m1", "m2"],
        score_lcb_c=0.0,
        scalarizer=scalarizer2,
        pareto=True,
    )
    pool2.add_many([a, b])
    assert {e.text for e in pool2.iter_elites()} == {"b"}


def test_pareto_weighted_selection_sweeps_tradeoffs():
    scalarizer = Scalarizer(["m1", "m2"], rng=random.Random(0))
    pool = CrowdedPool(
        max_size=10,
        rng=random.Random(0),
        score_fn=lambda r: float(r["m1"].mu + r["m2"].mu),
    )
    pool.add_many(
        [
            _elite("a", m1=10.0, m2=0.0),
            _elite("b", m1=0.0, m2=10.0),
            _elite("c", m1=6.0, m2=6.0),
        ]
    )

    selector = MixedParentSelector(
        uniform_probability=0.0,
        tournament_size=10,
        optimistic_beta=0.0,
        rng=random.Random(0),
        metrics=["m1", "m2"],
        scalarizer=scalarizer,
        pareto=True,
    )

    scalarizer.set_weights({"m1": 0.9, "m2": 0.1})
    assert selector.select_parent(pool).text == "a"

    scalarizer.set_weights({"m1": 0.1, "m2": 0.9})
    assert selector.select_parent(pool).text == "b"

    scalarizer.set_weights({"m1": 0.5, "m2": 0.5})
    assert selector.select_parent(pool).text == "c"
