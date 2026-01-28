"""Tests for mutation planning and uncertainty handling."""

from __future__ import annotations

import random

import numpy as np
import pytest
import trueskill as ts

from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.mutation import OperatorMutator, OperatorSpec
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import RatingSystem


class DummyOperator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str | None] = []
        self.partners_seen: list[tuple[str, ...]] = []

    def propose(
        self,
        *,
        parent: Elite,
        partners: list[Elite] | tuple[Elite, ...] | None = None,
        critique: Critique | None,
        focus: str | None = None,
    ):
        self.calls.append(focus)
        self.partners_seen.append(tuple(p.text for p in (partners or [])))
        suffix = focus or "none"
        return [f"{self.name}:{len(self.calls)}:{suffix}"]


def _make_parent() -> Elite:
    return Elite(
        text="parent",
        embedding=np.array([1.0], dtype=float),
        ratings={"m1": ts.Rating(mu=25.0, sigma=1.0)},
        age=0,
    )


def test_operator_mutator_assigns_distinct_focuses_and_uncertainty_scales():
    parent = _make_parent()
    critique = Critique(
        issues=("issue1", "issue2", "issue3"),
        routes=("route1", "route2", "route3"),
    )

    exploit = DummyOperator("exploit")
    explore = DummyOperator("explore")

    pool = CrowdedPool(max_size=10, rng=random.Random(0), score_fn=lambda _r: 0.0)
    mutator = OperatorMutator(
        pool=pool,
        operators={"exploit": exploit, "explore": explore},
        specs=[
            OperatorSpec(
                name="exploit",
                role="exploit",
                min_jobs=2,
                weight=1.0,
                uncertainty_scale=0.5,
            ),
            OperatorSpec(
                name="explore",
                role="explore",
                min_jobs=2,
                weight=1.0,
                uncertainty_scale=2.0,
            ),
        ],
        jobs_per_iteration=4,
        rng=random.Random(0),
    )

    candidates = mutator.propose(
        parent=parent,
        critique=critique,
        max_candidates=10,
        mutation_executor=None,
    )

    assert set(exploit.calls) == {"issue1", "issue2"}
    assert set(explore.calls) == {"route1", "route2"}
    assert len(candidates) == 4

    expected_scales = {"exploit": 0.5, "explore": 2.0}
    for cand in candidates:
        assert cand.uncertainty_scale == expected_scales[cand.operator]


def test_init_child_ratings_scales_uncertainty():
    rating = RatingSystem(["m1"], child_prior_tau=2.0)
    parent = Elite(
        text="p",
        embedding=np.array([1.0], dtype=float),
        ratings={"m1": ts.Rating(mu=25.0, sigma=1.0)},
        age=0,
    )
    child_ratings = rating.init_child_ratings(parent, uncertainty_scale=3.0)
    assert child_ratings["m1"].sigma == pytest.approx(
        (1.0**2 + (2.0 * 3.0) ** 2) ** 0.5
    )


def test_crossover_operator_receives_partners():
    rating = RatingSystem(["m1"], child_prior_tau=2.0)
    pool = CrowdedPool(max_size=10, rng=random.Random(0), score_fn=rating.score)

    parent = Elite(
        text="parent",
        embedding=np.array([1.0, 0.0], dtype=float),
        ratings=rating.new_ratings(),
        age=0,
    )
    close = Elite(
        text="close",
        embedding=np.array([0.9, 0.435889894], dtype=float),
        ratings=rating.new_ratings(),
        age=0,
    )
    far = Elite(
        text="far",
        embedding=np.array([0.0, 1.0], dtype=float),
        ratings=rating.new_ratings(),
        age=0,
    )
    farthest = Elite(
        text="farthest",
        embedding=np.array([-1.0, 0.0], dtype=float),
        ratings=rating.new_ratings(),
        age=0,
    )
    pool.add_many([parent, close, far, farthest])

    crossover = DummyOperator("crossover")
    mutator = OperatorMutator(
        pool=pool,
        operators={"crossover": crossover},
        specs=[
            OperatorSpec(
                name="crossover",
                role="crossover",
                min_jobs=1,
                weight=1.0,
                uncertainty_scale=1.0,
                committee_size=3,
                partner_selection="farthest",
                partner_farthest_k=3,
            )
        ],
        jobs_per_iteration=1,
        rng=random.Random(0),
    )

    mutator.propose(
        parent=parent,
        critique=None,
        max_candidates=1,
        mutation_executor=None,
    )

    assert crossover.partners_seen == [("farthest", "far")]
