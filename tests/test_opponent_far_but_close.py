"""Tests for far-but-close opponent selection."""

from __future__ import annotations

import random
from unittest.mock import Mock

import numpy as np
import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import BattleRanking, RatingSystem


def embed(text: str) -> np.ndarray:
    mapping: dict[str, np.ndarray] = {
        "seed": np.array([1.0, 0.0], dtype=float),
        "child": np.array([1.0, 0.0], dtype=float),
        "close_far": np.array([0.0, 1.0], dtype=float),
        "far_bad": np.array([-1.0, 0.0], dtype=float),
    }
    return mapping.get(text, np.array([1.0, 0.0], dtype=float))


def rank_parent_best(metrics: list[str], n: int) -> BattleRanking:
    tiers = [[0], *[[i] for i in range(1, n)]]
    return BattleRanking(tiers_by_metric={m: tiers for m in metrics})


def test_far_but_close_opponent_prefers_skill_close_among_far():
    cfg = Config()
    cfg.run.iterations = 1
    cfg.population.size = 10
    cfg.metrics.names = ["m1"]
    cfg.judging.opponent.kind = "far_but_close"
    cfg.judging.opponent.probability = 1.0
    cfg.judging.opponent.farthest_k = 2

    rating = RatingSystem(
        cfg.metrics.names,
        score_lcb_c=cfg.rating.score_lcb_c,
        child_prior_tau=cfg.rating.child_prior_tau,
    )

    pool = CrowdedPool(
        max_size=cfg.population.size,
        rng=random.Random(0),
        score_fn=rating.score,
    )

    seed = Elite(
        text="seed",
        embedding=embed("seed"),
        ratings=rating.new_ratings(),
        age=0,
    )
    pool.add(seed)

    pool.add(
        Elite(
            text="close_far",
            embedding=embed("close_far"),
            ratings={"m1": ts.Rating(mu=25.0, sigma=8.333)},
            age=0,
        )
    )
    pool.add(
        Elite(
            text="far_bad",
            embedding=embed("far_bad"),
            ratings={"m1": ts.Rating(mu=100.0, sigma=8.333)},
            age=0,
        )
    )

    mutator = Mock()
    mutator.propose = Mock(return_value=[MutationCandidate(text="child")])

    ranker = Mock()
    ranker.rank = Mock(
        side_effect=lambda **kw: rank_parent_best(
            list(kw["metrics"]), len(kw["battle"].participants)
        )
    )

    def selector(p: CrowdedPool) -> Elite:
        for elite in p.iter_elites():
            if elite.text == "seed":
                return elite
        raise AssertionError("seed not found")

    engine = EvolutionEngine(
        cfg=cfg,
        pool=pool,
        embed=embed,
        rating=rating,
        selector=selector,
        critic=None,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=None,
        rng=random.Random(0),
        store=None,
    )

    engine.run("seed")

    battle = ranker.rank.call_args.kwargs["battle"]
    assert {p.text for p in battle.participants} == {"seed", "child", "close_far"}
