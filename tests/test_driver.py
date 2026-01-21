"""Tests for the evolution engine orchestration."""

from __future__ import annotations

import random
from collections.abc import Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from fuzzyevolve.config import Config
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import BattleRanking, RatingSystem


def embed(text: str) -> np.ndarray:
    if text == "seed":
        return np.array([1.0, 0.0], dtype=float)
    if text == "other":
        return np.array([0.0, 1.0], dtype=float)
    return np.array([1.0, 0.0], dtype=float)


def make_engine(
    cfg: Config,
    *,
    mutator: Mock,
    ranker: Mock,
    selector,
    rng: random.Random | None = None,
) -> EvolutionEngine:
    rng = rng or random.Random(0)
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
    return EvolutionEngine(
        cfg=cfg,
        pool=pool,
        embed=embed,
        rating=rating,
        selector=selector,
        critic=None,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=None,
        rng=rng,
    )


def rank_parent_best(metrics: Sequence[str], n: int) -> BattleRanking:
    tiers = [[0], *[[i] for i in range(1, n)]]
    return BattleRanking(tiers_by_metric={m: tiers for m in metrics})


class TestEvolutionEngine:
    def test_batch_judging_multiple_children(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.size = 10
        cfg.metrics.names = ["m1"]
        cfg.judging.opponent.kind = "none"

        mutator = Mock()
        mutator.propose = Mock(
            return_value=[
                MutationCandidate(text="child1"),
                MutationCandidate(text="child2"),
            ]
        )

        ranker = Mock()

        def _rank(*, metrics, battle, metric_descriptions=None):
            return rank_parent_best(metrics, len(battle.participants))

        ranker.rank = Mock(side_effect=_rank)

        def selector(pool: CrowdedPool) -> Elite:
            return pool.random_elite()

        engine = make_engine(cfg, mutator=mutator, ranker=ranker, selector=selector)
        engine.run("seed")

        assert ranker.rank.call_count == 1
        battle = ranker.rank.call_args.kwargs["battle"]
        assert {p.text for p in battle.participants} == {"seed", "child1", "child2"}

    def test_empty_candidates_handling(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.size = 10
        cfg.metrics.names = ["m1"]
        cfg.judging.opponent.kind = "none"

        mutator = Mock()
        mutator.propose = Mock(return_value=[])
        ranker = Mock()
        ranker.rank = Mock()

        engine = make_engine(
            cfg, mutator=mutator, ranker=ranker, selector=lambda p: p.random_elite()
        )
        engine.run("seed")

        assert ranker.rank.call_count == 0
        assert len(engine.pool) == 1

    def test_no_children_added_on_judge_failure(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.size = 10
        cfg.metrics.names = ["m1"]
        cfg.judging.opponent.kind = "none"

        mutator = Mock()
        mutator.propose = Mock(return_value=[MutationCandidate(text="child")])
        ranker = Mock()
        ranker.rank = Mock(return_value=None)

        engine = make_engine(
            cfg, mutator=mutator, ranker=ranker, selector=lambda p: p.random_elite()
        )
        with pytest.raises(RuntimeError):
            engine.run("seed")
        assert len(engine.pool) == 1

    def test_step_skips_children_already_in_pool(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.size = 10
        cfg.metrics.names = ["m1"]
        cfg.judging.opponent.kind = "none"

        mutator = Mock()
        mutator.propose = Mock(
            return_value=[MutationCandidate(text="dup"), MutationCandidate(text="dup")]
        )

        ranker = Mock()
        ranker.rank = Mock(
            side_effect=lambda **kw: rank_parent_best(
                kw["metrics"], len(kw["battle"].participants)
            )
        )

        def selector(pool: CrowdedPool) -> Elite:
            for elite in pool.iter_elites():
                if elite.text == "seed":
                    return elite
            return pool.random_elite()

        engine = make_engine(cfg, mutator=mutator, ranker=ranker, selector=selector)
        engine.pool.add(
            Elite(
                text="dup",
                embedding=embed("dup"),
                ratings=engine.rating.new_ratings(),
                age=0,
            )
        )
        engine.run("seed")

        assert ranker.rank.call_count == 0

    def test_opponent_included_when_available(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.size = 10
        cfg.metrics.names = ["m1"]
        cfg.judging.opponent.kind = "farthest_from_parent"
        cfg.judging.opponent.probability = 1.0

        mutator = Mock()
        mutator.propose = Mock(return_value=[MutationCandidate(text="child")])

        ranker = Mock()
        ranker.rank = Mock(
            side_effect=lambda **kw: rank_parent_best(
                kw["metrics"], len(kw["battle"].participants)
            )
        )

        def selector(pool: CrowdedPool) -> Elite:
            for elite in pool.iter_elites():
                if elite.text == "seed":
                    return elite
            return pool.random_elite()

        engine = make_engine(cfg, mutator=mutator, ranker=ranker, selector=selector)
        engine.pool.add(
            Elite(
                text="other",
                embedding=embed("other"),
                ratings=engine.rating.new_ratings(),
                age=0,
            )
        )
        engine.run("seed")

        battle = ranker.rank.call_args.kwargs["battle"]
        assert {p.text for p in battle.participants} == {"seed", "child", "other"}
