"""Tests for on-disk run storage + resume."""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from fuzzyevolve.config import Config
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.core.models import MutationCandidate
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import BattleRanking, RatingSystem
from fuzzyevolve.run_store import RunStore


def embed(_text: str) -> np.ndarray:
    return np.array([1.0], dtype=float)


def rank_parent_best(metrics: list[str], n: int) -> BattleRanking:
    tiers = [[0], *[[i] for i in range(1, n)]]
    return BattleRanking(tiers_by_metric={m: tiers for m in metrics})


def test_run_store_checkpoint_and_resume(tmp_path: Path):
    cfg = Config()
    cfg.run.iterations = 2
    cfg.run.checkpoint_interval = 1
    cfg.population.size = 10
    cfg.metrics.names = ["m1"]
    cfg.judging.opponent.kind = "none"

    rating = RatingSystem(
        cfg.metrics.names,
        score_lcb_c=cfg.rating.score_lcb_c,
        child_prior_tau=cfg.rating.child_prior_tau,
    )

    store = RunStore.create(
        data_dir=tmp_path, cfg=cfg, seed_text="seed", config_path=None
    )

    mutator = Mock()
    mutator.propose = Mock(
        side_effect=[
            [MutationCandidate(text="child1")],
            [MutationCandidate(text="child2")],
        ]
    )
    ranker = Mock()
    ranker.rank = Mock(
        side_effect=lambda **kw: rank_parent_best(
            list(kw["metrics"]), len(kw["battle"].participants)
        )
    )

    pool = CrowdedPool(
        max_size=cfg.population.size, rng=random.Random(0), score_fn=rating.score
    )
    engine = EvolutionEngine(
        cfg=cfg,
        pool=pool,
        embed=embed,
        rating=rating,
        selector=lambda p: p.random_elite(),
        critic=None,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=None,
        rng=random.Random(0),
        store=store,
    )

    engine.run("seed")

    assert store.latest_checkpoint_path().is_file()
    assert (store.checkpoints_dir / "it000001.json").is_file()
    assert (store.checkpoints_dir / "it000002.json").is_file()
    assert len(list(store.texts_dir.glob("*.txt"))) >= 3

    cfg2 = Config.model_validate(cfg.model_dump())
    cfg2.run.iterations = 1
    rating2 = RatingSystem(
        cfg2.metrics.names,
        score_lcb_c=cfg2.rating.score_lcb_c,
        child_prior_tau=cfg2.rating.child_prior_tau,
    )
    store2 = RunStore.open(store.run_dir)
    pool2 = CrowdedPool(
        max_size=cfg2.population.size, rng=random.Random(1), score_fn=rating2.score
    )

    loaded = store2.load_checkpoint(
        cfg=cfg2,
        checkpoint_path=None,
        embed=embed,
        pool_factory=lambda: pool2,
        anchor_factory=lambda _cfg: None,
    )
    assert loaded.next_iteration == 2
    assert len(loaded.pool) >= 2

    mutator2 = Mock()
    mutator2.propose = Mock(return_value=[MutationCandidate(text="child3")])
    ranker2 = Mock()
    ranker2.rank = Mock(
        side_effect=lambda **kw: rank_parent_best(
            list(kw["metrics"]), len(kw["battle"].participants)
        )
    )

    engine2 = EvolutionEngine(
        cfg=cfg2,
        pool=loaded.pool,
        embed=embed,
        rating=rating2,
        selector=lambda p: p.random_elite(),
        critic=None,
        mutator=mutator2,
        ranker=ranker2,
        anchor_manager=loaded.anchors,
        rng=random.Random(0),
        store=store2,
    )
    engine2.resume(start_iteration=loaded.next_iteration)

    assert (store2.checkpoints_dir / "it000003.json").is_file()
