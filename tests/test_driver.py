"""Tests for the evolution engine orchestration."""

from __future__ import annotations

import random
from collections.abc import Sequence
from unittest.mock import Mock

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.ratings import BattleRanking, RatingSystem


def length_descriptor(text: str) -> dict[str, object]:
    return {"len": len(text)}


def make_engine(
    cfg: Config,
    *,
    mutator: Mock,
    ranker: Mock,
    rng: random.Random | None = None,
) -> EvolutionEngine:
    rng = rng or random.Random(0)
    space = build_descriptor_space({"len": {"bins": [0, 10, 100]}})
    rating = RatingSystem(
        cfg.metrics.names,
        score_lcb_c=cfg.rating.score_lcb_c,
        child_prior_tau=cfg.rating.child_prior_tau,
    )
    archive = MapElitesArchive(
        space,
        elites_per_cell=cfg.population.elites_per_cell,
        rng=random.Random(0),
        score_fn=rating.score,
    )

    def selector(arc: MapElitesArchive) -> Elite:
        return arc.random_elite()

    return EvolutionEngine(
        cfg=cfg,
        islands=[archive],
        describe=length_descriptor,
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
        cfg.population.elites_per_cell = 10
        cfg.metrics.names = ["m1"]
        cfg.mutation.max_children = 10
        cfg.judging.max_battle_size = 10

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

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.run("seed")

        assert ranker.rank.call_count == 1
        battle = ranker.rank.call_args.kwargs["battle"]
        assert {p.text for p in battle.participants} == {"seed", "child1", "child2"}

    def test_empty_candidates_handling(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.metrics.names = ["m1"]

        mutator = Mock()
        mutator.propose = Mock(return_value=[])
        ranker = Mock()
        ranker.rank = Mock()

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.run("seed")

        assert ranker.rank.call_count == 0
        assert sum(1 for _ in engine.islands[0].iter_elites()) == 1

    def test_no_children_added_on_judge_failure(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.metrics.names = ["m1"]
        cfg.new_cell_gate.kind = "none"

        mutator = Mock()
        mutator.propose = Mock(return_value=[MutationCandidate(text="child")])
        ranker = Mock()
        ranker.rank = Mock(return_value=None)

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.run("seed")

        assert sum(1 for _ in engine.islands[0].iter_elites()) == 1

    def test_max_battle_size_enforced(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.elites_per_cell = 10
        cfg.metrics.names = ["m1"]
        cfg.mutation.max_children = 10
        cfg.judging.max_battle_size = 3

        mutator = Mock()
        mutator.propose = Mock(
            return_value=[
                MutationCandidate(text="c1"),
                MutationCandidate(text="c2"),
                MutationCandidate(text="c3"),
                MutationCandidate(text="c4"),
            ]
        )

        ranker = Mock()

        def _rank(**kwargs):
            battle = kwargs["battle"]
            assert len(battle.participants) <= cfg.judging.max_battle_size
            return rank_parent_best(kwargs["metrics"], len(battle.participants))

        ranker.rank = Mock(side_effect=_rank)

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.run("seed")

    def test_new_cell_gate_blocks_low_child(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.elites_per_cell = 10
        cfg.metrics.names = ["m1"]
        cfg.descriptor.kind = "length"
        cfg.descriptor.length_bins = [0, 5, 100]
        cfg.new_cell_gate.kind = "parent_lcb"
        cfg.new_cell_gate.delta = 0.0
        cfg.judging.max_battle_size = 2

        mutator = Mock()
        mutator.propose = Mock(return_value=[MutationCandidate(text="this is long")])

        ranker = Mock()

        def _rank(**kwargs):
            # Put parent above child -> child gets worse after rating update.
            tiers = [[0], [1]]
            return BattleRanking(tiers_by_metric={m: tiers for m in kwargs["metrics"]})

        ranker.rank = Mock(side_effect=_rank)

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.run("seed")

        assert sum(1 for _ in engine.islands[0].iter_elites()) == 1

    def test_step_skips_children_already_in_archive(self):
        cfg = Config()
        cfg.run.iterations = 1
        cfg.population.elites_per_cell = 10
        cfg.metrics.names = ["m1"]
        cfg.mutation.max_children = 10
        cfg.judging.max_battle_size = 10

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

        engine = make_engine(cfg, mutator=mutator, ranker=ranker)
        engine.islands[0].add(
            Elite(
                text="dup",
                descriptor=length_descriptor("dup"),
                ratings=engine.rating.new_ratings(),
                age=0,
            )
        )
        engine.run("seed")

        assert ranker.rank.call_count == 0
