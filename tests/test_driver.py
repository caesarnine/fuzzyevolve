"""Tests for the evolution engine."""

import random
from unittest.mock import Mock

import pytest

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.mutation.mutator import MutationCandidate, MutationResult


class DummyRating:
    def __init__(self, mu: float = 25.0, sigma: float = 8.333):
        self.mu = mu
        self.sigma = sigma


def make_engine(cfg: Config, mutator: Mock, judge: Mock) -> EvolutionEngine:
    rng = random.Random(0)
    space = build_descriptor_space(cfg.axes)
    islands = [
        MapElitesArchive(space, cfg.elites_per_cell, rng=rng)
        for _ in range(cfg.island_count)
    ]
    return EvolutionEngine(cfg, mutator, judge, islands, rng=rng)


class TestEvolutionEngine:
    def setup_method(self):
        self.cfg = Config(
            iterations=2,
            island_count=1,
            elites_per_cell=3,
            metrics=["test_metric"],
            max_diffs=2,
            inspiration_count=0,
        )
        self.judge = Mock()
        self.judge.new_ratings = Mock(
            side_effect=lambda: {"test_metric": DummyRating()}
        )
        self.judge.rank_and_rate = Mock()
        self.mutator = Mock()

    def test_initialization(self):
        engine = make_engine(self.cfg, self.mutator, self.judge)
        assert engine.cfg == self.cfg
        assert len(engine.islands) == 1

    def test_batch_judging_multiple_children(self):
        cfg = Config(
            iterations=1,
            island_count=1,
            elites_per_cell=3,
            metrics=["test_metric"],
            max_diffs=2,
            inspiration_count=0,
        )
        judge = Mock()
        judge.new_ratings = Mock(side_effect=lambda: {"test_metric": DummyRating()})
        judge.rank_and_rate = Mock()
        mutator = Mock()
        mutator.propose = Mock(
            return_value=MutationResult(
                thinking="Test",
                candidates=[
                    MutationCandidate(text="modified1", diff="diff1"),
                    MutationCandidate(text="modified2", diff="diff2"),
                ],
            )
        )

        engine = make_engine(cfg, mutator, judge)
        engine.run("seed")

        assert judge.rank_and_rate.call_count == 1
        players = judge.rank_and_rate.call_args[0][0]
        assert len(players) == 3
        assert players[0].text == "seed"
        assert {players[1].text, players[2].text} == {"modified1", "modified2"}

    @pytest.mark.parametrize("include_inspirations", [True, False])
    def test_judge_include_inspirations_toggle(self, include_inspirations):
        cfg = Config(
            iterations=2,
            island_count=1,
            elites_per_cell=5,
            metrics=["test_metric"],
            max_diffs=1,
            inspiration_count=1,
            judge_include_inspirations=include_inspirations,
        )
        judge = Mock()
        judge.new_ratings = Mock(side_effect=lambda: {"test_metric": DummyRating()})
        judge.rank_and_rate = Mock()
        mutator = Mock()
        mutator.propose = Mock(
            return_value=MutationResult(
                thinking="Test",
                candidates=[MutationCandidate(text="seedX", diff="diff")],
            )
        )

        engine = make_engine(cfg, mutator, judge)
        engine.run("seed")

        assert judge.rank_and_rate.call_count == 2
        group_sizes = [len(call.args[0]) for call in judge.rank_and_rate.call_args_list]
        assert group_sizes[0] == 2
        assert group_sizes[1] == (3 if include_inspirations else 2)

    def test_empty_candidates_handling(self):
        self.mutator.propose = Mock(
            return_value=MutationResult(thinking="Test", candidates=[])
        )
        engine = make_engine(self.cfg, self.mutator, self.judge)
        engine.run("seed")

        assert self.judge.rank_and_rate.call_count == 0

    def test_migration(self):
        cfg = Config(
            iterations=1,
            island_count=2,
            elites_per_cell=5,
            metrics=["test_metric"],
            max_diffs=1,
            inspiration_count=0,
            migration_interval=1,
            migration_size=1,
            sparring_interval=1000,
        )
        judge = Mock()
        judge.new_ratings = Mock(side_effect=lambda: {"test_metric": DummyRating()})
        judge.rank_and_rate = Mock()
        mutator = Mock()
        mutator.propose = Mock(
            return_value=MutationResult(
                thinking="Test",
                candidates=[MutationCandidate(text="seedX", diff="diff")],
            )
        )

        engine = make_engine(cfg, mutator, judge)
        engine.run("seed")

        counts = [sum(1 for _ in island.iter_elites()) for island in engine.islands]
        assert all(count >= 2 for count in counts)

    def test_sparring(self):
        cfg = Config(
            iterations=1,
            island_count=2,
            elites_per_cell=5,
            metrics=["test_metric"],
            max_diffs=1,
            inspiration_count=0,
            sparring_interval=1,
            migration_interval=1000,
        )
        judge = Mock()
        judge.new_ratings = Mock(side_effect=lambda: {"test_metric": DummyRating()})
        judge.rank_and_rate = Mock()
        mutator = Mock()
        mutator.propose = Mock(
            return_value=MutationResult(
                thinking="Test",
                candidates=[MutationCandidate(text="seedX", diff="diff")],
            )
        )

        engine = make_engine(cfg, mutator, judge)
        engine.run("seed")

        assert judge.rank_and_rate.call_count == 2
