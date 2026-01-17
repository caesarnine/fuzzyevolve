"""Tests for LLM ranker validation + retry logic."""

from __future__ import annotations

import random
from types import SimpleNamespace

import trueskill as ts

from fuzzyevolve.adapters.llm.ranker import LLMRanker, MetricRanking, RankerOutput
from fuzzyevolve.core.battle import Battle
from fuzzyevolve.core.models import Elite


def make_elite(text: str, metric: str) -> Elite:
    return Elite(
        text=text,
        descriptor={"len": len(text)},
        ratings={metric: ts.Rating()},
        age=0,
    )


def make_battle(players: list[Elite]) -> Battle:
    return Battle(
        participants=tuple(players),
        judged_children=tuple(),
        resort_elites=tuple(),
        frozen_indices=frozenset(),
    )


def test_ranker_allows_ties():
    ranker = LLMRanker(model="mock", rng=random.Random(0), max_attempts=1)
    battle = make_battle([make_elite("a", "m1"), make_elite("b", "m1")])
    valid = RankerOutput(rankings=[MetricRanking(metric="m1", ranked_tiers=[[0, 1]])])
    ranker.agent.run_sync = lambda *args, **kwargs: SimpleNamespace(output=valid)
    ranking = ranker.rank(metrics=["m1"], battle=battle)
    assert ranking is not None
    assert ranking.tiers_by_metric["m1"] == [[0, 1]]


def test_ranker_repair_success():
    ranker = LLMRanker(model="mock", rng=random.Random(0), max_attempts=2, repair_enabled=True)
    battle = make_battle([make_elite("a", "m1"), make_elite("b", "m1")])

    invalid = RankerOutput(rankings=[MetricRanking(metric="m1", ranked_tiers=[[0]])])
    valid = RankerOutput(rankings=[MetricRanking(metric="m1", ranked_tiers=[[0], [1]])])
    calls = {"count": 0}

    def _run_sync(*args, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(output=invalid if calls["count"] == 1 else valid)

    ranker.agent.run_sync = _run_sync
    ranking = ranker.rank(metrics=["m1"], battle=battle)
    assert ranking is not None
    assert calls["count"] == 2


def test_ranker_invalid_after_retries():
    ranker = LLMRanker(model="mock", rng=random.Random(0), max_attempts=2, repair_enabled=True)
    battle = make_battle([make_elite("a", "m1"), make_elite("b", "m1")])
    invalid = RankerOutput(rankings=[MetricRanking(metric="m1", ranked_tiers=[[0]])])
    ranker.agent.run_sync = lambda *args, **kwargs: SimpleNamespace(output=invalid)
    assert ranker.rank(metrics=["m1"], battle=battle) is None

