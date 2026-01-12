"""Tests for judge validation and retry logic."""

import random
from types import SimpleNamespace

import trueskill as ts

from fuzzyevolve.core.judge import JudgeOutput, LLMJudge, MetricRanking
from fuzzyevolve.core.models import Elite


def make_elite(text: str, metric: str) -> Elite:
    return Elite(
        text=text,
        descriptor={"lang": "txt", "len": len(text)},
        ratings={metric: ts.Rating()},
        age=0,
    )


def test_validate_rankings_strict_permutation():
    judge = LLMJudge("mock", ["m1", "m2"], rng=random.Random(0))
    rankings = [
        MetricRanking(metric="m1", ranked_ids=[0, 1]),
        MetricRanking(metric="m2", ranked_ids=[0]),
    ]
    ranked_map, error = judge._validate_rankings(rankings, total_players=2)
    assert ranked_map is None
    assert error is not None


def test_rank_and_rate_repair_success():
    judge = LLMJudge("mock", ["m1"], rng=random.Random(0), max_attempts=2)
    players = [make_elite("a", "m1"), make_elite("b", "m1")]
    invalid = JudgeOutput(rankings=[MetricRanking(metric="m1", ranked_ids=[0])])
    valid = JudgeOutput(rankings=[MetricRanking(metric="m1", ranked_ids=[0, 1])])
    calls = {"count": 0}

    def _run_sync(*args, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(output=invalid if calls["count"] == 1 else valid)

    judge.agent.run_sync = _run_sync
    before = players[0].ratings["m1"].mu
    assert judge.rank_and_rate(players) is True
    assert calls["count"] == 2
    assert players[0].ratings["m1"].mu != before


def test_rank_and_rate_frozen_anchor_unchanged():
    judge = LLMJudge("mock", ["m1"], rng=random.Random(0), max_attempts=1)
    players = [make_elite("a", "m1"), make_elite("b", "m1")]
    players[0].frozen = True
    valid = JudgeOutput(rankings=[MetricRanking(metric="m1", ranked_ids=[0, 1])])
    judge.agent.run_sync = lambda *args, **kwargs: SimpleNamespace(output=valid)
    before = players[0].ratings["m1"].mu
    assert judge.rank_and_rate(players) is True
    assert players[0].ratings["m1"].mu == before


def test_rank_and_rate_invalid_after_retries():
    judge = LLMJudge("mock", ["m1"], rng=random.Random(0), max_attempts=2)
    players = [make_elite("a", "m1"), make_elite("b", "m1")]
    invalid = JudgeOutput(rankings=[MetricRanking(metric="m1", ranked_ids=[0])])
    judge.agent.run_sync = lambda *args, **kwargs: SimpleNamespace(output=invalid)
    assert judge.rank_and_rate(players) is False
