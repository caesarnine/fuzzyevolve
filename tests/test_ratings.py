"""Tests for TrueSkill-based rating system."""

from __future__ import annotations

import numpy as np
import pytest
import trueskill as ts

from fuzzyevolve.core.models import Anchor, Elite
from fuzzyevolve.core.ratings import BattleRanking, RatingSystem


def test_apply_ranking_skips_frozen_indices():
    rating = RatingSystem(["m1"], score_lcb_c=2.0)
    a = Elite(
        text="a",
        embedding=np.array([1.0], dtype=float),
        ratings=rating.new_ratings(),
        age=0,
    )
    b = Anchor(text="b", ratings=rating.new_ratings(), age=0)
    before = b.ratings["m1"].mu

    ranking = BattleRanking(tiers_by_metric={"m1": [[0], [1]]})
    rating.apply_ranking([a, b], ranking, frozen_indices={1})

    assert b.ratings["m1"].mu == before


def test_score_uses_lcb():
    rating = RatingSystem(["m1"], score_lcb_c=2.0)
    score = rating.score({"m1": ts.Rating(mu=10.0, sigma=1.5)})
    assert score == pytest.approx(10.0 - 2.0 * 1.5)
