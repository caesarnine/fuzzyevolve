"""
This module contains helper functions for calculating TrueSkill ratings.
"""

from typing import Dict, List

import trueskill as ts


def make_envs(metrics: List[str]) -> Dict[str, ts.TrueSkill]:
    """Creates a TrueSkill environment for each metric."""
    return {m: ts.TrueSkill(draw_probability=0.0) for m in metrics}


def ts_score(
    ratings: Dict[str, ts.Rating],
    weights: Dict[str, float] | None = None,
    c: float = 2.0,
) -> float:
    """Calculates a weighted score from TrueSkill ratings."""
    w = weights or {m: 1 / len(ratings) for m in ratings}
    return sum(w[m] * (ratings[m].mu - c * ratings[m].sigma) for m in ratings)
