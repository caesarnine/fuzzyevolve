from __future__ import annotations

from typing import Mapping, Sequence

import trueskill as ts


def make_envs(
    metrics: Sequence[str],
    *,
    mu: float = 25.0,
    sigma: float = 25.0 / 3.0,
    beta: float | None = None,
    tau: float | None = None,
    draw_probability: float = 0.2,
) -> dict[str, ts.TrueSkill]:
    beta = beta if beta is not None else sigma / 2.0
    tau = tau if tau is not None else sigma / 100.0
    return {
        metric: ts.TrueSkill(
            mu=mu,
            sigma=sigma,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
        )
        for metric in metrics
    }


def make_initial_ratings(
    metrics: Sequence[str],
    envs: Mapping[str, ts.TrueSkill] | None = None,
) -> dict[str, ts.Rating]:
    envs = envs or make_envs(metrics)
    return {metric: envs[metric].create_rating() for metric in metrics}


def score_ratings(
    ratings: Mapping[str, ts.Rating],
    weights: Mapping[str, float] | None = None,
    c: float = 2.0,
) -> float:
    if not ratings:
        return 0.0
    weight_map = weights or {metric: 1 / len(ratings) for metric in ratings}
    return sum(
        weight_map[metric] * (ratings[metric].mu - c * ratings[metric].sigma)
        for metric in ratings
    )
