from __future__ import annotations

import random

import numpy as np
import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import RatingSystem
from fuzzyevolve.reporting import render_top_by_fitness_markdown


def _elite(text: str, *, rating: ts.Rating) -> Elite:
    return Elite(
        text=text,
        embedding=np.array([1.0], dtype=float),
        ratings={"m1": rating},
        age=0,
    )


def test_top_report_limits_output():
    cfg = Config()
    cfg.metrics.names = ["m1"]
    cfg.rating.score_lcb_c = 1.0
    cfg.population.size = 10

    rating = RatingSystem(["m1"], score_lcb_c=1.0)
    pool = CrowdedPool(
        max_size=cfg.population.size, rng=random.Random(0), score_fn=rating.score
    )

    pool.add(_elite("LOW", rating=ts.Rating(mu=10, sigma=1)))
    pool.add(_elite("CHAMPION", rating=ts.Rating(mu=20, sigma=1)))

    report = render_top_by_fitness_markdown(cfg=cfg, pool=pool, rating=rating, top=1)

    assert "CHAMPION" in report
    assert "LOW" not in report
    assert "Top individuals" in report
    assert "| metric | μ | σ | LCB |" in report
