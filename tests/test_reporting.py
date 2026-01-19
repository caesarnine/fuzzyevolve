from __future__ import annotations

import random

import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.ratings import RatingSystem
from fuzzyevolve.reporting import render_best_by_cell_markdown


def _elite(text: str, *, length: int, rating: ts.Rating) -> Elite:
    return Elite(
        text=text,
        descriptor={"len": length},
        ratings={"m1": rating},
        age=0,
    )


def test_best_by_cell_report_merges_islands_and_limits_cells():
    cfg = Config()
    cfg.metrics.names = ["m1"]
    cfg.rating.score_lcb_c = 1.0
    cfg.descriptor.kind = "length"
    cfg.descriptor.length_bins = [0, 10, 100]

    rating = RatingSystem(["m1"], score_lcb_c=1.0)
    space = build_descriptor_space({"len": {"bins": [0, 10, 100]}})

    islands = [
        MapElitesArchive(
            space,
            elites_per_cell=4,
            rng=random.Random(0),
            score_fn=rating.score,
        )
        for _ in range(2)
    ]

    # Cell 0 champion should come from island 1.
    islands[0].add(_elite("LOW_CELL0", length=5, rating=ts.Rating(mu=10, sigma=1)))
    islands[1].add(_elite("CHAMPION_CELL0", length=5, rating=ts.Rating(mu=20, sigma=1)))

    # Cell 1 champion should come from island 0.
    islands[0].add(_elite("CHAMPION_CELL1", length=20, rating=ts.Rating(mu=18, sigma=1)))
    islands[1].add(_elite("LOW_CELL1", length=20, rating=ts.Rating(mu=1, sigma=1)))

    report = render_best_by_cell_markdown(
        cfg=cfg,
        islands=islands,
        rating=rating,
        top_cells=1,
    )

    assert "CHAMPION_CELL0" in report
    assert "CHAMPION_CELL1" not in report
    assert "Cell champions" in report
    assert "| metric | μ | σ | LCB |" in report

