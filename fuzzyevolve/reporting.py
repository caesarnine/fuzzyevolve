from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import RatingSystem


@dataclass(frozen=True, slots=True)
class RankedElite:
    rank: int
    elite: Elite
    score: float


def top_by_fitness(
    *,
    pool: CrowdedPool,
    rating: RatingSystem,
    top: int,
) -> list[RankedElite]:
    if top < 0:
        raise ValueError("top must be >= 0")
    elites = list(pool.iter_elites())
    elites.sort(key=lambda e: rating.score(e.ratings), reverse=True)
    if top != 0:
        elites = elites[: min(top, len(elites))]
    return [
        RankedElite(rank=idx, elite=elite, score=rating.score(elite.ratings))
        for idx, elite in enumerate(elites, start=1)
    ]


def render_top_by_fitness_markdown(
    *,
    cfg: Config,
    pool: CrowdedPool,
    rating: RatingSystem,
    top: int,
) -> str:
    if top < 0:
        raise ValueError("top must be >= 0")

    metrics = list(cfg.metrics.names)
    c = float(cfg.rating.score_lcb_c)

    ranked = top_by_fitness(pool=pool, rating=rating, top=top)
    total = len(pool)
    showing = len(ranked)

    lines: list[str] = []
    lines.append("# fuzzyevolve results")
    lines.append("")
    lines.append(f"- Goal: {cfg.task.goal}")
    lines.append(f"- Metrics: {', '.join(metrics)}")
    lines.append(f"- Population: {total}/{cfg.population.size}")
    lines.append(f"- Embeddings: {cfg.embeddings.model}")
    lines.append(f"- Score: average(metric μ - {c:g}·σ)")
    if top == 0:
        lines.append(f"- Showing: all {showing}")
    else:
        lines.append(f"- Showing: top {showing}")
    lines.append("")

    if not ranked:
        lines.append("_No individuals found._")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Top individuals (ranked)")
    lines.append("")
    lines.append("| rank | score | age | preview |")
    lines.append("|---:|---:|---:|---|")
    for item in ranked:
        preview = _preview_line(item.elite.text, max_len=72)
        lines.append(
            f"| {item.rank} | {item.score:.3f} | {item.elite.age} | {preview} |"
        )

    lines.append("")
    for item in ranked:
        lines.append("---")
        lines.append("")
        lines.append(f"## {item.rank}. score {item.score:.3f}")
        lines.append("")
        lines.append(f"- age: `{item.elite.age}`")
        lines.append("")
        lines.append(_format_metric_table(item.elite.ratings, metrics=metrics, c=c))
        lines.append("")
        lines.append("```text")
        lines.append(item.elite.text.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _preview_line(text: str, *, max_len: int) -> str:
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            line = stripped
            break
    else:
        line = text.strip()
    if len(line) > max_len:
        return line[: max(0, max_len - 1)].rstrip() + "…"
    return line


def _format_metric_table(
    ratings: dict[str, ts.Rating],
    *,
    metrics: Iterable[str],
    c: float,
) -> str:
    lines = ["| metric | μ | σ | LCB |", "|---|---:|---:|---:|"]
    for metric in metrics:
        r = ratings.get(metric)
        if r is None:
            continue
        lcb = r.mu - c * r.sigma
        lines.append(f"| {metric} | {r.mu:.2f} | {r.sigma:.2f} | {lcb:.2f} |")
    return "\n".join(lines)
