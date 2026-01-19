from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import BinnedAxis, DescriptorSpace
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.ratings import RatingSystem


@dataclass(frozen=True, slots=True)
class CellChampion:
    cell_key: tuple[Any, ...]
    elite: Elite
    island: int
    score: float


def gather_cell_champions(
    *,
    islands: Sequence[MapElitesArchive],
    rating: RatingSystem,
) -> list[CellChampion]:
    """Return the best elite per cell across all islands."""

    champions_by_cell: dict[tuple[Any, ...], CellChampion] = {}
    for island_idx, archive in enumerate(islands):
        for cell_key, bucket in archive.iter_cells():
            if not bucket:
                continue
            best = max(bucket, key=lambda elite: rating.score(elite.ratings))
            best_score = rating.score(best.ratings)
            candidate = CellChampion(
                cell_key=cell_key,
                elite=best,
                island=island_idx,
                score=best_score,
            )
            existing = champions_by_cell.get(cell_key)
            if existing is None or candidate.score > existing.score:
                champions_by_cell[cell_key] = candidate

    champions = list(champions_by_cell.values())
    champions.sort(key=lambda champ: champ.score, reverse=True)
    return champions


def render_best_by_cell_markdown(
    *,
    cfg: Config,
    islands: Sequence[MapElitesArchive],
    rating: RatingSystem,
    top_cells: int,
) -> str:
    if top_cells < 0:
        raise ValueError("top_cells must be >= 0")
    if not islands:
        raise ValueError("At least one island is required.")

    space: DescriptorSpace = islands[0].space
    metrics = list(cfg.metrics.names)
    c = float(cfg.rating.score_lcb_c)

    champions = gather_cell_champions(islands=islands, rating=rating)
    total_cells = len(champions)
    limit = total_cells if top_cells == 0 else min(top_cells, total_cells)
    champions = champions[:limit]

    total_elites = sum(1 for archive in islands for _ in archive.iter_elites())

    lines: list[str] = []
    lines.append("# fuzzyevolve results")
    lines.append("")
    lines.append(f"- Goal: {cfg.task.goal}")
    lines.append(f"- Metrics: {', '.join(metrics)}")
    lines.append(f"- Islands: {len(islands)}")
    lines.append(f"- Total elites: {total_elites}")
    lines.append(f"- Descriptor kind: {cfg.descriptor.kind}")
    lines.append(f"- Score: average(metric μ - {c:g}·σ)")
    if top_cells == 0:
        showing = f"showing all {limit}"
    else:
        showing = f"showing top {limit}"
    lines.append(f"- Non-empty cells: {total_cells}/{space.total_cells} ({showing})")
    lines.append("")

    if not champions:
        lines.append("_No elites found._")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Cell champions (ranked)")
    lines.append("")
    lines.append("| rank | score | cell | island | age | preview |")
    lines.append("|---:|---:|---|---:|---:|---|")
    for idx, champ in enumerate(champions, start=1):
        cell = format_cell(space, champ.cell_key)
        preview = _preview_line(champ.elite.text, max_len=72)
        lines.append(
            f"| {idx} | {champ.score:.3f} | `{cell}` | {champ.island} | {champ.elite.age} | {preview} |"
        )

    lines.append("")
    for idx, champ in enumerate(champions, start=1):
        lines.append("---")
        lines.append("")
        lines.append(f"## {idx}. score {champ.score:.3f} — cell `{format_cell(space, champ.cell_key)}`")
        lines.append("")
        lines.append(f"- island: `{champ.island}`")
        lines.append(f"- age: `{champ.elite.age}`")
        lines.append(f"- descriptor: `{_format_descriptor(champ.elite.descriptor)}`")
        lines.append("")
        lines.append(_format_metric_table(champ.elite.ratings, metrics=metrics, c=c))
        lines.append("")
        lines.append("```text")
        lines.append(champ.elite.text.rstrip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def format_cell(space: DescriptorSpace, cell_key: tuple[Any, ...]) -> str:
    """Format a cell key into readable axis labels."""

    parts: list[str] = []
    for (axis_name, axis), key_part in zip(space.axes.items(), cell_key):
        if isinstance(axis, BinnedAxis):
            idx = int(key_part)
            lo = axis.bins[idx]
            hi = axis.bins[idx + 1]
            parts.append(f"{axis_name}[{lo:g},{hi:g})")
        else:
            parts.append(f"{axis_name}={key_part}")
    return " ".join(parts) if parts else "(none)"


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


def _format_descriptor(desc: dict[str, Any]) -> str:
    if not desc:
        return "(none)"
    parts: list[str] = []
    for key, value in desc.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


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
