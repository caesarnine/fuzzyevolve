from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from fuzzyevolve.config import Config
from fuzzyevolve.run_store import RunStore


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, max_lines: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if max_lines is not None and max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _bin_index(value: float, bins: list[float]) -> int:
    if len(bins) < 2:
        return 0
    if value >= bins[-1]:
        return len(bins) - 2
    for idx in range(len(bins) - 1):
        if bins[idx] <= value < bins[idx + 1]:
            return idx
    return 0


@dataclass(frozen=True, slots=True)
class MetricRating:
    mu: float
    sigma: float

    @property
    def lcb(self) -> float:
        # c is applied elsewhere; this is raw mu/sigma.
        return self.mu - self.sigma


@dataclass(frozen=True, slots=True)
class EliteRecord:
    text_id: str
    descriptor: dict[str, Any]
    ratings: dict[str, MetricRating]
    age: int
    island: int
    cell_key: tuple[int, ...]
    score: float

    @property
    def preview(self) -> str:
        text = self.text_id
        return text[:8]


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_dir: Path
    run_id: str
    created_at: str | None
    metrics: list[str]
    iteration: int
    best_score: float | None


@dataclass(slots=True)
class RunState:
    run_dir: Path
    cfg: Config
    store: RunStore
    iteration: int
    islands: list[list[EliteRecord]]
    best: EliteRecord | None

    descriptor_kind: str
    bins_x: list[float] | None = None
    bins_y: list[float] | None = None
    length_bins: list[float] | None = None

    checkpoint_mtime: float = 0.0

    def score_from_ratings(self, ratings: Mapping[str, MetricRating]) -> float:
        c = float(self.cfg.rating.score_lcb_c)
        metrics = self.cfg.metrics.names
        if not metrics:
            return 0.0
        total = 0.0
        for metric in metrics:
            r = ratings.get(metric)
            if r is None:
                continue
            total += r.mu - c * r.sigma
        return total / len(metrics)

    def get_text(self, text_id: str) -> str:
        return self.store.get_text(text_id)

    def cell_counts(self) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = {}
        for island in self.islands:
            for elite in island:
                counts[elite.cell_key] = counts.get(elite.cell_key, 0) + 1
        return counts

    def cell_best_scores(self) -> dict[tuple[int, ...], float]:
        best: dict[tuple[int, ...], float] = {}
        for island in self.islands:
            for elite in island:
                prev = best.get(elite.cell_key)
                if prev is None or elite.score > prev:
                    best[elite.cell_key] = elite.score
        return best


def list_runs(data_dir: Path) -> list[RunSummary]:
    runs_root = data_dir / "runs"
    if not runs_root.is_dir():
        return []
    runs = [p for p in runs_root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)

    out: list[RunSummary] = []
    for run_dir in runs:
        run_id = run_dir.name
        created_at = None
        metrics: list[str] = []
        iteration = 0
        best_score: float | None = None

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = _read_json(meta_path)
                created_at = meta.get("created_at")
                metrics = list(meta.get("metrics") or [])
            except Exception:
                pass

        stats = _read_jsonl(run_dir / "stats.jsonl", max_lines=1)
        if stats:
            try:
                iteration = int(stats[-1].get("iteration", 0))
                best_score = float(stats[-1].get("best_score"))
            except Exception:
                pass

        out.append(
            RunSummary(
                run_dir=run_dir,
                run_id=run_id,
                created_at=created_at,
                metrics=metrics,
                iteration=iteration,
                best_score=best_score,
            )
        )
    return out


def load_run_state(run_dir: Path) -> RunState:
    store = RunStore.open(run_dir)
    cfg = store.load_config()

    cp_path = store.latest_checkpoint_path()
    checkpoint_mtime = cp_path.stat().st_mtime if cp_path.exists() else 0.0
    checkpoint = _read_json(cp_path) if cp_path.exists() else {}
    iteration = int(checkpoint.get("next_iteration", 0))

    kind = cfg.descriptor.kind
    bins_x = None
    bins_y = None
    length_bins = None
    if kind == "embedding_2d":
        bins_x = list(cfg.descriptor.embedding_2d.bins_x)
        bins_y = list(cfg.descriptor.embedding_2d.bins_y)
    elif kind == "length":
        length_bins = list(cfg.descriptor.length_bins)

    islands_out: list[list[EliteRecord]] = []
    for island_idx, island in enumerate(checkpoint.get("islands", [])):
        elites_out: list[EliteRecord] = []
        for elite in island.get("elites", []):
            text_id = str(elite["text_id"])
            desc = dict(elite.get("descriptor") or {})
            ratings_raw = elite.get("ratings") or {}
            ratings = {
                metric: MetricRating(
                    mu=float(rdict.get("mu", 0.0)),
                    sigma=float(rdict.get("sigma", 0.0)),
                )
                for metric, rdict in ratings_raw.items()
            }

            if kind == "embedding_2d" and bins_x and bins_y:
                cx = _bin_index(float(desc.get("embed_x", 0.0)), bins_x)
                cy = _bin_index(float(desc.get("embed_y", 0.0)), bins_y)
                cell_key = (cx, cy)
            elif kind == "length" and length_bins:
                cell_key = (_bin_index(float(desc.get("len", 0.0)), length_bins),)
            else:
                cell_key = tuple()

            age = int(elite.get("age", 0))
            # conservative score used throughout the project
            c = float(cfg.rating.score_lcb_c)
            metrics = cfg.metrics.names
            score = (
                sum(
                    (ratings[m].mu - c * ratings[m].sigma)
                    for m in metrics
                    if m in ratings
                )
                / len(metrics)
                if metrics
                else 0.0
            )

            elites_out.append(
                EliteRecord(
                    text_id=text_id,
                    descriptor=desc,
                    ratings=ratings,
                    age=age,
                    island=island_idx,
                    cell_key=cell_key,
                    score=score,
                )
            )
        elites_out.sort(key=lambda e: e.score, reverse=True)
        islands_out.append(elites_out)

    all_elites: list[EliteRecord] = [e for island in islands_out for e in island]
    best = max(all_elites, key=lambda e: e.score) if all_elites else None

    return RunState(
        run_dir=store.run_dir,
        cfg=cfg,
        store=store,
        iteration=iteration,
        islands=islands_out,
        best=best,
        descriptor_kind=kind,
        bins_x=bins_x,
        bins_y=bins_y,
        length_bins=length_bins,
        checkpoint_mtime=checkpoint_mtime,
    )


def tail_stats(run_dir: Path, *, max_lines: int = 300) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "stats.jsonl", max_lines=max_lines)


def tail_events(run_dir: Path, *, max_lines: int = 2000) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "events.jsonl", max_lines=max_lines)


def tail_llm_index(run_dir: Path, *, max_lines: int = 500) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "llm.jsonl", max_lines=max_lines)


def find_last_by_type(
    events: Iterable[dict[str, Any]], event_type: str
) -> dict[str, Any] | None:
    for ev in reversed(list(events)):
        if ev.get("type") == event_type:
            return ev
    return None

