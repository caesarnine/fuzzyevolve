from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from fuzzyevolve.config import Config
from fuzzyevolve.run_store import RunStore


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, max_lines: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if max_lines is not None and max_lines > 0:
        dq: deque[str] = deque(maxlen=max_lines)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                dq.append(line)
        lines = list(dq)
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
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


@dataclass(frozen=True, slots=True)
class MetricRating:
    mu: float
    sigma: float


@dataclass(frozen=True, slots=True)
class EliteRecord:
    text_id: str
    ratings: dict[str, MetricRating]
    age: int
    score: float
    preview_text: str = ""

    @property
    def preview(self) -> str:
        return self.preview_text or self.text_id[:8]


@dataclass(frozen=True, slots=True)
class StatsRecord:
    iteration: int
    best_score: float | None
    pool_size: int | None
    best_text_id: str | None = None

    mean_score: float | None = None
    p50_score: float | None = None
    p90_score: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    std_score: float | None = None

    diversity_nn_mean: float | None = None
    diversity_nn_p10: float | None = None
    diversity_nn_p50: float | None = None

    mean_sigma: float | None = None


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
    members: list[EliteRecord]
    best: EliteRecord | None
    checkpoint_mtime: float = 0.0
    stats: list[StatsRecord] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    stats_mtime: float = 0.0
    events_mtime: float = 0.0

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


def _preview_line(text: str, *, max_len: int = 72) -> str:
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


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_stats(lines: list[dict[str, Any]]) -> list[StatsRecord]:
    out: list[StatsRecord] = []
    for row in lines:
        iteration = _parse_optional_int(row.get("iteration"))
        if iteration is None:
            continue
        best_text_id = row.get("best_text_id")
        out.append(
            StatsRecord(
                iteration=iteration,
                best_score=_parse_optional_float(row.get("best_score")),
                pool_size=_parse_optional_int(row.get("pool_size")),
                best_text_id=(str(best_text_id) if best_text_id else None),
                mean_score=_parse_optional_float(row.get("mean_score")),
                p50_score=_parse_optional_float(row.get("p50_score")),
                p90_score=_parse_optional_float(row.get("p90_score")),
                min_score=_parse_optional_float(row.get("min_score")),
                max_score=_parse_optional_float(row.get("max_score")),
                std_score=_parse_optional_float(row.get("std_score")),
                diversity_nn_mean=_parse_optional_float(row.get("diversity_nn_mean")),
                diversity_nn_p10=_parse_optional_float(row.get("diversity_nn_p10")),
                diversity_nn_p50=_parse_optional_float(row.get("diversity_nn_p50")),
                mean_sigma=_parse_optional_float(row.get("mean_sigma")),
            )
        )
    out.sort(key=lambda r: r.iteration)
    return out


def load_run_state(run_dir: Path) -> RunState:
    store = RunStore.open(run_dir)
    cfg = store.load_config()

    cp_path = store.latest_checkpoint_path()
    checkpoint_mtime = cp_path.stat().st_mtime if cp_path.exists() else 0.0
    checkpoint = _read_json(cp_path) if cp_path.exists() else {}
    iteration = int(checkpoint.get("next_iteration", 0))

    stats_path = store.run_dir / "stats.jsonl"
    stats_mtime = stats_path.stat().st_mtime if stats_path.exists() else 0.0
    stats_lines = _read_jsonl(stats_path, max_lines=20000)
    stats = _parse_stats(stats_lines)

    events_path = store.run_dir / "events.jsonl"
    events_mtime = events_path.stat().st_mtime if events_path.exists() else 0.0
    events = _read_jsonl(events_path, max_lines=50000)

    members_out: list[EliteRecord] = []
    for elite in checkpoint.get("population", {}).get("members", []):
        text_id = str(elite["text_id"])
        ratings_raw = elite.get("ratings") or {}
        ratings = {
            metric: MetricRating(
                mu=float(rdict.get("mu", 0.0)),
                sigma=float(rdict.get("sigma", 0.0)),
            )
            for metric, rdict in ratings_raw.items()
        }
        age = int(elite.get("age", 0))

        c = float(cfg.rating.score_lcb_c)
        metrics = cfg.metrics.names
        score = (
            sum((ratings[m].mu - c * ratings[m].sigma) for m in metrics if m in ratings)
            / len(metrics)
            if metrics
            else 0.0
        )

        preview_text = ""
        try:
            preview_text = _preview_line(store.get_text(text_id))
        except Exception:
            preview_text = text_id[:8]

        members_out.append(
            EliteRecord(
                text_id=text_id,
                ratings=ratings,
                age=age,
                score=score,
                preview_text=preview_text,
            )
        )

    members_out.sort(key=lambda e: e.score, reverse=True)
    best = members_out[0] if members_out else None

    return RunState(
        run_dir=store.run_dir,
        cfg=cfg,
        store=store,
        iteration=iteration,
        members=members_out,
        best=best,
        checkpoint_mtime=checkpoint_mtime,
        stats=stats,
        events=events,
        stats_mtime=stats_mtime,
        events_mtime=events_mtime,
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
