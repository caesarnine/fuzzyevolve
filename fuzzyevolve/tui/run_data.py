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

    @property
    def preview(self) -> str:
        return self.text_id[:8]


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


def load_run_state(run_dir: Path) -> RunState:
    store = RunStore.open(run_dir)
    cfg = store.load_config()

    cp_path = store.latest_checkpoint_path()
    checkpoint_mtime = cp_path.stat().st_mtime if cp_path.exists() else 0.0
    checkpoint = _read_json(cp_path) if cp_path.exists() else {}
    iteration = int(checkpoint.get("next_iteration", 0))

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

        members_out.append(
            EliteRecord(text_id=text_id, ratings=ratings, age=age, score=score)
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
