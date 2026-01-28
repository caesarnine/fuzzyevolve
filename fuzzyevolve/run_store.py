from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import trueskill as ts

from fuzzyevolve.config import Config
from fuzzyevolve.core.anchors import AnchorManager, AnchorPool
from fuzzyevolve.core.models import Anchor, Elite
from fuzzyevolve.core.pool import CrowdedPool

SCHEMA_VERSION = 2


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rating_to_dict(r: ts.Rating) -> dict[str, float]:
    return {"mu": float(r.mu), "sigma": float(r.sigma)}


def _rating_from_dict(data: Mapping[str, Any]) -> ts.Rating:
    return ts.Rating(mu=float(data["mu"]), sigma=float(data["sigma"]))


def _to_jsonable(obj: Any) -> Any:
    # Pydantic models
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


@dataclass(frozen=True, slots=True)
class LoadedState:
    next_iteration: int
    pool: CrowdedPool
    anchors: AnchorManager | None


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.texts_dir = run_dir / "texts"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.llm_dir = run_dir / "llm"
        self.events_path = run_dir / "events.jsonl"
        self.llm_index_path = run_dir / "llm.jsonl"
        self.stats_path = run_dir / "stats.jsonl"

        self._lock = threading.Lock()
        self._llm_call_seq = 0
        self._current_iteration = 0

        self.texts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.llm_dir.mkdir(parents=True, exist_ok=True)

        if self.llm_index_path.exists():
            try:
                with self.llm_index_path.open("r", encoding="utf-8") as f:
                    self._llm_call_seq = sum(1 for _ in f)
            except Exception:
                # Best-effort; sequence collisions are still unlikely due to iteration prefixing.
                self._llm_call_seq = 0

    @classmethod
    def create(
        cls,
        *,
        data_dir: Path,
        cfg: Config,
        seed_text: str | None,
        config_path: Path | None,
    ) -> "RunStore":
        runs_root = data_dir / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        ts_part = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        rand_part = os.urandom(3).hex()
        run_dir = runs_root / f"{ts_part}_{rand_part}"
        run_dir.mkdir(parents=True, exist_ok=False)

        store = cls(run_dir)
        store._write_text(
            run_dir / "meta.json", _json_dump(store._build_meta(cfg, config_path))
        )
        store._write_text(
            run_dir / "config.json",
            _json_dump(
                {"schema": SCHEMA_VERSION, "config": cfg.model_dump(mode="json")}
            ),
        )
        if seed_text is not None:
            store._write_text(run_dir / "seed.txt", seed_text)
        if config_path is not None and config_path.is_file():
            try:
                store._write_text(
                    run_dir / "config.source.txt",
                    config_path.read_text(encoding="utf-8"),
                )
            except Exception:
                # Best-effort only; config may not be readable in some contexts.
                pass
        return store

    @classmethod
    def open(cls, run_dir: Path) -> "RunStore":
        if run_dir.is_file():
            if run_dir.parent.name == "checkpoints":
                run_dir = run_dir.parent.parent
            else:
                run_dir = run_dir.parent
        elif run_dir.is_dir() and run_dir.name == "checkpoints":
            run_dir = run_dir.parent
        if not run_dir.is_dir():
            raise ValueError(f"Run directory not found: {run_dir}")
        return cls(run_dir)

    @staticmethod
    def default_data_dir(*, cwd: Path | None = None) -> Path:
        return (cwd or Path.cwd()) / ".fuzzyevolve"

    def load_config(self) -> Config:
        raw = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        cfg_data = raw.get("config", raw)
        return Config.model_validate(cfg_data)

    def latest_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "latest.json"

    def set_iteration(self, iteration: int) -> None:
        self._current_iteration = max(0, int(iteration))

    def put_text(self, text: str) -> str:
        text_id = _hash_text(text)
        path = self.texts_dir / f"{text_id}.txt"
        if path.exists():
            return text_id
        with self._lock:
            if path.exists():
                return text_id
            path.write_text(text, encoding="utf-8")
        return text_id

    def get_text(self, text_id: str) -> str:
        path = self.texts_dir / f"{text_id}.txt"
        return path.read_text(encoding="utf-8")

    def record_event(
        self, kind: str, data: Mapping[str, Any], *, iteration: int | None = None
    ) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "iteration": int(
                iteration if iteration is not None else self._current_iteration
            ),
            "type": kind,
            "data": _to_jsonable(dict(data)),
        }
        self._append_jsonl(self.events_path, payload)

    def record_stats(
        self,
        *,
        iteration: int,
        best_score: float,
        pool_size: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "ts": _utc_now_iso(),
            "iteration": int(iteration),
            "best_score": float(best_score),
            "pool_size": int(pool_size),
        }
        if extra:
            payload.update(_to_jsonable(dict(extra)))
        self._append_jsonl(self.stats_path, payload)

    def record_llm_call(
        self,
        *,
        name: str,
        model: str,
        model_settings: Mapping[str, Any] | None,
        prompt: str,
        output: Any | None,
        error: str | None = None,
        iteration: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        it = int(iteration if iteration is not None else self._current_iteration)
        with self._lock:
            self._llm_call_seq += 1
            call_id = self._llm_call_seq

        stem = f"it{it:06d}_{call_id:05d}_{name}"
        prompt_path = self.llm_dir / f"{stem}.prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        output_path: Path | None = None
        if output is not None:
            out_obj = _to_jsonable(output)
            output_path = self.llm_dir / f"{stem}.output.json"
            output_path.write_text(_json_dump(out_obj), encoding="utf-8")

        payload = {
            "ts": _utc_now_iso(),
            "iteration": it,
            "name": name,
            "model": model,
            "model_settings": _to_jsonable(dict(model_settings or {})),
            "prompt_file": str(prompt_path.relative_to(self.run_dir)),
            "output_file": str(output_path.relative_to(self.run_dir))
            if output_path
            else None,
            "error": error,
            "extra": _to_jsonable(dict(extra or {})),
        }
        self._append_jsonl(self.llm_index_path, payload)

    def save_checkpoint(
        self,
        *,
        iteration: int,
        pool: CrowdedPool,
        anchor_manager: AnchorManager | None,
        keep: bool,
    ) -> None:
        data = self._serialize_checkpoint(
            iteration=iteration,
            pool=pool,
            anchor_manager=anchor_manager,
        )
        latest = self.checkpoints_dir / "latest.json"
        latest.write_text(_json_dump(data), encoding="utf-8")
        if keep:
            path = self.checkpoints_dir / f"it{int(iteration):06d}.json"
            path.write_text(_json_dump(data), encoding="utf-8")

    def load_checkpoint(
        self,
        *,
        cfg: Config,
        checkpoint_path: Path | None = None,
        embed: Callable[[str], np.ndarray],
        pool_factory: Callable[[], CrowdedPool],
        anchor_factory: Callable[[Config], AnchorManager | None],
    ) -> LoadedState:
        checkpoint_path = checkpoint_path or (self.checkpoints_dir / "latest.json")
        raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))

        if int(raw.get("schema", 0)) != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema.")

        next_iteration = int(raw.get("next_iteration", 0))

        pool = pool_factory()
        members = raw.get("population", {}).get("members", [])
        elites: list[Elite] = []
        for elite_data in members:
            text = self.get_text(str(elite_data["text_id"]))
            elite = Elite(
                text=text,
                embedding=embed(text),
                ratings={
                    metric: _rating_from_dict(rdict)
                    for metric, rdict in elite_data["ratings"].items()
                },
                age=int(elite_data["age"]),
            )
            elites.append(elite)
        pool.add_many(elites)

        anchors = None
        anchors_data = raw.get("anchors")
        if anchors_data and anchors_data.get("items"):
            anchors = anchor_factory(cfg)
            if anchors is not None:
                pool_obj: AnchorPool = anchors.pool
                items = []
                for a in anchors_data.get("items", []):
                    text = self.get_text(str(a["text_id"]))
                    items.append(
                        Anchor(
                            text=text,
                            ratings={
                                metric: _rating_from_dict(rdict)
                                for metric, rdict in a["ratings"].items()
                            },
                            age=int(a["age"]),
                            label=str(a.get("label", "")),
                        )
                    )
                pool_obj.load(items)

        return LoadedState(next_iteration=next_iteration, pool=pool, anchors=anchors)

    def _build_meta(self, cfg: Config, config_path: Path | None) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "created_at": _utc_now_iso(),
            "cwd": str(Path.cwd()),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "config_path": str(config_path) if config_path else None,
            "metrics": list(cfg.metrics.names),
            "population_size": int(cfg.population.size),
            "embeddings_model": cfg.embeddings.model,
        }

    def _serialize_checkpoint(
        self,
        *,
        iteration: int,
        pool: CrowdedPool,
        anchor_manager: AnchorManager | None,
    ) -> dict[str, Any]:
        members_out: list[dict[str, Any]] = []
        for elite in pool.iter_elites():
            text_id = self.put_text(elite.text)
            members_out.append(
                {
                    "text_id": text_id,
                    "ratings": {
                        metric: _rating_to_dict(rating)
                        for metric, rating in elite.ratings.items()
                    },
                    "age": int(elite.age),
                }
            )

        anchors_out: dict[str, Any] | None = None
        if anchor_manager is not None:
            anchor_pool = anchor_manager.pool
            items_out: list[dict[str, Any]] = []
            for anchor in anchor_pool.iter_anchors():
                text_id = self.put_text(anchor.text)
                items_out.append(
                    {
                        "text_id": text_id,
                        "ratings": {
                            metric: _rating_to_dict(rating)
                            for metric, rating in anchor.ratings.items()
                        },
                        "age": int(anchor.age),
                        "label": anchor.label,
                    }
                )
            anchors_out = {
                "seed_text_id": (
                    self.put_text(anchor_pool.seed_anchor.text)
                    if anchor_pool.seed_anchor
                    else None
                ),
                "items": items_out,
            }

        return {
            "schema": SCHEMA_VERSION,
            "saved_at": _utc_now_iso(),
            "next_iteration": int(iteration),
            "population": {
                "max_size": int(pool.max_size),
                "members": members_out,
            },
            "anchors": anchors_out,
        }

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _append_jsonl(self, path: Path, obj: Any) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
