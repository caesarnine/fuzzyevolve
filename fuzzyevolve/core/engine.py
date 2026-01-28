from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import numpy as np

from fuzzyevolve.config import Config
from fuzzyevolve.core.anchors import AnchorManager, AnchorPolicy
from fuzzyevolve.core.battle import build_battle
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Anchor, Elite, EvolutionResult, IterationSnapshot
from fuzzyevolve.core.models import MutationCandidate
from fuzzyevolve.core.multiobjective import Scalarizer
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.pool import cosine_distance
from fuzzyevolve.core.ports import Critic, Mutator, Ranker
from fuzzyevolve.core.ratings import RatingSystem

log_evo = logging.getLogger("evolution")


class Recorder(Protocol):
    def set_iteration(self, iteration: int) -> None: ...

    def put_text(self, text: str) -> str: ...

    def record_event(
        self, kind: str, data: Mapping[str, Any], *, iteration: int | None = None
    ) -> None: ...

    def record_stats(
        self,
        *,
        iteration: int,
        best_score: float,
        pool_size: int,
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...

    def save_checkpoint(
        self,
        *,
        iteration: int,
        pool: CrowdedPool,
        anchor_manager: AnchorManager | None,
        keep: bool,
    ) -> None: ...


class EvolutionEngine:
    def __init__(
        self,
        *,
        cfg: Config,
        pool: CrowdedPool,
        embed: Callable[[str], np.ndarray],
        rating: RatingSystem,
        selector: Callable[[CrowdedPool], Elite],
        critic: Critic | None,
        mutator: Mutator,
        ranker: Ranker,
        anchor_manager: AnchorManager | None,
        rng: random.Random,
        store: Recorder | None = None,
        scalarizer: Scalarizer | None = None,
    ) -> None:
        self.cfg = cfg
        self.pool = pool
        self.embed = embed
        self.rating = rating
        self.selector = selector
        self.critic = critic
        self.mutator = mutator
        self.ranker = ranker
        self.anchors = anchor_manager
        self.rng = rng
        self.store = store
        self.scalarizer = scalarizer

    def run(
        self,
        seed_text: str,
        *,
        on_iteration: Callable[[IterationSnapshot], None] | None = None,
    ) -> EvolutionResult:
        self._seed(seed_text)
        return self._run_loop(start_iteration=0, on_iteration=on_iteration)

    def resume(
        self,
        *,
        start_iteration: int,
        on_iteration: Callable[[IterationSnapshot], None] | None = None,
    ) -> EvolutionResult:
        if start_iteration < 0:
            raise ValueError("start_iteration must be >= 0.")
        return self._run_loop(
            start_iteration=start_iteration, on_iteration=on_iteration
        )

    def _run_loop(
        self,
        *,
        start_iteration: int,
        on_iteration: Callable[[IterationSnapshot], None] | None,
    ) -> EvolutionResult:
        mutation_executor: ThreadPoolExecutor | None = None
        if self.cfg.mutation.jobs_per_iteration > 1:
            max_workers = min(
                self.cfg.mutation.jobs_per_iteration,
                self.cfg.mutation.max_workers,
            )
            mutation_executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            end_iteration = start_iteration + self.cfg.run.iterations
            for iteration in range(start_iteration, end_iteration):
                if self.store:
                    self.store.set_iteration(iteration + 1)
                self.step(iteration, mutation_executor=mutation_executor)

                best = self.best_elite()
                snapshot = IterationSnapshot(
                    iteration=iteration + 1,
                    best_score=self.rating.score(best.ratings),
                    pool_size=len(self.pool),
                    best_elite=best,
                )
                if on_iteration:
                    on_iteration(snapshot)

                if self.anchors and self.anchors.maybe_add_ghost(
                    best, iteration=iteration + 1
                ):
                    log_evo.info("Added ghost anchor at iteration %d.", iteration + 1)

                if self.store:
                    try:
                        pool_elites = list(self.pool.iter_elites())
                        scores = np.array(
                            [self.rating.score(e.ratings) for e in pool_elites],
                            dtype=float,
                        )
                        extra: dict[str, Any] = {
                            "best_text_id": self.store.put_text(best.text),
                        }
                        if scores.size:
                            extra.update(
                                {
                                    "mean_score": float(scores.mean()),
                                    "p50_score": float(np.percentile(scores, 50)),
                                    "p90_score": float(np.percentile(scores, 90)),
                                    "min_score": float(scores.min()),
                                    "max_score": float(scores.max()),
                                    "std_score": float(scores.std()),
                                }
                            )

                        if len(pool_elites) >= 2:
                            embeddings = np.stack([e.embedding for e in pool_elites])
                            sims = embeddings @ embeddings.T
                            np.fill_diagonal(sims, -np.inf)
                            nn_sim = sims.max(axis=1)
                            nn_dist = 1.0 - nn_sim
                            extra.update(
                                {
                                    "diversity_nn_mean": float(nn_dist.mean()),
                                    "diversity_nn_p10": float(
                                        np.percentile(nn_dist, 10)
                                    ),
                                    "diversity_nn_p50": float(
                                        np.percentile(nn_dist, 50)
                                    ),
                                }
                            )

                        sigmas: list[float] = []
                        for elite in pool_elites:
                            for metric in self.cfg.metrics.names:
                                r = elite.ratings.get(metric)
                                if r is None:
                                    continue
                                sigmas.append(float(r.sigma))
                        if sigmas:
                            extra["mean_sigma"] = float(np.mean(sigmas))

                        self.store.record_stats(
                            iteration=snapshot.iteration,
                            best_score=snapshot.best_score,
                            pool_size=snapshot.pool_size,
                            extra=extra,
                        )
                        checkpoint_interval = getattr(
                            self.cfg.run, "checkpoint_interval", 1
                        )
                        keep = checkpoint_interval > 0 and (
                            snapshot.iteration % checkpoint_interval == 0
                        )
                        self.store.save_checkpoint(
                            iteration=snapshot.iteration,
                            pool=self.pool,
                            anchor_manager=self.anchors,
                            keep=keep,
                        )
                        self.store.record_event(
                            "iteration",
                            {
                                "best_text_id": self.store.put_text(best.text),
                                "best_score": snapshot.best_score,
                                "pool_size": snapshot.pool_size,
                            },
                            iteration=snapshot.iteration,
                        )
                    except Exception:
                        log_evo.exception("Failed to record iteration state.")

        finally:
            if mutation_executor is not None:
                mutation_executor.shutdown(wait=True)

        best = self.best_elite()
        return EvolutionResult(
            best_elite=best, best_score=self.rating.score(best.ratings)
        )

    def best_elite(self) -> Elite:
        return self.pool.best

    def _seed(self, seed_text: str) -> None:
        seed = Elite(
            text=seed_text,
            embedding=self.embed(seed_text),
            ratings=self.rating.new_ratings(),
            age=0,
        )
        self.pool.add(seed)

        if self.anchors:
            self.anchors.seed(seed_text)

    def step(
        self,
        iteration: int,
        *,
        mutation_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        scalarization: dict[str, float] | None = None
        scalarization_source: str | None = None
        if self.scalarizer is not None:
            scalarization = self.scalarizer.sample()
            scalarization_source = self.scalarizer.last_source

        parent = self.selector(self.pool)
        self.rating.ensure_ratings(parent)

        if self.store:
            try:
                parent_metrics = {
                    metric: {
                        "mu": float(parent.ratings[metric].mu),
                        "sigma": float(parent.ratings[metric].sigma),
                    }
                    for metric in self.cfg.metrics.names
                    if metric in parent.ratings
                }
                self.store.record_event(
                    "step_start",
                    {
                        "parent_text_id": self.store.put_text(parent.text),
                        "pool_size": len(self.pool),
                        "scalarization": scalarization,
                        "scalarization_source": scalarization_source,
                        "parent_age": int(parent.age),
                        "parent_score": float(self.rating.score(parent.ratings)),
                        "parent_ratings": parent_metrics,
                    },
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record step_start.")

        critique = None
        if self.critic:
            critique = self.critic.critique(parent=parent)
            if critique and self.store:
                try:
                    self.store.record_event(
                        "critique",
                        {
                            "summary": critique.summary,
                            "preserve": list(critique.preserve),
                            "issues": list(critique.issues),
                            "routes": list(critique.routes),
                            "constraints": list(critique.constraints),
                        },
                        iteration=iteration + 1,
                    )
                except Exception:
                    log_evo.exception("Failed to record critique.")

        candidates = self._propose_children(
            parent,
            critique=critique,
            mutation_executor=mutation_executor,
        )
        if not candidates:
            return

        if self.store:
            try:
                self.store.record_event(
                    "candidates",
                    {
                        "items": [
                            {
                                "text_id": self.store.put_text(c.text),
                                "operator": c.operator,
                                "uncertainty_scale": c.uncertainty_scale,
                                "focus": c.focus,
                                "partner_text_ids": [
                                    self.store.put_text(t) for t in c.partner_texts
                                ],
                            }
                            for c in candidates
                        ]
                    },
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record candidates.")

        children = self._make_children(parent, candidates, age=iteration)
        if not children:
            return

        if self.store:
            try:
                parent_text_id = self.store.put_text(parent.text)
                operator_roles = {
                    op.name: op.role
                    for op in getattr(self.cfg.mutation, "operators", [])
                    if getattr(op, "enabled", True)
                }
                edges = []
                for cand, child in zip(candidates, children):
                    child_text_id = self.store.put_text(child.text)
                    partner_text_ids = [
                        self.store.put_text(t) for t in cand.partner_texts
                    ]
                    edges.append(
                        {
                            "parent_text_id": parent_text_id,
                            "child_text_id": child_text_id,
                            "operator": cand.operator,
                            "role": operator_roles.get(cand.operator, ""),
                            "uncertainty_scale": float(cand.uncertainty_scale),
                            "focus": cand.focus,
                            "partner_text_ids": partner_text_ids,
                            "parent_score": float(self.rating.score(parent.ratings)),
                            "child_score_prior": float(
                                self.rating.score(child.ratings)
                            ),
                            "embedding_distance": float(
                                cosine_distance(parent.embedding, child.embedding)
                            ),
                        }
                    )
                self.store.record_event(
                    "lineage",
                    {"edges": edges},
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record lineage.")

        anchors = self._maybe_pick_anchors([parent, *children])
        opponent = self._maybe_pick_opponent(parent, [parent, *children, *anchors])

        battle = build_battle(
            parent=parent,
            children=children,
            anchors=anchors,
            opponent=opponent,
        )
        if battle.size < 2:
            return

        if self.store:
            try:
                child_set = {id(child) for child in battle.judged_children}
                anchor_set = {id(a) for a in anchors}
                opponent_id = id(opponent) if opponent is not None else None
                participants = []
                for idx, p in enumerate(battle.participants):
                    if isinstance(p, Anchor):
                        role = "anchor"
                    elif id(p) == id(parent):
                        role = "parent"
                    elif id(p) in child_set:
                        role = "child"
                    elif opponent_id is not None and id(p) == opponent_id:
                        role = "opponent"
                    else:
                        role = "elite"
                    participants.append(
                        {
                            "idx": idx,
                            "role": role,
                            "text_id": self.store.put_text(p.text),
                            "frozen": idx in battle.frozen_indices,
                            "is_anchor": id(p) in anchor_set,
                        }
                    )
                self.store.record_event(
                    "battle",
                    {"participants": participants},
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record battle.")

        ranking = self.ranker.rank(
            metrics=self.cfg.metrics.names,
            battle=battle,
            metric_descriptions=self.cfg.metrics.descriptions,
        )
        if ranking is None:
            raise RuntimeError(
                f"Ranker returned no ranking at iteration {iteration + 1}."
            )

        if self.store:
            try:
                self.store.record_event(
                    "ranking",
                    {"tiers_by_metric": ranking.tiers_by_metric},
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record ranking.")

        ratings_before: list[dict[str, dict[str, float]]] | None = None
        if self.store:
            try:
                for player in battle.participants:
                    self.rating.ensure_ratings(player)
                ratings_before = [
                    {
                        metric: {
                            "mu": float(player.ratings[metric].mu),
                            "sigma": float(player.ratings[metric].sigma),
                        }
                        for metric in self.cfg.metrics.names
                        if metric in player.ratings
                    }
                    for player in battle.participants
                ]
            except Exception:
                log_evo.exception("Failed to snapshot ratings (before).")
                ratings_before = None

        self.rating.apply_ranking(
            battle.participants,
            ranking,
            frozen_indices=set(battle.frozen_indices),
        )

        if self.store and ratings_before is not None:
            try:
                participants = []
                for idx, player in enumerate(battle.participants):
                    after = {
                        metric: {
                            "mu": float(player.ratings[metric].mu),
                            "sigma": float(player.ratings[metric].sigma),
                        }
                        for metric in self.cfg.metrics.names
                        if metric in player.ratings
                    }
                    participants.append(
                        {
                            "idx": idx,
                            "text_id": self.store.put_text(player.text),
                            "frozen": idx in battle.frozen_indices,
                            "before": ratings_before[idx],
                            "after": after,
                        }
                    )
                self.store.record_event(
                    "ratings_update",
                    {"participants": participants},
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record ratings_update.")

        # Absorb children into the fixed-size pool (crowding handles removal).
        pool_before: set[str] | None = None
        if self.store:
            try:
                pool_before = {e.text for e in self.pool.iter_elites()}
            except Exception:
                pool_before = None
        self.pool.add_many(battle.judged_children)

        if self.store and pool_before is not None:
            try:
                pool_after = {e.text for e in self.pool.iter_elites()}
                removed = sorted(pool_before - pool_after)
                inserted = [child.text for child in battle.judged_children]
                kept = [t for t in inserted if t in pool_after]
                self.store.record_event(
                    "pool_delta",
                    {
                        "inserted_child_text_ids": [
                            self.store.put_text(t) for t in inserted
                        ],
                        "kept_child_text_ids": [self.store.put_text(t) for t in kept],
                        "removed_text_ids": [self.store.put_text(t) for t in removed],
                        "pool_size": len(self.pool),
                    },
                    iteration=iteration + 1,
                )
            except Exception:
                log_evo.exception("Failed to record pool_delta.")

    def _propose_children(
        self,
        parent: Elite,
        *,
        critique: Critique | None,
        mutation_executor: ThreadPoolExecutor | None,
    ) -> list[MutationCandidate]:
        try:
            raw = self.mutator.propose(
                parent=parent,
                critique=critique,
                max_candidates=self.cfg.mutation.max_children,
                mutation_executor=mutation_executor,
            )
        except Exception:
            log_evo.exception("Mutation step failed; skipping iteration.")
            raw = []

        seen: set[str] = set()
        unique: list[MutationCandidate] = []
        for cand in raw:
            text = cand.text
            if text in seen:
                continue
            if self.pool.contains_text(text):
                continue
            seen.add(text)
            unique.append(cand)
        return unique

    def _make_children(
        self,
        parent: Elite,
        candidates: Sequence[MutationCandidate],
        *,
        age: int,
    ) -> list[Elite]:
        children: list[Elite] = []
        for cand in candidates:
            text = cand.text
            child = Elite(
                text=text,
                embedding=self.embed(text),
                ratings=self.rating.init_child_ratings(
                    parent, uncertainty_scale=cand.uncertainty_scale
                ),
                age=age,
            )
            children.append(child)
        return children

    def _maybe_pick_anchors(self, group: Sequence[Elite]) -> list[Anchor]:
        if not self.anchors:
            return []
        exclude = {e.text for e in group}
        return self.anchors.maybe_sample(exclude_texts=exclude)

    def _maybe_pick_opponent(
        self, parent: Elite, group: Sequence[Elite | Anchor]
    ) -> Elite | None:
        opponent_cfg = self.cfg.judging.opponent
        if opponent_cfg.kind == "none":
            return None
        if self.rng.random() >= opponent_cfg.probability:
            return None

        exclude_texts = {e.text for e in group}

        if opponent_cfg.kind == "random":
            candidates = [
                e for e in self.pool.iter_elites() if e.text not in exclude_texts
            ]
            if not candidates:
                return None
            return candidates[self.rng.randrange(len(candidates))]

        if opponent_cfg.kind == "farthest_from_parent":
            return self.pool.farthest_from(parent, exclude_texts=exclude_texts)

        if opponent_cfg.kind == "far_but_close":
            candidates = [
                e for e in self.pool.iter_elites() if e.text not in exclude_texts
            ]
            if not candidates:
                return None

            scored = [
                (cosine_distance(parent.embedding, e.embedding), e) for e in candidates
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            k = min(int(getattr(opponent_cfg, "farthest_k", 32)), len(scored))
            far = [e for _dist, e in scored[:k]]
            return max(far, key=lambda e: self.rating.match_quality(parent, e))

        raise ValueError(f"Unknown opponent kind '{opponent_cfg.kind}'.")


def build_anchor_manager(
    *,
    cfg: Config,
    rng: random.Random,
) -> AnchorManager | None:
    if cfg.anchors.injection_probability <= 0 and cfg.anchors.ghost_interval <= 0:
        return None
    from fuzzyevolve.core.anchors import AnchorPool

    pool = AnchorPool(cfg.metrics.names, rng=rng)
    policy = AnchorPolicy(
        injection_probability=cfg.anchors.injection_probability,
        max_per_battle=cfg.anchors.max_per_battle,
        ghost_interval=cfg.anchors.ghost_interval,
        seed_mu=cfg.anchors.seed_mu,
        seed_sigma=cfg.anchors.seed_sigma,
    )
    return AnchorManager(pool, policy, rng=rng)
