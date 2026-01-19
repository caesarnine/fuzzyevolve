from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from fuzzyevolve.config import Config
from fuzzyevolve.core.anchors import AnchorManager, AnchorPolicy
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.battle import Battle, build_battle
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Anchor, Elite, EvolutionResult, IterationSnapshot
from fuzzyevolve.core.models import MutationCandidate
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
        self, *, iteration: int, best_score: float, empty_cells: int
    ) -> None: ...

    def save_checkpoint(
        self,
        *,
        iteration: int,
        islands: Sequence[MapElitesArchive],
        anchor_manager: AnchorManager | None,
        keep: bool,
    ) -> None: ...


class EvolutionEngine:
    def __init__(
        self,
        *,
        cfg: Config,
        islands: Sequence[MapElitesArchive],
        describe: Callable[[str], dict],
        rating: RatingSystem,
        selector: Callable[[MapElitesArchive], Elite],
        critic: Critic | None,
        mutator: Mutator,
        ranker: Ranker,
        anchor_manager: AnchorManager | None,
        rng: random.Random,
        store: Recorder | None = None,
    ) -> None:
        self.cfg = cfg
        self.islands = list(islands)
        if not self.islands:
            raise ValueError("At least one island archive is required.")
        self.describe = describe
        self.rating = rating
        self.selector = selector
        self.critic = critic
        self.mutator = mutator
        self.ranker = ranker
        self.anchors = anchor_manager
        self.rng = rng
        self.store = store

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
        return self._run_loop(start_iteration=start_iteration, on_iteration=on_iteration)

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
                    empty_cells=self.islands[0].empty_cells,
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
                        self.store.record_stats(
                            iteration=snapshot.iteration,
                            best_score=snapshot.best_score,
                            empty_cells=snapshot.empty_cells,
                        )
                        checkpoint_interval = getattr(
                            self.cfg.run, "checkpoint_interval", 1
                        )
                        keep = checkpoint_interval > 0 and (
                            snapshot.iteration % checkpoint_interval == 0
                        )
                        self.store.save_checkpoint(
                            iteration=snapshot.iteration,
                            islands=self.islands,
                            anchor_manager=self.anchors,
                            keep=keep,
                        )
                        self.store.record_event(
                            "iteration",
                            {
                                "best_text_id": self.store.put_text(best.text),
                                "best_score": snapshot.best_score,
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
        all_elites = [
            elite for archive in self.islands for elite in archive.iter_elites()
        ]
        if not all_elites:
            raise ValueError("No elites available.")
        return max(all_elites, key=lambda e: self.rating.score(e.ratings))

    def _seed(self, seed_text: str) -> None:
        seed = Elite(
            text=seed_text,
            descriptor=self.describe(seed_text),
            ratings=self.rating.new_ratings(),
            age=0,
        )
        for archive in self.islands:
            archive.add(seed.clone())

        if self.anchors:
            self.anchors.seed(seed_text, descriptor_fn=self.describe)

    def step(
        self,
        iteration: int,
        *,
        mutation_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        island_idx = self.rng.randrange(len(self.islands))
        archive = self.islands[island_idx]
        parent = self.selector(archive)
        self.rating.ensure_ratings(parent)
        if self.store:
            try:
                self.store.record_event(
                    "step_start",
                    {
                        "island": island_idx,
                        "parent_text_id": self.store.put_text(parent.text),
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
            archive,
            parent,
            critique=critique,
            mutation_executor=mutation_executor,
        )
        if not candidates:
            self._maintenance(iteration)
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
            self._maintenance(iteration)
            return

        anchors = self._maybe_pick_anchors([parent, *children])
        opponent = self._maybe_pick_opponent(
            archive, parent, [parent, *children, *anchors]
        )

        battle = build_battle(
            parent=parent,
            children=children,
            anchors=anchors,
            opponent=opponent,
        )
        if battle.size < 2:
            self._maintenance(iteration)
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

        self.rating.apply_ranking(
            battle.participants,
            ranking,
            frozen_indices=set(battle.frozen_indices),
        )
        for elite in battle.resort_elites:
            archive.resort(elite)

        for child in battle.judged_children:
            if self._passes_new_cell_gate(archive, parent, child):
                archive.add(child)
                if self.store:
                    try:
                        self.store.record_event(
                            "archive_add",
                            {"text_id": self.store.put_text(child.text)},
                            iteration=iteration + 1,
                        )
                    except Exception:
                        log_evo.exception("Failed to record archive_add.")

        self._maintenance(iteration)

    def _propose_children(
        self,
        archive: MapElitesArchive,
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
            if archive.contains_text(text):
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
                descriptor=self.describe(text),
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
        self,
        archive: MapElitesArchive,
        parent: Elite,
        group: Sequence[Elite | Anchor],
    ) -> Elite | None:
        opponent_cfg = self.cfg.judging.opponent
        if opponent_cfg.kind == "none":
            return None
        if self.rng.random() >= opponent_cfg.probability:
            return None

        exclude_texts = {e.text for e in group}

        if opponent_cfg.kind == "cell_champion":
            parent_key = archive.space.cell_key(parent.descriptor)
            bucket = next(
                (bucket for key, bucket in archive.iter_cells() if key == parent_key),
                None,
            )
            if not bucket:
                return None
            for elite in bucket:
                if elite.text not in exclude_texts:
                    return elite
            return None

        if opponent_cfg.kind == "global_best":
            candidates = [
                e for e in archive.iter_elites() if e.text not in exclude_texts
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda e: self.rating.score(e.ratings))

        if opponent_cfg.kind == "topk_other_cell_champion":
            top_k = int(opponent_cfg.top_k)
            parent_key = archive.space.cell_key(parent.descriptor)

            champions: list[tuple[float, Elite]] = []
            for cell_key, bucket in archive.iter_cells():
                if not bucket or cell_key == parent_key:
                    continue
                best: Elite | None = None
                best_score: float | None = None
                for elite in bucket:
                    if elite.text in exclude_texts:
                        continue
                    score = self.rating.score(elite.ratings)
                    if best is None or best_score is None or score > best_score:
                        best = elite
                        best_score = score
                if best is not None and best_score is not None:
                    champions.append((best_score, best))

            if not champions:
                return None

            champions.sort(key=lambda item: item[0], reverse=True)
            pool = champions if top_k <= 0 else champions[: min(top_k, len(champions))]
            return pool[self.rng.randrange(len(pool))][1]

        raise ValueError(f"Unknown opponent kind '{opponent_cfg.kind}'.")

    def _passes_new_cell_gate(
        self, archive: MapElitesArchive, parent: Elite, child: Elite
    ) -> bool:
        gate = self.cfg.new_cell_gate
        if gate.kind == "none":
            return True
        if not archive.is_new_cell(child.descriptor):
            return True
        if gate.kind == "parent_lcb":
            return (
                self.rating.score(child.ratings)
                >= self.rating.score(parent.ratings) + gate.delta
            )
        raise ValueError(f"Unknown new cell gate kind '{gate.kind}'.")

    def _maintenance(self, iteration: int) -> None:
        mig = self.cfg.maintenance.migration
        if mig.interval > 0 and (iteration + 1) % mig.interval == 0:
            self._migrate(mig.size)

        spar = self.cfg.maintenance.sparring
        if spar.interval > 0 and (iteration + 1) % spar.interval == 0:
            self._spar()

    def _migrate(self, size: int) -> None:
        if len(self.islands) <= 1 or size <= 0:
            return
        for idx, src in enumerate(self.islands):
            migrants = src.sample_elites(size)
            dst = self.islands[(idx + 1) % len(self.islands)]
            for elite in migrants:
                dst.add(elite.clone())

    def _spar(self) -> None:
        pool: list[Elite] = []
        elite_to_island: dict[int, MapElitesArchive] = {}
        for archive in self.islands:
            for elite in archive.sample_one_per_cell():
                pool.append(elite)
                elite_to_island[id(elite)] = archive

        if len(pool) <= 1:
            return

        anchors = self._maybe_pick_anchors(pool)
        battle = Battle(
            participants=tuple([*pool, *anchors]),
            judged_children=tuple(),
            resort_elites=tuple(pool),
            frozen_indices=frozenset(
                idx
                for idx, player in enumerate([*pool, *anchors])
                if isinstance(player, Anchor)
            ),
        )
        ranking = self.ranker.rank(
            metrics=self.cfg.metrics.names,
            battle=battle,
            metric_descriptions=self.cfg.metrics.descriptions,
        )

        self.rating.apply_ranking(
            battle.participants,
            ranking,
            frozen_indices=set(battle.frozen_indices),
        )
        for elite in pool:
            origin = elite_to_island.get(id(elite))
            if origin:
                origin.resort(elite)


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
