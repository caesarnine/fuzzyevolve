from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import default_text_descriptor
import trueskill as ts

from fuzzyevolve.core.models import Elite, EvolutionResult, IterationSnapshot
from fuzzyevolve.core.reference_pool import ReferencePool
from fuzzyevolve.core.scoring import score_ratings
from fuzzyevolve.core.selection import ParentSelector
from fuzzyevolve.core.stats import EvolutionStats
from fuzzyevolve.mutation.mutator import LLMMutator
from fuzzyevolve.core.judge import LLMJudge

log_evo = logging.getLogger("evolution")


class EvolutionEngine:
    def __init__(
        self,
        cfg: Config,
        mutator: LLMMutator,
        judge: LLMJudge,
        islands: Sequence[MapElitesArchive],
        reference_pool: ReferencePool | None = None,
        selector: ParentSelector | None = None,
        descriptor_fn: Callable[[str], dict[str, Any]] = default_text_descriptor,
        rng: random.Random | None = None,
        stats: EvolutionStats | None = None,
    ) -> None:
        self.cfg = cfg
        self.mutator = mutator
        self.judge = judge
        self.islands = list(islands)
        if not self.islands:
            raise ValueError("At least one island archive is required.")
        self.descriptor_fn = descriptor_fn
        self.rng = rng or random.Random()
        self.reference_pool = reference_pool or ReferencePool(cfg.metrics, rng=self.rng)
        self.selector = selector
        self.stats = stats

    def run(
        self,
        seed_text: str,
        on_iteration: Callable[[IterationSnapshot], None] | None = None,
    ) -> EvolutionResult:
        self._seed_islands(seed_text)

        for iteration in range(self.cfg.iterations):
            self._step(iteration)
            best_elite = self.best_elite()
            snapshot = IterationSnapshot(
                iteration=iteration + 1,
                best_score=score_ratings(best_elite.ratings, c=self.cfg.report_score_c),
                empty_cells=self.islands[0].empty_cells,
                best_elite=best_elite,
            )
            if on_iteration:
                on_iteration(snapshot)
            self._maybe_add_ghost_anchor(iteration + 1)

        best_final = self.best_elite()
        return EvolutionResult(
            best_elite=best_final,
            best_score=score_ratings(best_final.ratings, c=self.cfg.report_score_c),
        )

    def best_elite(self) -> Elite:
        all_elites = [
            elite for archive in self.islands for elite in archive.iter_elites()
        ]
        if not all_elites:
            raise ValueError("No elites available.")
        return max(
            all_elites,
            key=lambda elite: score_ratings(elite.ratings, c=self.cfg.report_score_c),
        )

    def _seed_islands(self, seed_text: str) -> None:
        seed_elite = Elite(
            text=seed_text,
            descriptor=self.descriptor_fn(seed_text),
            ratings=self.judge.new_ratings(),
            age=0,
        )
        for archive in self.islands:
            archive.add(seed_elite.clone())
        if self.reference_pool:
            self.reference_pool.add_seed_anchor(
                seed_text,
                descriptor_fn=self.descriptor_fn,
                mu=self.cfg.anchor_mu,
                sigma=self.cfg.anchor_sigma,
            )

    def _step(self, iteration: int) -> None:
        archive = self.rng.choice(self.islands)
        parent = (
            self.selector.select_parent(archive)
            if self.selector
            else archive.random_elite()
        )
        self.judge.ensure_ratings(parent)
        inspirations = self._pick_inspirations(archive, parent)

        mutation = self.mutator.propose(parent, inspirations)
        if not mutation.candidates:
            return

        children: list[Elite] = []
        for candidate in mutation.candidates:
            child = Elite(
                text=candidate.text,
                descriptor=self.descriptor_fn(candidate.text),
                ratings=self._init_child_ratings(parent),
                age=iteration,
            )
            children.append(child)
        children = self._downsample_children(children)
        if not children:
            return

        judge_inspiration = None
        if self.cfg.judge_include_inspirations and inspirations:
            judge_inspiration = self.rng.choice(inspirations)
        anchors = self._maybe_pick_anchors([parent] + children)
        opponent = self._maybe_pick_opponent(
            archive, parent, [parent] + children + anchors
        )
        group, judged_children = self._assemble_battle_group(
            parent, children, anchors, opponent, judge_inspiration
        )
        if len(group) < 2:
            return

        if self.stats:
            self.stats.record_battle_size(len(group))
            child_ids = {id(child) for child in judged_children}
            self.stats.children_judged += sum(
                1 for elite in group if id(elite) in child_ids
            )
            anchor_ids = {id(anchor) for anchor in anchors}
            if anchor_ids:
                self.stats.anchor_injected_total += sum(
                    1 for elite in group if id(elite) in anchor_ids
                )

        frozen_ids = {id(elite) for elite in group if elite.frozen}
        judge_success = self.judge.rank_and_rate(group, frozen=frozen_ids)
        if judge_success:
            archive.resort(parent)
            if opponent is not None:
                archive.resort(opponent)
            if judge_inspiration is not None:
                archive.resort(judge_inspiration)

            for child in judged_children:
                if self._passes_new_cell_gate(archive, parent, child):
                    archive.add(child)
                    if self.stats:
                        self.stats.children_inserted += 1
                elif self.stats:
                    self.stats.children_rejected_new_cell_gate += 1
        else:
            log_evo.warning(
                "Judge failed; skipping archive update for iteration %d.",
                iteration + 1,
            )

        if self.cfg.migration_interval > 0 and (
            (iteration + 1) % self.cfg.migration_interval == 0
        ):
            self._migrate()

        if self.cfg.sparring_interval > 0 and (
            (iteration + 1) % self.cfg.sparring_interval == 0
        ):
            self._spar()

    def _pick_inspirations(
        self, archive: MapElitesArchive, parent: Elite
    ) -> list[Elite]:
        if self.cfg.inspiration_count <= 0:
            return []
        candidates = [elite for elite in archive.iter_elites() if elite is not parent]
        if not candidates:
            return []
        return self.rng.sample(
            candidates, k=min(self.cfg.inspiration_count, len(candidates))
        )

    def _migrate(self) -> None:
        for idx, src in enumerate(self.islands):
            migrants = src.sample_elites(self.cfg.migration_size)
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

        log_evo.info("Global sparring with %d elites.", len(pool))
        pool = pool + self._maybe_pick_anchors(pool)
        frozen_ids = {id(elite) for elite in pool if elite.frozen}
        if not self.judge.rank_and_rate(pool, frozen=frozen_ids):
            log_evo.warning("Global sparring judge failed - skipping resort.")
            return
        for elite in pool:
            origin = elite_to_island.get(id(elite))
            if origin:
                origin.resort(elite)

    def _maybe_pick_anchors(self, group: list[Elite]) -> list[Elite]:
        if not self.reference_pool:
            return []
        if self.cfg.anchor_injection_prob <= 0:
            return []
        if self.rng.random() >= self.cfg.anchor_injection_prob:
            return []
        max_count = self.cfg.anchor_max_per_judgement
        if max_count <= 0:
            return []
        return self.reference_pool.sample(
            max_count,
            exclude_texts={elite.text for elite in group},
        )

    def _maybe_add_ghost_anchor(self, iteration: int) -> None:
        if not self.reference_pool:
            return
        if self.cfg.ghost_anchor_interval <= 0:
            return
        if iteration % self.cfg.ghost_anchor_interval != 0:
            return
        best = self.best_elite()
        self.reference_pool.add_ghost_anchor(best, age=iteration)
        log_evo.info("Added ghost anchor at iteration %d.", iteration)

    def _init_child_ratings(self, parent: Elite) -> dict[str, ts.Rating]:
        if self.cfg.child_prior_mode != "inherit":
            return self.judge.new_ratings()
        ratings: dict[str, ts.Rating] = {}
        for metric, rating in parent.ratings.items():
            sigma = math.sqrt(rating.sigma * rating.sigma + self.cfg.child_prior_tau**2)
            ratings[metric] = ts.Rating(mu=rating.mu, sigma=sigma)
        return ratings

    def _downsample_children(self, children: list[Elite]) -> list[Elite]:
        if len(children) <= self.cfg.max_children_judged:
            return children
        return self.rng.sample(children, k=self.cfg.max_children_judged)

    def _maybe_pick_opponent(
        self, archive: MapElitesArchive, parent: Elite, group: list[Elite]
    ) -> Elite | None:
        if self.cfg.judge_opponent_mode == "none":
            return None
        if self.rng.random() >= self.cfg.judge_opponent_p:
            return None
        exclude = set(group)

        if self.cfg.judge_opponent_mode == "cell_champion":
            if parent.cell_key is None:
                return None
            bucket = next(
                (
                    bucket
                    for key, bucket in archive.iter_cells()
                    if key == parent.cell_key
                ),
                None,
            )
            if not bucket:
                return None
            for elite in bucket:
                if elite not in exclude:
                    return elite
            return None

        if self.cfg.judge_opponent_mode == "global_top_sample":
            candidates = [
                elite for elite in archive.iter_elites() if elite not in exclude
            ]
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda elite: score_ratings(
                    elite.ratings, c=self.cfg.archive_score_c
                ),
            )

        raise ValueError(
            f"Unknown judge_opponent_mode '{self.cfg.judge_opponent_mode}'."
        )

    def _assemble_battle_group(
        self,
        parent: Elite,
        children: list[Elite],
        anchors: list[Elite],
        opponent: Elite | None,
        inspiration: Elite | None,
    ) -> tuple[list[Elite], list[Elite]]:
        group_children = list(children)
        extras: list[Elite] = []
        max_size = self.cfg.max_battle_size

        if len(group_children) > max_size - 1:
            group_children = self.rng.sample(group_children, k=max_size - 1)

        available = max_size - 1 - len(group_children)
        if anchors:
            for anchor in anchors:
                if available <= 0 and group_children:
                    drop_idx = self.rng.randrange(len(group_children))
                    group_children.pop(drop_idx)
                    available += 1
                if available <= 0:
                    break
                extras.append(anchor)
                available -= 1

        if opponent is not None and available > 0:
            extras.append(opponent)
            available -= 1

        if inspiration is not None and inspiration not in extras and available > 0:
            extras.append(inspiration)

        return [parent] + group_children + extras, group_children

    def _passes_new_cell_gate(
        self, archive: MapElitesArchive, parent: Elite, child: Elite
    ) -> bool:
        if self.cfg.new_cell_gate_mode == "none":
            return True
        if not archive.is_new_cell(child.descriptor):
            return True
        parent_score = score_ratings(parent.ratings, c=self.cfg.report_score_c)
        child_score = score_ratings(child.ratings, c=self.cfg.report_score_c)
        if self.cfg.new_cell_gate_mode == "parent_lcb":
            return child_score >= parent_score + self.cfg.new_cell_gate_delta
        raise ValueError(f"Unknown new_cell_gate_mode '{self.cfg.new_cell_gate_mode}'.")
