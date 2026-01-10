from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from typing import Any

from fuzzyevolve.config import Config
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import default_text_descriptor
from fuzzyevolve.core.models import Elite, EvolutionResult, IterationSnapshot
from fuzzyevolve.core.scoring import score_ratings
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
        descriptor_fn: Callable[[str], dict[str, Any]] = default_text_descriptor,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.mutator = mutator
        self.judge = judge
        self.islands = list(islands)
        if not self.islands:
            raise ValueError("At least one island archive is required.")
        self.descriptor_fn = descriptor_fn
        self.rng = rng or random.Random()

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
                best_score=score_ratings(best_elite.ratings),
                empty_cells=self.islands[0].empty_cells,
                best_elite=best_elite,
            )
            if on_iteration:
                on_iteration(snapshot)

        best_final = self.best_elite()
        return EvolutionResult(
            best_elite=best_final, best_score=score_ratings(best_final.ratings)
        )

    def best_elite(self) -> Elite:
        return max(
            (archive.best for archive in self.islands),
            key=lambda elite: score_ratings(elite.ratings),
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

    def _step(self, iteration: int) -> None:
        archive = self.rng.choice(self.islands)
        parent = archive.random_elite()
        inspirations = self._pick_inspirations(archive, parent)

        mutation = self.mutator.propose(parent, inspirations)
        if not mutation.candidates:
            return

        children: list[Elite] = []
        for candidate in mutation.candidates:
            child = Elite(
                text=candidate.text,
                descriptor=self.descriptor_fn(candidate.text),
                ratings=self.judge.new_ratings(),
                age=iteration,
            )
            children.append(child)

        opponents = inspirations if self.cfg.judge_include_inspirations else []
        group = [parent] + opponents + children
        self.judge.rank_and_rate(group)

        for elite in [parent] + opponents:
            archive.resort(elite)

        for child in children:
            archive.add(child)

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
        self.judge.rank_and_rate(pool)
        for elite in pool:
            origin = elite_to_island.get(id(elite))
            if origin:
                origin.resort(elite)
