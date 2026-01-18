from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from fuzzyevolve.config import Config
from fuzzyevolve.core.anchors import AnchorManager, AnchorPolicy
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.battle import Battle, build_battle
from fuzzyevolve.core.inspirations import InspirationPicker
from fuzzyevolve.core.models import Anchor, Elite, EvolutionResult, IterationSnapshot
from fuzzyevolve.core.ports import Mutator, Ranker
from fuzzyevolve.core.ratings import RatingSystem

log_evo = logging.getLogger("evolution")


class EvolutionEngine:
    def __init__(
        self,
        *,
        cfg: Config,
        islands: Sequence[MapElitesArchive],
        describe: Callable[[str], dict],
        rating: RatingSystem,
        selector: Callable[[MapElitesArchive], Elite],
        inspirations: InspirationPicker,
        mutator: Mutator,
        ranker: Ranker,
        anchor_manager: AnchorManager | None,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.islands = list(islands)
        if not self.islands:
            raise ValueError("At least one island archive is required.")
        self.describe = describe
        self.rating = rating
        self.selector = selector
        self.inspirations = inspirations
        self.mutator = mutator
        self.ranker = ranker
        self.anchors = anchor_manager
        self.rng = rng

    def run(
        self,
        seed_text: str,
        *,
        on_iteration: Callable[[IterationSnapshot], None] | None = None,
    ) -> EvolutionResult:
        self._seed(seed_text)

        mutation_executor: ThreadPoolExecutor | None = None
        if self.cfg.mutation.calls_per_iteration > 1:
            max_workers = min(
                self.cfg.mutation.calls_per_iteration,
                self.cfg.mutation.max_workers,
            )
            mutation_executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
            for iteration in range(self.cfg.run.iterations):
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
        archive = self.rng.choice(self.islands)
        parent = self.selector(archive)
        self.rating.ensure_ratings(parent)

        inspiration_picks = self.inspirations.pick(
            archive,
            parent,
            count=self.cfg.mutation.inspiration_count,
        )
        inspirations = [pick.elite for pick in inspiration_picks]
        inspiration_labels = [pick.label for pick in inspiration_picks]

        candidates = self._propose_children(
            archive,
            parent,
            inspirations,
            inspiration_labels=inspiration_labels,
            mutation_executor=mutation_executor,
        )
        if not candidates:
            self._maintenance(iteration)
            return

        children = self._make_children(parent, candidates, age=iteration)
        if not children:
            self._maintenance(iteration)
            return

        judge_inspiration = None
        if self.cfg.judging.include_inspiration and inspirations:
            judge_inspiration = self.rng.choice(inspirations)

        anchors = self._maybe_pick_anchors([parent, *children])
        opponent = self._maybe_pick_opponent(
            archive, parent, [parent, *children, *anchors]
        )

        battle = build_battle(
            parent=parent,
            children=children,
            anchors=anchors,
            opponent=opponent,
            inspiration=judge_inspiration,
            max_battle_size=self.cfg.judging.max_battle_size,
            rng=self.rng,
        )
        if battle.size < 2:
            self._maintenance(iteration)
            return

        ranking = self.ranker.rank(
            metrics=self.cfg.metrics.names,
            battle=battle,
            metric_descriptions=self.cfg.metrics.descriptions,
        )
        if ranking is None:
            log_evo.warning(
                "Judge failed; skipping archive update for iteration %d.", iteration + 1
            )
            self._maintenance(iteration)
            return

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

        self._maintenance(iteration)

    def _propose_children(
        self,
        archive: MapElitesArchive,
        parent: Elite,
        inspirations: Sequence[Elite],
        *,
        inspiration_labels: Sequence[str],
        mutation_executor: ThreadPoolExecutor | None,
    ) -> list[str]:
        candidate_texts: list[str] = []
        call_count = self.cfg.mutation.calls_per_iteration

        def call_mutator() -> Sequence[str]:
            out = self.mutator.propose(
                parent=parent,
                inspirations=inspirations,
                inspiration_labels=inspiration_labels or None,
            )
            return [c.text for c in out]

        if call_count <= 1 or mutation_executor is None:
            batches = [call_mutator()]
        else:
            futures = [
                mutation_executor.submit(call_mutator) for _ in range(call_count)
            ]
            batches = []
            for fut in as_completed(futures):
                try:
                    batches.append(fut.result())
                except Exception:
                    log_evo.exception("Mutation call failed; skipping.")

        seen = set()
        for batch in batches:
            for text in batch:
                if text in seen:
                    continue
                if archive.contains_text(text):
                    continue
                seen.add(text)
                candidate_texts.append(text)

        if len(candidate_texts) > self.cfg.mutation.max_children:
            candidate_texts = self.rng.sample(
                candidate_texts, k=self.cfg.mutation.max_children
            )
        return candidate_texts

    def _make_children(
        self, parent: Elite, texts: Sequence[str], *, age: int
    ) -> list[Elite]:
        children: list[Elite] = []
        for text in texts:
            child = Elite(
                text=text,
                descriptor=self.describe(text),
                ratings=self.rating.init_child_ratings(parent),
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
        if ranking is None:
            log_evo.warning("Global sparring judge failed; skipping resort.")
            return

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
