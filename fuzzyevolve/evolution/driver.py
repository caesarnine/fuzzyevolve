"""
This module contains the EvolutionaryDriver class, which orchestrates the main evolutionary loop.
"""

import copy
import logging
import random
from pathlib import Path

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from fuzzyevolve.config import Config
from fuzzyevolve.evolution.archive import MixedArchive
from fuzzyevolve.evolution.judge import MultiMetricJudge
from fuzzyevolve.evolution.scoring import ts_score
from fuzzyevolve.llm.client import LLMProvider
from fuzzyevolve.llm.parsers import parse_llm_mutation_response
from fuzzyevolve.llm.prompts import build_mut_prompt
from fuzzyevolve.utils.diff import apply_patch, split_blocks

log_mut = logging.getLogger("mutation")


class EvolutionaryDriver:
    """Orchestrates the main evolutionary loop."""

    def __init__(
        self,
        cfg: Config,
        llm_provider: LLMProvider,
        judge: MultiMetricJudge,
        islands: list[MixedArchive],
    ):
        self.cfg = cfg
        self.llm_provider = llm_provider
        self.judge = judge
        self.islands = islands

    def run(self, seed_text: str, output_path: Path, quiet: bool):
        """Runs the main evolutionary loop."""
        # seed
        seed_desc = {"lang": "txt", "len": len(seed_text)}
        seed_elite = {
            "txt": seed_text,
            "rating": {m: self.judge.envs[m].create_rating() for m in self.cfg.metrics},
            "age": 0,
        }
        for arc in self.islands:
            arc.add(seed_desc, copy.deepcopy(seed_elite))

        # ─ Rich progress bar setup ─────────────────────────────────────
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• best {task.fields[best]:.3f}"),
            TextColumn("• empty {task.fields[empty]}"),
            TimeElapsedColumn(),
            transient=quiet,
        )

        with progress:
            task = progress.add_task(
                "evolving",
                total=self.cfg.iterations,
                best=0.0,
                empty=self.islands[0].empty_cells,
            )

            # ─ main loop ────────────────────────────────────────────────
            for it in range(self.cfg.iterations):
                isl_idx = random.randrange(self.cfg.num_islands)
                arc = self.islands[isl_idx]

                parent = arc.random_elite(self.cfg.youth_bias)
                cand = [e for b in arc.cell.values() for e in b if e is not parent]
                inspirations = random.sample(cand, k=min(3, len(cand))) if cand else []

                prompt = build_mut_prompt(
                    parent,
                    inspirations,
                    self.cfg.mutation_prompt_goal,
                    self.cfg.mutation_prompt_instructions,
                )
                reply = self.llm_provider.call(prompt)

                thinking, diff_content = parse_llm_mutation_response(reply, log_mut)

                if thinking:
                    log_mut.info("Mutator rationale (iter %d):\n%s", it, thinking)

                if not diff_content:
                    log_mut.warning(
                        "No diff content found in LLM reply, skipping iter %d.", it
                    )
                    continue

                log_mut.debug("Diff block (iter %d):\n%s", it, diff_content)

                for blk in split_blocks(diff_content)[: self.cfg.n_diffs]:
                    child_txt = apply_patch(parent["txt"], blk)
                    if child_txt == parent["txt"]:
                        continue
                    child = {
                        "txt": child_txt,
                        "rating": {
                            m: self.judge.envs[m].create_rating()
                            for m in self.cfg.metrics
                        },
                        "age": it,
                    }
                    desc_child = {"lang": "txt", "len": len(child_txt)}
                    child["cell_key"] = arc._key(desc_child)

                    group = [parent] + inspirations + [child]
                    self.judge.rank_and_rate(group)
                    for e in group:
                        arc.resort_elite(e)

                    arc.add(desc_child, child)

                if (it + 1) % self.cfg.migration_every == 0:
                    for idx, src in enumerate(self.islands):
                        migrants = random.sample(
                            [e for b in src.cell.values() for e in b],
                            k=min(
                                self.cfg.migrants_per_island,
                                sum(len(b) for b in src.cell.values()),
                            ),
                        )
                        dst = self.islands[(idx + 1) % self.cfg.num_islands]
                        for e in migrants:
                            desc = {"lang": "txt", "len": len(e["txt"])}
                            dst.add(desc, copy.deepcopy(e))

                if (it + 1) % self.cfg.sparring_every == 0:
                    elite_to_island_map = {}
                    pool = []
                    for isl_idx, sparring_arc in enumerate(self.islands):
                        for cell_key_in_arc in list(sparring_arc.cell.keys()):
                            bucket = sparring_arc.cell.get(cell_key_in_arc)
                            if bucket:
                                chosen_elite = random.choice(bucket)
                                pool.append(chosen_elite)
                                elite_to_island_map[id(chosen_elite)] = sparring_arc

                    if len(pool) > 1:
                        logging.info("Global sparring with %d elites.", len(pool))
                        self.judge.rank_and_rate(pool)
                        for e_updated in pool:
                            original_island = elite_to_island_map.get(id(e_updated))
                            if original_island:
                                original_island.resort_elite(e_updated)

                best_global = max(
                    (arc.best for arc in self.islands),
                    key=lambda e: ts_score(e["rating"]),
                )
                best_score = ts_score(best_global["rating"])
                if (it + 1) % self.cfg.log_every == 0:
                    metric_parts = [
                        f'{m}(μ={r.mu:.2f}, σ={r.sigma:.2f})'
                        for m, r in best_global["rating"].items()
                    ]
                    logging.info(
                        "it %d best_score %.3f | %s",
                        it + 1,
                        best_score,
                        " | ".join(metric_parts),
                    )

                progress.update(
                    task,
                    advance=1,
                    best=best_score,
                    empty=self.islands[0].empty_cells,
                )

        best_final = max(
            (arc.best for arc in self.islands), key=lambda e: ts_score(e["rating"])
        )
        output_path.write_text(best_final["txt"])
        logging.info(
            "DONE – best saved to %s (score %.3f)",
            output_path,
            ts_score(best_final["rating"]),
        )
