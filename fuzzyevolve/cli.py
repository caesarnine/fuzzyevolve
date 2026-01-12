"""Command-line interface for fuzzyevolve."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from fuzzyevolve.config import load_cfg
from fuzzyevolve.console.logging import setup_logging
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptors import build_descriptor_space
from fuzzyevolve.core.descriptors_semantic import build_descriptor_fn
from fuzzyevolve.core.engine import EvolutionEngine
from fuzzyevolve.core.judge import LLMJudge
from fuzzyevolve.core.reference_pool import ReferencePool
from fuzzyevolve.core.scoring import score_ratings
from fuzzyevolve.core.selection import ParentSelector
from fuzzyevolve.mutation.mutator import LLMMutator
from fuzzyevolve.mutation.patcher import PatchConfig

app = typer.Typer()


def _parse_log_level(value: str) -> int:
    text = value.strip()
    if text.isdigit():
        return int(text)
    level = getattr(logging, text.upper(), None)
    if isinstance(level, int):
        return level
    raise typer.BadParameter(
        f"Invalid log level '{value}'. Use debug, info, warning, error, critical, or a number."
    )


@app.command()
def cli(
    input: Optional[str] = typer.Argument(
        None, help="Initial text to evolve. Can be a string or a file path."
    ),
    config: Optional[Path] = typer.Option(
        None, "-c", "--config", help="Path to a TOML or JSON config file."
    ),
    output: Path = typer.Option(
        Path("best.txt"), "-o", "--output", help="Path to save the best final result."
    ),
    iterations: Optional[int] = typer.Option(
        None, "-i", "--iterations", help="Number of evolution iterations."
    ),
    goal: Optional[str] = typer.Option(
        None, "-g", "--goal", help="The high-level goal for the mutation prompt."
    ),
    metric: Optional[List[str]] = typer.Option(
        None,
        "-m",
        "--metric",
        help="A metric to evaluate against. Can be specified multiple times.",
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="The LLM to use for judging candidates."
    ),
    log_level: str = typer.Option(
        "info",
        "-l",
        "--log-level",
        help="Logging level (debug, info, warning, error, critical) or a number.",
    ),
    log_file: Optional[Path] = typer.Option(None, help="Path to write detailed logs."),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="Suppress the progress bar and non-essential logging.",
    ),
):
    """Evolve text with a language model."""
    cfg = load_cfg(config)

    if iterations is not None:
        cfg.iterations = iterations
    if goal is not None:
        cfg.mutation_prompt_goal = goal
    if metric:
        cfg.metrics = metric
    if judge_model is not None:
        cfg.judge_model = judge_model

    seed_text = _read_seed_text(input)

    setup_logging(level=_parse_log_level(log_level), quiet=quiet, log_file=log_file)

    seed = cfg.random_seed
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        logging.info("Generated random seed: %d", seed)
    master_rng = random.Random(seed)
    rng_engine = random.Random(master_rng.randrange(2**32))
    rng_selection = random.Random(master_rng.randrange(2**32))
    rng_judge = random.Random(master_rng.randrange(2**32))
    rng_models = random.Random(master_rng.randrange(2**32))
    rng_reference = random.Random(master_rng.randrange(2**32))
    archive_rngs = [
        random.Random(master_rng.randrange(2**32)) for _ in range(cfg.island_count)
    ]

    axes_spec = dict(cfg.axes)
    if cfg.descriptor_mode == "semantic":
        axes_spec.setdefault("semantic_x", {"bins": cfg.semantic_bins_x})
        axes_spec.setdefault("semantic_y", {"bins": cfg.semantic_bins_y})
    space = build_descriptor_space(axes_spec)

    def score_fn(ratings):
        return score_ratings(ratings, c=cfg.archive_score_c)

    islands = [
        MapElitesArchive(
            space, cfg.elites_per_cell, rng=archive_rngs[idx], score_fn=score_fn
        )
        for idx in range(cfg.island_count)
    ]

    judge = LLMJudge(
        cfg.judge_model,
        cfg.metrics,
        rng=rng_judge,
        max_attempts=cfg.judge_max_attempts,
        repair_enabled=cfg.judge_repair_enabled,
    )
    patch_cfg = PatchConfig(
        fuzzy_enabled=cfg.fuzzy_patch_enabled,
        threshold=cfg.fuzzy_patch_threshold,
        margin=cfg.fuzzy_patch_margin,
        min_search_len=cfg.fuzzy_patch_min_search_len,
        max_window_expansion=cfg.fuzzy_patch_max_window_expansion,
    )
    mutator = LLMMutator(
        cfg.llm_ensemble,
        cfg.mutation_prompt_goal,
        cfg.mutation_prompt_instructions,
        cfg.max_diffs,
        patch_cfg=patch_cfg,
        show_metric_stats=cfg.mutation_prompt_show_metric_stats,
        metric_c=cfg.mutation_prompt_c,
        rng=rng_models,
    )
    reference_pool = ReferencePool(cfg.metrics, rng=rng_reference)
    selector = ParentSelector(
        cfg.selection_mode,
        cfg.selection_beta,
        cfg.selection_temp,
        rng=rng_selection,
    )
    descriptor_fn = build_descriptor_fn(cfg)

    engine = EvolutionEngine(
        cfg,
        mutator,
        judge,
        islands,
        reference_pool=reference_pool,
        selector=selector,
        descriptor_fn=descriptor_fn,
        rng=rng_engine,
    )

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
            total=cfg.iterations,
            best=0.0,
            empty=islands[0].empty_cells,
        )

        def on_iteration(snapshot):
            if cfg.log_interval and snapshot.iteration % cfg.log_interval == 0:
                metric_parts = [
                    f"{metric}(μ={rating.mu:.2f}, σ={rating.sigma:.2f})"
                    for metric, rating in snapshot.best_elite.ratings.items()
                ]
                logging.info(
                    "it %d best_score %.3f | %s",
                    snapshot.iteration,
                    snapshot.best_score,
                    " | ".join(metric_parts),
                )
            progress.update(
                task,
                advance=1,
                best=snapshot.best_score,
                empty=snapshot.empty_cells,
            )

        result = engine.run(seed_text, on_iteration=on_iteration)

    output.write_text(result.best_elite.text)
    logging.info("DONE – best saved to %s (score %.3f)", output, result.best_score)


def _read_seed_text(user_input: str | None) -> str:
    if user_input is None:
        if not sys.stdin.isatty():
            seed_text = sys.stdin.read()
        else:
            typer.echo(
                "Error: No input provided. Please provide an input string, file path, or pipe data via stdin."
            )
            raise typer.Exit(code=1)
    elif Path(user_input).is_file():
        seed_text = Path(user_input).read_text()
    else:
        seed_text = user_input

    if not seed_text.strip():
        typer.echo("Error: Input text is empty.")
        raise typer.Exit(code=1)
    return seed_text


if __name__ == "__main__":
    app()
