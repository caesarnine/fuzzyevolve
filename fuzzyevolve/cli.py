"""Command-line interface for fuzzyevolve."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from fuzzyevolve.adapters.llm.mutator import LLMMutator
from fuzzyevolve.adapters.llm.ranker import LLMRanker
from fuzzyevolve.config import load_config
from fuzzyevolve.console.logging import setup_logging
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptor_system import build_descriptor_system
from fuzzyevolve.core.engine import EvolutionEngine, build_anchor_manager
from fuzzyevolve.core.inspirations import InspirationPicker
from fuzzyevolve.core.ratings import RatingSystem
from fuzzyevolve.core.selection import ParentSelector

app = typer.Typer(add_completion=False, no_args_is_help=False)

_DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("config.toml", "config.json")


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


def _resolve_config_path(
    config: Path | None, *, cwd: Path | None = None
) -> tuple[Path | None, str]:
    if config is not None:
        if not config.is_file():
            raise typer.BadParameter(f"Config file not found: {config}")
        return config, f"Using config file: {config}"

    cwd = cwd or Path.cwd()
    for filename in _DEFAULT_CONFIG_FILENAMES:
        candidate = cwd / filename
        if candidate.is_file():
            return candidate, f"Using config file from CWD: {candidate}"

    return None, "Using built-in default config (no config file found)."


@app.callback(invoke_without_command=True)
def main(
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
        None, "-i", "--iterations", help="Override iterations from config."
    ),
    goal: Optional[str] = typer.Option(
        None, "-g", "--goal", help="Override the mutation goal from config."
    ),
    metric: Optional[list[str]] = typer.Option(
        None,
        "-m",
        "--metric",
        help="Override metrics (can be specified multiple times).",
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
) -> None:
    """Evolve text with LLM-backed mutation + ranking."""
    config_path, config_message = _resolve_config_path(config)
    cfg = load_config(str(config_path) if config_path else None)

    if iterations is not None:
        cfg.run.iterations = iterations
    if goal is not None:
        cfg.mutation.prompt.goal = goal
    if metric:
        cfg.metrics.names = metric

    seed_text = _read_seed_text(input)

    setup_logging(level=_parse_log_level(log_level), quiet=quiet, log_file=log_file)
    logging.info("%s", config_message)

    seed = cfg.run.random_seed
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        logging.info("Generated random seed: %d", seed)

    master_rng = random.Random(seed)
    rng_engine = random.Random(master_rng.randrange(2**32))
    rng_selection = random.Random(master_rng.randrange(2**32))
    rng_inspirations = random.Random(master_rng.randrange(2**32))
    rng_ranker = random.Random(master_rng.randrange(2**32))
    rng_models = random.Random(master_rng.randrange(2**32))
    rng_anchors = random.Random(master_rng.randrange(2**32))
    archive_rngs = [
        random.Random(master_rng.randrange(2**32))
        for _ in range(cfg.population.islands)
    ]

    describe, space = build_descriptor_system(cfg)

    rating = RatingSystem(
        cfg.metrics.names,
        mu=cfg.rating.mu,
        sigma=cfg.rating.sigma,
        beta=cfg.rating.beta,
        tau=cfg.rating.tau,
        draw_probability=cfg.rating.draw_probability,
        score_lcb_c=cfg.rating.score_lcb_c,
        child_prior_tau=cfg.rating.child_prior_tau,
    )

    islands = [
        MapElitesArchive(
            space,
            elites_per_cell=cfg.population.elites_per_cell,
            rng=archive_rngs[idx],
            score_fn=rating.score,
        )
        for idx in range(cfg.population.islands)
    ]

    selector = ParentSelector(
        mode=cfg.selection.kind,
        beta=cfg.selection.ucb_beta,
        temp=cfg.selection.temperature,
        rng=rng_selection,
    )

    inspiration_picker = InspirationPicker(rating=rating, rng=rng_inspirations)

    mutator = LLMMutator(
        ensemble=cfg.llm.ensemble,
        metrics=cfg.metrics.names,
        metric_descriptions=cfg.metrics.descriptions,
        goal=cfg.mutation.prompt.goal,
        instructions=cfg.mutation.prompt.instructions,
        max_edits=cfg.mutation.max_edits,
        show_metric_stats=cfg.mutation.prompt.show_metric_stats,
        score_lcb_c=cfg.rating.score_lcb_c,
        rng=rng_models,
    )
    ranker = LLMRanker(
        model=cfg.llm.judge_model,
        rng=rng_ranker,
        max_attempts=cfg.judging.max_attempts,
        repair_enabled=cfg.judging.repair_enabled,
    )
    anchor_manager = build_anchor_manager(cfg=cfg, rng=rng_anchors)

    engine = EvolutionEngine(
        cfg=cfg,
        islands=islands,
        describe=describe,
        rating=rating,
        selector=selector.select_parent,
        inspirations=inspiration_picker,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=anchor_manager,
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
            total=cfg.run.iterations,
            best=0.0,
            empty=islands[0].empty_cells,
        )

        def on_iteration(snapshot):
            if cfg.run.log_interval and snapshot.iteration % cfg.run.log_interval == 0:
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
                "Error: No input provided. Provide an input string, file path, or pipe via stdin."
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
