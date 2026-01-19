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

from pydantic_ai.settings import ModelSettings

from fuzzyevolve.adapters.llm.critic import LLMCritic
from fuzzyevolve.adapters.llm.operators import LLMRewriteOperator
from fuzzyevolve.adapters.llm.ranker import LLMRanker
from fuzzyevolve.config import load_config
from fuzzyevolve.console.logging import setup_logging
from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.descriptor_system import build_descriptor_system
from fuzzyevolve.core.engine import EvolutionEngine, build_anchor_manager
from fuzzyevolve.core.mutation import OperatorMutator, OperatorSpec
from fuzzyevolve.core.ratings import RatingSystem
from fuzzyevolve.core.selection import ParentSelector
from fuzzyevolve.run_store import RunStore

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True},
)

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
    ctx: typer.Context,
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
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from a previous run directory (or a checkpoint file).",
    ),
    store: bool = typer.Option(
        True,
        "--store/--no-store",
        help="Record run state, checkpoints, and LLM I/O under .fuzzyevolve/.",
    ),
) -> None:
    """Evolve text with LLM-backed mutation + ranking."""
    if ctx.invoked_subcommand is not None:
        return

    setup_logging(level=_parse_log_level(log_level), quiet=quiet, log_file=log_file)
    run_store: RunStore | None = None
    seed_text: str | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None

    if resume is not None:
        if config is not None:
            raise typer.BadParameter("Do not use --config when resuming.")
        if ctx.args:
            raise typer.BadParameter("Do not provide input text when resuming.")
        run_store = RunStore.open(resume)
        checkpoint_path = resume if resume.is_file() else None
        cfg = run_store.load_config()
        logging.info("Resuming run from %s", run_store.run_dir)
    else:
        config_path, config_message = _resolve_config_path(config)
        cfg = load_config(str(config_path) if config_path else None)
        logging.info("%s", config_message)
        raw_input = " ".join(ctx.args) if ctx.args else None
        seed_text = _read_seed_text(raw_input)

    if iterations is not None:
        cfg.run.iterations = iterations
    if goal is not None:
        cfg.task.goal = goal
    if metric:
        cfg.metrics.names = metric

    seed = cfg.run.random_seed
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**32 - 1)
        logging.info("Generated random seed: %d", seed)
        cfg.run.random_seed = seed

    master_rng = random.Random(seed)
    rng_engine = random.Random(master_rng.randrange(2**32))
    rng_selection = random.Random(master_rng.randrange(2**32))
    rng_ranker = random.Random(master_rng.randrange(2**32))
    rng_mutation = random.Random(master_rng.randrange(2**32))
    rng_anchors = random.Random(master_rng.randrange(2**32))

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

    recorder = run_store if (store and run_store is not None) else None

    if run_store is None and store:
        data_dir = RunStore.default_data_dir(cwd=Path.cwd())
        run_store = RunStore.create(
            data_dir=data_dir,
            cfg=cfg,
            seed_text=seed_text,
            config_path=config_path,
        )
        recorder = run_store
        logging.info("Recording run to %s", run_store.run_dir)

    islands: list[MapElitesArchive]
    anchor_manager = None
    start_iteration = 0
    if resume is not None:
        def _space_factory(_cfg):
            return space

        def _archive_factory(space_obj):
            return MapElitesArchive(
                space_obj,
                elites_per_cell=cfg.population.elites_per_cell,
                rng=random.Random(master_rng.randrange(2**32)),
                score_fn=rating.score,
            )

        def _anchor_factory(_cfg):
            return build_anchor_manager(cfg=cfg, rng=rng_anchors)

        loaded = run_store.load_checkpoint(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            space_factory=_space_factory,
            archive_factory=_archive_factory,
            anchor_factory=_anchor_factory,
        )
        islands = loaded.islands
        anchor_manager = loaded.anchors
        start_iteration = loaded.next_iteration
    else:
        archive_rngs = [
            random.Random(master_rng.randrange(2**32))
            for _ in range(cfg.population.islands)
        ]
        islands = [
            MapElitesArchive(
                space,
                elites_per_cell=cfg.population.elites_per_cell,
                rng=archive_rngs[idx],
                score_fn=rating.score,
            )
            for idx in range(cfg.population.islands)
        ]
        anchor_manager = build_anchor_manager(cfg=cfg, rng=rng_anchors)

    selector = ParentSelector(
        mode=cfg.selection.kind,
        beta=cfg.selection.ucb_beta,
        temp=cfg.selection.temperature,
        rng=rng_selection,
    )

    critic = None
    if cfg.critic.enabled:
        critic_model = cfg.llm.critic_model or cfg.llm.judge_model
        critic_settings: ModelSettings = {"temperature": cfg.llm.critic_temperature}
        critic = LLMCritic(
            model=critic_model,
            model_settings=critic_settings,
            goal=cfg.task.goal,
            metrics=cfg.metrics.names,
            metric_descriptions=cfg.metrics.descriptions,
            routes=cfg.critic.routes,
            instructions=cfg.critic.instructions,
            show_metric_stats=cfg.prompts.show_metric_stats,
            score_lcb_c=cfg.rating.score_lcb_c,
            store=recorder,
        )

    operators = {}
    specs: list[OperatorSpec] = []
    for op_cfg in [op for op in cfg.mutation.operators if op.enabled]:
        op_rng = random.Random(master_rng.randrange(2**32))
        ensemble = op_cfg.ensemble or cfg.llm.ensemble
        operators[op_cfg.name] = LLMRewriteOperator(
            name=op_cfg.name,
            role=op_cfg.role,
            ensemble=ensemble,
            temperature=op_cfg.temperature,
            goal=cfg.task.goal,
            metrics=cfg.metrics.names,
            metric_descriptions=cfg.metrics.descriptions,
            instructions=op_cfg.instructions,
            show_metric_stats=cfg.prompts.show_metric_stats,
            score_lcb_c=cfg.rating.score_lcb_c,
            rng=op_rng,
            store=recorder,
        )
        specs.append(
            OperatorSpec(
                name=op_cfg.name,
                role=op_cfg.role,
                min_jobs=op_cfg.min_jobs,
                weight=op_cfg.weight,
                uncertainty_scale=op_cfg.uncertainty_scale,
            )
        )

    mutator = OperatorMutator(
        operators=operators,
        specs=specs,
        jobs_per_iteration=cfg.mutation.jobs_per_iteration,
        rng=rng_mutation,
    )
    ranker = LLMRanker(
        model=cfg.llm.judge_model,
        rng=rng_ranker,
        max_attempts=cfg.judging.max_attempts,
        repair_enabled=cfg.judging.repair_enabled,
        store=recorder,
    )

    engine = EvolutionEngine(
        cfg=cfg,
        islands=islands,
        describe=describe,
        rating=rating,
        selector=selector.select_parent,
        critic=critic,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=anchor_manager,
        rng=rng_engine,
        store=recorder,
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

        if resume is not None:
            result = engine.resume(
                start_iteration=start_iteration,
                on_iteration=on_iteration,
            )
        else:
            result = engine.run(seed_text or "", on_iteration=on_iteration)

    output.write_text(result.best_elite.text)
    if run_store and store:
        (run_store.run_dir / "best.txt").write_text(result.best_elite.text)
    logging.info("DONE – best saved to %s (score %.3f)", output, result.best_score)


@app.command()
def tui(
    run: Optional[Path] = typer.Option(
        None,
        "--run",
        help="Run directory (or checkpoint file) to open. If omitted, shows a run picker.",
    ),
    data_dir: Path = typer.Option(
        Path(".fuzzyevolve"),
        "--data-dir",
        help="Directory containing runs/ (defaults to .fuzzyevolve).",
    ),
    attach: bool = typer.Option(
        True,
        "--attach/--no-attach",
        help="Periodically refresh from disk (follow a running evolution).",
    ),
) -> None:
    """Browse recorded runs in a Textual TUI."""
    from fuzzyevolve.tui.app import run_tui

    run_tui(data_dir=data_dir, run_dir=run, attach=attach)


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
