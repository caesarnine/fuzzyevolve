"""Command-line interface for fuzzyevolve.

Design goals:
- `fuzzyevolve "seed text"` should work (run is the default command).
- Subcommands like `fuzzyevolve tui` should remain intuitive.
- Keep parsing predictable: a real `run` command owns its options; we don't
  rely on Click's "extra args" escape hatches.
"""

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
from typer.core import TyperGroup

from pydantic_ai.settings import ModelSettings

from fuzzyevolve.adapters.llm.critic import LLMCritic
from fuzzyevolve.adapters.llm.operators import LLMRewriteOperator
from fuzzyevolve.adapters.llm.ranker import LLMRanker
from fuzzyevolve.config import load_config
from fuzzyevolve.console.logging import setup_logging
from fuzzyevolve.core.embeddings import (
    SentenceTransformerProvider,
)
from fuzzyevolve.core.engine import EvolutionEngine, build_anchor_manager
from fuzzyevolve.core.multiobjective import Scalarizer
from fuzzyevolve.core.mutation import OperatorMutator, OperatorSpec
from fuzzyevolve.core.pool import CrowdedPool
from fuzzyevolve.core.ratings import RatingSystem
from fuzzyevolve.core.selection import MixedParentSelector
from fuzzyevolve.reporting import render_top_by_fitness_markdown
from fuzzyevolve.run_store import RunStore

_HELP_FLAG = {"-h", "--help"}


class DefaultToRunGroup(TyperGroup):
    """A Typer/Click group that defaults to `run`.

    Click groups treat the first non-option token as a subcommand name, which
    means `fuzzyevolve "seed text"` fails unless we provide a default command.

    We implement the common UX pattern of "default to run" by inserting `run`
    into argv before Click parses options/args, unless the first token is an
    explicitly known subcommand (or the user asked for top-level help).
    """

    default_command = "run"

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        if not args:
            args = [self.default_command]
        elif args[0] not in self.commands and args[0] not in _HELP_FLAG:
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


app = typer.Typer(add_completion=False, no_args_is_help=False, cls=DefaultToRunGroup)

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


def _seed_parts_to_input(seed_parts: list[str] | None) -> str | None:
    if not seed_parts:
        return None
    if len(seed_parts) == 1:
        return seed_parts[0]
    return " ".join(seed_parts)


def _execute_run(
    *,
    seed_parts: list[str] | None,
    config: Path | None,
    output: Path,
    top: int,
    iterations: int | None,
    goal: str | None,
    metric: list[str] | None,
    log_level: str,
    log_file: Path | None,
    quiet: bool,
    resume: Path | None,
    store: bool,
) -> None:
    if top < 0:
        raise typer.BadParameter("--top must be >= 0 (use 0 for no limit).")

    setup_logging(level=_parse_log_level(log_level), quiet=quiet, log_file=log_file)

    run_store: RunStore | None = None
    seed_text: str | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None

    if resume is not None:
        if config is not None:
            raise typer.BadParameter("Do not use --config when resuming.")
        if seed_parts:
            raise typer.BadParameter("Do not provide input text when resuming.")
        run_store = RunStore.open(resume)
        checkpoint_path = resume if resume.is_file() else None
        cfg = run_store.load_config()
        logging.info("Resuming run from %s", run_store.run_dir)
    else:
        config_path, config_message = _resolve_config_path(config)
        cfg = load_config(str(config_path) if config_path else None)
        logging.info("%s", config_message)
        seed_text = _read_seed_text(_seed_parts_to_input(seed_parts))

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
    rng_pool = random.Random(master_rng.randrange(2**32))
    rng_anchors = random.Random(master_rng.randrange(2**32))
    rng_scalarizer = random.Random(master_rng.randrange(2**32))

    provider = SentenceTransformerProvider(cfg.embeddings.model)

    def embed(text: str):
        return provider.embed(text)

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

    scalarizer: Scalarizer | None = None
    pareto_enabled = bool(cfg.multiobjective.enabled and cfg.multiobjective.pareto)
    if cfg.multiobjective.enabled:
        scalarizer = Scalarizer(
            cfg.metrics.names,
            rng=rng_scalarizer,
            dirichlet_alpha=cfg.multiobjective.dirichlet_alpha,
            balanced_probability=cfg.multiobjective.balanced_probability,
            enabled=True,
        )

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

    pool = CrowdedPool(
        max_size=cfg.population.size,
        rng=rng_pool,
        score_fn=rating.score,
        pruning_strategy=cfg.population.pruning,
        knn_k=cfg.population.knn_k,
        metrics=cfg.metrics.names,
        score_lcb_c=cfg.rating.score_lcb_c,
        scalarizer=scalarizer,
        pareto=pareto_enabled,
    )

    anchor_manager = None
    start_iteration = 0
    if resume is not None:
        loaded = run_store.load_checkpoint(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            embed=embed,
            pool_factory=lambda: pool,
            anchor_factory=lambda _cfg: build_anchor_manager(cfg=cfg, rng=rng_anchors),
        )
        pool = loaded.pool
        anchor_manager = loaded.anchors
        start_iteration = loaded.next_iteration
    else:
        anchor_manager = build_anchor_manager(cfg=cfg, rng=rng_anchors)

    selector = MixedParentSelector(
        uniform_probability=cfg.selection.uniform_probability,
        tournament_size=cfg.selection.tournament_size,
        optimistic_beta=cfg.selection.optimistic_beta,
        rng=rng_selection,
        metrics=cfg.metrics.names,
        scalarizer=scalarizer,
        pareto=pareto_enabled,
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
        goal=cfg.task.goal,
        rng=rng_ranker,
        max_attempts=cfg.judging.max_attempts,
        repair_enabled=cfg.judging.repair_enabled,
        store=recorder,
    )

    engine = EvolutionEngine(
        cfg=cfg,
        pool=pool,
        embed=embed,
        rating=rating,
        selector=selector.select_parent,
        critic=critic,
        mutator=mutator,
        ranker=ranker,
        anchor_manager=anchor_manager,
        rng=rng_engine,
        store=recorder,
        scalarizer=scalarizer,
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("• best {task.fields[best]:.3f}"),
        TextColumn("• pool {task.fields[pool]}"),
        TimeElapsedColumn(),
        transient=quiet,
    )

    with progress:
        task = progress.add_task(
            "evolving",
            total=cfg.run.iterations,
            best=0.0,
            pool=len(pool),
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
                pool=snapshot.pool_size,
            )

        if resume is not None:
            result = engine.resume(
                start_iteration=start_iteration,
                on_iteration=on_iteration,
            )
        else:
            result = engine.run(seed_text or "", on_iteration=on_iteration)

    report = render_top_by_fitness_markdown(
        cfg=cfg,
        pool=pool,
        rating=rating,
        top=top,
    )
    output.write_text(report)
    if run_store and store:
        (run_store.run_dir / "best.md").write_text(report)
    logging.info(
        "DONE – report saved to %s (best score %.3f)", output, result.best_score
    )


@app.command()
def run(
    ctx: typer.Context,
    seed: list[str] = typer.Argument(
        None,
        help=(
            "Seed text, a file path, or '-' for stdin. "
            "If omitted, reads stdin when piped."
        ),
    ),
    config: Optional[Path] = typer.Option(
        None, "-c", "--config", help="Path to a TOML or JSON config file."
    ),
    output: Path = typer.Option(
        Path("best.md"),
        "-o",
        "--output",
        help="Path to save the final Markdown report (top by fitness).",
    ),
    top: int = typer.Option(
        20,
        "--top",
        help="How many top individuals to include in the report (0 = all).",
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
    if resume is None and not seed and sys.stdin.isatty():
        typer.echo(ctx.get_help())
        raise typer.Exit(code=1)
    _execute_run(
        seed_parts=seed,
        config=config,
        output=output,
        top=top,
        iterations=iterations,
        goal=goal,
        metric=metric,
        log_level=log_level,
        log_file=log_file,
        quiet=quiet,
        resume=resume,
        store=store,
    )


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
    if user_input == "-":
        seed_text = sys.stdin.read()
    elif user_input is None:
        if not sys.stdin.isatty():
            seed_text = sys.stdin.read()
        else:
            raise typer.BadParameter(
                "No input provided. Provide a seed string, a file path, '-' for stdin, or pipe via stdin."
            )
    elif Path(user_input).is_file():
        seed_text = Path(user_input).read_text()
    else:
        seed_text = user_input

    if not seed_text.strip():
        raise typer.BadParameter("Input text is empty.")
    return seed_text


if __name__ == "__main__":
    app()
