"""
This module contains the command-line interface for fuzzyevolve.
"""

import logging
import random
import sys
from pathlib import Path
from typing import List, Optional

import typer

from fuzzyevolve.config import load_cfg, LLMEntry
from fuzzyevolve.evolution.archive import MixedArchive
from fuzzyevolve.evolution.driver import EvolutionaryDriver
from fuzzyevolve.evolution.judge import MultiMetricJudge
from fuzzyevolve.llm.client import LLMProvider
from fuzzyevolve.utils.logging import setup_logging

app = typer.Typer()


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
    log_file: Optional[Path] = typer.Option(
        None, help="Path to write detailed logs."
    ),
    quiet: bool = typer.Option(
        False, "-q", "--quiet", help="Suppress the progress bar and non-essential logging."
    ),
):
    """Evolve text with a language model."""
    # --- Configuration Layering ---
    cfg = load_cfg(config)

    if iterations is not None:
        cfg.iterations = iterations
    if goal is not None:
        cfg.mutation_prompt_goal = goal
    if metric:
        cfg.metrics = metric
    if judge_model is not None:
        cfg.judge_model = judge_model

    # --- Input Handling ---
    seed_text = ""
    if input is None:
        if not sys.stdin.isatty():
            seed_text = sys.stdin.read()
        else:
            typer.echo(
                "Error: No input provided. Please provide an input string, file path, or pipe data via stdin."
            )
            raise typer.Exit(code=1)
    elif Path(input).is_file():
        seed_text = Path(input).read_text()
    else:
        seed_text = input

    if not seed_text.strip():
        typer.echo("Error: Input text is empty.")
        raise typer.Exit(code=1)

    # ─ setup logging & PRNG ─────────────────────────────────────────
    setup_logging(level=logging.INFO, quiet=quiet, log_file=log_file)
    random.seed(42)

    # llm providers
    # - mutations: use ensemble
    mut_llm_provider = LLMProvider(cfg.llm_ensemble)
    # - judge: force single model from cfg.judge_model
    judge_llm_provider = LLMProvider([LLMEntry(model=cfg.judge_model, p=1.0)])

    # islands & judge
    islands = [MixedArchive(cfg.axes, cfg.k_top) for _ in range(cfg.num_islands)]
    judge = MultiMetricJudge(judge_llm_provider, cfg.metrics)

    # driver
    driver = EvolutionaryDriver(cfg, mut_llm_provider, judge, islands)
    driver.run(seed_text, output, quiet)


if __name__ == "__main__":
    app()
