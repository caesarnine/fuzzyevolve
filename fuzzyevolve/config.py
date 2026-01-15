from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib

from pydantic import BaseModel, Field

from fuzzyevolve.llm.models import ModelSpec

DEFAULT_SEMANTIC_BINS = [-2.0, -1.0, 0.0, 1.0, 2.0]


class Config(BaseModel):
    # core loop
    iterations: int = 10
    log_interval: int = 1

    # map-elites + islands
    island_count: int = 1
    elites_per_cell: int = 4
    migration_interval: int = 0
    migration_size: int = 4
    sparring_interval: int = 0

    # sampling
    inspiration_count: int = 3
    max_diffs: int = 4
    judge_include_inspirations: bool = False

    # anchors
    anchor_injection_prob: float = Field(0.2, ge=0.0, le=1.0)
    anchor_mu: float = 25.0
    anchor_sigma: float = 2.0
    anchor_max_per_judgement: int = Field(2, ge=0)
    ghost_anchor_interval: int = Field(10, ge=0)

    # judge robustness
    max_battle_size: int = Field(6, ge=2)
    max_children_judged: int = Field(4, ge=1)
    judge_max_attempts: int = Field(2, ge=1)
    judge_repair_enabled: bool = True

    # opponent selection
    judge_opponent_mode: str = "none"
    judge_opponent_p: float = Field(0.1, ge=0.0, le=1.0)

    # child priors
    child_prior_mode: str = "inherit"
    child_prior_tau: float = Field(4.0, ge=0.0)

    # scoring
    archive_score_c: float = Field(2.0, ge=0.0)
    report_score_c: float = Field(2.0, ge=0.0)

    # trueskill (LLM-as-judge is noisy; favor conservative updates + allow ties)
    trueskill_mu: float = 25.0
    trueskill_sigma: float = Field(25.0 / 3.0, gt=0.0)
    trueskill_beta: float = Field(25.0 / 3.0, gt=0.0)
    trueskill_tau: float = Field(25.0 / 3.0 / 50.0, ge=0.0)
    trueskill_draw_probability: float = Field(0.2, ge=0.0, le=1.0)

    # new cell gate
    new_cell_gate_mode: str = "none"
    new_cell_gate_delta: float = -0.5

    # selection
    selection_mode: str = "optimistic_cell_softmax"
    selection_beta: float = Field(1.0, ge=0.0)
    selection_temp: float = Field(1.0, gt=0.0)

    # descriptors
    descriptor_mode: str = "semantic"
    semantic_projection_seed: int = 123
    semantic_bins_x: list[float] = Field(
        default_factory=lambda: list(DEFAULT_SEMANTIC_BINS)
    )
    semantic_bins_y: list[float] = Field(
        default_factory=lambda: list(DEFAULT_SEMANTIC_BINS)
    )
    semantic_embedding_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"

    # reproducibility
    random_seed: int | None = None

    # llm ensemble + judge
    llm_ensemble: list[ModelSpec] = Field(
        default_factory=lambda: [
            ModelSpec(
                model="google-gla:gemini-3-flash-preview",
                p=0.85,
                temperature=1,
            ),
            ModelSpec(
                model="google-gla:gemini-3-pro-preview",
                p=0.15,
                temperature=1,
            ),
        ]
    )
    judge_model: str = "google-gla:gemini-3-pro-preview"
    metrics: list[str] = ["atmosphere", "creativity"]

    # descriptor space
    axes: dict[str, Any] = Field(
        default_factory=lambda: {
            "semantic_x": {"bins": list(DEFAULT_SEMANTIC_BINS)},
            "semantic_y": {"bins": list(DEFAULT_SEMANTIC_BINS)},
        }
    )

    # mutation prompt
    mutation_prompt_goal: str = "Write me a riveting short story."
    mutation_prompt_instructions: str = (
        "Propose one or more alternative edits to improve the PARENT text. "
        "Each edit must be independently applicable to the original PARENT text (do not chain edits). "
        "Use exact substring search/replace semantics: your `search` must appear verbatim in the PARENT text. "
        "You can rewrite, shorten, or completely change the text."
    )
    mutation_prompt_show_metric_stats: bool = True
    mutation_prompt_c: float = Field(1.0, ge=0.0)


def load_cfg(path: str | None) -> Config:
    if not path:
        return Config()
    data = Path(path).read_text()
    try:
        cfg_dict = json.loads(data)
    except json.JSONDecodeError:
        cfg_dict = tomllib.loads(data)
    return Config.model_validate(cfg_dict)
