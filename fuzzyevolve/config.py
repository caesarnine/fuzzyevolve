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


class Config(BaseModel):
    # core loop
    iterations: int = 10
    log_interval: int = 1

    # map-elites + islands
    island_count: int = 1
    elites_per_cell: int = 4
    migration_interval: int = 300
    migration_size: int = 4
    sparring_interval: int = 50

    # sampling
    inspiration_count: int = 3
    max_diffs: int = 4
    judge_include_inspirations: bool = True

    # reproducibility
    random_seed: int | None = None

    # llm ensemble + judge
    llm_ensemble: list[ModelSpec] = Field(
        default_factory=lambda: [
            ModelSpec(
                model="vertex_ai/gemini-2.5-flash",
                p=0.85,
                temperature=1,
            ),
            ModelSpec(model="vertex_ai/gemini-2.5-pro", p=0.15, temperature=1),
        ]
    )
    judge_model: str = "vertex_ai/gemini-2.5-pro"
    metrics: list[str] = ["clarity", "conciseness", "creativity"]

    # descriptor space
    axes: dict[str, Any] = Field(
        default_factory=lambda: {
            "lang": ["txt"],
            "len": {"bins": [0, 500, 1000, 2000, 1e9]},
        }
    )

    # mutation prompt
    mutation_prompt_goal: str = "Improve the text based on the metrics provided."
    mutation_prompt_instructions: str = (
        "Propose one or more SEARCH/REPLACE diff blocks to improve the PARENT text. "
        "If you provide multiple diff blocks, each block must be a standalone alternative that applies to the original PARENT text as-is. "
        "You can rewrite, shorten, or completely change the text. "
        "First, explain your reasoning in a <thinking> block. Then, provide the diffs in a <diffs> block."
    )


def load_cfg(path: str | None) -> Config:
    if not path:
        return Config()
    data = Path(path).read_text()
    try:
        cfg_dict = json.loads(data)
    except json.JSONDecodeError:
        cfg_dict = tomllib.loads(data)
    return Config.model_validate(cfg_dict)
