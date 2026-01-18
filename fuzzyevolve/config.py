from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for <3.11
    import tomli as tomllib

from pydantic import BaseModel, Field, model_validator

DEFAULT_EMBEDDING_BINS = [-2.0, -1.0, 0.0, 1.0, 2.0]


class RunConfig(BaseModel):
    iterations: int = Field(10, ge=1)
    log_interval: int = Field(1, ge=0)
    random_seed: int | None = None


class PopulationConfig(BaseModel):
    islands: int = Field(1, ge=1)
    elites_per_cell: int = Field(4, ge=1)


class MigrationConfig(BaseModel):
    interval: int = Field(0, ge=0)
    size: int = Field(4, ge=1)


class SparringConfig(BaseModel):
    interval: int = Field(0, ge=0)


class MaintenanceConfig(BaseModel):
    migration: MigrationConfig = Field(default_factory=MigrationConfig)
    sparring: SparringConfig = Field(default_factory=SparringConfig)


class MetricsConfig(BaseModel):
    names: list[str] = Field(default_factory=lambda: ["atmosphere", "creativity"])
    descriptions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_names(self) -> "MetricsConfig":
        names = [name.strip() for name in self.names if name.strip()]
        if not names:
            raise ValueError("metrics.names must contain at least one metric name.")
        self.names = names
        return self


class RatingConfig(BaseModel):
    mu: float = 25.0
    sigma: float = Field(25.0 / 3.0, gt=0.0)
    beta: float = Field(25.0 / 3.0, gt=0.0)
    tau: float = Field(25.0 / 3.0 / 50.0, ge=0.0)
    draw_probability: float = Field(0.2, ge=0.0, le=1.0)

    score_lcb_c: float = Field(2.0, ge=0.0)
    child_prior_tau: float = Field(4.0, ge=0.0)


class Embedding2DConfig(BaseModel):
    embedding_model: str | None = "hash"
    projection_seed: int = 123
    bins_x: list[float] = Field(default_factory=lambda: list(DEFAULT_EMBEDDING_BINS))
    bins_y: list[float] = Field(default_factory=lambda: list(DEFAULT_EMBEDDING_BINS))

    @model_validator(mode="after")
    def _validate_bins(self) -> "Embedding2DConfig":
        if len(self.bins_x) < 2 or len(self.bins_y) < 2:
            raise ValueError("embedding_2d bins must each have at least two values.")
        return self


class DescriptorConfig(BaseModel):
    kind: Literal["embedding_2d", "length"] = "embedding_2d"
    embedding_2d: Embedding2DConfig = Field(default_factory=Embedding2DConfig)
    length_bins: list[float] = Field(default_factory=lambda: [0, 50, 200, 1000, 10_000])

    @model_validator(mode="after")
    def _validate_descriptor(self) -> "DescriptorConfig":
        if self.kind == "length" and len(self.length_bins) < 2:
            raise ValueError("descriptor.length_bins must have at least two values.")
        return self


class SelectionConfig(BaseModel):
    kind: Literal["uniform_cell", "optimistic_cell_softmax"] = "uniform_cell"
    ucb_beta: float = Field(1.0, ge=0.0)
    temperature: float = Field(1.0, gt=0.0)


class MutationPromptConfig(BaseModel):
    goal: str = "Write me a riveting short story."
    instructions: str = (
        "Propose one or more edits to improve the PARENT text.\n"
        "Use exact substring search/replace semantics: your `search` must appear verbatim in the PARENT text.\n"
        "Return multiple edits only when they should be applied together to create one improved child.\n"
        "You can rewrite, shorten, or completely change the text.\n"
    )
    show_metric_stats: bool = True


class MutationConfig(BaseModel):
    calls_per_iteration: int = Field(4, ge=1)
    max_workers: int = Field(8, ge=1)
    max_edits: int = Field(4, ge=1)
    max_children: int = Field(4, ge=1)
    inspiration_count: int = Field(0, ge=0)
    prompt: MutationPromptConfig = Field(default_factory=MutationPromptConfig)


class OpponentConfig(BaseModel):
    kind: Literal["none", "cell_champion", "global_best"] = "none"
    probability: float = Field(0.1, ge=0.0, le=1.0)


class JudgingConfig(BaseModel):
    max_attempts: int = Field(2, ge=1)
    repair_enabled: bool = True
    max_battle_size: int = Field(6, ge=2)
    include_inspiration: bool = False
    opponent: OpponentConfig = Field(default_factory=OpponentConfig)


class AnchorsConfig(BaseModel):
    injection_probability: float = Field(0.2, ge=0.0, le=1.0)
    max_per_battle: int = Field(2, ge=0)
    seed_mu: float = 25.0
    seed_sigma: float = Field(2.0, gt=0.0)
    ghost_interval: int = Field(10, ge=0)


class NewCellGateConfig(BaseModel):
    kind: Literal["none", "parent_lcb"] = "none"
    delta: float = -0.5


class ModelSpec(BaseModel):
    model: str
    weight: float = Field(..., gt=0.0)
    temperature: float = 0.7


class LLMConfig(BaseModel):
    ensemble: list[ModelSpec] = Field(
        default_factory=lambda: [
            ModelSpec(
                model="google-gla:gemini-3-flash-preview",
                weight=0.85,
                temperature=1.0,
            ),
            ModelSpec(
                model="google-gla:gemini-3-pro-preview",
                weight=0.15,
                temperature=1.0,
            ),
        ]
    )
    judge_model: str = "google-gla:gemini-3-pro-preview"

    @model_validator(mode="after")
    def _validate_ensemble(self) -> "LLMConfig":
        if not self.ensemble:
            raise ValueError("llm.ensemble must contain at least one model spec.")
        return self


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    rating: RatingConfig = Field(default_factory=RatingConfig)
    descriptor: DescriptorConfig = Field(default_factory=DescriptorConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    judging: JudgingConfig = Field(default_factory=JudgingConfig)
    anchors: AnchorsConfig = Field(default_factory=AnchorsConfig)
    new_cell_gate: NewCellGateConfig = Field(default_factory=NewCellGateConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: str | None) -> Config:
    if not path:
        return Config()
    data = Path(path).read_text()
    try:
        cfg_dict = json.loads(data)
    except json.JSONDecodeError:
        cfg_dict = tomllib.loads(data)
    return Config.model_validate(cfg_dict)
