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
    checkpoint_interval: int = Field(
        1,
        ge=0,
        description="Save a checkpoint every N iterations (0 disables periodic checkpoints; latest is still written).",
    )
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


class TaskConfig(BaseModel):
    goal: str = "Write me a riveting short story."


class PromptConfig(BaseModel):
    show_metric_stats: bool = True


class CriticConfig(BaseModel):
    enabled: bool = True
    routes: int = Field(
        8,
        ge=1,
        description="How many distinct rewrite routes to generate per critique.",
    )
    instructions: str = (
        "You are a critique agent helping an evolutionary text system.\n"
        "Analyze the PARENT text and propose actionable guidance.\n"
        "Prefer concrete, specific feedback over generic advice.\n"
        "Do not quote large spans of the parent text.\n"
    )


class MutationOperatorConfig(BaseModel):
    name: str
    role: Literal["exploit", "explore"] = "exploit"
    enabled: bool = True
    min_jobs: int = Field(0, ge=0)
    weight: float = Field(1.0, gt=0.0)
    uncertainty_scale: float = Field(
        1.0,
        ge=0.0,
        description="Multiplier on rating.child_prior_tau for children from this operator.",
    )
    temperature: float | None = Field(
        None,
        description="Optional temperature override for this operator (otherwise uses model spec temperature).",
    )
    instructions: str = ""
    ensemble: list[ModelSpec] | None = None

    @model_validator(mode="after")
    def _validate_operator(self) -> "MutationOperatorConfig":
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("mutation.operators.name must be non-empty.")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("mutation.operators.temperature must be >= 0.")
        return self


class MutationConfig(BaseModel):
    jobs_per_iteration: int = Field(4, ge=1)
    max_workers: int = Field(8, ge=1)
    max_children: int = Field(4, ge=1)
    operators: list[MutationOperatorConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_mutation(self) -> "MutationConfig":
        if not self.operators:
            self.operators = [
                MutationOperatorConfig(
                    name="exploit",
                    role="exploit",
                    min_jobs=2,
                    weight=1.0,
                    uncertainty_scale=0.7,
                    temperature=0.7,
                    instructions=(
                        "Rewrite the PARENT to improve quality and maximize the metrics.\n"
                        "Keep the core premise and intent, but you may restructure as needed.\n"
                        "Use the critique issues as a to-do list when provided.\n"
                        "Do not mention evaluation metrics.\n"
                    ),
                ),
                MutationOperatorConfig(
                    name="explore",
                    role="explore",
                    min_jobs=2,
                    weight=1.0,
                    uncertainty_scale=2.5,
                    temperature=1.2,
                    instructions=(
                        "Rewrite freely from scratch for exploration.\n"
                        "You may change everything: plot, voice, POV, style, genre, structure.\n"
                        "Use a provided rewrite route as the main creative constraint.\n"
                        "Do not copy phrases from the parent.\n"
                        "Do not mention evaluation metrics.\n"
                    ),
                ),
            ]

        enabled = [op for op in self.operators if op.enabled]
        if not enabled:
            raise ValueError("mutation.operators must contain at least one enabled operator.")

        names = [op.name for op in enabled]
        if len(set(names)) != len(names):
            raise ValueError("mutation.operators names must be unique.")

        min_sum = sum(op.min_jobs for op in enabled)
        if min_sum > self.jobs_per_iteration:
            raise ValueError(
                "sum(mutation.operators.min_jobs) must be <= mutation.jobs_per_iteration."
            )
        return self


class OpponentConfig(BaseModel):
    kind: Literal["none", "cell_champion", "global_best"] = "none"
    probability: float = Field(0.1, ge=0.0, le=1.0)


class JudgingConfig(BaseModel):
    max_attempts: int = Field(2, ge=1)
    repair_enabled: bool = True
    max_battle_size: int = Field(6, ge=2)
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
    critic_model: str | None = None
    critic_temperature: float = Field(0.2, ge=0.0)

    @model_validator(mode="after")
    def _validate_ensemble(self) -> "LLMConfig":
        if not self.ensemble:
            raise ValueError("llm.ensemble must contain at least one model spec.")
        return self


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
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
