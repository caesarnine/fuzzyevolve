from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from fuzzyevolve.adapters.llm.prompts import build_rank_prompt
from fuzzyevolve.core.battle import Battle
from fuzzyevolve.core.ratings import BattleRanking

log_llm = logging.getLogger("llm.ranker")


class MetricRanking(BaseModel):
    metric: str = Field(..., description="Metric name from the prompt.")
    ranked_tiers: list[list[int]] = Field(
        ...,
        description=(
            "Candidate IDs grouped into tiers from best to worst. "
            "Each tier is a list of IDs; IDs within a tier are tied. "
            "IDs must correspond to the bracketed IDs in the prompt."
        ),
    )


class RankerOutput(BaseModel):
    rankings: list[MetricRanking] = Field(default_factory=list)


class LLMRanker:
    def __init__(
        self,
        *,
        model: str,
        rng: random.Random | None = None,
        model_settings: ModelSettings | None = None,
        max_attempts: int = 2,
        repair_enabled: bool = True,
    ) -> None:
        self.model = model
        self.rng = rng or random.Random()
        self.model_settings = model_settings or {"temperature": 0.0}
        self.max_attempts = max(1, max_attempts)
        self.repair_enabled = repair_enabled
        self.agent = Agent(
            output_type=RankerOutput,
            name="ranker",
            instructions=(
                "Rank the provided candidates independently for each metric.\n"
                "- Return ONLY the structured output (no prose).\n"
                "- For each metric, group ALL candidate IDs into ordered tiers.\n"
                "- Use ties when candidates are effectively indistinguishable.\n"
                "- Provide `rankings` as a list of {metric, ranked_tiers} objects.\n"
            ),
        )

    def rank(
        self,
        *,
        metrics: Sequence[str],
        battle: Battle,
        metric_descriptions: Mapping[str, str] | None = None,
    ) -> BattleRanking | None:
        if len(battle.participants) < 2:
            return None

        shuffled_indices = list(range(len(battle.participants)))
        self.rng.shuffle(shuffled_indices)

        prompt_items: list[tuple[int, str]] = []
        prompt_id_to_original: dict[int, int] = {}
        for prompt_id, original_index in enumerate(shuffled_indices):
            prompt_items.append((prompt_id, battle.participants[original_index].text))
            prompt_id_to_original[prompt_id] = original_index

        prompt = build_rank_prompt(
            metrics=metrics,
            items=prompt_items,
            metric_descriptions=metric_descriptions,
        )
        log_llm.debug("Ranker prompt:\n%s", prompt)

        for attempt in range(1, self.max_attempts + 1):
            try:
                rsp = self.agent.run_sync(
                    prompt,
                    model=self.model,
                    model_settings=self.model_settings,
                )
            except Exception:
                log_llm.error(
                    "Ranker call failed outright — attempt %d/%d.",
                    attempt,
                    self.max_attempts,
                )
                if attempt >= self.max_attempts:
                    return None
                continue

            parsed = rsp.output.rankings
            ranked_map, error = _validate_rankings(
                parsed, metrics, len(battle.participants)
            )
            if ranked_map is None:
                log_llm.warning(
                    "Ranker returned invalid rankings (%s) — attempt %d/%d.",
                    error,
                    attempt,
                    self.max_attempts,
                )
                if not self.repair_enabled or attempt >= self.max_attempts:
                    return None
                prompt = _build_repair_prompt(prompt, error or "invalid output")
                continue

            tiers_by_metric: dict[str, list[list[int]]] = {}
            for metric in metrics:
                tiers_by_metric[metric] = [
                    [prompt_id_to_original[prompt_id] for prompt_id in tier]
                    for tier in ranked_map[metric]
                ]
            return BattleRanking(tiers_by_metric=tiers_by_metric)

        return None


def _validate_rankings(
    rankings: Sequence[MetricRanking],
    metrics: Sequence[str],
    total_players: int,
) -> tuple[dict[str, list[list[int]]] | None, str | None]:
    expected_ids = set(range(total_players))
    ranked_map: dict[str, list[list[int]]] = {}
    errors: list[str] = []

    for ranking in rankings:
        metric_name = ranking.metric
        if metric_name not in metrics:
            errors.append(f"unknown metric '{metric_name}'")
            continue
        if metric_name in ranked_map:
            errors.append(f"duplicate metric '{metric_name}'")
            continue
        tiers = ranking.ranked_tiers
        if not tiers:
            errors.append(f"metric '{metric_name}' has no tiers")
            continue
        empty_tiers = [idx for idx, tier in enumerate(tiers) if not tier]
        if empty_tiers:
            errors.append(f"metric '{metric_name}' has empty tiers at {empty_tiers}")
            continue
        ranked_ids = [player_id for tier in tiers for player_id in tier]
        ranked_set = set(ranked_ids)
        if ranked_set != expected_ids or len(ranked_ids) != total_players:
            missing = expected_ids - ranked_set
            extra = ranked_set - expected_ids
            errors.append(
                f"metric '{metric_name}' ids mismatch missing={sorted(missing)} extra={sorted(extra)}"
            )
            continue
        ranked_map[metric_name] = tiers

    missing_metrics = [m for m in metrics if m not in ranked_map]
    if missing_metrics:
        errors.append(f"missing metrics {missing_metrics}")

    if errors:
        return None, "; ".join(errors)
    return ranked_map, None


def _build_repair_prompt(prompt: str, error_msg: str) -> str:
    return (
        "The previous output was invalid.\n"
        f"Issues: {error_msg}\n\n"
        "Return corrected structured output only.\n\n"
        f"Original prompt:\n{prompt}"
    )
