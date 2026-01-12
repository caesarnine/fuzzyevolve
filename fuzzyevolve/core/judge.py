from __future__ import annotations

import logging
import random
from typing import Iterable, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

import trueskill as ts
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.stats import EvolutionStats
from fuzzyevolve.core.scoring import make_envs
from fuzzyevolve.llm.prompts import build_rank_prompt

log_llm = logging.getLogger("llm")


class MetricRanking(BaseModel):
    metric: str = Field(
        ...,
        description="Metric name from the prompt.",
    )
    ranked_ids: list[int] = Field(
        ...,
        description=(
            "Candidate IDs ordered best to worst. "
            "IDs must correspond to the bracketed IDs in the prompt."
        ),
    )


class JudgeOutput(BaseModel):
    rankings: list[MetricRanking] = Field(
        default_factory=list,
        description="Per-metric rankings as a list of objects.",
    )


class LLMJudge:
    def __init__(
        self,
        model: str,
        metrics: Sequence[str],
        rng: random.Random | None = None,
        model_settings: ModelSettings | None = None,
        max_attempts: int = 2,
        repair_enabled: bool = True,
        stats: EvolutionStats | None = None,
    ) -> None:
        self.model = model
        self.metrics = list(metrics)
        self.envs = make_envs(self.metrics)
        self.rng = rng or random.Random()
        self.model_settings = model_settings or {"temperature": 0.0}
        self.max_attempts = max(1, max_attempts)
        self.repair_enabled = repair_enabled
        self.stats = stats
        self.agent = Agent(
            output_type=JudgeOutput,
            name="judge",
            instructions=(
                "Rank the provided candidates independently for each metric.\n"
                "- Return ONLY the structured output (no prose).\n"
                "- For each metric, rank ALL candidate IDs exactly once.\n"
                "- Provide `rankings` as a list of {metric, ranked_ids} objects.\n"
            ),
        )

    def new_ratings(self) -> dict[str, ts.Rating]:
        return {metric: self.envs[metric].create_rating() for metric in self.metrics}

    def ensure_ratings(self, elite: Elite) -> None:
        if not elite.ratings:
            elite.ratings = self.new_ratings()
            return
        for metric in self.metrics:
            if metric not in elite.ratings:
                elite.ratings[metric] = self.envs[metric].create_rating()

    def rank_and_rate(
        self, players: Sequence[Elite], frozen: set[int] | None = None
    ) -> bool:
        for player in players:
            self.ensure_ratings(player)

        if not players:
            log_llm.warning("Rank and rate called with no players. Skipping.")
            return False

        if self.stats:
            self.stats.judge_calls_total += 1

        shuffled_indices = list(range(len(players)))
        self.rng.shuffle(shuffled_indices)

        prompt_items: list[tuple[int, Elite]] = []
        id_to_player: dict[int, Elite] = {}
        for prompt_id, original_index in enumerate(shuffled_indices):
            player = players[original_index]
            prompt_items.append((prompt_id, player))
            id_to_player[prompt_id] = player

        prompt = build_rank_prompt(self.metrics, prompt_items)
        log_llm.debug("Judge prompt:\n%s", prompt)
        frozen_ids = frozen or set()

        for attempt in range(1, self.max_attempts + 1):
            try:
                rsp = self.agent.run_sync(
                    prompt,
                    model=self.model,
                    model_settings=self.model_settings,
                )
                log_llm.debug("Judge response: %s", rsp.output)
            except Exception:
                log_llm.error(
                    "Judge agent call failed outright — attempt %d/%d.",
                    attempt,
                    self.max_attempts,
                )
                if self.stats and attempt >= self.max_attempts:
                    self.stats.judge_calls_failed += 1
                if attempt >= self.max_attempts:
                    return False
                continue

            parsed_rankings = rsp.output.rankings
            if not parsed_rankings:
                log_llm.error(
                    "Judge agent: No rankings returned — attempt %d/%d.",
                    attempt,
                    self.max_attempts,
                )
                if self.stats and attempt >= self.max_attempts:
                    self.stats.judge_calls_failed += 1
                if attempt >= self.max_attempts:
                    return False
                continue

            ranked_map, error_msg = self._validate_rankings(
                parsed_rankings, len(players)
            )
            if ranked_map is None:
                log_llm.warning(
                    "Judge agent: Invalid rankings (%s) — attempt %d/%d.",
                    error_msg,
                    attempt,
                    self.max_attempts,
                )
                if self.stats:
                    self.stats.judge_invalid_total += 1
                if not self.repair_enabled or attempt >= self.max_attempts:
                    if self.stats:
                        self.stats.judge_calls_failed += 1
                    return False
                if self.stats:
                    self.stats.judge_repair_attempts += 1
                prompt = self._build_repair_prompt(
                    prompt, error_msg or "invalid output"
                )
                continue

            updates: list[tuple[Elite, str, ts.Rating]] = []
            try:
                for metric_name in self.metrics:
                    ranked_ids = ranked_map[metric_name]
                    ranked_players = [
                        id_to_player[prompt_id] for prompt_id in ranked_ids
                    ]
                    rating_groups = [
                        [player.ratings[metric_name]] for player in ranked_players
                    ]
                    updated_ratings = self.envs[metric_name].rate(
                        rating_groups,
                        ranks=list(range(len(ranked_players))),
                    )
                    for player, new_rating in zip(ranked_players, updated_ratings):
                        if player.frozen or id(player) in frozen_ids:
                            continue
                        updates.append((player, metric_name, new_rating[0]))
            except Exception as exc:
                log_llm.error("Judge agent: TrueSkill update failed: %s", exc)
                if self.stats:
                    self.stats.judge_calls_failed += 1
                return False

            if not updates:
                log_llm.error("Judge agent: No ratings updated. Skipping update.")
                if self.stats:
                    self.stats.judge_calls_failed += 1
                return False

            for player, metric_name, rating in updates:
                player.ratings[metric_name] = rating
            return True

        return False

    def _validate_rankings(
        self, rankings: Iterable[MetricRanking], total_players: int
    ) -> tuple[dict[str, list[int]] | None, str | None]:
        expected_ids = set(range(total_players))
        ranked_map: dict[str, list[int]] = {}
        errors: list[str] = []

        for ranking in rankings:
            metric_name = ranking.metric
            if metric_name not in self.metrics:
                errors.append(f"unknown metric '{metric_name}'")
                continue
            if metric_name in ranked_map:
                errors.append(f"duplicate metric '{metric_name}'")
                continue
            ranked_ids = ranking.ranked_ids
            if len(ranked_ids) != total_players:
                errors.append(
                    f"metric '{metric_name}' has {len(ranked_ids)} ids, expected {total_players}"
                )
                continue
            if set(ranked_ids) != expected_ids:
                missing = expected_ids - set(ranked_ids)
                extra = set(ranked_ids) - expected_ids
                errors.append(
                    f"metric '{metric_name}' ids mismatch missing={sorted(missing)} extra={sorted(extra)}"
                )
                continue
            ranked_map[metric_name] = ranked_ids

        missing_metrics = [m for m in self.metrics if m not in ranked_map]
        if missing_metrics:
            errors.append(f"missing metrics {missing_metrics}")

        if errors:
            return None, "; ".join(errors)
        return ranked_map, None

    def _build_repair_prompt(self, prompt: str, error_msg: str) -> str:
        return (
            "The previous output was invalid.\n"
            f"Issues: {error_msg}\n\n"
            "Return corrected structured output only.\n\n"
            f"Original prompt:\n{prompt}"
        )
