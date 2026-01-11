from __future__ import annotations

import logging
import random
from typing import Iterable, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

import trueskill as ts
from fuzzyevolve.core.models import Elite
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
    ) -> None:
        self.model = model
        self.metrics = list(metrics)
        self.envs = make_envs(self.metrics)
        self.rng = rng or random.Random()
        self.model_settings = model_settings or {"temperature": 0.0}
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

    def rank_and_rate(self, players: Sequence[Elite]) -> None:
        for player in players:
            self.ensure_ratings(player)

        if not players:
            log_llm.warning("Rank and rate called with no players. Skipping.")
            return

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
        try:
            rsp = self.agent.run_sync(
                prompt,
                model=self.model,
                model_settings=self.model_settings,
            )
            log_llm.debug("Judge response: %s", rsp.output)
        except Exception:
            log_llm.error(
                "Judge agent call failed outright — skipping rating update for this batch."
            )
            return
        parsed_rankings = rsp.output.rankings
        if not parsed_rankings:
            log_llm.error("Judge agent: No rankings returned. Skipping update.")
            return

        ranked_map: dict[str, list[int]] = {}
        for ranking in parsed_rankings:
            metric_name = ranking.metric
            if metric_name in ranked_map:
                log_llm.warning(
                    "Judge agent: Duplicate ranking for metric '%s'.",
                    metric_name,
                )
                continue
            if metric_name not in self.metrics:
                log_llm.warning(
                    "Judge agent: Ranking returned for unknown metric '%s'.",
                    metric_name,
                )
                continue
            ranked_map[metric_name] = ranking.ranked_ids

        if not ranked_map:
            log_llm.error("Judge agent: No usable rankings returned. Skipping update.")
            return

        for metric_name in self.metrics:
            ranked_ids = ranked_map.get(metric_name)
            if not ranked_ids:
                log_llm.warning(
                    "Judge agent: Missing ranking for metric '%s'.",
                    metric_name,
                )
                continue

            ranked_players = self._resolve_ranked_players(
                metric_name, ranked_ids, id_to_player
            )
            if not ranked_players:
                continue

            rating_groups = [[p.ratings[metric_name]] for p in ranked_players]
            try:
                updated_ratings = self.envs[metric_name].rate(
                    rating_groups,
                    ranks=list(range(len(ranked_players))),
                )
            except Exception as exc:
                log_llm.error(
                    "Judge agent: TrueSkill update failed for metric '%s': %s",
                    metric_name,
                    exc,
                )
                continue

            for player, new_rating in zip(ranked_players, updated_ratings):
                player.ratings[metric_name] = new_rating[0]

    def _resolve_ranked_players(
        self,
        metric_name: str,
        ranked_ids: Iterable[int],
        id_to_player: dict[int, Elite],
    ) -> list[Elite]:
        ranked_players: list[Elite] = []
        seen_ids: set[int] = set()
        for prompt_id in ranked_ids:
            if prompt_id in seen_ids:
                log_llm.warning(
                    "Judge agent: Metric '%s' ranking contains duplicate ID %s.",
                    metric_name,
                    prompt_id,
                )
                continue
            player = id_to_player.get(prompt_id)
            if player is None:
                log_llm.warning(
                    "Judge agent: Metric '%s' ranking contains invalid ID %s.",
                    metric_name,
                    prompt_id,
                )
                continue
            ranked_players.append(player)
            seen_ids.add(prompt_id)
        return ranked_players
