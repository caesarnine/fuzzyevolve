from __future__ import annotations

import logging
import random
from typing import Iterable, Sequence

import trueskill as ts
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.scoring import make_envs
from fuzzyevolve.llm.client import LLMProvider
from fuzzyevolve.llm.parsing import parse_judge_response
from fuzzyevolve.llm.prompts import build_rank_prompt

log_llm = logging.getLogger("llm")


class LLMJudge:
    def __init__(
        self,
        llm_provider: LLMProvider,
        metrics: Sequence[str],
        rng: random.Random | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.metrics = list(metrics)
        self.envs = make_envs(self.metrics)
        self.rng = rng or random.Random()

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
        try:
            raw_response = self.llm_provider.call(prompt=prompt)
        except Exception:
            log_llm.error(
                "Judge LLM call failed outright — skipping rating update for this batch."
            )
            return

        if not raw_response:
            log_llm.error("Judge LLM returned an empty response — skipping update.")
            return

        thinking, parsed_rankings = parse_judge_response(
            raw_response, self.metrics, log_llm
        )
        if thinking:
            log_llm.info("Judge LLM rationale:\n%s", thinking)
        else:
            log_llm.warning("Judge LLM: No thinking process extracted from response.")

        if not parsed_rankings:
            log_llm.error(
                "Judge LLM: No valid rankings extracted. Skipping rating update."
            )
            return

        for metric_name in self.metrics:
            ranked_ids = parsed_rankings.get(metric_name)
            if not ranked_ids:
                log_llm.warning(
                    "Judge LLM: Missing ranking for metric '%s'.",
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
                    "Judge LLM: TrueSkill update failed for metric '%s': %s",
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
                    "Judge LLM: Metric '%s' ranking contains duplicate ID %s.",
                    metric_name,
                    prompt_id,
                )
                continue
            player = id_to_player.get(prompt_id)
            if player is None:
                log_llm.warning(
                    "Judge LLM: Metric '%s' ranking contains invalid ID %s.",
                    metric_name,
                    prompt_id,
                )
                continue
            ranked_players.append(player)
            seen_ids.add(prompt_id)
        return ranked_players
