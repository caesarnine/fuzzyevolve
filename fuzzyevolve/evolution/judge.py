"""
This module contains the MultiMetricJudge class, which is responsible for evaluating and ranking elites.
"""

import logging
import random
from typing import Dict, List, Tuple

from fuzzyevolve.llm.client import LLMProvider
from fuzzyevolve.evolution.scoring import make_envs
from fuzzyevolve.llm.parsers import parse_llm_judge_response
from fuzzyevolve.llm.prompts import make_rank_prompt

log_llm = logging.getLogger("llm")


class MultiMetricJudge:
    """A class that uses a language model to judge elites based on multiple metrics."""

    def __init__(self, llm_provider: LLMProvider, metrics: List[str]):
        self.llm_provider = llm_provider
        self.metrics = metrics
        self.envs = make_envs(metrics)

    def ensure_ratings(self, elite: Dict):
        """Ensures that an elite has a rating for each metric."""
        if "rating" not in elite:
            elite["rating"] = {m: self.envs[m].create_rating() for m in self.metrics}

    def rank_and_rate(self, players: List[Dict]):
        """Ranks and rates a list of players using the language model."""
        for p in players:
            self.ensure_ratings(p)

        if not players:
            log_llm.warning("Rank and Rate called with no players. Skipping.")
            return

        num_players = len(players)

        # 1. Create a list of original indices and shuffle them.
        # These original_indices refer to the positions in the input 'players' list.
        shuffled_original_indices = list(range(num_players))
        random.shuffle(shuffled_original_indices)

        # 2. Prepare items for the prompt with new, temporary IDs (0 to N-1 for LLM).
        # And create a mapping from these temporary prompt_ids back to original player objects.
        prompt_items_for_llm: List[Tuple[int, Dict]] = []
        # Holds the player object at the position of its temporary prompt_id
        temp_prompt_id_to_player_obj_map: List[Dict] = [{} for _ in range(num_players)]

        for temp_prompt_id, original_idx_val in enumerate(shuffled_original_indices):
            player_obj = players[original_idx_val]
            prompt_items_for_llm.append((temp_prompt_id, player_obj))
            temp_prompt_id_to_player_obj_map[temp_prompt_id] = player_obj

        prompt = make_rank_prompt(self.metrics, prompt_items_for_llm)

        raw_response: str | None = None
        try:
            # Expecting raw string with pseudo-XML, so response_format is None
            raw_response = self.llm_provider.call(prompt=prompt)
        except Exception:
            # llm_call already logs details of the exception
            log_llm.error(
                "Judge LLM call failed outright — skipping rating update for this batch."
            )
            return

        if not raw_response:
            log_llm.error(
                "Judge LLM returned an empty response — skipping rating update."
            )
            return

        # Log the raw response before parsing attempt, can be very verbose
        # log_llm.debug("Judge LLM raw response:\n%s", raw_response)

        thinking_process, parsed_rankings = parse_llm_judge_response(
            raw_response, self.metrics, log_llm
        )

        if thinking_process:
            log_llm.info(
                "Judge LLM rationale:\n%s", thinking_process
            )  # Log thinking as INFO
        else:
            log_llm.warning("Judge LLM: No thinking process extracted from response.")

        if not parsed_rankings:
            log_llm.error(
                "Judge LLM: No valid rankings extracted from response after parsing. Skipping rating update for this batch."
            )
            return

        # Iterate through metrics defined in config to update ratings
        for metric_name_from_config in self.metrics:
            # 'ranked_temp_prompt_ids' contains the temporary prompt IDs (0..N-1)
            # as ranked by the LLM for this metric (e.g., [2, 0, 1])
            ranked_temp_prompt_ids = parsed_rankings.get(metric_name_from_config)

            if (
                not isinstance(ranked_temp_prompt_ids, list)
                or len(ranked_temp_prompt_ids) < 1
            ):
                log_llm.warning(
                    "Judge LLM: Insufficient or invalid ranking list for metric '%s'. Found: %s. Skipping this metric.",
                    metric_name_from_config,
                    ranked_temp_prompt_ids,
                )
                continue

            # Validate temporary prompt IDs returned by LLM
            valid_ranked_players_for_metric = []
            # Ensure no duplicates from LLM for this metric, using temporary prompt IDs
            seen_temp_prompt_ids_in_ranking = set()
            # The set of valid temporary prompt IDs LLM could have returned
            valid_temp_prompt_id_set = set(range(num_players))

            for temp_prompt_id_from_llm in ranked_temp_prompt_ids:
                if (
                    temp_prompt_id_from_llm in valid_temp_prompt_id_set
                    and temp_prompt_id_from_llm not in seen_temp_prompt_ids_in_ranking
                ):
                    # Map temp_prompt_id back to the actual player object
                    player_obj = temp_prompt_id_to_player_obj_map[
                        temp_prompt_id_from_llm
                    ]
                    valid_ranked_players_for_metric.append(player_obj)
                    seen_temp_prompt_ids_in_ranking.add(temp_prompt_id_from_llm)
                else:
                    log_llm.warning(
                        "Judge LLM: Metric '%s' ranking contains invalid or duplicate temporary prompt ID: %s. It will be ignored.",
                        metric_name_from_config,
                        temp_prompt_id_from_llm,
                    )
            if (
                len(valid_ranked_players_for_metric) < 1
            ):  # Can we rate a single player? TrueSkill typically needs >=2 teams
                # For single player "rating", it's more like assigning a score.
                # If TrueSkill needs >=2, this check should be < 2.
                # ts.rate() needs at least one rating group. A "team" of one is one rating group.
                # If only one player ranked, they are rank 0.
                log_llm.warning(
                    "Judge LLM: Not enough valid players for metric '%s' after validation (%d found). Skipping rating update for this metric.",
                    metric_name_from_config,
                    len(valid_ranked_players_for_metric),
                )
                continue

            # Proceed with TrueSkill update using valid_ranked_players_for_metric
            # Each player is a "team" of one.
            current_ratings_for_trueskill = [
                [p["rating"][metric_name_from_config]]
                for p in valid_ranked_players_for_metric
            ]

            try:
                # Ranks are 0, 1, 2... for the best, second best, etc.
                updated_ratings_tuples = self.envs[metric_name_from_config].rate(
                    current_ratings_for_trueskill,
                    ranks=list(range(len(valid_ranked_players_for_metric))),
                )
                # Update player objects with new TrueSkill Rating objects
                for player_obj, new_rating_tuple in zip(
                    valid_ranked_players_for_metric, updated_ratings_tuples
                ):
                    player_obj["rating"][metric_name_from_config] = new_rating_tuple[
                        0
                    ]  # new_rating_tuple is (Rating(),)
            except IndexError as e:
                log_llm.error(
                    "Judge LLM: IndexError during TrueSkill update for metric '%s', possibly due to issues with "
                    "valid_ranked_players_for_metric or their ratings. Error: %s. Players: %s",
                    metric_name_from_config,
                    e,
                    valid_ranked_players_for_metric,
                )
            except Exception as e:
                log_llm.error(
                    "Judge LLM: Unexpected error during TrueSkill update for metric '%s': %s. Skipping.",
                    metric_name_from_config,
                    e,
                )
