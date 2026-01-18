from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import trueskill as ts

from fuzzyevolve.core.models import RatedText, Ratings


@dataclass(frozen=True, slots=True)
class BattleRanking:
    tiers_by_metric: dict[str, list[list[int]]]


class RatingSystem:
    def __init__(
        self,
        metrics: Sequence[str],
        *,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 3.0,
        tau: float = 25.0 / 3.0 / 50.0,
        draw_probability: float = 0.2,
        score_lcb_c: float = 2.0,
        child_prior_tau: float = 4.0,
    ) -> None:
        self.metrics = list(metrics)
        if not self.metrics:
            raise ValueError("At least one metric is required.")
        self.score_lcb_c = score_lcb_c
        self.child_prior_tau = child_prior_tau
        self.envs = {
            metric: ts.TrueSkill(
                mu=mu,
                sigma=sigma,
                beta=beta,
                tau=tau,
                draw_probability=draw_probability,
            )
            for metric in self.metrics
        }

    def new_ratings(self) -> Ratings:
        return {metric: self.envs[metric].create_rating() for metric in self.metrics}

    def ensure_ratings(self, player: RatedText) -> None:
        missing = [metric for metric in self.metrics if metric not in player.ratings]
        if not missing:
            return
        for metric in missing:
            player.ratings[metric] = self.envs[metric].create_rating()

    def score(self, ratings: Mapping[str, ts.Rating]) -> float:
        c = self.score_lcb_c
        weight = 1.0 / len(self.metrics)
        return sum(
            weight * (ratings[m].mu - c * ratings[m].sigma) for m in self.metrics
        )

    def metric_lcb(self, rating: ts.Rating) -> float:
        return rating.mu - self.score_lcb_c * rating.sigma

    def init_child_ratings(self, parent: RatedText) -> Ratings:
        if self.child_prior_tau <= 0:
            return {
                metric: ts.Rating(rating.mu, rating.sigma)
                for metric, rating in parent.ratings.items()
                if metric in self.metrics
            }
        ratings: Ratings = {}
        for metric in self.metrics:
            base = parent.ratings[metric]
            sigma = math.sqrt(base.sigma * base.sigma + self.child_prior_tau**2)
            ratings[metric] = ts.Rating(mu=base.mu, sigma=sigma)
        return ratings

    def validate_ranking(
        self, ranking: BattleRanking, *, total_players: int
    ) -> tuple[bool, str | None]:
        expected_ids = set(range(total_players))
        errors: list[str] = []

        for metric in self.metrics:
            tiers = ranking.tiers_by_metric.get(metric)
            if tiers is None:
                errors.append(f"missing metric '{metric}'")
                continue
            if not tiers:
                errors.append(f"metric '{metric}' has no tiers")
                continue
            empty_tiers = [idx for idx, tier in enumerate(tiers) if not tier]
            if empty_tiers:
                errors.append(f"metric '{metric}' has empty tiers at {empty_tiers}")
                continue
            ranked_ids = [player_id for tier in tiers for player_id in tier]
            ranked_set = set(ranked_ids)
            if ranked_set != expected_ids or len(ranked_ids) != total_players:
                missing = expected_ids - ranked_set
                extra = ranked_set - expected_ids
                errors.append(
                    f"metric '{metric}' ids mismatch missing={sorted(missing)} extra={sorted(extra)}"
                )

        extra_metrics = [m for m in ranking.tiers_by_metric if m not in self.metrics]
        if extra_metrics:
            errors.append(f"unknown metrics {sorted(extra_metrics)}")

        if errors:
            return False, "; ".join(errors)
        return True, None

    def apply_ranking(
        self,
        players: Sequence[RatedText],
        ranking: BattleRanking,
        *,
        frozen_indices: set[int] | None = None,
    ) -> None:
        frozen_indices = frozen_indices or set()
        for player in players:
            self.ensure_ratings(player)

        ok, err = self.validate_ranking(ranking, total_players=len(players))
        if not ok:
            raise ValueError(f"Invalid ranking: {err}")

        for metric in self.metrics:
            tiers = ranking.tiers_by_metric[metric]
            ranked_players: list[RatedText] = []
            rating_groups: list[list[ts.Rating]] = []
            ranks: list[int] = []
            for rank, tier in enumerate(tiers):
                for player_idx in tier:
                    ranked_players.append(players[player_idx])
                    rating_groups.append([players[player_idx].ratings[metric]])
                    ranks.append(rank)
            updated = self.envs[metric].rate(rating_groups, ranks=ranks)
            for player, idx, new_rating in zip(
                ranked_players,
                [idx for tier in tiers for idx in tier],
                updated,
            ):
                if idx in frozen_indices:
                    continue
                player.ratings[metric] = new_rating[0]
