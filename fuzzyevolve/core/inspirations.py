from __future__ import annotations

import random
from dataclasses import dataclass

from fuzzyevolve.core.archive import MapElitesArchive
from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.ratings import RatingSystem


@dataclass(frozen=True, slots=True)
class Inspiration:
    elite: Elite
    label: str


class InspirationPicker:
    def __init__(
        self,
        *,
        rating: RatingSystem,
        rng: random.Random,
    ) -> None:
        self.rating = rating
        self.rng = rng

    def pick(
        self,
        archive: MapElitesArchive,
        parent: Elite,
        *,
        count: int,
    ) -> list[Inspiration]:
        if count <= 0:
            return []

        candidates = [elite for elite in archive.iter_elites() if elite is not parent]
        if not candidates:
            return []

        for elite in [parent, *candidates]:
            self.rating.ensure_ratings(elite)

        budget = min(count, len(candidates))
        if budget <= 0:
            return []

        chosen: list[Inspiration] = []
        seen_texts = {parent.text}

        def available() -> list[Elite]:
            return [e for e in candidates if e.text not in seen_texts]

        def add(elite: Elite, label: str) -> None:
            chosen.append(Inspiration(elite=elite, label=label))
            seen_texts.add(elite.text)

        # 1) Mentor: best on parent's weakest metric (by LCB).
        mentor_metric = min(
            self.rating.metrics,
            key=lambda m: self.rating.metric_lcb(parent.ratings[m]),
        )
        remaining = available()
        if remaining:
            parent_lcb = self.rating.metric_lcb(parent.ratings[mentor_metric])
            mentor = max(
                remaining,
                key=lambda e: self.rating.metric_lcb(e.ratings[mentor_metric]),
            )
            mentor_lcb = self.rating.metric_lcb(mentor.ratings[mentor_metric])
            if mentor_lcb > parent_lcb:
                add(
                    mentor,
                    f"MENTOR (for {mentor_metric}, Δlcb={mentor_lcb - parent_lcb:+.3f})",
                )
                if len(chosen) >= budget:
                    return chosen

        # 2) Champion (or runner-up): best archive score.
        remaining = available()
        if remaining:
            parent_score = self.rating.score(parent.ratings)
            island_best = max(
                [parent, *candidates], key=lambda e: self.rating.score(e.ratings)
            )
            if island_best.text not in seen_texts:
                champion = island_best
                label = f"CHAMPION (island best, Δscore={self.rating.score(champion.ratings) - parent_score:+.3f})"
            else:
                champion = max(remaining, key=lambda e: self.rating.score(e.ratings))
                label = f"RUNNER-UP (island #2, Δscore={self.rating.score(champion.ratings) - parent_score:+.3f})"
            add(champion, label)
            if len(chosen) >= budget:
                return chosen

        # 3) Random fill.
        while len(chosen) < budget:
            remaining = available()
            if not remaining:
                break
            pick = self.rng.choice(remaining)
            add(pick, "RANDOM")

        return chosen
