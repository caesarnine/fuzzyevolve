from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import trueskill as ts

from fuzzyevolve.core.models import Anchor, Elite, Ratings


class AnchorPool:
    def __init__(
        self, metrics: Sequence[str], rng: random.Random | None = None
    ) -> None:
        self.metrics = list(metrics)
        if not self.metrics:
            raise ValueError("At least one metric is required for anchors.")
        self.rng = rng or random.Random()
        self._anchors: list[Anchor] = []
        self._text_index: dict[str, Anchor] = {}
        self.seed_anchor: Anchor | None = None

    def _make_ratings(self, mu: float, sigma: float) -> Ratings:
        return {metric: ts.Rating(mu=mu, sigma=sigma) for metric in self.metrics}

    def _add_anchor(self, anchor: Anchor, *, set_seed: bool = False) -> Anchor:
        existing = self._text_index.get(anchor.text)
        if existing is not None:
            if set_seed and self.seed_anchor is None:
                self.seed_anchor = existing
            return existing
        self._anchors.append(anchor)
        self._text_index[anchor.text] = anchor
        if set_seed:
            self.seed_anchor = anchor
        return anchor

    def add_seed(
        self,
        text: str,
        descriptor_fn: Callable[[str], dict[str, Any]],
        *,
        mu: float,
        sigma: float,
    ) -> Anchor:
        if self.seed_anchor is not None:
            return self.seed_anchor
        anchor = Anchor(
            text=text,
            descriptor=descriptor_fn(text),
            ratings=self._make_ratings(mu, sigma),
            age=0,
            label="SEED",
        )
        return self._add_anchor(anchor, set_seed=True)

    def add_ghost(self, elite: Elite, *, age: int | None = None) -> Anchor:
        anchor = Anchor(
            text=elite.text,
            descriptor=dict(elite.descriptor),
            ratings={
                name: ts.Rating(rating.mu, rating.sigma)
                for name, rating in elite.ratings.items()
            },
            age=elite.age if age is None else age,
            label="GHOST",
        )
        return self._add_anchor(anchor)

    def sample(
        self, max_count: int, *, exclude_texts: set[str] | None = None
    ) -> list[Anchor]:
        if max_count <= 0:
            return []
        if not self._anchors:
            return []
        exclude_texts = exclude_texts or set()

        chosen: list[Anchor] = []
        seen = set(exclude_texts)

        if self.seed_anchor and self.seed_anchor.text not in seen:
            chosen.append(self.seed_anchor)
            seen.add(self.seed_anchor.text)

        if len(chosen) >= max_count:
            return chosen[:max_count]

        candidates = [
            anchor
            for anchor in self._anchors
            if anchor is not self.seed_anchor and anchor.text not in seen
        ]
        self.rng.shuffle(candidates)
        for anchor in candidates:
            if anchor.text in seen:
                continue
            chosen.append(anchor)
            seen.add(anchor.text)
            if len(chosen) >= max_count:
                break
        return chosen


@dataclass(frozen=True, slots=True)
class AnchorPolicy:
    injection_probability: float
    max_per_battle: int
    ghost_interval: int
    seed_mu: float
    seed_sigma: float


class AnchorManager:
    def __init__(
        self,
        pool: AnchorPool,
        policy: AnchorPolicy,
        rng: random.Random,
    ) -> None:
        self.pool = pool
        self.policy = policy
        self.rng = rng

    def seed(
        self, text: str, *, descriptor_fn: Callable[[str], dict[str, Any]]
    ) -> Anchor:
        return self.pool.add_seed(
            text,
            descriptor_fn=descriptor_fn,
            mu=self.policy.seed_mu,
            sigma=self.policy.seed_sigma,
        )

    def maybe_sample(self, *, exclude_texts: set[str]) -> list[Anchor]:
        if self.policy.injection_probability <= 0:
            return []
        if self.rng.random() >= self.policy.injection_probability:
            return []
        if self.policy.max_per_battle <= 0:
            return []
        return self.pool.sample(self.policy.max_per_battle, exclude_texts=exclude_texts)

    def maybe_add_ghost(self, best: Elite, *, iteration: int) -> bool:
        if self.policy.ghost_interval <= 0:
            return False
        if iteration % self.policy.ghost_interval != 0:
            return False
        self.pool.add_ghost(best, age=iteration)
        return True
