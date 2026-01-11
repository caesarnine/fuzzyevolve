from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import trueskill as ts

from fuzzyevolve.core.models import Elite


class ReferencePool:
    def __init__(self, metrics: Sequence[str], rng: random.Random | None = None) -> None:
        self.metrics = list(metrics)
        self.rng = rng or random.Random()
        self._anchors: list[Elite] = []
        self._text_index: dict[str, Elite] = {}
        self.seed_anchor: Elite | None = None

    def _make_ratings(self, mu: float, sigma: float) -> dict[str, ts.Rating]:
        return {metric: ts.Rating(mu=mu, sigma=sigma) for metric in self.metrics}

    def _add_anchor(self, anchor: Elite, *, set_seed: bool = False) -> Elite:
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

    def add_seed_anchor(
        self,
        text: str,
        descriptor_fn: Callable[[str], dict[str, Any]],
        mu: float,
        sigma: float,
    ) -> Elite:
        if self.seed_anchor is not None:
            return self.seed_anchor
        anchor = Elite(
            text=text,
            descriptor=descriptor_fn(text),
            ratings=self._make_ratings(mu, sigma),
            age=0,
            frozen=True,
        )
        return self._add_anchor(anchor, set_seed=True)

    def add_ghost_anchor(self, elite: Elite, age: int | None = None) -> Elite:
        anchor = elite.clone()
        anchor.frozen = True
        anchor.cell_key = None
        if age is not None:
            anchor.age = age
        return self._add_anchor(anchor)

    def sample(
        self, max_count: int, exclude_texts: set[str] | None = None
    ) -> list[Elite]:
        if max_count <= 0 or not self._anchors:
            return []
        exclude_texts = exclude_texts or set()
        anchors: list[Elite] = []
        seen_texts = set(exclude_texts)

        if self.seed_anchor and self.seed_anchor.text not in seen_texts:
            anchors.append(self.seed_anchor)
            seen_texts.add(self.seed_anchor.text)

        if len(anchors) >= max_count:
            return anchors[:max_count]

        candidates = [
            anchor
            for anchor in self._anchors
            if anchor is not self.seed_anchor and anchor.text not in seen_texts
        ]
        self.rng.shuffle(candidates)
        for anchor in candidates:
            if anchor.text in seen_texts:
                continue
            anchors.append(anchor)
            seen_texts.add(anchor.text)
            if len(anchors) >= max_count:
                break
        return anchors
