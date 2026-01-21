from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import trueskill as ts

from fuzzyevolve.core.models import Elite


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance for unit-normalized vectors: 1 - dot(a, b)."""
    return 1.0 - float(np.dot(a, b))


class CrowdedPool:
    """Fixed-size population with crowding via closest-pair elimination.

    Invariant: after insertion, the pool size is <= max_size. If size exceeds
    max_size, repeatedly find the closest pair in embedding space and remove the
    weaker individual by score (with deterministic tie-breaks).
    """

    def __init__(
        self,
        *,
        max_size: int,
        rng: random.Random | None = None,
        score_fn: Callable[[dict[str, ts.Rating]], float] | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        if score_fn is None:
            raise ValueError("score_fn is required.")
        self.max_size = int(max_size)
        self.rng = rng or random.Random()
        self._score = score_fn

        self._members: list[Elite] = []
        self._text_index: dict[str, Elite] = {}

    def __len__(self) -> int:
        return len(self._members)

    def contains_text(self, text: str) -> bool:
        return text in self._text_index

    def iter_elites(self) -> Iterable[Elite]:
        return tuple(self._members)

    @property
    def best(self) -> Elite:
        if not self._members:
            raise ValueError("Pool is empty.")
        return max(self._members, key=lambda elite: self._score(elite.ratings))

    def random_elite(self) -> Elite:
        if not self._members:
            raise ValueError("Cannot sample from an empty pool.")
        return self.rng.choice(self._members)

    def sample(self, k: int) -> list[Elite]:
        if k <= 0 or not self._members:
            return []
        return self.rng.sample(self._members, k=min(int(k), len(self._members)))

    def farthest_from(
        self, reference: Elite, *, exclude_texts: set[str] | None = None
    ) -> Elite | None:
        exclude_texts = exclude_texts or set()
        best: Elite | None = None
        best_dist: float | None = None
        for elite in self._members:
            if elite.text in exclude_texts:
                continue
            dist = cosine_distance(reference.embedding, elite.embedding)
            if best is None or best_dist is None or dist > best_dist:
                best = elite
                best_dist = dist
        return best

    def add_many(self, elites: Sequence[Elite]) -> None:
        """Add a batch, then eliminate until within max_size (order-independent)."""
        for elite in elites:
            if elite.text in self._text_index:
                continue
            self._members.append(elite)
            self._text_index[elite.text] = elite
        self._eliminate_until_limit()

    def add(self, elite: Elite) -> None:
        self.add_many([elite])

    def _eliminate_until_limit(self) -> None:
        while len(self._members) > self.max_size:
            i, j = self._closest_pair_indices()
            loser = self._pick_loser(self._members[i], self._members[j])
            self._remove_by_text(loser.text)

    def _remove_by_text(self, text: str) -> None:
        elite = self._text_index.pop(text, None)
        if elite is None:
            return
        self._members = [e for e in self._members if e.text != text]

    def _closest_pair_indices(self) -> tuple[int, int]:
        if len(self._members) < 2:
            raise ValueError("Need at least two elites to find a closest pair.")
        best_i = 0
        best_j = 1
        best_dist = cosine_distance(
            self._members[0].embedding, self._members[1].embedding
        )
        for i in range(len(self._members)):
            emb_i = self._members[i].embedding
            for j in range(i + 1, len(self._members)):
                dist = cosine_distance(emb_i, self._members[j].embedding)
                if dist < best_dist:
                    best_dist = dist
                    best_i, best_j = i, j
        return best_i, best_j

    def _pick_loser(self, a: Elite, b: Elite) -> Elite:
        """Return the elite to remove (lower is worse)."""
        a_score = float(self._score(a.ratings))
        b_score = float(self._score(b.ratings))
        if a_score != b_score:
            return a if a_score < b_score else b

        a_mu = _avg_mu(a.ratings)
        b_mu = _avg_mu(b.ratings)
        if a_mu != b_mu:
            return a if a_mu < b_mu else b

        a_sigma = _avg_sigma(a.ratings)
        b_sigma = _avg_sigma(b.ratings)
        if a_sigma != b_sigma:
            return a if a_sigma > b_sigma else b

        return a if a.text > b.text else b


def _avg_mu(ratings: dict[str, ts.Rating]) -> float:
    if not ratings:
        return 0.0
    return float(sum(r.mu for r in ratings.values()) / len(ratings))


def _avg_sigma(ratings: dict[str, ts.Rating]) -> float:
    if not ratings:
        return 0.0
    return float(sum(r.sigma for r in ratings.values()) / len(ratings))
