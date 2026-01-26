from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Callable

import numpy as np
import trueskill as ts

from fuzzyevolve.core.multiobjective import Scalarizer, dominates
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
        pruning_strategy: str = "closest_pair",
        knn_k: int = 8,
        metrics: Sequence[str] | None = None,
        score_lcb_c: float = 2.0,
        scalarizer: Scalarizer | None = None,
        pareto: bool = False,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        if score_fn is None:
            raise ValueError("score_fn is required.")
        self.max_size = int(max_size)
        self.rng = rng or random.Random()
        self._score = score_fn
        self.pruning_strategy = str(pruning_strategy)
        self.knn_k = int(knn_k)
        self.scalarizer = scalarizer
        if metrics is None and scalarizer is not None:
            metrics = list(scalarizer.metrics)
        self.metrics = [m.strip() for m in (metrics or []) if m.strip()]
        self.score_lcb_c = float(score_lcb_c)
        self.pareto = bool(pareto)

        if self.pruning_strategy not in {"closest_pair", "knn_local_competition"}:
            raise ValueError(
                "pruning_strategy must be 'closest_pair' or 'knn_local_competition'."
            )
        if self.knn_k <= 0:
            raise ValueError("knn_k must be a positive integer.")
        if self.pareto and not self.metrics:
            raise ValueError("pareto=True requires a non-empty metrics list.")

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
        """Add a batch, pruning until within max_size."""
        if self.pruning_strategy == "closest_pair":
            # Order-independent: insert everything, then prune globally.
            for elite in elites:
                if elite.text in self._text_index:
                    continue
                self._members.append(elite)
                self._text_index[elite.text] = elite
            self._eliminate_until_limit()
            return

        if self.pruning_strategy == "knn_local_competition":
            # Order-dependent by nature; shuffle to reduce systematic bias.
            incoming = list(elites)
            self.rng.shuffle(incoming)
            for elite in incoming:
                self._add_knn_local_competition(elite)
            return

        raise ValueError(f"Unknown pruning_strategy '{self.pruning_strategy}'.")

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
        use_multiobjective = bool(self.metrics) and (
            self.pareto or self.scalarizer is not None
        )
        if use_multiobjective:
            a_lcb = self._lcb_vector(a)
            b_lcb = self._lcb_vector(b)

            if self.pareto:
                if dominates(a_lcb, b_lcb):
                    return b
                if dominates(b_lcb, a_lcb):
                    return a

            a_score = self._scalarized_lcb(a_lcb)
            b_score = self._scalarized_lcb(b_lcb)
        else:
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

    def _lcb_vector(self, elite: Elite) -> list[float]:
        c = float(self.score_lcb_c)
        out: list[float] = []
        for metric in self.metrics:
            r = elite.ratings.get(metric)
            if r is None:
                out.append(float("-inf"))
            else:
                out.append(float(r.mu) - c * float(r.sigma))
        return out

    def _scalarized_lcb(self, values: Sequence[float]) -> float:
        if not values:
            return float("-inf")
        if self.scalarizer is None:
            weight = 1.0 / len(values)
            return sum(weight * float(v) for v in values)
        weights = self.scalarizer.weights_for(self.metrics)
        return sum(w * float(v) for w, v in zip(weights, values))

    def _add_knn_local_competition(self, elite: Elite) -> None:
        if elite.text in self._text_index:
            return
        if len(self._members) < self.max_size:
            self._members.append(elite)
            self._text_index[elite.text] = elite
            return
        if not self._members:
            self._members.append(elite)
            self._text_index[elite.text] = elite
            return

        k = min(self.knn_k, len(self._members))
        neighbor_indices = self._knn_indices(elite.embedding, k=k)
        neighborhood = [self._members[idx] for idx in neighbor_indices]

        loser = self._worst_of([elite, *neighborhood])
        if loser is elite:
            return

        self._remove_by_text(loser.text)
        self._members.append(elite)
        self._text_index[elite.text] = elite

    def _knn_indices(self, embedding: np.ndarray, *, k: int) -> list[int]:
        if k <= 0 or not self._members:
            return []
        scored: list[tuple[float, int]] = []
        for idx, member in enumerate(self._members):
            scored.append((cosine_distance(embedding, member.embedding), idx))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [idx for _dist, idx in scored[:k]]

    def _worst_of(self, elites: Sequence[Elite]) -> Elite:
        if not elites:
            raise ValueError("Need at least one elite to pick a worst.")
        worst = elites[0]
        for other in elites[1:]:
            worst = self._pick_loser(worst, other)
        return worst


def _avg_mu(ratings: dict[str, ts.Rating]) -> float:
    if not ratings:
        return 0.0
    return float(sum(r.mu for r in ratings.values()) / len(ratings))


def _avg_sigma(ratings: dict[str, ts.Rating]) -> float:
    if not ratings:
        return 0.0
    return float(sum(r.sigma for r in ratings.values()) / len(ratings))
