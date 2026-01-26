from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def dominates(a: Sequence[float], b: Sequence[float], *, eps: float = 1e-12) -> bool:
    """Return True if vector `a` Pareto-dominates `b` (with tolerance)."""
    if len(a) != len(b):
        raise ValueError("dominates() requires vectors of equal length.")
    ge_all = True
    gt_any = False
    for av, bv in zip(a, b):
        if av < bv - eps:
            ge_all = False
            break
        if av > bv + eps:
            gt_any = True
    return ge_all and gt_any


def nondominated_indices(
    vectors: Sequence[Sequence[float]], *, eps: float = 1e-12
) -> list[int]:
    """Indices of vectors that are not dominated by any other vector."""
    n = len(vectors)
    if n <= 1:
        return list(range(n))
    out: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(vectors[j], vectors[i], eps=eps):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return out


class Scalarizer:
    """Samples random metric weights (Dirichlet) for scalarization.

    Weights are stored on the instance so one sample can be reused across an
    iteration by both selection and pruning.
    """

    def __init__(
        self,
        metrics: Sequence[str],
        *,
        rng: random.Random,
        dirichlet_alpha: float = 1.0,
        balanced_probability: float = 0.2,
        enabled: bool = True,
    ) -> None:
        self.metrics = [m.strip() for m in metrics if m.strip()]
        if not self.metrics:
            raise ValueError("Scalarizer requires at least one metric.")
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0.")
        if not (0.0 <= balanced_probability <= 1.0):
            raise ValueError("balanced_probability must be between 0 and 1.")
        self.rng = rng
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.balanced_probability = float(balanced_probability)
        self.enabled = bool(enabled)
        self.weights = self._equal_weights()
        self.last_source = "balanced"

    def _equal_weights(self) -> list[float]:
        return [1.0 / len(self.metrics)] * len(self.metrics)

    def _sample_dirichlet(self) -> list[float]:
        if len(self.metrics) == 1:
            return [1.0]
        draws = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in self.metrics]
        total = float(sum(draws))
        if total <= 0.0:
            return self._equal_weights()
        return [float(d) / total for d in draws]

    def sample(self) -> dict[str, float]:
        """Sample and set current weights; returns weights by metric."""
        if not self.enabled:
            self.weights = self._equal_weights()
            self.last_source = "disabled"
            return self.weights_by_metric()

        if self.rng.random() < self.balanced_probability:
            self.weights = self._equal_weights()
            self.last_source = "balanced"
        else:
            self.weights = self._sample_dirichlet()
            self.last_source = "dirichlet"
        return self.weights_by_metric()

    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Set current weights from a mapping; used for deterministic tests."""
        if not weights:
            self.weights = self._equal_weights()
            self.last_source = "manual"
            return
        w = [float(weights.get(metric, 0.0)) for metric in self.metrics]
        total = float(sum(w))
        if total <= 0.0:
            self.weights = self._equal_weights()
            self.last_source = "manual"
            return
        self.weights = [v / total for v in w]
        self.last_source = "manual"

    def weights_by_metric(self) -> dict[str, float]:
        return {metric: float(self.weights[i]) for i, metric in enumerate(self.metrics)}

    def weights_for(self, metrics: Sequence[str]) -> list[float]:
        """Return weights aligned to `metrics` (renormalized if needed)."""
        requested = [m.strip() for m in metrics if m.strip()]
        if not requested:
            return []
        if requested == self.metrics:
            return list(self.weights)

        wmap = self.weights_by_metric()
        aligned = [float(wmap.get(metric, 0.0)) for metric in requested]
        total = float(sum(aligned))
        if total <= 0.0:
            return [1.0 / len(requested)] * len(requested)
        return [w / total for w in aligned]

