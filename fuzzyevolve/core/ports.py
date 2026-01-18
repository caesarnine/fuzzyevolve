from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from fuzzyevolve.core.battle import Battle
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.ratings import BattleRanking


class Mutator(Protocol):
    def propose(
        self,
        *,
        parent: Elite,
        inspirations: Sequence[Elite],
        inspiration_labels: Sequence[str] | None = None,
    ) -> Sequence[MutationCandidate]: ...


class Ranker(Protocol):
    def rank(
        self,
        *,
        metrics: Sequence[str],
        battle: Battle,
        metric_descriptions: Mapping[str, str] | None = None,
    ) -> BattleRanking | None: ...
