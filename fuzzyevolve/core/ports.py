from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from typing import Protocol

from fuzzyevolve.core.battle import Battle
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite, MutationCandidate
from fuzzyevolve.core.ratings import BattleRanking


class Critic(Protocol):
    def critique(
        self,
        *,
        parent: Elite,
    ) -> Critique | None: ...


class MutationOperator(Protocol):
    def propose(
        self,
        *,
        parent: Elite,
        critique: Critique | None,
        focus: str | None = None,
    ) -> Sequence[str]: ...


class Mutator(Protocol):
    def propose(
        self,
        *,
        parent: Elite,
        critique: Critique | None,
        max_candidates: int,
        mutation_executor: ThreadPoolExecutor | None = None,
    ) -> Sequence[MutationCandidate]: ...


class Ranker(Protocol):
    def rank(
        self,
        *,
        metrics: Sequence[str],
        battle: Battle,
        metric_descriptions: Mapping[str, str] | None = None,
    ) -> BattleRanking: ...
