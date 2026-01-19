from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fuzzyevolve.core.models import Anchor, Elite, RatedText


@dataclass(frozen=True, slots=True)
class Battle:
    participants: tuple[RatedText, ...]
    judged_children: tuple[Elite, ...]
    resort_elites: tuple[Elite, ...]
    frozen_indices: frozenset[int]

    @property
    def size(self) -> int:
        return len(self.participants)


def build_battle(
    *,
    parent: Elite,
    children: Sequence[Elite],
    anchors: Sequence[Anchor] = (),
    opponent: Elite | None = None,
) -> Battle:
    chosen_children = list(children)
    participants: list[RatedText] = [parent, *chosen_children, *anchors]
    if opponent is not None:
        participants.append(opponent)

    frozen_indices = frozenset(
        idx for idx, player in enumerate(participants) if isinstance(player, Anchor)
    )

    resort_elites: list[Elite] = [parent]
    if opponent is not None:
        resort_elites.append(opponent)

    return Battle(
        participants=tuple(participants),
        judged_children=tuple(chosen_children),
        resort_elites=tuple(resort_elites),
        frozen_indices=frozen_indices,
    )
