from __future__ import annotations

import random
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
    inspiration: Elite | None = None,
    max_battle_size: int,
    rng: random.Random,
) -> Battle:
    if max_battle_size < 2:
        raise ValueError("max_battle_size must be >= 2")

    child_budget = max_battle_size - 1
    chosen_children = list(children)
    if len(chosen_children) > child_budget:
        chosen_children = rng.sample(chosen_children, k=child_budget)

    participants: list[RatedText] = [parent, *chosen_children]

    available = max_battle_size - len(participants)
    for anchor in anchors:
        if available <= 0:
            break
        participants.append(anchor)
        available -= 1

    if opponent is not None and available > 0:
        participants.append(opponent)
        available -= 1

    if inspiration is not None and inspiration not in participants and available > 0:
        participants.append(inspiration)

    frozen_indices = frozenset(
        idx for idx, player in enumerate(participants) if isinstance(player, Anchor)
    )

    resort_elites: list[Elite] = [parent]
    if opponent is not None and opponent in participants:
        resort_elites.append(opponent)
    if (
        inspiration is not None
        and inspiration in participants
        and inspiration is not parent
    ):
        resort_elites.append(inspiration)

    return Battle(
        participants=tuple(participants),
        judged_children=tuple(chosen_children),
        resort_elites=tuple(resort_elites),
        frozen_indices=frozen_indices,
    )
