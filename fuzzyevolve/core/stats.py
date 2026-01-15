from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvolutionStats:
    judge_calls_total: int = 0
    judge_calls_failed: int = 0
    judge_invalid_total: int = 0
    judge_repair_attempts: int = 0

    mutations_proposed: int = 0
    patch_exact_success: int = 0
    patch_fail: int = 0

    children_judged: int = 0
    children_inserted: int = 0
    children_rejected_new_cell_gate: int = 0

    anchor_injected_total: int = 0
    battle_sizes: Counter[int] = field(default_factory=Counter)

    def record_battle_size(self, size: int) -> None:
        self.battle_sizes[size] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "judge_calls_total": self.judge_calls_total,
            "judge_calls_failed": self.judge_calls_failed,
            "judge_invalid_total": self.judge_invalid_total,
            "judge_repair_attempts": self.judge_repair_attempts,
            "mutations_proposed": self.mutations_proposed,
            "patch_exact_success": self.patch_exact_success,
            "patch_fail": self.patch_fail,
            "children_judged": self.children_judged,
            "children_inserted": self.children_inserted,
            "children_rejected_new_cell_gate": self.children_rejected_new_cell_gate,
            "anchor_injected_total": self.anchor_injected_total,
            "battle_sizes": dict(sorted(self.battle_sizes.items())),
        }
