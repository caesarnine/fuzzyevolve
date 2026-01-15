from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatchResult:
    success: bool
    new_text: str | None


def apply_patch(text: str, search: str, replace: str) -> PatchResult:
    if not search:
        return PatchResult(False, None)

    idx = text.find(search)
    if idx == -1:
        return PatchResult(False, None)

    new_text = text.replace(search, replace, 1)
    return PatchResult(True, new_text)
