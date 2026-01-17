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


def apply_edits(text: str, edits: list[tuple[str, str]]) -> PatchResult:
    if not edits:
        return PatchResult(False, None)

    spans: list[tuple[int, int, str]] = []
    for search, replace in edits:
        if not search:
            return PatchResult(False, None)
        start = text.find(search)
        if start == -1:
            return PatchResult(False, None)
        end = start + len(search)
        spans.append((start, end, replace))

    spans.sort(key=lambda item: item[0])
    prev_end = -1
    for start, end, _ in spans:
        if start < prev_end:
            return PatchResult(False, None)
        prev_end = end

    new_text = text
    for start, end, replace in reversed(spans):
        new_text = new_text[:start] + replace + new_text[end:]
    return PatchResult(True, new_text)
