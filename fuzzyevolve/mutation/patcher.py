from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

log_patch = logging.getLogger("patcher")


@dataclass(frozen=True, slots=True)
class PatchResult:
    success: bool
    used_fuzzy: bool
    match_score: float | None
    new_text: str | None


@dataclass(frozen=True, slots=True)
class PatchConfig:
    fuzzy_enabled: bool
    threshold: float
    margin: float
    min_search_len: int
    max_window_expansion: float


def apply_patch(text: str, search: str, replace: str, cfg: PatchConfig) -> PatchResult:
    if not search:
        return PatchResult(False, False, None, None)

    idx = text.find(search)
    if idx != -1:
        new_text = text.replace(search, replace, 1)
        return PatchResult(True, False, 1.0, new_text)

    if not cfg.fuzzy_enabled:
        return PatchResult(False, False, None, None)
    if len(search) < cfg.min_search_len:
        return PatchResult(False, False, None, None)

    match = _find_best_fuzzy_match(text, search, cfg.max_window_expansion)
    if match is None:
        return PatchResult(False, False, None, None)

    best_score, best_adjusted, start, end, runner_up = match
    if best_score < cfg.threshold:
        return PatchResult(False, True, best_score, None)
    if runner_up is not None and (best_adjusted - runner_up) < cfg.margin:
        return PatchResult(False, True, best_score, None)

    new_text = text[:start] + replace + text[end:]
    log_patch.debug("Fuzzy patch applied score=%.3f span=%d:%d", best_score, start, end)
    return PatchResult(True, True, best_score, new_text)


def _find_best_fuzzy_match(
    text: str, search: str, max_window_expansion: float
) -> tuple[float, float, int, int, float | None] | None:
    if not text:
        return None
    target_len = len(search)
    min_len = max(1, int(target_len * (1 - max_window_expansion)))
    max_len = max(min_len, int(target_len * (1 + max_window_expansion)))
    if len(text) < min_len:
        return None

    best_score = -1.0
    best_adjusted = -1.0
    best_span: tuple[int, int] | None = None
    length_penalty = 0.5
    for length in range(min_len, max_len + 1):
        for start in range(0, len(text) - length + 1):
            candidate = text[start : start + length]
            score = fuzz.ratio(search, candidate) / 100.0
            len_diff = abs(length - target_len)
            adjusted = score - (len_diff / target_len) * length_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_score = score
                best_span = (start, start + length)

    if best_span is None:
        return None
    runner_up = -1.0
    for length in range(min_len, max_len + 1):
        for start in range(0, len(text) - length + 1):
            if (start, start + length) == best_span:
                continue
            candidate = text[start : start + length]
            score = fuzz.ratio(search, candidate) / 100.0
            len_diff = abs(length - target_len)
            adjusted = score - (len_diff / target_len) * length_penalty
            if adjusted > runner_up:
                runner_up = adjusted
    runner_up_val = runner_up if runner_up >= 0 else None
    return best_score, best_adjusted, best_span[0], best_span[1], runner_up_val
