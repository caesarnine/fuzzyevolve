from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Critique:
    """Structured, reusable guidance for a single parent text."""

    summary: str = ""
    preserve: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
