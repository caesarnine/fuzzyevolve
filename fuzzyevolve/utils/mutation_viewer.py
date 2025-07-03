# utils/mutation_viewer.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class MutationEvent:
    iteration: int
    island: int
    parent_s: float  # μ-3σ
    child_s: float
    diff: str  # unified diff text


class MutationViewer:
    """
    Keeps the last N mutation events and renders them.

    Usage:
        viewer = MutationViewer(history=6)
        ...
        viewer.push(event)   # whenever you accept a child
        live.update(viewer)  # will auto-refresh
    """

    def __init__(self, history: int = 6):
        self.events: Deque[MutationEvent] = deque(maxlen=history)

    # ───────── public ────────────────────────────────────────────────
    def push(self, e: MutationEvent) -> None:
        self.events.appendleft(e)

    # ──────── render helpers ─────────────────────────────────────────
    def _colour_diff(self, diff: str) -> RenderableType:
        """
        Simple colourisation: red for removed lines, green for added.
        """
        text = Text()
        for ln in diff.splitlines():
            if ln.startswith("+") and not ln.startswith("+++"):
                text.append(ln + "\n", style="green")
            elif ln.startswith("-") and not ln.startswith("---"):
                text.append(ln + "\n", style="red")
            else:
                text.append(ln + "\n")
        return Panel(text, padding=(0, 1), border_style="dim")

    # Rich will call this automatically inside Live.
    def __rich__(self) -> RenderableType:
        if not self.events:
            return Panel("no mutations yet", padding=(1, 2), border_style="dim")

        tbl = Table(show_edge=False, pad_edge=False)
        tbl.add_column("#", justify="right", style="bold")
        tbl.add_column("isle", justify="right")
        tbl.add_column("parent", justify="right")
        tbl.add_column("child", justify="right")

        diff_panels: List[RenderableType] = []
        for e in self.events:
            tbl.add_row(
                str(e.iteration),
                str(e.island),
                f"{e.parent_s:.3f}",
                f"{e.child_s:.3f}",
            )
            diff_panels.append(self._colour_diff(e.diff))

        return Group(tbl, *diff_panels)
