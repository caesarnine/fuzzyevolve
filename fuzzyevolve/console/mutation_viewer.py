from __future__ import annotations

from collections import deque
from typing import Deque, List

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fuzzyevolve.core.models import MutationEvent


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

    def push(self, event: MutationEvent) -> None:
        self.events.appendleft(event)

    def _colour_diff(self, diff: str) -> RenderableType:
        text = Text()
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                text.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                text.append(line + "\n", style="red")
            else:
                text.append(line + "\n")
        return Panel(text, padding=(0, 1), border_style="dim")

    def __rich__(self) -> RenderableType:
        if not self.events:
            return Panel("no mutations yet", padding=(1, 2), border_style="dim")

        tbl = Table(show_edge=False, pad_edge=False)
        tbl.add_column("#", justify="right", style="bold")
        tbl.add_column("isle", justify="right")
        tbl.add_column("parent", justify="right")
        tbl.add_column("child", justify="right")

        diff_panels: List[RenderableType] = []
        for event in self.events:
            tbl.add_row(
                str(event.iteration),
                str(event.island),
                f"{event.parent_score:.3f}",
                f"{event.child_score:.3f}",
            )
            diff_panels.append(self._colour_diff(event.diff))

        return Group(tbl, *diff_panels)
