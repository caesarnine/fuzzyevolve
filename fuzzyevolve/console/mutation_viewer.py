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

    def _render_edit(self, search: str, replace: str) -> RenderableType:
        text = Text()
        text.append("SEARCH\n", style="bold yellow")
        if search:
            text.append(search.rstrip("\n") + "\n", style="red")
        text.append("REPLACE\n", style="bold green")
        if replace:
            text.append(replace.rstrip("\n"), style="green")
        return Panel(text, padding=(0, 1), border_style="dim")

    def __rich__(self) -> RenderableType:
        if not self.events:
            return Panel("no mutations yet", padding=(1, 2), border_style="dim")

        tbl = Table(show_edge=False, pad_edge=False)
        tbl.add_column("#", justify="right", style="bold")
        tbl.add_column("isle", justify="right")
        tbl.add_column("parent", justify="right")
        tbl.add_column("child", justify="right")

        edit_panels: List[RenderableType] = []
        for event in self.events:
            tbl.add_row(
                str(event.iteration),
                str(event.island),
                f"{event.parent_score:.3f}",
                f"{event.child_score:.3f}",
            )
            edit_panels.append(self._render_edit(event.search, event.replace))

        return Group(tbl, *edit_panels)
