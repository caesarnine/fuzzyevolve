from __future__ import annotations

from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from fuzzyevolve.tui.run_data import (
    EliteRecord,
    RunState,
    RunSummary,
    list_runs,
    load_run_state,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _format_metric_table(elite: EliteRecord, *, c: float) -> str:
    lines = ["| metric | μ | σ | LCB |", "|---|---:|---:|---:|"]
    for metric, rating in elite.ratings.items():
        lcb = rating.mu - c * rating.sigma
        lines.append(f"| {metric} | {rating.mu:.2f} | {rating.sigma:.2f} | {lcb:.2f} |")
    return "\n".join(lines)


class Inspector(Static):
    def compose(self) -> ComposeResult:
        with TabbedContent(id="inspector_tabs"):
            with TabPane("Content", id="inspector_tab_content"):
                yield TextArea("", id="inspector_content", read_only=True)
            with TabPane("Meta", id="inspector_tab_meta"):
                yield Markdown("", id="inspector_meta")

    def show_elite(self, elite: EliteRecord, *, get_text, score_lcb_c: float) -> None:
        text = get_text(elite.text_id)
        self.query_one("#inspector_content", TextArea).text = text

        c = float(score_lcb_c)
        meta = [
            f"**score (LCB avg)**: `{elite.score:.3f}`",
            f"**age**: `{elite.age}`",
            "",
            _format_metric_table(elite, c=c),
        ]
        self.query_one("#inspector_meta", Markdown).update("\n".join(meta))


class RunPicker(Screen[RunSummary | None]):
    def __init__(self, *, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.runs: list[RunSummary] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("Select a run:", id="run_picker_title")
        yield ListView(id="run_picker_list")
        yield Footer()

    def on_mount(self) -> None:
        self._reload()

    def _reload(self) -> None:
        self.runs = list_runs(self.data_dir)
        lv = self.query_one("#run_picker_list", ListView)
        lv.clear()
        if not self.runs:
            lv.append(ListItem(Label("No runs found under .fuzzyevolve/runs/")))
            return
        for run in self.runs:
            label = f"{run.run_id}  it={run.iteration}  best={_format_float(run.best_score)}"
            lv.append(ListItem(Label(label)))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None:
            return
        if idx >= len(self.runs):
            self.dismiss(None)
            return
        self.dismiss(self.runs[idx])


class RunViewer(Screen[None]):
    def __init__(self, *, run_dir: Path, attach: bool) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.attach = attach
        self.state: RunState | None = None
        self._last_checkpoint_mtime = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Label("", id="run_header")
                yield DataTable(id="elite_table")
            yield Inspector(id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        self._load()
        if self.attach:
            self.set_interval(1.0, self._maybe_refresh)

    def _maybe_refresh(self) -> None:
        if self.state is None:
            return
        cp_path = self.state.store.latest_checkpoint_path()
        if not cp_path.exists():
            return
        mtime = cp_path.stat().st_mtime
        if mtime <= self._last_checkpoint_mtime:
            return
        self._load()

    def _load(self) -> None:
        self.state = load_run_state(self.run_dir)
        self._last_checkpoint_mtime = self.state.checkpoint_mtime

        header = self.query_one("#run_header", Label)
        best = self.state.best.score if self.state.best else None
        header.update(
            f"run: {self.state.run_dir.name}  it={self.state.iteration}  best={_format_float(best)}"
        )

        table = self.query_one("#elite_table", DataTable)
        table.clear(columns=True)
        table.add_columns("rank", "score", "age", "preview")
        for idx, elite in enumerate(self.state.members, start=1):
            table.add_row(
                str(idx),
                f"{elite.score:.3f}",
                str(elite.age),
                elite.preview,
                key=str(idx),
            )

        if self.state.members:
            self._show_elite(self.state.members[0])

    def _show_elite(self, elite: EliteRecord) -> None:
        if self.state is None:
            return
        inspector = self.query_one(Inspector)
        inspector.show_elite(
            elite,
            get_text=self.state.get_text,
            score_lcb_c=self.state.cfg.rating.score_lcb_c,
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.state is None:
            return
        row_key = event.row_key.value if event.row_key else None
        if not row_key:
            return
        try:
            idx = int(row_key) - 1
        except ValueError:
            return
        if 0 <= idx < len(self.state.members):
            self._show_elite(self.state.members[idx])

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()


class FuzzyEvolveTUI(App[None]):
    CSS = """
    #left {
        width: 50%;
    }
    #inspector {
        width: 50%;
    }
    #elite_table {
        height: 1fr;
    }
    """

    def __init__(self, *, data_dir: Path, run_dir: Path | None, attach: bool) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.attach = attach

    def on_mount(self) -> None:
        if self.run_dir is not None:
            self.push_screen(RunViewer(run_dir=self.run_dir, attach=self.attach))
            return

        async def _pick_and_open():
            summary = await self.push_screen_wait(RunPicker(data_dir=self.data_dir))
            if summary is None:
                self.exit()
                return
            self.push_screen(RunViewer(run_dir=summary.run_dir, attach=self.attach))

        self.call_later(_pick_and_open)


def run_tui(*, data_dir: Path, run_dir: Path | None, attach: bool) -> None:
    app = FuzzyEvolveTUI(data_dir=data_dir, run_dir=run_dir, attach=attach)
    app.run()
