from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Markdown,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from fuzzyevolve.tui.run_data import (
    EliteRecord,
    RunSummary,
    find_last_by_type,
    list_runs,
    load_run_state,
    tail_events,
    tail_llm_index,
    tail_stats,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _format_cell_key(cell_key: tuple[int, ...]) -> str:
    if not cell_key:
        return "—"
    return ",".join(str(v) for v in cell_key)


def _format_descriptor(desc: dict[str, Any]) -> str:
    if not desc:
        return "(none)"
    parts: list[str] = []
    for key, value in desc.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_metric_table(elite: EliteRecord, *, c: float) -> str:
    lines = ["| metric | μ | σ | LCB |", "|---|---:|---:|---:|"]
    for metric, rating in elite.ratings.items():
        lcb = rating.mu - c * rating.sigma
        lines.append(
            f"| {metric} | {rating.mu:.2f} | {rating.sigma:.2f} | {lcb:.2f} |"
        )
    return "\n".join(lines)


class Inspector(Static):
    def compose(self) -> ComposeResult:
        with TabbedContent(id="inspector_tabs"):
            with TabPane("Content", id="inspector_tab_content"):
                yield TextArea("", id="inspector_content", read_only=True)
            with TabPane("Meta", id="inspector_tab_meta"):
                yield Markdown("", id="inspector_meta")

    def show_elite(
        self,
        elite: EliteRecord,
        *,
        get_text,
        score_lcb_c: float,
    ) -> None:
        text = get_text(elite.text_id)
        self.query_one("#inspector_content", TextArea).text = text

        c = float(score_lcb_c)
        meta = [
            f"**score (LCB avg)**: `{elite.score:.3f}`",
            f"**island**: `{elite.island}`",
            f"**age**: `{elite.age}`",
            f"**cell**: `{_format_cell_key(elite.cell_key)}`",
            "",
            "**descriptor**",
            f"`{_format_descriptor(elite.descriptor)}`",
            "",
        ]

        # Show per-metric info as a Markdown table
        # (we don't assume ordering beyond dict insertion; it's fine for inspection)
        meta.append(_format_metric_table(elite, c=c))
        self.query_one("#inspector_meta", Markdown).update("\n".join(meta))

    def show_llm_call(self, run_dir: Path, call: dict[str, Any]) -> None:
        prompt_file = call.get("prompt_file")
        output_file = call.get("output_file")
        prompt = ""
        output = ""
        if isinstance(prompt_file, str):
            try:
                prompt = (run_dir / prompt_file).read_text(encoding="utf-8")
            except Exception:
                prompt = "(failed to read prompt)"
        if isinstance(output_file, str):
            try:
                output = (run_dir / output_file).read_text(encoding="utf-8")
            except Exception:
                output = "(failed to read output)"

        content_parts = []
        if prompt:
            content_parts.append("### Prompt\n")
            content_parts.append(prompt)
        if output:
            content_parts.append("\n\n### Output\n")
            content_parts.append(output)
        if not content_parts:
            content_parts = ["(no content)"]

        self.query_one("#inspector_content", TextArea).text = "".join(content_parts)

        meta_lines = [
            f"**iteration**: `{call.get('iteration', '—')}`",
            f"**name**: `{call.get('name', '—')}`",
            f"**model**: `{call.get('model', '—')}`",
            f"**error**: `{call.get('error') or '—'}`",
        ]
        self.query_one("#inspector_meta", Markdown).update("\n".join(meta_lines))

    def clear(self) -> None:
        self.query_one("#inspector_content", TextArea).text = ""
        self.query_one("#inspector_meta", Markdown).update("")


class RunPickerScreen(Screen[Path]):
    BINDINGS = [("r", "refresh", "Refresh")]

    def __init__(self, *, data_dir: Path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self._runs: list[RunSummary] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Select a run", classes="title")
        yield ListView(id="run_list")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        self._runs = list_runs(self.data_dir)
        lv = self.query_one("#run_list", ListView)
        lv.clear()
        if not self._runs:
            lv.append(ListItem(Label("No runs found under .fuzzyevolve/runs/")))
            return
        for run in self._runs:
            summary = f"{run.run_id}  • it {run.iteration}  • best {_format_float(run.best_score)}"
            if run.metrics:
                summary += f"  • {', '.join(run.metrics)}"
            lv.append(ListItem(Label(summary)))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.index
        if idx < 0 or idx >= len(self._runs):
            return
        self.dismiss(self._runs[idx].run_dir)


class RunScreen(Screen[None]):
    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+r", "refresh", "Refresh"),
    ]

    def __init__(self, *, run_dir: Path, attach: bool = True) -> None:
        super().__init__()
        self.attach = attach
        self.state = load_run_state(run_dir)
        self.run_dir = self.state.run_dir
        self._stats: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._llm: list[dict[str, Any]] = []
        self._selected_archive_text_id: str | None = None
        self._selected_llm_prompt_file: str | None = None
        self._selected_map_cell: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="run_bar")
        with Horizontal(id="main"):
            with TabbedContent(id="tabs"):
                with TabPane("Dashboard"):
                    yield Static("", id="dash_summary")
                    yield Sparkline([], id="dash_sparkline")
                    yield DataTable(id="dash_metrics")
                    yield Static("", id="dash_best_preview")
                with TabPane("Map"):
                    yield Static(
                        "MAP-Elites occupancy / best score heatmap (select a cell)",
                        id="map_help",
                    )
                    yield DataTable(id="map_table")
                with TabPane("Archive"):
                    yield DataTable(id="archive_table")
                with TabPane("Battle"):
                    yield Static("", id="battle_summary")
                    yield DataTable(id="battle_table")
                    yield Markdown("", id="battle_ranking")
                with TabPane("LLM"):
                    yield DataTable(id="llm_table")
            yield Inspector(id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_all()
        if self.attach:
            self.set_interval(0.75, self._refresh_live)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        self._refresh_all(force=True)

    def _refresh_live(self) -> None:
        self._refresh_all(force=False)

    def _refresh_all(self, *, force: bool = False) -> None:
        cp = self.state.store.latest_checkpoint_path()
        mtime = cp.stat().st_mtime if cp.exists() else 0.0
        if force or mtime != self.state.checkpoint_mtime:
            self.state = load_run_state(self.run_dir)

        self._stats = tail_stats(self.run_dir, max_lines=240)
        self._events = tail_events(self.run_dir, max_lines=2500)
        self._llm = tail_llm_index(self.run_dir, max_lines=600)

        self._update_run_bar()
        self._update_dashboard()
        self._update_map()
        self._update_archive()
        self._update_battle()
        self._update_llm()

    def _update_run_bar(self) -> None:
        best = self.state.best
        best_score = best.score if best else None
        metrics = ", ".join(self.state.cfg.metrics.names)
        descriptor = self.state.descriptor_kind
        self.query_one("#run_bar", Static).update(
            f"[run] {self.state.run_dir.name}  • it {self.state.iteration}  • best {_format_float(best_score)}"
            f"  • metrics: {metrics}  • descriptor: {descriptor}"
        )

    def _update_dashboard(self) -> None:
        best = self.state.best
        empty_cells = None
        if self._stats:
            try:
                empty_cells = int(self._stats[-1].get("empty_cells", 0))
            except Exception:
                empty_cells = None

        summary = [
            f"Iterations: {self.state.iteration}",
            f"Islands: {len(self.state.islands)}",
        ]
        if empty_cells is not None:
            summary.append(f"Empty cells: {empty_cells}")
        self.query_one("#dash_summary", Static).update("  •  ".join(summary))

        scores = []
        for row in self._stats:
            if "best_score" in row:
                try:
                    scores.append(float(row["best_score"]))
                except Exception:
                    continue
        self.query_one("#dash_sparkline", Sparkline).data = scores[-240:]

        metrics_table = self.query_one("#dash_metrics", DataTable)
        if not metrics_table.columns:
            metrics_table.add_columns("metric", "μ", "σ", "LCB")
        metrics_table.clear()
        if best:
            c = float(self.state.cfg.rating.score_lcb_c)
            for metric in self.state.cfg.metrics.names:
                r = best.ratings.get(metric)
                if r is None:
                    continue
                lcb = r.mu - c * r.sigma
                metrics_table.add_row(
                    metric,
                    f"{r.mu:.2f}",
                    f"{r.sigma:.2f}",
                    f"{lcb:.2f}",
                )

        preview = "(no best elite yet)"
        if best:
            text = self.state.get_text(best.text_id)
            preview = text.strip()
            if len(preview) > 1200:
                preview = preview[:1200].rstrip() + "\n…"
        self.query_one("#dash_best_preview", Static).update(preview)

    def _update_map(self) -> None:
        table = self.query_one("#map_table", DataTable)
        table.cursor_type = "cell"
        table.clear(columns=True)

        counts = self.state.cell_counts()
        best_scores = self.state.cell_best_scores()

        if self.state.descriptor_kind == "embedding_2d" and self.state.bins_x and self.state.bins_y:
            bins_x = self.state.bins_x
            bins_y = self.state.bins_y
            nx = len(bins_x) - 1
            ny = len(bins_y) - 1

            table.add_columns(*[f"[{bins_x[i]:g},{bins_x[i+1]:g})" for i in range(nx)])

            max_count = max(counts.values(), default=0)
            palette = ["#111827", "#1f2937", "#0f766e", "#14b8a6", "#67e8f9"]

            for y in range(ny):
                label = f"[{bins_y[y]:g},{bins_y[y+1]:g})"
                row = []
                for x in range(nx):
                    key = (x, y)
                    n = counts.get(key, 0)
                    level = 0
                    if max_count > 0:
                        level = min(
                            len(palette) - 1,
                            int((n / max_count) * (len(palette) - 1)),
                        )
                    style = f"on {palette[level]}" if n else "dim"
                    row.append(Text(str(n) if n else "·", style=style))
                table.add_row(*row, label=label)

            if self._selected_map_cell is not None:
                x, y = self._selected_map_cell
                if 0 <= x < nx and 0 <= y < ny:
                    table.move_cursor(row=y, column=x, animate=False)

        elif self.state.descriptor_kind == "length" and self.state.length_bins:
            bins = self.state.length_bins
            nb = len(bins) - 1
            table.add_columns("length bin", "count", "best score")
            max_count = max(counts.values(), default=0)
            palette = ["#111827", "#1f2937", "#0f766e", "#14b8a6", "#67e8f9"]
            for i in range(nb):
                key = (i,)
                n = counts.get(key, 0)
                s = best_scores.get(key)
                level = 0
                if max_count > 0:
                    level = min(
                        len(palette) - 1,
                        int((n / max_count) * (len(palette) - 1)),
                    )
                style = f"on {palette[level]}" if n else "dim"
                table.add_row(
                    Text(f"[{bins[i]:g},{bins[i+1]:g})", style=style),
                    Text(str(n), style=style),
                    Text(_format_float(s), style=style),
                )
        else:
            table.add_columns("map")
            table.add_row("(unknown descriptor kind)")

    def _update_archive(self) -> None:
        table = self.query_one("#archive_table", DataTable)
        table.cursor_type = "row"
        if not table.columns:
            table.add_columns("score", "age", "island", "cell", "text_id")
        table.clear()

        elites: list[EliteRecord] = [e for island in self.state.islands for e in island]
        elites.sort(key=lambda e: e.score, reverse=True)
        for elite in elites[:2000]:
            table.add_row(
                f"{elite.score:.3f}",
                str(elite.age),
                str(elite.island),
                _format_cell_key(elite.cell_key),
                elite.text_id[:10],
                key=elite.text_id,
            )
        if self._selected_archive_text_id:
            try:
                row_idx = table.get_row_index(self._selected_archive_text_id)
                table.move_cursor(row=row_idx, column=0, animate=False)
            except Exception:
                pass

    def _update_battle(self) -> None:
        summary = self.query_one("#battle_summary", Static)
        table = self.query_one("#battle_table", DataTable)
        ranking_md = self.query_one("#battle_ranking", Markdown)

        table.clear()
        if not table.columns:
            table.add_columns("idx", "role", "frozen", "text")

        battle_ev = find_last_by_type(self._events, "battle")
        ranking_ev = find_last_by_type(self._events, "ranking")
        if not battle_ev:
            summary.update("No battle data yet.")
            ranking_md.update("")
            return

        it = battle_ev.get("iteration", "—")
        summary.update(f"Last battle at iteration {it}")
        participants = (battle_ev.get("data") or {}).get("participants") or []
        idx_to_label: dict[int, str] = {}
        for p in participants:
            idx = p.get("idx")
            role = p.get("role", "")
            frozen = p.get("frozen", False)
            text_id = p.get("text_id", "")
            text_preview = ""
            if isinstance(text_id, str) and text_id:
                try:
                    text_preview = self.state.get_text(text_id).strip().splitlines()[0][:80]
                except Exception:
                    text_preview = text_id[:12]
            try:
                idx_int = int(idx)
                idx_to_label[idx_int] = f"{idx_int}:{role} {text_preview}"
            except Exception:
                pass
            table.add_row(str(idx), str(role), "yes" if frozen else "", text_preview)

        if ranking_ev:
            tiers_by_metric = (ranking_ev.get("data") or {}).get("tiers_by_metric") or {}
            lines = ["### Rankings"]
            for metric, tiers in tiers_by_metric.items():
                lines.append(f"\n**{metric}**")
                for rank, tier in enumerate(tiers):
                    rendered = []
                    for player_idx in tier:
                        if isinstance(player_idx, int) and player_idx in idx_to_label:
                            rendered.append(idx_to_label[player_idx])
                        else:
                            rendered.append(str(player_idx))
                    lines.append(f"- tier {rank}: " + " · ".join(rendered))
            ranking_md.update("\n".join(lines))
        else:
            ranking_md.update("")

    def _update_llm(self) -> None:
        table = self.query_one("#llm_table", DataTable)
        table.cursor_type = "row"
        if not table.columns:
            table.add_columns("it", "name", "model", "error")
        table.clear()

        for call in reversed(self._llm[-400:]):
            prompt_file = call.get("prompt_file")
            if not isinstance(prompt_file, str) or not prompt_file:
                continue
            table.add_row(
                str(call.get("iteration", "")),
                str(call.get("name", "")),
                str(call.get("model", "")),
                str(call.get("error") or ""),
                key=prompt_file,
            )
        if self._selected_llm_prompt_file:
            try:
                row_idx = table.get_row_index(self._selected_llm_prompt_file)
                table.move_cursor(row=row_idx, column=0, animate=False)
            except Exception:
                pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "archive_table":
            if not event.data_table.has_focus:
                return
            text_id = event.row_key.value if event.row_key else None
            if not text_id:
                return
            self._selected_archive_text_id = text_id
            elite = next(
                (
                    e
                    for island in self.state.islands
                    for e in island
                    if e.text_id == text_id
                ),
                None,
            )
            if elite:
                self.query_one("#inspector", Inspector).show_elite(
                    elite,
                    get_text=self.state.get_text,
                    score_lcb_c=self.state.cfg.rating.score_lcb_c,
                )
        elif event.data_table.id == "llm_table":
            if not event.data_table.has_focus:
                return
            prompt_file = event.row_key.value if event.row_key else None
            if not isinstance(prompt_file, str) or not prompt_file:
                return
            self._selected_llm_prompt_file = prompt_file
            call = next(
                (
                    c
                    for c in reversed(self._llm[-400:])
                    if str(c.get("prompt_file") or "") == prompt_file
                ),
                None,
            )
            if call is not None:
                self.query_one("#inspector", Inspector).show_llm_call(self.run_dir, call)

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        if event.data_table.id != "map_table":
            return
        # Map cells do not have stable keys in DataTable; compute from position.
        if self.state.descriptor_kind != "embedding_2d":
            return
        if not event.data_table.has_focus:
            return
        x = event.coordinate.column
        y = event.coordinate.row
        cell_key = (x, y)
        self._selected_map_cell = cell_key
        elites = [
            e
            for island in self.state.islands
            for e in island
            if e.cell_key == cell_key
        ]
        if not elites:
            return
        best = max(elites, key=lambda e: e.score)
        self.query_one("#inspector", Inspector).show_elite(
            best,
            get_text=self.state.get_text,
            score_lcb_c=self.state.cfg.rating.score_lcb_c,
        )


class FuzzyEvolveTUI(App[None]):
    CSS = """
    Screen {
        background: #0b0f14;
    }

    .title {
        padding: 1 2;
        color: #93c5fd;
        text-style: bold;
    }

    #run_bar {
        padding: 0 2;
        height: 1;
        color: #e5e7eb;
        background: #111827;
    }

    #main {
        height: 1fr;
    }

    #tabs {
        width: 1fr;
        border: round #1f2937;
    }

    #inspector {
        width: 60;
        border: round #1f2937;
        padding: 0 1;
    }

    #dash_best_preview {
        height: 1fr;
        border-top: solid #1f2937;
        padding: 1 1;
        overflow: auto;
    }

    #dash_sparkline {
        height: 3;
        padding: 0 1;
        color: #67e8f9;
    }

    DataTable {
        height: 1fr;
    }
    """

    def __init__(self, *, data_dir: Path, run_dir: Path | None = None, attach: bool = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.initial_run_dir = run_dir
        self.attach = attach

    def on_mount(self) -> None:
        if self.initial_run_dir is not None:
            self.push_screen(RunScreen(run_dir=self.initial_run_dir, attach=self.attach))
        else:
            self.push_screen(RunPickerScreen(data_dir=self.data_dir), self._open_run)

    def _open_run(self, run_dir: Path) -> None:
        self.push_screen(RunScreen(run_dir=run_dir, attach=True))

    async def on_key(self, event: events.Key) -> None:
        if event.key == "q":
            self.exit()


def run_tui(*, data_dir: Path, run_dir: Path | None = None, attach: bool = True) -> None:
    app = FuzzyEvolveTUI(data_dir=data_dir, run_dir=run_dir, attach=attach)
    app.run()
