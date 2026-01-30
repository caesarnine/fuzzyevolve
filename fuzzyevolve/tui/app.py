from __future__ import annotations

from pathlib import Path

from textual import events, on
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
    StatsRecord,
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


def _sparkline(values: list[float], *, width: int = 64) -> str:
    if not values:
        return ""
    width = max(1, int(width))
    if len(values) > width:
        step = len(values) / width
        sampled: list[float] = []
        for i in range(width):
            idx = int(i * step)
            sampled.append(values[min(idx, len(values) - 1)])
        values = sampled
    low = min(values)
    high = max(values)
    if high <= low:
        return "▁" * len(values)
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for v in values:
        t = (v - low) / (high - low)
        idx = int(round(t * (len(blocks) - 1)))
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


def _text_preview(text: str, *, max_len: int = 56) -> str:
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            line = stripped
            break
    else:
        line = text.strip()
    if len(line) > max_len:
        return line[: max(0, max_len - 1)].rstrip() + "…"
    return line


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


class IterationInspector(Static):
    def compose(self) -> ComposeResult:
        with TabbedContent(id="iteration_tabs"):
            with TabPane("Best", id="iteration_tab_best"):
                yield TextArea("", id="iteration_best", read_only=True)
            with TabPane("Parent", id="iteration_tab_parent"):
                yield TextArea("", id="iteration_parent", read_only=True)
            with TabPane("Details", id="iteration_tab_details"):
                yield Markdown("", id="iteration_details")

    def show_iteration(
        self,
        *,
        state: RunState,
        stats: StatsRecord,
        events_for_iteration: list[dict[str, object]],
    ) -> None:
        best_text = ""
        if stats.best_text_id:
            try:
                best_text = state.get_text(stats.best_text_id)
            except Exception:
                best_text = ""
        self.query_one("#iteration_best", TextArea).text = best_text

        step_start = None
        for ev in events_for_iteration:
            if ev.get("type") == "step_start":
                step_start = ev
                break
        parent_text = ""
        if step_start:
            data = step_start.get("data") or {}
            parent_text_id = None
            if isinstance(data, dict):
                parent_text_id = data.get("parent_text_id")
            if isinstance(parent_text_id, str):
                try:
                    parent_text = state.get_text(parent_text_id)
                except Exception:
                    parent_text = ""
        self.query_one("#iteration_parent", TextArea).text = parent_text

        lines: list[str] = []
        lines.append(f"**iteration**: `{stats.iteration}`")
        if stats.best_score is not None:
            lines.append(f"**best_score**: `{stats.best_score:.3f}`")
        if stats.pool_size is not None:
            lines.append(f"**pool_size**: `{stats.pool_size}`")
        if stats.mean_score is not None:
            lines.append(f"**mean_score**: `{stats.mean_score:.3f}`")
        if stats.p90_score is not None:
            lines.append(f"**p90_score**: `{stats.p90_score:.3f}`")
        if stats.diversity_nn_mean is not None:
            lines.append(f"**diversity_nn_mean**: `{stats.diversity_nn_mean:.3f}`")
        if stats.mean_sigma is not None:
            lines.append(f"**mean_sigma**: `{stats.mean_sigma:.3f}`")

        if step_start and isinstance(step_start.get("data"), dict):
            d = step_start["data"]
            lines.append("")
            lines.append("### step_start")
            if "parent_age" in d:
                lines.append(f"- parent_age: `{d['parent_age']}`")
            if "parent_score" in d:
                try:
                    lines.append(f"- parent_score: `{float(d['parent_score']):.3f}`")
                except Exception:
                    lines.append(f"- parent_score: `{d['parent_score']}`")
            scalarization = d.get("scalarization")
            if isinstance(scalarization, dict) and scalarization:
                parts = [f"{k}={float(v):.2f}" for k, v in scalarization.items()]
                lines.append(f"- scalarization: `{', '.join(parts)}`")

        # Pull out some common, high-signal events for the iteration.
        by_type: dict[str, dict] = {}
        for ev in events_for_iteration:
            ev_type = ev.get("type")
            if isinstance(ev_type, str) and isinstance(ev.get("data"), dict):
                by_type[ev_type] = ev["data"]

        critique = by_type.get("critique")
        if isinstance(critique, dict) and critique:
            summary = critique.get("summary")
            issues = critique.get("issues")
            lines.append("")
            lines.append("### critique")
            if isinstance(summary, str) and summary.strip():
                lines.append(f"- summary: {summary.strip()}")
            if isinstance(issues, list) and issues:
                for issue in issues[:5]:
                    if isinstance(issue, str) and issue.strip():
                        lines.append(f"- issue: {issue.strip()}")

        candidates = by_type.get("candidates", {}).get("items")
        if isinstance(candidates, list) and candidates:
            lines.append("")
            lines.append("### candidates")
            for cand in candidates[:20]:
                if not isinstance(cand, dict):
                    continue
                tid = cand.get("text_id")
                op = cand.get("operator")
                focus = cand.get("focus")
                preview = ""
                if isinstance(tid, str):
                    try:
                        preview = _text_preview(state.get_text(tid))
                    except Exception:
                        preview = tid[:8]
                focus_part = (
                    f" | focus={focus}" if isinstance(focus, str) and focus else ""
                )
                lines.append(f"- `{op}` `{str(tid)[:8]}` {preview}{focus_part}")

        battle = by_type.get("battle")
        idx_to_label: dict[int, str] = {}
        if isinstance(battle, dict):
            participants = battle.get("participants")
            if isinstance(participants, list) and participants:
                lines.append("")
                lines.append("### battle")
                for p in participants:
                    if not isinstance(p, dict):
                        continue
                    idx = p.get("idx")
                    role = p.get("role", "")
                    text_id = p.get("text_id")
                    frozen = p.get("frozen", False)
                    if not isinstance(idx, int) or not isinstance(text_id, str):
                        continue
                    try:
                        preview = _text_preview(state.get_text(text_id))
                    except Exception:
                        preview = text_id[:8]
                    tag = " (frozen)" if frozen else ""
                    label = f"[{idx}] {role} {text_id[:8]} {preview}{tag}"
                    idx_to_label[idx] = label
                    lines.append(f"- {label}")

        ranking = by_type.get("ranking", {}).get("tiers_by_metric")
        if isinstance(ranking, dict) and ranking:
            lines.append("")
            lines.append("### ranking")
            for metric, tiers in ranking.items():
                if not isinstance(metric, str) or not isinstance(tiers, list):
                    continue
                lines.append(f"- **{metric}**")
                for tier_idx, tier in enumerate(tiers, start=1):
                    if not isinstance(tier, list):
                        continue
                    items = []
                    for idx in tier:
                        if not isinstance(idx, int):
                            continue
                        items.append(idx_to_label.get(idx, f"[{idx}]"))
                    if items:
                        lines.append(f"  - tier {tier_idx}: " + " | ".join(items))

        ratings_update = by_type.get("ratings_update", {}).get("participants")
        if isinstance(ratings_update, list) and ratings_update:
            lines.append("")
            lines.append("### ratings_update (avg LCB)")
            metrics = list(state.cfg.metrics.names)
            c = float(state.cfg.rating.score_lcb_c)

            def _score_lcb(rmap: dict) -> float | None:
                if not metrics:
                    return None
                total = 0.0
                count = 0
                for m in metrics:
                    r = rmap.get(m)
                    if not isinstance(r, dict):
                        continue
                    mu = r.get("mu")
                    sigma = r.get("sigma")
                    try:
                        total += float(mu) - c * float(sigma)
                        count += 1
                    except Exception:
                        continue
                if count <= 0:
                    return None
                return total / len(metrics)

            for p in ratings_update:
                if not isinstance(p, dict):
                    continue
                idx = p.get("idx")
                if not isinstance(idx, int):
                    continue
                before = p.get("before")
                after = p.get("after")
                if not isinstance(before, dict) or not isinstance(after, dict):
                    continue
                b = _score_lcb(before)
                a = _score_lcb(after)
                if b is None or a is None:
                    continue
                label = idx_to_label.get(idx, f"[{idx}]")
                lines.append(f"- {label}: `{b:.3f} → {a:.3f}`")

        pool_delta = by_type.get("pool_delta", {})
        if isinstance(pool_delta, dict) and pool_delta:
            removed = pool_delta.get("removed_text_ids")
            kept = pool_delta.get("kept_child_text_ids")
            lines.append("")
            lines.append("### pool_delta")
            if isinstance(kept, list):
                lines.append(f"- kept_children: `{len(kept)}`")
            if isinstance(removed, list):
                lines.append(f"- removed: `{len(removed)}`")

        self.query_one("#iteration_details", Markdown).update("\n".join(lines))


class EdgeInspector(Static):
    def compose(self) -> ComposeResult:
        with TabbedContent(id="edge_tabs"):
            with TabPane("Child", id="edge_tab_child"):
                yield TextArea("", id="edge_child", read_only=True)
            with TabPane("Parent", id="edge_tab_parent"):
                yield TextArea("", id="edge_parent", read_only=True)
            with TabPane("Meta", id="edge_tab_meta"):
                yield Markdown("", id="edge_meta")

    def show_edge(self, edge: dict[str, object], *, state: RunState) -> None:
        parent_id = edge.get("parent_text_id")
        child_id = edge.get("child_text_id")
        parent_text = ""
        child_text = ""
        if isinstance(parent_id, str):
            try:
                parent_text = state.get_text(parent_id)
            except Exception:
                parent_text = ""
        if isinstance(child_id, str):
            try:
                child_text = state.get_text(child_id)
            except Exception:
                child_text = ""

        self.query_one("#edge_parent", TextArea).text = parent_text
        self.query_one("#edge_child", TextArea).text = child_text

        meta: list[str] = []
        meta.append(f"**iteration**: `{edge.get('iteration', '—')}`")
        meta.append(f"**operator**: `{edge.get('operator', '')}`")
        role = edge.get("role")
        if role:
            meta.append(f"**role**: `{role}`")
        dist = edge.get("embedding_distance")
        if dist is not None:
            try:
                meta.append(f"**embedding_distance**: `{float(dist):.3f}`")
            except Exception:
                meta.append(f"**embedding_distance**: `{dist}`")
        focus = edge.get("focus")
        if isinstance(focus, str) and focus.strip():
            meta.append(f"**focus**: `{focus.strip()}`")

        partners = edge.get("partner_text_ids")
        if isinstance(partners, list) and partners:
            meta.append("")
            meta.append("Partners:")
            for pid in partners[:8]:
                if not isinstance(pid, str):
                    continue
                try:
                    preview = _text_preview(state.get_text(pid))
                except Exception:
                    preview = pid[:8]
                meta.append(f"- `{pid[:8]}` {preview}")

        self.query_one("#edge_meta", Markdown).update("\n".join(meta))


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
        idx = int(event.index)
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
        self._last_stats_mtime = 0.0
        self._last_events_mtime = 0.0
        self._selected_text_id: str | None = None
        self._selected_iteration: int | None = None
        self._lineage_edge_by_key: dict[str, dict[str, object]] = {}
        self._selected_edge_key: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("", id="run_header")
        with TabbedContent(id="run_tabs"):
            with TabPane("Population", id="tab_population"):
                with Horizontal(id="population_view"):
                    with Vertical(id="population_left"):
                        yield DataTable(
                            id="elite_table", cursor_type="row", show_row_labels=False
                        )
                    yield Inspector(id="inspector")
            with TabPane("Timeline", id="tab_timeline"):
                with Vertical(id="timeline_root"):
                    yield Label("", id="timeline_header")
                    with Horizontal(id="timeline_view"):
                        with Vertical(id="timeline_left"):
                            yield DataTable(
                                id="stats_table",
                                cursor_type="row",
                                show_row_labels=False,
                            )
                        yield IterationInspector(id="iteration_inspector")
            with TabPane("Lineage", id="tab_lineage"):
                with Horizontal(id="lineage_view"):
                    with Vertical(id="lineage_left"):
                        yield DataTable(
                            id="lineage_table",
                            cursor_type="row",
                            show_row_labels=False,
                        )
                    yield EdgeInspector(id="edge_inspector")
        yield Footer()

    def on_mount(self) -> None:
        self._load()
        if self.attach:
            self.set_interval(1.0, self._maybe_refresh)
        self.query_one("#elite_table", DataTable).focus()

    def _maybe_refresh(self) -> None:
        if self.state is None:
            return
        store = self.state.store
        cp_path = store.latest_checkpoint_path()
        stats_path = store.run_dir / "stats.jsonl"
        events_path = store.run_dir / "events.jsonl"

        cp_mtime = cp_path.stat().st_mtime if cp_path.exists() else 0.0
        stats_mtime = stats_path.stat().st_mtime if stats_path.exists() else 0.0
        events_mtime = events_path.stat().st_mtime if events_path.exists() else 0.0

        if (
            cp_mtime <= self._last_checkpoint_mtime
            and stats_mtime <= self._last_stats_mtime
            and events_mtime <= self._last_events_mtime
        ):
            return
        self._load()

    def _load(self) -> None:
        self.state = load_run_state(self.run_dir)
        self._last_checkpoint_mtime = self.state.checkpoint_mtime
        self._last_stats_mtime = self.state.stats_mtime
        self._last_events_mtime = self.state.events_mtime

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
                key=elite.text_id,
            )

        stats_table = self.query_one("#stats_table", DataTable)
        stats_table.clear(columns=True)
        stats_table.add_columns("it", "best", "mean", "p90", "pool", "div_nn", "σ")
        for row in self.state.stats:
            stats_table.add_row(
                str(row.iteration),
                _format_float(row.best_score),
                _format_float(row.mean_score),
                _format_float(row.p90_score),
                str(row.pool_size) if row.pool_size is not None else "—",
                _format_float(row.diversity_nn_mean),
                _format_float(row.mean_sigma),
                key=str(row.iteration),
            )

        timeline_header = self.query_one("#timeline_header", Label)
        best_scores = [
            r.best_score for r in self.state.stats if r.best_score is not None
        ]
        spark = _sparkline([float(v) for v in best_scores][-120:], width=72)
        if best_scores:
            timeline_header.update(
                f"best score: {spark}  (min={min(best_scores):.3f} max={max(best_scores):.3f})"
            )
        else:
            timeline_header.update("best score: —")

        self._load_lineage()

        if not self.state.members:
            return

        target = (
            self._selected_text_id
            if self._selected_text_id
            and any(e.text_id == self._selected_text_id for e in self.state.members)
            else self.state.members[0].text_id
        )
        self._show_selected_row(target, force=True)
        try:
            table.move_cursor(row=table.get_row_index(target), column=0, scroll=False)
        except Exception:
            pass

        if self.state.stats:
            if self._selected_iteration is None:
                self._selected_iteration = self.state.stats[-1].iteration
            self._show_selected_iteration(self._selected_iteration, force=True)
            try:
                stats_table.move_cursor(
                    row=stats_table.get_row_index(str(self._selected_iteration)),
                    column=0,
                    scroll=False,
                )
            except Exception:
                pass

        if self._lineage_edge_by_key:
            if (
                self._selected_edge_key is None
                or self._selected_edge_key not in self._lineage_edge_by_key
            ):
                self._selected_edge_key = next(iter(self._lineage_edge_by_key.keys()))
            self._show_selected_edge(self._selected_edge_key, force=True)
            lineage_table = self.query_one("#lineage_table", DataTable)
            try:
                lineage_table.move_cursor(
                    row=lineage_table.get_row_index(self._selected_edge_key),
                    column=0,
                    scroll=False,
                )
            except Exception:
                pass

    def _show_elite(self, elite: EliteRecord) -> None:
        if self.state is None:
            return
        self._selected_text_id = elite.text_id
        inspector = self.query_one(Inspector)
        inspector.show_elite(
            elite,
            get_text=self.state.get_text,
            score_lcb_c=self.state.cfg.rating.score_lcb_c,
        )

    @on(DataTable.RowHighlighted, "#elite_table")
    @on(DataTable.RowSelected, "#elite_table")
    @on(DataTable.CellHighlighted, "#elite_table")
    @on(DataTable.CellSelected, "#elite_table")
    def _elite_table_changed(self, event) -> None:
        if isinstance(event, (DataTable.CellHighlighted, DataTable.CellSelected)):
            text_id = str(event.cell_key.row_key.value)
        else:
            text_id = str(event.row_key.value)
        self._show_selected_row(text_id)

    def _show_selected_row(self, text_id: str, *, force: bool = False) -> None:
        if self.state is None:
            return
        if not force and text_id == self._selected_text_id:
            return
        for elite in self.state.members:
            if elite.text_id == text_id:
                self._show_elite(elite)
                break

    @on(DataTable.RowHighlighted, "#stats_table")
    @on(DataTable.RowSelected, "#stats_table")
    @on(DataTable.CellHighlighted, "#stats_table")
    @on(DataTable.CellSelected, "#stats_table")
    def _stats_table_changed(self, event) -> None:
        if isinstance(event, (DataTable.CellHighlighted, DataTable.CellSelected)):
            key = str(event.cell_key.row_key.value)
        else:
            key = str(event.row_key.value)
        try:
            iteration = int(key)
        except Exception:
            return
        self._show_selected_iteration(iteration)

    def _show_selected_iteration(self, iteration: int, *, force: bool = False) -> None:
        if self.state is None:
            return
        if not force and self._selected_iteration == iteration:
            return
        self._selected_iteration = iteration
        stats = next((s for s in self.state.stats if s.iteration == iteration), None)
        if stats is None:
            return
        events_for_it: list[dict[str, object]] = []
        for ev in self.state.events:
            try:
                if int(ev.get("iteration", -1)) == iteration:
                    events_for_it.append(ev)
            except Exception:
                continue
        inspector = self.query_one(IterationInspector)
        inspector.show_iteration(
            state=self.state,
            stats=stats,
            events_for_iteration=events_for_it,
        )

    def _load_lineage(self) -> None:
        if self.state is None:
            return
        self._lineage_edge_by_key = {}
        lineage_table = self.query_one("#lineage_table", DataTable)
        lineage_table.clear(columns=True)
        lineage_table.add_columns("it", "op", "dist", "parent", "child")

        edges: list[dict[str, object]] = []
        for ev in self.state.events:
            if ev.get("type") != "lineage":
                continue
            it = int(ev.get("iteration", 0))
            data = ev.get("data") or {}
            if not isinstance(data, dict):
                continue
            edge_list = data.get("edges") or []
            if not isinstance(edge_list, list):
                continue
            for edge in edge_list:
                if not isinstance(edge, dict):
                    continue
                row = {"iteration": it, **edge}
                edges.append(row)

        edges.sort(key=lambda e: int(e.get("iteration", 0)), reverse=True)
        for idx, edge in enumerate(edges):
            it = int(edge.get("iteration", 0))
            op = edge.get("operator", "")
            dist = edge.get("embedding_distance")
            if dist is None:
                dist_str = "—"
            else:
                try:
                    dist_str = _format_float(float(dist))
                except Exception:
                    dist_str = "—"
            parent_id = edge.get("parent_text_id")
            child_id = edge.get("child_text_id")
            parent_preview = ""
            child_preview = ""
            if isinstance(parent_id, str):
                try:
                    parent_preview = _text_preview(self.state.get_text(parent_id))
                except Exception:
                    parent_preview = parent_id[:8]
            if isinstance(child_id, str):
                try:
                    child_preview = _text_preview(self.state.get_text(child_id))
                except Exception:
                    child_preview = child_id[:8]
            key = f"{it}:{idx}"
            self._lineage_edge_by_key[key] = edge
            lineage_table.add_row(
                str(it),
                str(op),
                dist_str,
                parent_preview,
                child_preview,
                key=key,
            )

    @on(DataTable.RowHighlighted, "#lineage_table")
    @on(DataTable.RowSelected, "#lineage_table")
    @on(DataTable.CellHighlighted, "#lineage_table")
    @on(DataTable.CellSelected, "#lineage_table")
    def _lineage_table_changed(self, event) -> None:
        if isinstance(event, (DataTable.CellHighlighted, DataTable.CellSelected)):
            key = str(event.cell_key.row_key.value)
        else:
            key = str(event.row_key.value)
        self._show_selected_edge(key)

    def _show_selected_edge(self, key: str, *, force: bool = False) -> None:
        if self.state is None:
            return
        if not force and key == self._selected_edge_key:
            return
        edge = self._lineage_edge_by_key.get(key)
        if edge is None:
            return
        self._selected_edge_key = key
        inspector = self.query_one(EdgeInspector)
        inspector.show_edge(edge, state=self.state)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()


class FuzzyEvolveTUI(App[None]):
    CSS = """
    #run_header {
        padding: 0 1;
    }
    #population_left {
        width: 45%;
    }
    #inspector {
        width: 55%;
    }
    #timeline_left {
        width: 45%;
    }
    #iteration_inspector {
        width: 55%;
    }
    #lineage_left {
        width: 45%;
    }
    #edge_inspector {
        width: 55%;
    }
    #elite_table, #stats_table, #lineage_table {
        height: 1fr;
    }
    #timeline_header {
        padding: 0 1;
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

        def _open_selected(summary: RunSummary | None) -> None:
            if summary is None:
                self.exit()
                return
            self.push_screen(RunViewer(run_dir=summary.run_dir, attach=self.attach))

        self.push_screen(RunPicker(data_dir=self.data_dir), _open_selected)


def run_tui(*, data_dir: Path, run_dir: Path | None, attach: bool) -> None:
    app = FuzzyEvolveTUI(data_dir=data_dir, run_dir=run_dir, attach=attach)
    app.run()
