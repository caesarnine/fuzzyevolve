from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from fuzzyevolve.core.models import Elite

_MUTATION_TEMPLATE = """
Overall goal: {goal}
Task instructions: {instructions}
{metric_section}

Return up to {max_edits} edits to apply to the PARENT text (in order).
Each edit must specify:
- `search`: an exact substring from the PARENT text (verbatim)
- `replace`: replacement text for that substring

Constraints:
- All edits will be applied together to produce ONE child candidate.
- Each `search` must appear verbatim in the PARENT text.
- Edits must not overlap.
- Prefer a `search` span with enough context to be unique.
- If you can't find a safe exact-match `search` span, return no edits.

──────────────── PARENT ────────────────
Score: {p_score:.3f}
{p_stats}
{p_text}

──────────── INSPIRATIONS (optional) ────────────
Use these as reference examples only.
- They may be worse than the PARENT.
- Do NOT copy phrasing verbatim; prefer improving the PARENT directly.
{insp_text}
──────────────────────────────────────────
"""

_RANK_TEMPLATE = """You are judging {n} candidate texts.

Metrics: {metrics_list_str}
{metric_section}

For each metric, group ALL candidates into tiers from best to worst.
If candidates are effectively indistinguishable for a metric, you may tie them by placing them in the same tier.
Use the metric names exactly as provided above.

Candidates:
{candidates_str}
"""


def build_mutation_prompt(
    *,
    parent: Elite,
    inspirations: Sequence[Elite],
    goal: str,
    instructions: str,
    max_edits: int,
    metrics: Sequence[str],
    metric_descriptions: Mapping[str, str] | None,
    show_metric_stats: bool,
    score_lcb_c: float,
    inspiration_labels: Sequence[str] | None,
) -> str:
    p_stats = (
        _format_metric_stats(parent, metrics, score_lcb_c) if show_metric_stats else ""
    )
    metric_section = _format_metric_definitions(metrics, metric_descriptions)
    insp_lines = [
        _format_inspiration(
            elite,
            i,
            metrics=metrics,
            score_lcb_c=score_lcb_c,
            show_metric_stats=show_metric_stats,
            label=inspiration_labels[i - 1]
            if inspiration_labels and i - 1 < len(inspiration_labels)
            else None,
        )
        for i, elite in enumerate(inspirations, 1)
    ]
    return _MUTATION_TEMPLATE.format(
        goal=goal,
        instructions=instructions,
        metric_section=metric_section,
        max_edits=max_edits,
        p_score=_score_lcb(parent, metrics, score_lcb_c),
        p_stats=p_stats,
        p_text=parent.text,
        insp_text="\n\n".join(insp_lines) or "(none)",
    )


def build_rank_prompt(
    *,
    metrics: Sequence[str],
    items: Sequence[tuple[int, str]],
    metric_descriptions: Mapping[str, str] | None,
) -> str:
    candidate_lines = []
    for idx, text in items:
        candidate_lines.append(f"[{idx}]\n{text}\n")
    candidates_str = "\n".join(candidate_lines)
    metrics_list_str = ", ".join(metrics)
    metric_section = _format_metric_definitions(metrics, metric_descriptions)
    return _RANK_TEMPLATE.format(
        n=len(items),
        metrics_list_str=metrics_list_str,
        metric_section=metric_section,
        candidates_str=candidates_str,
    )


def _score_lcb(elite: Elite, metrics: Sequence[str], c: float) -> float:
    if not metrics:
        return 0.0
    total = 0.0
    for metric in metrics:
        rating = elite.ratings.get(metric)
        if rating is None:
            continue
        total += rating.mu - c * rating.sigma
    return total / len(metrics)


def _format_metric_stats(elite: Elite, metrics: Sequence[str], c: float) -> str:
    lines = []
    for metric in metrics:
        rating = elite.ratings.get(metric)
        if rating is None:
            continue
        lcb = rating.mu - c * rating.sigma
        lines.append(
            f"{metric}: mu={rating.mu:.2f}, sigma={rating.sigma:.2f}, lcb={lcb:.2f}"
        )
    return "Per-metric stats:\n" + "\n".join(lines)


def _format_inspiration(
    elite: Elite,
    idx: int,
    *,
    metrics: Sequence[str],
    score_lcb_c: float,
    show_metric_stats: bool,
    label: str | None,
) -> str:
    label_part = f" {label}" if label else ""
    header = f"[{idx}]{label_part} score={_score_lcb(elite, metrics, score_lcb_c):.3f}"
    if show_metric_stats:
        stats = _format_metric_stats(elite, metrics, score_lcb_c)
        return f"{header}\n{stats}\n{elite.text}"
    return f"{header}\n{elite.text}"


def _format_metric_definitions(
    metrics: Iterable[str],
    descriptions: Mapping[str, str] | None,
) -> str:
    if not descriptions:
        return ""
    lines: list[str] = []
    for metric_name in metrics:
        desc = descriptions.get(metric_name)
        if not desc or not desc.strip():
            continue
        desc_lines = [
            line.strip() for line in desc.strip().splitlines() if line.strip()
        ]
        if not desc_lines:
            continue
        lines.append(f"- {metric_name}: {desc_lines[0]}")
        lines.extend(f"  {line}" for line in desc_lines[1:])
    if not lines:
        return ""
    return "Metric definitions:\n" + "\n".join(lines) + "\n"
