from __future__ import annotations

from typing import Sequence

from fuzzyevolve.core.models import Elite
from fuzzyevolve.core.scoring import score_ratings

_MUT_PROMPT_TEMPLATE = """
Overall goal: {goal}
Task instructions: {instructions}

Return up to {max_diffs} alternative edits.
Each edit must specify:
- `search`: an exact substring from the PARENT text (verbatim)
- `replace`: replacement text for that substring

Constraints:
- Each edit is evaluated independently; do not chain edits.
- Prefer a `search` span with enough context to be unique.
- If you can't find a safe exact-match `search` span, return no edits.

──────────────── PARENT ────────────────
Score: {p_score:.3f}
{p_stats}
{p_text}

──────────────── INSPIRATIONS ───────────
{insp_text}
──────────────────────────────────────────
"""

_RANK_PROMPT_TEMPLATE = """You are judging {n} candidate texts.

Metrics: {metrics_list_str}

For each metric, rank ALL candidates from best to worst.
Use the metric names exactly as provided above.

Candidates:
{candidates_str}
"""


def build_mutation_prompt(
    parent: Elite,
    inspirations: Sequence[Elite],
    goal: str,
    instructions: str,
    max_diffs: int,
    show_metric_stats: bool,
    metric_c: float,
) -> str:
    p_stats = _format_metric_stats(parent, metric_c) if show_metric_stats else ""
    insp_lines = [
        _format_inspiration(elite, i, show_metric_stats, metric_c)
        for i, elite in enumerate(inspirations, 1)
    ]
    return _MUT_PROMPT_TEMPLATE.format(
        goal=goal,
        instructions=instructions,
        max_diffs=max_diffs,
        p_score=score_ratings(parent.ratings, c=metric_c),
        p_stats=p_stats,
        p_text=parent.text,
        insp_text="\n\n".join(insp_lines) or "(none)",
    )


def build_rank_prompt(
    metrics: Sequence[str], items: Sequence[tuple[int, Elite]]
) -> str:
    candidate_lines = []
    for idx, elite_data in items:
        candidate_lines.append(f"[{idx}]\n{elite_data.text}\n")

    candidates_str = "\n".join(candidate_lines)
    metrics_list_str = ", ".join(metrics)

    return _RANK_PROMPT_TEMPLATE.format(
        n=len(items),
        metrics_list_str=metrics_list_str,
        candidates_str=candidates_str,
    )


def _format_metric_stats(elite: Elite, c: float) -> str:
    lines = []
    for metric, rating in elite.ratings.items():
        lcb = rating.mu - c * rating.sigma
        lines.append(
            f"{metric}: mu={rating.mu:.2f}, sigma={rating.sigma:.2f}, lcb={lcb:.2f}"
        )
    return "Per-metric stats:\n" + "\n".join(lines)


def _format_inspiration(
    elite: Elite, idx: int, show_metric_stats: bool, metric_c: float
) -> str:
    header = f"[{idx}] score={score_ratings(elite.ratings, c=metric_c):.3f}"
    if show_metric_stats:
        stats = _format_metric_stats(elite, metric_c)
        return f"{header}\n{stats}\n{elite.text}"
    return f"{header}\n{elite.text}"
