from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite

_CRITIQUE_TEMPLATE = """You are critiquing a text for an evolutionary rewriting system.

Overall goal: {goal}
Metrics: {metrics_list_str}
{metric_section}

Return structured critique with:
- summary: 1–3 sentences on what's working and what's not.
- preserve: 3–7 things worth keeping (voice, constraints, standout elements).
- issues: 3–10 prioritized, actionable improvements.
- routes: {routes} distinct rewrite routes. Each route is a short directive (1–3 sentences) that could guide a full rewrite.
- constraints: 0–6 hard constraints (optional).

Do not write any new story text.
Do not mention TrueSkill, ratings, or evaluation mechanics.

──────────────── PARENT ────────────────
Score (LCB avg): {p_score:.3f}
{p_stats}
{p_text}
────────────────────────────────────────
"""


_REWRITE_TEMPLATE = """You are generating ONE child text for an evolutionary rewriting system.

Overall goal: {goal}
Operator: {operator_name} ({role})
Operator instructions: {operator_instructions}
Metrics: {metrics_list_str}
{metric_section}

Use the critique as guidance. If a focus is provided, prioritize it.
Do not mention evaluation metrics, ratings, or judging.
Return structured output only.

Focus (optional):
{focus}

Critique summary:
{summary}

Preserve:
{preserve}

Issues:
{issues}

Constraints:
{constraints}

{parent_section}
"""

_REWRITE_TEMPLATE_EXPLORE = """You are generating ONE child text for an evolutionary rewriting system.

Overall goal: {goal}
Operator: {operator_name} ({role})
Operator instructions: {operator_instructions}
Metrics: {metrics_list_str}
{metric_section}

Use the provided focus as the primary creative constraint when present.
Stay faithful to the overall goal/premise, but otherwise explore widely (voice, structure, POV, genre, format).

Do NOT reuse specific names, phrases, or concrete props from the parent unless they are required by the overall goal/premise.
Do not mention evaluation metrics, ratings, or judging.
Return structured output only.

Focus (optional):
{focus}

(Parent text intentionally omitted for exploration.)
"""


_PARENT_SECTION = """──────────────── PARENT ────────────────
Score (LCB avg): {p_score:.3f}
{p_stats}
{p_text}
────────────────────────────────────────
"""


def build_critique_prompt(
    *,
    parent: Elite,
    goal: str,
    metrics: Sequence[str],
    metric_descriptions: Mapping[str, str] | None,
    routes: int,
    show_metric_stats: bool,
    score_lcb_c: float,
) -> str:
    p_stats = (
        _format_metric_stats(parent, metrics, score_lcb_c) if show_metric_stats else ""
    )
    metrics_list_str = ", ".join(metrics)
    metric_section = _format_metric_definitions(metrics, metric_descriptions)
    return _CRITIQUE_TEMPLATE.format(
        goal=goal,
        metrics_list_str=metrics_list_str,
        metric_section=metric_section,
        routes=routes,
        p_score=_score_lcb(parent, metrics, score_lcb_c),
        p_stats=p_stats,
        p_text=parent.text,
    )


def build_rewrite_prompt(
    *,
    parent: Elite,
    goal: str,
    operator_name: str,
    role: str,
    operator_instructions: str,
    critique: Critique | None,
    focus: str | None,
    metrics: Sequence[str],
    metric_descriptions: Mapping[str, str] | None,
    show_metric_stats: bool,
    score_lcb_c: float,
) -> str:
    metrics_list_str = ", ".join(metrics)
    metric_section = _format_metric_definitions(metrics, metric_descriptions)
    p_stats = (
        _format_metric_stats(parent, metrics, score_lcb_c) if show_metric_stats else ""
    )

    if role == "explore":
        return _REWRITE_TEMPLATE_EXPLORE.format(
            goal=goal,
            operator_name=operator_name,
            role=role,
            operator_instructions=operator_instructions,
            metrics_list_str=metrics_list_str,
            metric_section=metric_section,
            focus=(focus.strip() if focus else "(none)"),
        )
    else:
        parent_section = _PARENT_SECTION.format(
            p_score=_score_lcb(parent, metrics, score_lcb_c),
            p_stats=p_stats,
            p_text=parent.text,
        )

    return _REWRITE_TEMPLATE.format(
        goal=goal,
        operator_name=operator_name,
        role=role,
        operator_instructions=operator_instructions,
        metrics_list_str=metrics_list_str,
        metric_section=metric_section,
        focus=(focus.strip() if focus else "(none)"),
        summary=(
            critique.summary.strip() if critique and critique.summary else "(none)"
        ),
        preserve=_format_lines(critique.preserve if critique else ()),
        issues=_format_lines(critique.issues if critique else ()),
        constraints=_format_lines(critique.constraints if critique else ()),
        parent_section=parent_section,
    )


_RANK_TEMPLATE = """You are judging {n} candidate texts.

Overall goal: {goal}

Metrics: {metrics_list_str}
{metric_section}

For each metric, group ALL candidates into tiers from best to worst.
If candidates are effectively indistinguishable for a metric, you may tie them by placing them in the same tier.
Use the metric names exactly as provided above.

Primary requirement:
- Candidates should adhere to the overall goal/premise.
- If a candidate clearly violates the goal/premise, place it in the worst tier for EVERY metric, regardless of writing quality.

Candidates:
{candidates_str}
"""


def build_rank_prompt(
    *,
    goal: str | None = None,
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
        goal=(goal.strip() if goal else "(none provided)"),
        metrics_list_str=metrics_list_str,
        metric_section=metric_section,
        candidates_str=candidates_str,
    )


def _format_lines(lines: Iterable[str]) -> str:
    cleaned = [line.strip() for line in lines if line and line.strip()]
    if not cleaned:
        return "(none)"
    return "\n".join(f"- {line}" for line in cleaned)


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
    lines: list[str] = []
    for metric in metrics:
        rating = elite.ratings.get(metric)
        if rating is None:
            continue
        lcb = rating.mu - c * rating.sigma
        lines.append(
            f"{metric}: mu={rating.mu:.2f}, sigma={rating.sigma:.2f}, lcb={lcb:.2f}"
        )
    if not lines:
        return ""
    return "Per-metric stats:\n" + "\n".join(lines)


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
