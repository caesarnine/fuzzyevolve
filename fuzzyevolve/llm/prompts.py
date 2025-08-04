from __future__ import annotations
import random
from typing import Dict, List, Tuple

from fuzzyevolve.datamodels import Elite
from fuzzyevolve.evolution.scoring import ts_score

_MUT_PROMPT_TEMPLATE = '''
Your overall goal is: {goal}
Your task is: {instructions}

First, provide your step-by-step thinking process on how to improve the PARENT text.

Analyze its weaknesses and explain the changes you will propose, potentially drawing inspiration from the other texts provided.

Then, provide the diffs using the exact SEARCH/REPLACE syntax inside a <diffs> block.

Use exactly this diff syntax:
<<<<<<< SEARCH
<text to match>
=======
<replacement>
>>>>>>> REPLACE

Example Response:

<thinking>
[Your detailed thought process for improving the parent text.]
</thinking>
<diffs>
<<<<<<< SEARCH
<a paragraph from the middle of the parent text>
=======
<a new, improved paragraph with better pacing>
>>>>>>> REPLACE

<<<<<<< SEARCH
<the last sentence of the parent text>
=======
<a new, more impactful ending>
>>>>>>> REPLACE
</diffs>

──────────────── PARENT ────────────────
Score   : {p_score:.3f}
{p_text}

──────────────── INSPIRATIONS ───────────
{insp_text}
──────────────────────────────────────────

Remember to follow the response format exactly: <thinking>...</thinking><diffs>...</diffs>
'''

_RANK_PROMPT_TEMPLATE = '''Below are {n} texts, each tagged with its [ID].
Your task is to evaluate these texts based on the following metrics: {metrics_list_str}.

First, provide your step-by-step thinking process within <thinking> tags.
Then, for each metric, provide a comma-separated list of IDs, ordered from best to worst, within its own XML-like tag. Tag names should be lowercase.

Example for metrics {metrics_list_str}:
<response_format>
<thinking>
[Your detailed rationale for rankings. Explain your reasoning for each metric, comparing the candidates.
For instance, for metric 'metric_name_1', candidate [id_x] was ranked higher than [id_y] because...
For metric 'metric_name_2', candidate [id_z] demonstrated stronger qualities in X, Y, Z leading to its top rank.]
</thinking>
<output>
{metric_tags_str}
</output>
</response_format>

Ensure your response strictly follows this format.

Metrics: {metrics_list_str}

Candidates:
{candidates_str}

Follow this exact response format:
<response_format>
<thinking>[Your step-by-step thinking process and rationale for rankings for each metric]</thinking>
<output>
{metric_tags_str}
</output>
</response_format>
'''

def build_mut_prompt(
    parent: Elite, inspirations: List[Elite], goal: str, instructions: str
) -> str:
    insp_lines = [
        f"[{i}] score={ts_score(e['rating']):.3f}\n{e['txt']}"
        for i, e in enumerate(inspirations, 1)
    ]
    return _MUT_PROMPT_TEMPLATE.format(
        goal=goal,
        instructions=instructions,
        p_score=ts_score(parent["rating"]),
        p_text=parent["txt"],
        insp_text="\n\n".join(insp_lines) or "(none)",
    )

def make_rank_prompt(metrics: List[str], items: List[Tuple[int, Elite]]) -> str:
    candidate_lines = []
    for idx, elite_data in items:
        candidate_lines.append(f"[{idx}]\n{elite_data['txt']}\n")

    candidates_str = "\n".join(candidate_lines)
    metrics_list_str = ", ".join(
        f'"{m}"' for m in metrics
    )

    metric_tags_str_parts = []
    example_ids_str = (
        ", ".join(
            str(i) for i in random.sample(range(len(items)), k=min(len(items), 3))
        )
        if items
        else "id_1, id_2"
    )
    for metric_name in metrics:
        tag_name = metric_name.lower()
        metric_tags_str_parts.append(f"<{tag_name}>[{example_ids_str}]</{tag_name}>")
    metric_tags_str = "\n".join(metric_tags_str_parts)

    return _RANK_PROMPT_TEMPLATE.format(
        n=len(items),
        metrics_list_str=metrics_list_str,
        candidates_str=candidates_str,
        metric_tags_str=metric_tags_str,
    )
