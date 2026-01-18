from __future__ import annotations

import logging
import random
import threading
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from fuzzyevolve.adapters.llm.ensemble import ModelEnsemble
from fuzzyevolve.adapters.llm.prompts import build_mutation_prompt
from fuzzyevolve.config import ModelSpec
from fuzzyevolve.core.models import Elite, MutationCandidate, TextEdit
from fuzzyevolve.mutation.patcher import apply_edits

log_llm = logging.getLogger("llm.mutator")


class MutationDiff(BaseModel):
    search: str = Field(
        ...,
        description=(
            "Exact substring to search for in the PARENT text. "
            "Must appear verbatim in the PARENT text."
        ),
    )
    replace: str = Field(..., description="Replacement text for the matched substring.")


class MutationOutput(BaseModel):
    diffs: list[MutationDiff] = Field(
        default_factory=list,
        description="Edits to apply to the original PARENT (in order) to produce one child.",
    )


class LLMMutator:
    def __init__(
        self,
        *,
        ensemble: Sequence[ModelSpec],
        metrics: Sequence[str],
        metric_descriptions: Mapping[str, str] | None,
        goal: str,
        instructions: str,
        max_edits: int,
        show_metric_stats: bool,
        score_lcb_c: float,
        rng: random.Random | None = None,
    ) -> None:
        self.ensemble = ModelEnsemble(ensemble, rng=rng)
        self.metrics = list(metrics)
        self.metric_descriptions = dict(metric_descriptions or {})
        self.goal = goal
        self.instructions = instructions
        self.max_edits = max_edits
        self.show_metric_stats = show_metric_stats
        self.score_lcb_c = score_lcb_c
        self._agent_local = threading.local()
        self._agent_instructions = (
            "Propose high-signal SEARCH/REPLACE style edits.\n"
            "- Return ONLY the structured output (no prose).\n"
            "- Return multiple diffs only when they should be applied together.\n"
            "- `search` must be an exact substring of the parent.\n"
        )

    def _get_agent(self) -> Agent:
        agent = getattr(self._agent_local, "agent", None)
        if agent is None:
            agent = Agent(
                output_type=MutationOutput,
                name="mutator",
                instructions=self._agent_instructions,
            )
            self._agent_local.agent = agent
        return agent

    def propose(
        self,
        *,
        parent: Elite,
        inspirations: Sequence[Elite],
        inspiration_labels: Sequence[str] | None = None,
    ) -> Sequence[MutationCandidate]:
        prompt = build_mutation_prompt(
            parent=parent,
            inspirations=inspirations,
            goal=self.goal,
            instructions=self.instructions,
            max_edits=self.max_edits,
            metrics=self.metrics,
            metric_descriptions=self.metric_descriptions,
            show_metric_stats=self.show_metric_stats,
            score_lcb_c=self.score_lcb_c,
            inspiration_labels=inspiration_labels,
        )
        log_llm.debug("Mutation prompt:\n%s", prompt)

        model, model_settings = self.ensemble.pick()
        agent = self._get_agent()
        try:
            rsp = agent.run_sync(prompt, model=model, model_settings=model_settings)
        except Exception:
            log_llm.exception("Mutator call failed; returning no candidates.")
            return []

        diffs = rsp.output.diffs[: self.max_edits]
        if not diffs:
            return []

        edits = tuple(
            TextEdit(search=diff.search, replace=diff.replace) for diff in diffs
        )
        patch = apply_edits(parent.text, [(e.search, e.replace) for e in edits])
        if not patch.success or patch.new_text is None or patch.new_text == parent.text:
            return []

        return [MutationCandidate(text=patch.new_text, edits=edits)]
