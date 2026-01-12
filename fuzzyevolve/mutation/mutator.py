from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from fuzzyevolve.core.models import Elite
from fuzzyevolve.llm.client import ModelEnsemble
from fuzzyevolve.llm.models import ModelSpec
from fuzzyevolve.llm.prompts import build_mutation_prompt
from fuzzyevolve.mutation.patcher import PatchConfig, apply_patch

log_mut = logging.getLogger("mutation")


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    text: str
    search: str
    replace: str


@dataclass(frozen=True, slots=True)
class MutationResult:
    candidates: list[MutationCandidate]


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
        description=(
            "Alternative edits; apply each diff independently to the original PARENT."
        ),
    )


class LLMMutator:
    def __init__(
        self,
        llm_ensemble: Sequence[ModelSpec],
        goal: str,
        instructions: str,
        max_diffs: int,
        patch_cfg: PatchConfig,
        show_metric_stats: bool,
        metric_c: float,
        rng: random.Random | None = None,
    ) -> None:
        self.model_ensemble = ModelEnsemble(llm_ensemble, rng=rng)
        self.goal = goal
        self.instructions = instructions
        self.max_diffs = max_diffs
        self.patch_cfg = patch_cfg
        self.show_metric_stats = show_metric_stats
        self.metric_c = metric_c
        self.agent = Agent(
            output_type=MutationOutput,
            name="mutator",
            instructions=(
                "Propose high-signal SEARCH/REPLACE style edits.\n"
                "- Return ONLY the structured output (no prose).\n"
                "- Each diff must be independently applicable to the original parent.\n"
                "- `search` must be an exact substring of the parent.\n"
            ),
        )

    def propose(self, parent: Elite, inspirations: Sequence[Elite]) -> MutationResult:
        prompt = build_mutation_prompt(
            parent,
            inspirations,
            goal=self.goal,
            instructions=self.instructions,
            max_diffs=self.max_diffs,
            show_metric_stats=self.show_metric_stats,
            metric_c=self.metric_c,
        )
        log_mut.debug("Mutation prompt:\n%s", prompt)
        model, model_settings = self.model_ensemble.pick()
        try:
            rsp = self.agent.run_sync(
                prompt,
                model=model,
                model_settings=model_settings,
            )
            log_mut.debug("Mutation response: %s", rsp.output)
        except Exception:
            log_mut.exception(
                "Mutator agent call failed — skipping mutation proposal batch."
            )
            return MutationResult(candidates=[])

        diffs = rsp.output.diffs[: self.max_diffs]
        candidates: list[MutationCandidate] = []
        for diff in diffs:
            patch = apply_patch(parent.text, diff.search, diff.replace, self.patch_cfg)
            if not patch.success or patch.new_text is None:
                continue
            if patch.new_text == parent.text:
                continue
            candidates.append(
                MutationCandidate(
                    text=patch.new_text,
                    search=diff.search,
                    replace=diff.replace,
                )
            )
        if candidates:
            log_mut.debug("Mutation candidates: %s", candidates)

        return MutationResult(candidates=candidates)
