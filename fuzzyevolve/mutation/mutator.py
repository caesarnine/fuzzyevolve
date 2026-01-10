from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from fuzzyevolve.core.models import Elite
from fuzzyevolve.llm.client import LLMProvider
from fuzzyevolve.llm.parsing import parse_mutation_response
from fuzzyevolve.llm.prompts import build_mutation_prompt
from fuzzyevolve.mutation.diff import DiffBlock, extract_blocks

log_mut = logging.getLogger("mutation")


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    text: str
    diff: str


@dataclass(frozen=True, slots=True)
class MutationResult:
    thinking: str | None
    candidates: list[MutationCandidate]


class LLMMutator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        goal: str,
        instructions: str,
        max_diffs: int,
    ) -> None:
        self.llm_provider = llm_provider
        self.goal = goal
        self.instructions = instructions
        self.max_diffs = max_diffs

    def propose(self, parent: Elite, inspirations: Sequence[Elite]) -> MutationResult:
        prompt = build_mutation_prompt(
            parent,
            inspirations,
            goal=self.goal,
            instructions=self.instructions,
        )
        reply = self.llm_provider.call(prompt)
        thinking, diff_content = parse_mutation_response(reply, log_mut)

        if thinking:
            log_mut.info("Mutator rationale:\n%s", thinking)

        if not diff_content:
            return MutationResult(thinking=thinking, candidates=[])

        blocks = extract_blocks(diff_content, log_mut)[: self.max_diffs]
        candidates: list[MutationCandidate] = []
        for block in blocks:
            new_text = block.apply(parent.text)
            if new_text is None or new_text == parent.text:
                continue
            candidates.append(MutationCandidate(text=new_text, diff=block.raw))

        return MutationResult(thinking=thinking, candidates=candidates)
