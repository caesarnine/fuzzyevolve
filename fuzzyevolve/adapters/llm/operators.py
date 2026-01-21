from __future__ import annotations

import logging
import random
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from fuzzyevolve.adapters.llm.ensemble import ModelEnsemble
from fuzzyevolve.adapters.llm.prompts import build_rewrite_prompt
from fuzzyevolve.config import ModelSpec
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite

log_llm = logging.getLogger("llm.operator")


class Recorder(Protocol):
    def put_text(self, text: str) -> str: ...

    def record_llm_call(
        self,
        *,
        name: str,
        model: str,
        model_settings: Mapping[str, Any] | None,
        prompt: str,
        output: Any | None,
        error: str | None = None,
        iteration: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None: ...


class RewriteOutput(BaseModel):
    text: str = Field(..., description="The full rewritten child text.")


class LLMRewriteOperator:
    def __init__(
        self,
        *,
        name: str,
        role: str,
        ensemble: Sequence[ModelSpec],
        temperature: float | None,
        goal: str,
        metrics: Sequence[str],
        metric_descriptions: Mapping[str, str] | None,
        instructions: str,
        show_metric_stats: bool,
        score_lcb_c: float,
        rng: random.Random | None = None,
        store: Recorder | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.ensemble = ModelEnsemble(ensemble, rng=rng)
        self.temperature = temperature
        self.goal = goal
        self.metrics = list(metrics)
        self.metric_descriptions = dict(metric_descriptions or {})
        self.instructions = instructions
        self.show_metric_stats = show_metric_stats
        self.score_lcb_c = score_lcb_c
        self.store = store
        self._agent_local = threading.local()
        self._agent_instructions = (
            "Generate exactly one rewritten child text.\n"
            "- Return ONLY the structured output (no prose).\n"
            "- Do not mention evaluation metrics, ratings, or judging.\n"
        )

    def _get_agent(self) -> Agent:
        agent = getattr(self._agent_local, "agent", None)
        if agent is None:
            agent = Agent(
                output_type=RewriteOutput,
                name=f"operator.{self.name}",
                instructions=self._agent_instructions,
            )
            self._agent_local.agent = agent
        return agent

    def propose(
        self,
        *,
        parent: Elite,
        critique: Critique | None,
        focus: str | None = None,
    ) -> Sequence[str]:
        prompt = build_rewrite_prompt(
            parent=parent,
            goal=self.goal,
            operator_name=self.name,
            role=self.role,
            operator_instructions=self.instructions,
            critique=critique,
            focus=focus,
            metrics=self.metrics,
            metric_descriptions=self.metric_descriptions,
            show_metric_stats=self.show_metric_stats,
            score_lcb_c=self.score_lcb_c,
        )
        log_llm.debug("Operator '%s' prompt:\n%s", self.name, prompt)

        model, model_settings = self.ensemble.pick()
        if self.temperature is not None:
            model_settings = dict(model_settings)
            model_settings["temperature"] = self.temperature

        agent = self._get_agent()
        try:
            rsp = agent.run_sync(prompt, model=model, model_settings=model_settings)
        except Exception:
            log_llm.exception(
                "Operator '%s' call failed; returning no candidates.", self.name
            )
            if self.store:
                try:
                    self.store.record_llm_call(
                        name=f"operator.{self.name}",
                        model=model,
                        model_settings=model_settings,
                        prompt=prompt,
                        output=None,
                        error="operator_call_failed",
                        extra={
                            "operator": self.name,
                            "role": self.role,
                            "focus": focus,
                        },
                    )
                except Exception:
                    log_llm.exception("Failed to record operator call.")
            return []

        out = rsp.output
        if self.store:
            try:
                self.store.record_llm_call(
                    name=f"operator.{self.name}",
                    model=model,
                    model_settings=model_settings,
                    prompt=prompt,
                    output=out,
                    extra={"operator": self.name, "role": self.role, "focus": focus},
                )
            except Exception:
                log_llm.exception("Failed to record operator call.")

        text = out.text.strip()
        if not text or text == parent.text:
            return []
        if self.store:
            try:
                self.store.put_text(text)
            except Exception:
                log_llm.exception("Failed to store operator output text.")
        return [text]
