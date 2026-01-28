from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from fuzzyevolve.adapters.llm.prompts import build_critique_prompt
from fuzzyevolve.core.critique import Critique
from fuzzyevolve.core.models import Elite

log_llm = logging.getLogger("llm.critic")


class Recorder(Protocol):
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


class CritiqueOutput(BaseModel):
    summary: str = ""
    preserve: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class LLMCritic:
    def __init__(
        self,
        *,
        model: str,
        model_settings: ModelSettings | None = None,
        goal: str,
        metrics: Sequence[str],
        metric_descriptions: Mapping[str, str] | None,
        routes: int,
        instructions: str,
        show_metric_stats: bool,
        score_lcb_c: float,
        store: Recorder | None = None,
    ) -> None:
        self.model = model
        self.model_settings = model_settings or {"temperature": 0.2}
        self.goal = goal
        self.metrics = list(metrics)
        self.metric_descriptions = dict(metric_descriptions or {})
        self.routes = routes
        self.instructions = instructions
        self.show_metric_stats = show_metric_stats
        self.score_lcb_c = score_lcb_c
        self.store = store

        self.agent = Agent(
            output_type=CritiqueOutput,
            name="critic",
            instructions=(
                f"{self.instructions.strip()}\n\n"
                "Return ONLY the structured output (no prose).\n"
                "- `routes` must be meaningfully distinct.\n"
                "- Use concrete, actionable language.\n"
                "- Do not include any new story text.\n"
            ),
        )

    def critique(
        self,
        *,
        parent: Elite,
    ) -> Critique | None:
        parent_text_id: str | None = None
        if self.store:
            try:
                parent_text_id = getattr(self.store, "put_text", lambda _t: None)(
                    parent.text
                )
            except Exception:
                parent_text_id = None

        prompt = build_critique_prompt(
            parent=parent,
            goal=self.goal,
            metrics=self.metrics,
            metric_descriptions=self.metric_descriptions,
            routes=self.routes,
            show_metric_stats=self.show_metric_stats,
            score_lcb_c=self.score_lcb_c,
        )
        log_llm.debug("Critic prompt:\n%s", prompt)

        try:
            rsp = self.agent.run_sync(
                prompt,
                model=self.model,
                model_settings=self.model_settings,
            )
        except Exception:
            log_llm.exception("Critic call failed; continuing without critique.")
            if self.store:
                try:
                    self.store.record_llm_call(
                        name="critic",
                        model=self.model,
                        model_settings=self.model_settings,
                        prompt=prompt,
                        output=None,
                        error="critic_call_failed",
                        extra={"parent_text_id": parent_text_id},
                    )
                except Exception:
                    log_llm.exception("Failed to record critic call.")
            return None

        out = rsp.output
        if self.store:
            try:
                self.store.record_llm_call(
                    name="critic",
                    model=self.model,
                    model_settings=self.model_settings,
                    prompt=prompt,
                    output=out,
                    extra={"parent_text_id": parent_text_id},
                )
            except Exception:
                log_llm.exception("Failed to record critic call.")
        return Critique(
            summary=(out.summary or "").strip(),
            preserve=tuple(item.strip() for item in out.preserve if item.strip()),
            issues=tuple(item.strip() for item in out.issues if item.strip()),
            routes=tuple(item.strip() for item in out.routes if item.strip()),
            constraints=tuple(item.strip() for item in out.constraints if item.strip()),
        )
