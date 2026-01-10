"""LLM provider wrapper around LiteLLM."""

from __future__ import annotations

import logging
import random
from typing import Sequence

from litellm import completion

from fuzzyevolve.llm.models import ModelSpec

log_llm = logging.getLogger("llm")


class LLMProvider:
    """Wrapper around LiteLLM for model selection and API calls."""

    def __init__(
        self, llm_ensemble: Sequence[ModelSpec], rng: random.Random | None = None
    ):
        self.llm_ensemble = list(llm_ensemble)
        if not self.llm_ensemble:
            raise ValueError("LLM ensemble cannot be empty.")
        self.rng = rng or random.Random()

    def _pick_model(self) -> tuple[str, float]:
        models, probs, temps = zip(
            *[(e.model, e.p, e.temperature) for e in self.llm_ensemble]
        )
        idx = self.rng.choices(range(len(models)), weights=probs)[0]
        return models[idx], temps[idx]

    def call(self, prompt: str, response_format: dict | None = None) -> str:
        model, temperature = self._pick_model()
        try:
            rsp = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format=response_format,
            )
        except Exception as exc:
            log_llm.exception("LLM call failed: %s", exc)
            raise

        log_llm.debug("PROMPT\n%s", prompt)
        log_llm.debug("RAW RESPONSE\n%s", rsp)
        return rsp.choices[0].message.content.strip()
