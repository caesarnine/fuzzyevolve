"""
This module abstracts the interaction with the language model provider.
"""

import logging
import random
from typing import List, Tuple

from litellm import completion

from fuzzyevolve.config import LLMEntry

log_llm = logging.getLogger("llm")


class LLMProvider:
    """A wrapper around LiteLLM to handle model selection and API calls."""

    def __init__(self, llm_ensemble: List[LLMEntry]):
        self.llm_ensemble = llm_ensemble

    def _pick_model(self) -> Tuple[str, float]:
        """Selects a model from the ensemble based on defined probabilities."""
        models, probs, temps = zip(
            *[(e.model, e.p, e.temperature) for e in self.llm_ensemble]
        )
        idx = random.choices(range(len(models)), weights=probs)[0]
        return models[idx], temps[idx]

    def call(self, prompt: str, response_format: dict | None = None) -> str:
        """Selects a model and makes a call to the LLM API."""
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
