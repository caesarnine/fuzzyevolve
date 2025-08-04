from __future__ import annotations
import json
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field

class LLMEntry(BaseModel):
    model: str
    p: float = Field(..., ge=0)
    temperature: float = 0.7


class Config(BaseModel):
    # core loop
    seed_path: str = "seed.txt"
    iterations: int = 10
    log_every: int = 1

    # map-elites + islands
    num_islands: int = 1
    k_top: int = 4
    migration_every: int = 300
    migrants_per_island: int = 4

    # llm ensemble + judge
    llm_ensemble: List[LLMEntry] = Field(
        default_factory=lambda: [
            LLMEntry(
                model="vertex_ai/gemini-2.5-flash",
                p=0.85,
                temperature=1,
            ),
            LLMEntry(
                model="vertex_ai/gemini-2.5-pro", p=0.15, temperature=1
            ),
        ]
    )
    judge_model: str = "vertex_ai/gemini-2.5-pro"
    metrics: List[str] = ["clarity", "conciseness", "creativity"]
    # descriptor space
    axes: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lang": ["txt"],
            "len": {"bins": [0, 500, 1000, 2000, 1e9]},
        }
    )

    n_diffs: int = 4
    youth_bias: float = 0.30
    # rarely-needed global sparring (still helpful early on)
    sparring_every: int = 100

    # mutation prompt
    mutation_prompt_goal: str = "Improve the text based on the metrics provided."
    mutation_prompt_instructions: str = (
        "Propose one or more SEARCH/REPLACE diff blocks to improve the PARENT text. "
        "You can rewrite, shorten, or completely change the text. "
        "First, explain your reasoning in a <thinking> block. Then, provide the diffs in a <diffs> block."
    )


def load_cfg(path: str | None) -> Config:
    if not path:
        return Config()
    data = Path(path).read_text()
    try:
        cfg_dict = json.loads(data)
    except json.JSONDecodeError:
        cfg_dict = tomllib.loads(data)
    return Config.model_validate(cfg_dict)
