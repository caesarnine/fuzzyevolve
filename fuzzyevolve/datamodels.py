from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

class Rating(BaseModel):
    mu: float
    sigma: float

class Elite(BaseModel):
    txt: str
    rating: Dict[str, Rating]
    age: int
    cell_key: Optional[Tuple] = None

@dataclass
class MutationEvent:
    iteration: int
    island: int
    parent_s: float
    child_s: float
    diff: str
