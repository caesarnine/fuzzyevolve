from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    model: str
    p: float = Field(..., ge=0)
    temperature: float = 0.7
