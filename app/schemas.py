"""Request and response models."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    gemini: Literal["configured", "missing"]
    chroma: str
    document_count: int
