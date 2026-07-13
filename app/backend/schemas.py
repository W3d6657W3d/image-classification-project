"""Pydantic response models for FastAPI endpoints.

Keeping response schemas explicit makes the API contract visible in OpenAPI
docs and gives the Streamlit frontend predictable field names to consume.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """One class-probability pair in a Top-K prediction response."""

    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health-check payload returned by ``GET /health``."""

    status: str
    model_loaded: bool
    device: str
    classes: list[str]


class PredictionResponse(BaseModel):
    """Prediction payload returned by ``POST /predict``."""

    id: int
    filename: str
    stored_path: str
    top_k: int
    predictions: list[PredictionItem]
    inference_ms: float


class HistoryItem(BaseModel):
    """One stored prediction item returned by ``GET /history``."""

    id: int
    filename: str
    stored_path: str
    top_k: int
    predictions: list[PredictionItem]
    inference_ms: float
    created_at: str
