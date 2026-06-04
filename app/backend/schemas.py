"""Pydantic response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str
    probability: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes: list[str]


class PredictionResponse(BaseModel):
    id: int
    filename: str
    stored_path: str
    top_k: int
    predictions: list[PredictionItem]
    inference_ms: float


class HistoryItem(BaseModel):
    id: int
    filename: str
    stored_path: str
    top_k: int
    predictions: list[PredictionItem]
    inference_ms: float
    created_at: str
