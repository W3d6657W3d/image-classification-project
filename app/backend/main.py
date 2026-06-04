"""FastAPI inference service for garbage image classification."""

from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.backend.config import (
    ALLOWED_EXTENSIONS,
    DATABASE_PATH,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MODEL_PATH,
    UPLOAD_DIR,
)
from app.backend.database import init_db, insert_prediction, list_predictions
from app.backend.schemas import HealthResponse, HistoryItem, PredictionResponse
from src.inference.service import ImageClassifier


app = FastAPI(
    title="Garbage Classification API",
    description="Upload an image and receive Top-K garbage classification predictions.",
    version="0.1.0",
)

classifier: ImageClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    init_db(DATABASE_PATH)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")
    classifier = ImageClassifier(MODEL_PATH)
    yield


app.router.lifespan_context = lifespan


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        device=str(classifier.device) if classifier else "unavailable",
        classes=classifier.class_names if classifier else [],
    )


@app.get("/classes", response_model=list[str])
def classes() -> list[str]:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return classifier.class_names


def save_upload(file: UploadFile) -> Path:
    original_name = file.filename or "upload.jpg"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    target_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    with target_path.open("wb") as output_file:
        shutil.copyfileobj(file.file, output_file)
    return target_path


@app.post("/predict", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
) -> PredictionResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    stored_path = save_upload(file)
    try:
        predictions, inference_ms = classifier.predict_path(stored_path, top_k=top_k)
    except ValueError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row_id = insert_prediction(
        DATABASE_PATH,
        filename=file.filename or stored_path.name,
        stored_path=str(stored_path),
        top_k=top_k,
        predictions=predictions,
        inference_ms=inference_ms,
    )

    return PredictionResponse(
        id=row_id,
        filename=file.filename or stored_path.name,
        stored_path=str(stored_path),
        top_k=top_k,
        predictions=predictions,
        inference_ms=inference_ms,
    )


@app.get("/history", response_model=list[HistoryItem])
def history(limit: int = Query(default=20, ge=1, le=100)) -> list[HistoryItem]:
    return [HistoryItem(**item) for item in list_predictions(DATABASE_PATH, limit=limit)]
