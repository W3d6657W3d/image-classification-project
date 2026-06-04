# Backend Inference API

## Goal

The backend exposes the trained MobileNetV2 checkpoint as a FastAPI service.
It supports image upload, Top-K prediction, class lookup, health checks, and
SQLite prediction history.

## Local Run

The model checkpoint must exist locally:

```text
models/best_mobilenet_v2.pth
```

Start the API:

```powershell
uvicorn app.backend.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

```text
GET /health
GET /classes
POST /predict
GET /history
```

## Prediction Request

Upload an image with form-data field name `file`.

Optional query parameter:

```text
top_k: 1 to 5, default 3
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=3" `
  -F "file=@data/processed/garbage_classification/test/battery/battery_0001.jpg"
```

## Runtime Artifacts

Uploaded files:

```text
uploads/
```

SQLite prediction history:

```text
logs/predictions.db
```

Both paths are ignored by Git because they are runtime artifacts.
