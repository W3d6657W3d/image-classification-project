# Docker Deployment

## Goal

This project provides a lightweight Docker setup for local deployment and
portfolio demonstration. Docker runs the FastAPI backend and Streamlit frontend
as separate services while reusing the same Python project image.

## Prerequisites

- Docker Desktop for Windows, macOS, or Linux
- Local model checkpoint:

```text
models/best_mobilenet_v2.pth
```

The model file is not committed to Git because `models/` is treated as a local
runtime artifact.

## Services

```text
backend   FastAPI inference API      http://localhost:8000
frontend  Streamlit demo UI          http://localhost:8501
```

Inside Docker Compose, the frontend calls the backend through:

```text
http://backend:8000
```

## Run With Docker Compose

From the project root:

```powershell
docker compose up --build
```

Open the frontend:

```text
http://localhost:8501
```

Open the FastAPI docs:

```text
http://localhost:8000/docs
```

## Runtime Volumes

The Compose setup mounts local runtime folders into the backend container:

```text
./models   -> /app/models   read-only model checkpoint
./logs     -> /app/logs     SQLite prediction history
./uploads  -> /app/uploads  uploaded images
```

This keeps the Docker image smaller and avoids committing trained model files,
uploaded images, or SQLite databases to Git.

## Stop Services

```powershell
docker compose down
```

## Verification Checklist

After Docker is installed, verify:

- `docker --version` works
- `docker compose version` works
- `models/best_mobilenet_v2.pth` exists locally
- `docker compose up --build` starts both services
- `http://localhost:8501` opens the Streamlit frontend
- image upload, Top-K prediction, inference time, and history display work

## Verification Status

Docker Compose deployment has been verified locally. The Streamlit frontend,
FastAPI backend, image upload flow, Top-K prediction display, inference time,
and recent prediction history were tested successfully.
