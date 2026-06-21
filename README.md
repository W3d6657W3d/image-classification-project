# Garbage Image Classification Training and Deployment System

A lightweight machine learning engineering portfolio project for 12-class garbage image classification.

This repository demonstrates a complete baseline workflow: dataset processing, exploratory analysis, transfer-learning training, model evaluation, FastAPI inference service, Streamlit demo UI, SQLite prediction history, and Docker Compose local deployment.

The project is designed for junior machine learning engineer, algorithm application engineer, data science, and AI solution engineer applications. The focus is engineering completeness, reproducibility, and clear explanation rather than large-scale model training.

## Project Highlights

- Built an end-to-end image classification pipeline for 12 garbage categories.
- Processed an imbalanced raw dataset into a balanced train/validation/test split.
- Trained a MobileNetV2 transfer-learning baseline with PyTorch and torchvision.
- Evaluated the model with accuracy, macro precision, macro recall, macro F1, per-class metrics, and confusion matrix.
- Packaged the trained model behind a FastAPI service with image upload, Top-K prediction, health check, class lookup, and prediction history endpoints.
- Built a Streamlit frontend for image preview, configurable Top-K inference, latency display, and recent prediction history.
- Used SQLite to persist prediction records and Docker Compose to run backend and frontend services locally.
- Documented the project as a job-application-ready ML engineering case study.

## Results

The current baseline uses MobileNetV2 on a held-out test split.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9537 |
| Macro precision | 0.9547 |
| Macro recall | 0.9537 |
| Macro F1 | 0.9538 |

Dataset split:

| Split | Images |
| --- | ---: |
| Train | 5,040 |
| Validation | 1,080 |
| Test | 1,080 |

Classes:

```text
battery, biological, brown-glass, cardboard, clothes, green-glass,
metal, paper, plastic, shoes, trash, white-glass
```

See [reports/model_report.md](reports/model_report.md) for per-class metrics and [docs/dataset_report.md](docs/dataset_report.md) for dataset processing details.

## System Architecture

```text
raw dataset
  -> dataset analysis and validation
  -> balanced processed dataset
  -> MobileNetV2 training and evaluation
  -> saved PyTorch checkpoint
  -> FastAPI inference service
  -> Streamlit demo UI
  -> SQLite prediction history
```

Runtime services:

```text
User browser
  -> Streamlit frontend :8501
  -> FastAPI backend :8000
  -> PyTorch model checkpoint
  -> SQLite database in logs/predictions.db
```

## Tech Stack

- Python
- PyTorch, torchvision
- scikit-learn, matplotlib
- FastAPI, Uvicorn
- Streamlit
- SQLite
- Docker, Docker Compose

## Repository Structure

```text
image-classification-project/
  app/
    backend/          FastAPI inference service
    frontend/         Streamlit UI
  data/
    raw/              Original dataset files, ignored by git
    processed/        Processed dataset files, ignored by git
  docs/               Project, dataset, API, frontend, and deployment docs
  logs/               Runtime logs and SQLite database, ignored by git
  models/             Trained model checkpoint, ignored by git
  reports/            Metrics and generated reports
  src/
    data/             Dataset analysis and preparation scripts
    training/         Training, model, metrics, and evaluation code
    inference/        Model loading and prediction logic
  tests/              Test directory
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Local Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Prepare the processed dataset under:

```text
data/processed/garbage_classification/
```

Train the baseline model:

```powershell
python -m src.training.train --epochs 5 --batch-size 16
```

Evaluate the saved checkpoint:

```powershell
python -m src.training.evaluate
```

Start the FastAPI backend:

```powershell
uvicorn app.backend.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Start the Streamlit frontend in another terminal:

```powershell
streamlit run app/frontend/streamlit_app.py
```

Open the demo UI:

```text
http://localhost:8501
```

The frontend calls `http://127.0.0.1:8000` by default. To use another backend:

```powershell
$env:BACKEND_URL="http://127.0.0.1:8000"
streamlit run app/frontend/streamlit_app.py
```

## Docker Run

The trained checkpoint must exist locally:

```text
models/best_mobilenet_v2.pth
```

Start backend and frontend:

```powershell
docker compose up --build
```

Open:

```text
Frontend: http://localhost:8501
API docs: http://localhost:8000/docs
```

Stop services:

```powershell
docker compose down
```

Docker Compose mounts local runtime folders instead of baking artifacts into the image:

```text
./models   -> /app/models
./logs     -> /app/logs
./uploads  -> /app/uploads
```

See [docs/deployment.md](docs/deployment.md) for deployment notes and verification checklist.

## API Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Service status, device, and loaded classes |
| GET | `/classes` | Class label list |
| POST | `/predict` | Upload image and return Top-K predictions |
| GET | `/history` | Recent prediction records |

Example prediction request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=3" `
  -F "file=@path\to\image.jpg"
```

See [docs/backend_api.md](docs/backend_api.md) for API details.

## Frontend

The Streamlit frontend supports:

- image upload and preview
- Top-K selection from 1 to 5
- prediction class and probability display
- inference latency display
- backend health status
- recent prediction history

See [docs/frontend_streamlit.md](docs/frontend_streamlit.md) for frontend usage.

## Documentation

- [docs/project_plan.md](docs/project_plan.md): scope and project positioning
- [docs/dataset_report.md](docs/dataset_report.md): dataset analysis and split strategy
- [docs/training_baseline.md](docs/training_baseline.md): training workflow
- [reports/model_report.md](reports/model_report.md): model evaluation report
- [docs/backend_api.md](docs/backend_api.md): FastAPI service
- [docs/frontend_streamlit.md](docs/frontend_streamlit.md): Streamlit frontend
- [docs/deployment.md](docs/deployment.md): Docker Compose deployment
- [docs/job_application_materials.md](docs/job_application_materials.md): resume bullets and interview preparation

## Notes

The dataset files, model checkpoint, uploaded images, and SQLite runtime database are intentionally not committed to Git. They are local artifacts generated or mounted during training and deployment.

This project does not claim production-grade high-concurrency serving, large-scale MLOps automation, cloud deployment, or model monitoring. Those are possible future extensions, but the completed scope is a compact and runnable ML engineering baseline.
