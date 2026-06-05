# Lightweight Image Classification Training and Deployment System

A lightweight machine learning engineering project for portfolio and job applications.

The goal is to finish a complete image classification workflow within two weeks:
dataset preparation, model training, model evaluation, inference API, simple user
interface, prediction history, Docker deployment, and project documentation.

Current phase: project initialization. The final project name can be refined later.

## Project Goal

This project is not only an image recognition demo. It is designed as a small but
complete machine learning engineering system.

It aims to demonstrate:

- dataset preparation, splitting, and exploratory analysis
- transfer learning with lightweight image classification models
- model evaluation with accuracy, precision, recall, F1, and confusion matrix
- model packaging as an inference service
- image upload, Top-K prediction, error handling, and prediction history
- reproducible deployment with Docker
- clear documentation for GitHub and resume presentation

## Target Roles

This project is designed to support applications for:

- junior machine learning engineer
- algorithm application engineer
- data science related roles
- AI solution engineer
- algorithm application roles in fintech or banking technology

## Tech Stack

Planned stack:

- Language: Python
- Model: PyTorch, torchvision, transfer learning
- API: FastAPI
- Frontend: Streamlit
- Database: SQLite
- Evaluation: scikit-learn, matplotlib
- Deployment: Docker

## Planned Features

- dataset preparation
- class distribution analysis
- sample visualization
- train, validation, and test split
- lightweight pretrained model fine-tuning
- baseline model comparison if time allows
- model metrics and confusion matrix
- image upload inference API
- Top-K prediction response
- prediction history
- simple usable frontend
- Docker-based startup

## Project Structure

```text
image-classification-project/
  app/
    backend/          FastAPI service
    frontend/         Streamlit UI
  data/
    raw/              Original dataset files, ignored by git
    processed/        Processed dataset files, ignored by git
  docs/
    project_plan.md   Scope, timeline, and delivery plan
    task_prompts.md   Prompts for future focused conversations
  logs/               Runtime logs, ignored by git
  models/             Trained model files, ignored by git
  notebooks/          Data exploration and experiments
  reports/
    figures/          Evaluation charts, ignored by git
  src/
    data/             Data loading and preprocessing
    training/         Training and evaluation pipeline
    inference/        Model loading and prediction logic
  tests/              Unit and integration tests
  uploads/            Uploaded images, ignored by git
  .env.example
  .gitignore
  LICENSE
  README.md
```

## Two-Week Roadmap

| Phase | Days | Deliverable |
| --- | --- | --- |
| Initialization | Day 1 | Project scope, structure, README, planning docs |
| Dataset | Days 2-3 | Dataset selection, folder structure, EDA, split plan |
| Training | Days 4-7 | Transfer learning training pipeline and saved best model |
| Evaluation | Days 7-8 | Metrics, confusion matrix, model report |
| Backend | Days 9-10 | FastAPI upload and prediction API |
| Frontend | Days 11-12 | Usable image upload and result display page |
| Deployment | Days 13-14 | Docker, deployment notes, final README, resume bullets |

## Current Status

- [x] GitHub repository connected
- [x] Initial project scope drafted
- [x] Directory structure planned
- [x] Dataset selected
- [x] Dataset distribution analyzed
- [x] Balanced train/validation/test split prepared
- [x] Training pipeline implemented
- [x] Model evaluated
- [x] API implemented
- [x] Frontend implemented
- [ ] Docker deployment completed

## Local Demo

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the FastAPI backend:

```powershell
uvicorn app.backend.main:app --reload
```

Start the Streamlit frontend in another terminal:

```powershell
streamlit run app/frontend/streamlit_app.py
```

The Streamlit app opens at:

```text
http://localhost:8501
```

The frontend expects the backend at `http://127.0.0.1:8000` by default. Set
`BACKEND_URL` to use a different backend address.

## Notes

Because the local machine has limited hardware, this project will prioritize small
public datasets, lightweight pretrained models, and transfer learning. The focus is
engineering completeness, reproducibility, deployment, and clear explanation rather
than large-scale training.
