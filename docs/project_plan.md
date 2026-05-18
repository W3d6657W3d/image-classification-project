# Project Plan

## Positioning

This project is a lightweight image classification training and deployment system.
It should show a complete machine learning engineering workflow instead of an
isolated notebook or a simple demo.

Core story:

> Build a practical image classification solution for small business scenarios,
> covering data preparation, model training, evaluation, inference service, and
> deployment.

## Scope

Must have:

- use a small public image classification dataset
- complete dataset split and basic exploratory analysis
- train a lightweight model with transfer learning
- report accuracy, precision, recall, F1, and confusion matrix
- provide a FastAPI inference API
- provide a simple frontend page
- save prediction history
- provide a Docker-based deployment path
- finish README, model report, and resume project description

Can be delayed:

- multi-user system
- complex permission management
- large-scale training
- cloud GPU training
- complex MLOps platform features

Out of scope:

- training a large CNN from scratch
- industrial high-concurrency service design
- complex UI animation
- OCR, object detection, or image segmentation

## Recommended Dataset Direction

Prefer a dataset with:

- moderate size for a low-end personal machine
- 5 to 20 classes
- clear image quality
- easy-to-explain business scenario
- good fit for transfer learning

Candidate directions:

- flower classification
- small food classification subset
- waste classification
- clothing classification
- animal or scene classification

## Recommended Model Direction

Preferred models:

- MobileNetV2
- ResNet18
- EfficientNet-B0

Start with one stable baseline first. Add model comparison only after the full
workflow is running.

## Delivery Materials

Final deliverables:

- runnable source code
- trained lightweight model file
- model evaluation figures
- FastAPI service
- frontend page
- SQLite prediction history
- Dockerfile or docker-compose
- README
- model evaluation report
- resume-ready project description

## Risk Control

Main risks:

- slow local training
- dataset download or cleanup taking too long
- deployment platform model size limits
- frontend and backend integration taking more time than expected

Control strategy:

- start with a small dataset and lightweight model
- build an end-to-end baseline first
- improve metrics and presentation after the baseline is stable
- prioritize runnable, deployable, and explainable results
