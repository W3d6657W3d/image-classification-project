# Streamlit Frontend

## Goal

The frontend provides a lightweight browser-based demo for the garbage
classification inference service. It focuses on low-risk delivery and clear
machine learning engineering presentation instead of complex frontend
customization.

## Features

- image upload
- image preview
- configurable Top-K prediction
- FastAPI `/predict` integration
- Top-K prediction table
- inference latency display
- recent prediction history from `/history`
- backend health status display

## Local Run

Start the FastAPI backend first:

```powershell
uvicorn app.backend.main:app --reload
```

Start the Streamlit frontend in another terminal:

```powershell
streamlit run app/frontend/streamlit_app.py
```

Open the app:

```text
http://localhost:8501
```

## Configuration

By default, the frontend calls:

```text
http://127.0.0.1:8000
```

Use `BACKEND_URL` to point the frontend to another backend:

```powershell
$env:BACKEND_URL="http://127.0.0.1:8000"
streamlit run app/frontend/streamlit_app.py
```

## Expected Workflow

1. Upload a garbage image.
2. Preview the uploaded image.
3. Select Top-K from 1 to 5.
4. Click `Run Prediction`.
5. Review the predicted classes, probabilities, and inference time.
6. Check recent prediction records in the history table.

## Notes

The frontend does not load the PyTorch model directly. Model inference,
uploaded file persistence, and SQLite prediction history are handled by the
FastAPI backend.
