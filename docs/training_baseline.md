# Training Baseline

## Goal

The first model should prove the full machine learning loop:

```text
processed dataset -> train model -> validate model -> save checkpoint
-> evaluate on test set -> load checkpoint for single-image prediction
```

The baseline uses MobileNetV2 because it is lightweight enough for local CPU
training and practical for later FastAPI deployment.

## Dataset

Input directory:

```text
data/processed/garbage_classification/
```

Split summary:

```text
train: 5040 images
validation: 1080 images
test: 1080 images
classes: 12
```

## Recommended First Run

CPU-only machines should start small:

```powershell
python -m src.training.train --epochs 5 --batch-size 16
```

If memory is tight:

```powershell
python -m src.training.train --epochs 5 --batch-size 8
```

After training:

```powershell
python -m src.training.evaluate
```

Single-image prediction:

```powershell
python -m src.inference.predict path\to\image.jpg --top-k 3
```

## Outputs

Model checkpoint:

```text
models/best_mobilenet_v2.pth
```

Training and evaluation artifacts:

```text
reports/training_history.json
reports/model_report.md
reports/test_metrics.csv
reports/figures/confusion_matrix.png
```

Model files and figures are ignored by Git. Reports and scripts are tracked.
