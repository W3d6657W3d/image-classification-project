"""Backend configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "best_mobilenet_v2.pth"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", PROJECT_ROOT / "logs" / "predictions.db"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", PROJECT_ROOT / "uploads"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "5"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
