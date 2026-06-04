"""SQLite persistence for prediction history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(database_path: Path) -> Iterator[sqlite3.Connection]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(database_path: Path) -> None:
    with connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                predictions_json TEXT NOT NULL,
                inference_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def insert_prediction(
    database_path: Path,
    filename: str,
    stored_path: str,
    top_k: int,
    predictions: list[dict[str, float | str]],
    inference_ms: float,
) -> int:
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO prediction_history (
                filename,
                stored_path,
                top_k,
                predictions_json,
                inference_ms,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                stored_path,
                top_k,
                json.dumps(predictions, ensure_ascii=False),
                inference_ms,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)


def list_predictions(database_path: Path, limit: int = 20) -> list[dict[str, object]]:
    with connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, filename, stored_path, top_k, predictions_json, inference_ms, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "stored_path": row["stored_path"],
            "top_k": row["top_k"],
            "predictions": json.loads(row["predictions_json"]),
            "inference_ms": row["inference_ms"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
