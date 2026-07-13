"""Streamlit frontend for the garbage classification demo.

The frontend is intentionally thin: it collects an uploaded image, calls the
FastAPI backend, renders Top-K probabilities, and displays recent prediction
history from the backend API.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30


def get_backend_url() -> str:
    """Resolve backend URL from environment for local and Docker runs."""
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def request_json(method: str, url: str, **kwargs: Any) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    """Call the backend and normalize success/error handling for the UI."""
    try:
        response = requests.request(
            method,
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        return None, f"Could not connect to backend: {exc}"

    if response.ok:
        return response.json(), None

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"Backend returned {response.status_code}: {detail}"


def get_health(backend_url: str) -> dict[str, Any] | None:
    """Fetch backend health and render sidebar errors if unavailable."""
    data, error = request_json("GET", f"{backend_url}/health")
    if error:
        st.sidebar.error(error)
        return None
    if not isinstance(data, dict):
        st.sidebar.error("Unexpected health response from backend.")
        return None
    return data


def predict_image(
    backend_url: str,
    uploaded_file: Any,
    top_k: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send the uploaded image to the backend prediction endpoint."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return request_json("POST", f"{backend_url}/predict", params={"top_k": top_k}, files=files)


def get_history(backend_url: str, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch recent prediction history from the backend."""
    data, error = request_json("GET", f"{backend_url}/history", params={"limit": limit})
    if error:
        return [], error
    if not isinstance(data, list):
        return [], "Unexpected history response from backend."
    return data, None


def render_prediction_result(result: dict[str, Any]) -> None:
    """Render the top prediction, inference latency, and full Top-K table."""
    predictions = result.get("predictions", [])
    if not predictions:
        st.warning("No predictions returned.")
        return

    top_prediction = predictions[0]
    confidence = float(top_prediction["probability"])

    # Keep the highest-confidence class prominent, while the table preserves
    # the full Top-K distribution for inspection.
    left, right = st.columns([2, 1])
    with left:
        st.subheader(top_prediction["class_name"])
        st.progress(confidence, text=f"Top confidence: {confidence:.1%}")
    with right:
        st.metric("Inference time", f"{float(result['inference_ms']):.1f} ms")

    rows = [
        {
            "Rank": index,
            "Class": item["class_name"],
            "Probability": f"{float(item['probability']):.2%}",
        }
        for index, item in enumerate(predictions, start=1)
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_history(backend_url: str) -> None:
    """Render recent predictions stored by the backend service."""
    st.header("Recent Prediction History")
    history_limit = st.slider("History items", min_value=5, max_value=50, value=10, step=5)
    history, error = get_history(backend_url, history_limit)

    if error:
        st.warning(error)
        return
    if not history:
        st.info("No prediction history yet.")
        return

    rows = []
    for item in history:
        predictions = item.get("predictions") or []
        top_prediction = predictions[0] if predictions else {}
        rows.append(
            {
                "ID": item.get("id"),
                "Filename": item.get("filename"),
                "Top class": top_prediction.get("class_name", "-"),
                "Confidence": (
                    f"{float(top_prediction.get('probability', 0.0)):.2%}"
                    if top_prediction
                    else "-"
                ),
                "Top-K": item.get("top_k"),
                "Inference": f"{float(item.get('inference_ms', 0.0)):.1f} ms",
                "Created at": item.get("created_at"),
            }
        )

    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    """Build the Streamlit page and wire UI events to backend calls."""
    st.set_page_config(
        page_title="Garbage Classification Demo",
        layout="wide",
    )

    backend_url = get_backend_url()

    st.title("Garbage Classification Demo")
    st.caption("MobileNetV2 baseline inference with FastAPI, Streamlit, and SQLite history.")

    with st.sidebar:
        st.header("Backend")
        st.code(backend_url)
        health = get_health(backend_url)
        if health:
            st.success(f"Status: {health.get('status', 'unknown')}")
            st.write(f"Device: `{health.get('device', 'unknown')}`")
            st.write(f"Classes: {len(health.get('classes', []))}")

        top_k = st.slider("Top-K", min_value=1, max_value=5, value=3)

    upload_col, result_col = st.columns([1, 1])

    with upload_col:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a garbage image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded_file is not None:
            st.image(uploaded_file.getvalue(), caption=uploaded_file.name, use_column_width=True)

    with result_col:
        st.header("Prediction")
        if uploaded_file is None:
            st.info("Upload an image to run prediction.")
        else:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Calling inference API..."):
                    result, error = predict_image(backend_url, uploaded_file, top_k)

                if error:
                    st.error(error)
                elif isinstance(result, dict):
                    st.session_state["last_prediction"] = result
                    render_prediction_result(result)
                else:
                    st.error("Unexpected prediction response from backend.")
            elif (
                "last_prediction" in st.session_state
                and st.session_state["last_prediction"].get("filename") == uploaded_file.name
            ):
                # Preserve the last result across Streamlit reruns caused by
                # widget changes such as moving the Top-K slider.
                render_prediction_result(st.session_state["last_prediction"])

    st.divider()
    render_history(backend_url)


if __name__ == "__main__":
    main()
