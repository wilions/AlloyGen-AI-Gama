from __future__ import annotations
import asyncio
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import joblib
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class Session:
    history: list[dict] = field(default_factory=list)
    data_path: str = ""
    state: str = "requirements"
    model_path: Optional[str] = None
    targets: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    has_dataset: bool = True
    models_to_train: list[str] = field(default_factory=list)
    ready_for_pipeline: bool = False
    data_summary: Optional[str] = None
    awaiting_target: bool = False
    clean_data_path: Optional[str] = None
    last_active: float = field(default_factory=time.time)
    cancel_requested: bool = False
    pipeline_task: Optional[asyncio.Task] = field(default=None, repr=False)
    # Inverse design: pending constraints waiting for element selection
    pending_inverse_constraints: Optional[list[dict]] = None
    pending_inverse_fixed: Optional[dict] = None

    @property
    def target(self) -> Optional[str]:
        return self.targets[0] if self.targets else None

    @property
    def task_type(self) -> Optional[str]:
        return self.task_types[0] if self.task_types else None

    def reset_for_retry(self) -> None:
        self.state = "requirements"
        self.ready_for_pipeline = False
        self.awaiting_target = True
        self.targets = []
        self.task_types = []
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()


async def send_ws(
    websocket: WebSocket,
    msg_type: str,
    content: str,
    state: str,
    chart: Optional[dict] = None,
) -> None:
    try:
        payload: dict = {"type": msg_type, "content": content, "state": state}
        if chart:
            payload["chart"] = chart
        await websocket.send_json(payload)
    except Exception:
        logger.debug("Could not send WS message (client may have disconnected)")


async def run_pipeline(session: Session, websocket: WebSocket, agents: dict) -> None:
    """Run the full ML pipeline with per-model progress and cancel support."""
    try:
        await _run_pipeline_inner(session, websocket, agents)
    except Exception as e:
        logger.exception("Pipeline failed")
        session.reset_for_retry()
        error_msg = (
            f"The pipeline encountered an error: {e}\n\n"
            "Your dataset is still loaded. Please try again — "
            "specify which column(s) you'd like to predict."
        )
        session.history.append({"role": "assistant", "content": error_msg})
        await send_ws(websocket, "error", error_msg, session.state)


async def _run_pipeline_inner(session: Session, websocket: WebSocket, agents: dict) -> None:
    """Inner pipeline logic."""
    original_data_path = session.data_path
    targets = session.targets
    task_type = session.task_types[0]
    targets_display = ", ".join(f"**{t}**" for t in targets)

    session.cancel_requested = False
    await send_ws(websocket, "status", "pipeline_start", "data_prep")

    # --- Data Prep ---
    session.state = "data_prep"
    await send_ws(
        websocket, "status",
        f"Requirements gathered. Preparing data for {targets_display}...",
        "data_prep",
    )

    if session.cancel_requested:
        return await _handle_cancel(session, websocket)

    prep_res, clean_data_path = await agents["prep"].process(
        session.data_path, targets
    )

    if clean_data_path is None:
        session.reset_for_retry()
        await send_ws(
            websocket, "message",
            f"Warning: Data Preparation failed: {prep_res}\n\n"
            "Your dataset file is still loaded — you don't need to re-upload it. "
            "Please tell me which column from the list above you'd like to predict.",
            "requirements",
        )
        return

    session.clean_data_path = clean_data_path
    session.state = "online_search"
    await send_ws(websocket, "message", prep_res, "data_prep")

    if session.cancel_requested:
        return await _handle_cancel(session, websocket)

    # --- Online Search ---
    online_search_res = await agents["online_search"].process(
        ", ".join(targets), task_type
    )
    session.state = "model_search"
    await send_ws(websocket, "message", online_search_res, "online_search")

    if session.cancel_requested:
        return await _handle_cancel(session, websocket)

    # --- Model Search ---
    search_res, models = agents["model_search"].process(task_type)
    session.models_to_train = models
    session.state = "training"
    await send_ws(websocket, "message", search_res, "model_search")

    if session.cancel_requested:
        return await _handle_cancel(session, websocket)

    # --- Training with per-model progress ---
    async def on_training_progress(model_name: str, completed: int, total: int):
        if model_name == "done":
            await send_ws(
                websocket, "status",
                f"All {total} models trained. Selecting best...",
                "training",
            )
        else:
            await send_ws(
                websocket, "status",
                f"Training model {completed + 1}/{total}: **{model_name}**...",
                "training",
            )

    train_res, model_path = await agents["train"].process(
        clean_data_path, targets, task_type, models,
        on_progress=on_training_progress,
    )

    if session.cancel_requested:
        return await _handle_cancel(session, websocket)

    session.model_path = model_path
    session.state = "prediction"

    # Build chart data from training results
    training_chart = None
    if model_path and os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            all_results = model_data.get("all_results", [])
            if all_results:
                training_chart = {
                    "type": "model_comparison",
                    "data": [
                        {"name": r["name"], "score": r["score"], "cv_score": r.get("cv_score")}
                        for r in all_results if r.get("score") is not None
                    ],
                }
                # Also add feature importance chart if available
                fi = model_data.get("feature_importance")
                if fi:
                    training_chart = {
                        "type": "feature_importance",
                        "data": [{"name": k, "importance": v} for k, v in fi.items()],
                    }
                # Add SHAP data if available
                shap_data = model_data.get("shap_data")
                if shap_data and shap_data.get("mean_abs_shap"):
                    training_chart = {
                        "type": "shap_waterfall",
                        "data": [{"name": k, "value": v} for k, v in shap_data["mean_abs_shap"].items()],
                        "base_value": shap_data.get("base_value", 0),
                    }
        except Exception as e:
            logger.debug("Could not build training chart data: %s", e)

    await send_ws(websocket, "message", train_res, "training", chart=training_chart)

    capabilities = (
        "\n\nYou can now:\n"
        "1. **Forward prediction** — provide alloy composition to predict properties\n"
        "   _e.g., \"predict hardness for 0.4% C, 1.2% Mn, 18% Cr\"_\n"
        "2. **Inverse design** — describe desired properties to find optimal compositions\n"
        "   _e.g., \"design an alloy with hardness > 50 HRC\"_\n"
        "3. **Batch prediction** — upload a CSV for bulk predictions"
    )

    if len(targets) > 1:
        await send_ws(
            websocket, "message",
            f"Pipeline complete! I trained a multi-output model predicting {targets_display} simultaneously."
            f"{capabilities}",
            "prediction",
        )
    else:
        await send_ws(
            websocket, "message",
            f"Pipeline complete! The model is ready to predict {targets_display}."
            f"{capabilities}",
            "prediction",
        )

    _cleanup_uploads(original_data_path)


async def _handle_cancel(session: Session, websocket: WebSocket) -> None:
    """Handle pipeline cancellation."""
    session.reset_for_retry()
    await send_ws(
        websocket, "message",
        "Pipeline cancelled. You can start again by specifying which column(s) to predict.",
        "requirements",
    )


def _cleanup_uploads(original_data_path: Optional[str]) -> None:
    if not original_data_path:
        return
    try:
        if os.path.exists(original_data_path) and original_data_path.startswith("uploads/"):
            os.remove(original_data_path)
            logger.info("Cleaned up original upload: %s", original_data_path)
    except OSError as e:
        logger.warning("Could not delete %s: %s", original_data_path, e)
