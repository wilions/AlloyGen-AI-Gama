from __future__ import annotations
import glob as globmod
import json
import logging
import os
import re
import time

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.agents import (
    RequirementsAgent,
    DataPrepAgent,
    OnlineSearchAgent,
    ModelSearchAgent,
    TrainingAgent,
    PredictionAgent,
    InverseDesignAgent,
)
from backend.pipeline import Session, send_ws, run_pipeline
from backend.config import SESSION_TTL_SECONDS, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AlloyGen AI (Gama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5178", "http://127.0.0.1:5178", "http://localhost:5176", "http://127.0.0.1:5176"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, Session] = {}

GREETING = (
    "Hello! I am your Metallurgic & Mechanical AI expert. "
    "To get started, please upload your dataset and I'll help you explore it!"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitize_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", raw)[:64]


def _sanitize_filename(raw: str) -> str:
    name = os.path.basename(raw)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)[:128]


def _purge_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in expired:
        session = sessions.pop(sid, None)
        if session:
            for path in (session.data_path, session.clean_data_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            logger.info("Purged expired session %s", sid)


def _touch(session: Session) -> None:
    session.last_active = time.time()


_INVERSE_KEYWORDS = [
    "design", "find a composition", "find an alloy", "find a material",
    "suggest a composition", "suggest an alloy", "what composition",
    "what alloy", "optimize", "inverse", "achieve", "reach", "get a",
    "need an alloy", "need a material", "want an alloy", "want a material",
    "target of", "target value", "desired", "i want", "i need",
    "how to get", "how to achieve", "how can i get",
]


def _is_inverse_design_request(msg: str) -> bool:
    """Detect if the user is asking for inverse design (desired property → composition)
    vs forward prediction (composition → property)."""
    lower = msg.lower()
    return any(kw in lower for kw in _INVERSE_KEYWORDS)


def _get_or_create_session(session_id: str) -> Session:
    if session_id not in sessions:
        sessions[session_id] = Session()
    _touch(sessions[session_id])
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    session_id = _sanitize_id(session_id)
    if not session_id:
        raise HTTPException(400, "Invalid session ID")

    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_UPLOAD_BYTES // 1024 // 1024} MB")

    _purge_expired_sessions()
    session = _get_or_create_session(session_id)

    os.makedirs("uploads", exist_ok=True)
    safe_name = _sanitize_filename(file.filename or "upload.csv")
    file_path = f"uploads/{session_id}_{safe_name}"
    with open(file_path, "wb") as f:
        f.write(content)

    session.data_path = file_path

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            df = None

        if df is not None:
            cols_info = []
            for col in df.columns:
                non_null = int(df[col].notna().sum())
                if pd.api.types.is_numeric_dtype(df[col]):
                    cols_info.append(
                        f"  - **{col}** (numeric, {non_null} values, "
                        f"range: {df[col].min():.4g} - {df[col].max():.4g})"
                    )
                else:
                    n_unique = df[col].nunique()
                    cols_info.append(
                        f"  - **{col}** (categorical, {non_null} values, {n_unique} unique)"
                    )

            summary = (
                f"**Dataset loaded:** `{safe_name}`\n"
                f"**Shape:** {df.shape[0]} rows x {df.shape[1]} columns\n\n"
                f"**Columns:**\n" + "\n".join(cols_info)
            )
            session.data_summary = summary
        else:
            session.data_summary = f"File `{safe_name}` uploaded (unsupported format for preview)."
    except Exception as e:
        session.data_summary = f"File `{safe_name}` uploaded (could not preview: {e})."

    return {"status": "success", "filename": safe_name, "message": "File uploaded successfully."}


# ---------------------------------------------------------------------------
# Batch prediction upload
# ---------------------------------------------------------------------------
@app.post("/batch-predict")
async def batch_predict(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a CSV for batch predictions using the trained model."""
    session_id = _sanitize_id(session_id)
    if not session_id or session_id not in sessions:
        raise HTTPException(400, "Invalid or unknown session")

    session = sessions[session_id]
    if not session.model_path or not os.path.exists(session.model_path):
        raise HTTPException(400, "No trained model available. Run the pipeline first.")

    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() != ".csv":
        raise HTTPException(400, "Batch prediction only supports CSV files.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")

    os.makedirs("uploads", exist_ok=True)
    safe_name = _sanitize_filename(file.filename or "batch.csv")
    csv_path = f"uploads/{session_id}_batch_{safe_name}"
    with open(csv_path, "wb") as f:
        f.write(content)

    prediction_agent = PredictionAgent()
    summary, output_path = await prediction_agent.batch_predict(session.model_path, csv_path)

    # Clean up input
    try:
        os.remove(csv_path)
    except OSError:
        pass

    if output_path and os.path.exists(output_path):
        return {"status": "success", "message": summary, "download_path": output_path}
    return {"status": "error", "message": summary}


@app.get("/download/{filepath:path}")
async def download_file(filepath: str):
    """Download a prediction results file."""
    # Only allow files from uploads/ or models/ directories
    if not (filepath.startswith("uploads/") or filepath.startswith("models/")):
        raise HTTPException(403, "Access denied")
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")
    return FileResponse(filepath, filename=os.path.basename(filepath))


# ---------------------------------------------------------------------------
# Model registry endpoint
# ---------------------------------------------------------------------------
@app.get("/models")
async def list_models():
    """List all saved models with metadata."""
    model_files = globmod.glob("models/model_*.joblib")
    models = []
    for path in model_files:
        try:
            data = joblib.load(path)
            models.append({
                "path": path,
                "filename": os.path.basename(path),
                "best_model_name": data.get("best_model_name", "Unknown"),
                "targets": data.get("targets", []),
                "task_type": data.get("task_type", "unknown"),
                "score": round(data.get("score") or 0, 4),
                "cv_score": round(data.get("cv_score") or 0, 4),
                "timestamp": data.get("timestamp", 0),
                "features": data.get("features", []),
                "feature_count": len(data.get("features", [])),
            })
        except Exception as e:
            logger.warning("Could not load model metadata from %s: %s", path, e)
    # Sort by score descending (best first)
    models.sort(key=lambda m: m["score"], reverse=True)
    return {"models": models}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session_id = _sanitize_id(session_id)
    await websocket.accept()

    session = _get_or_create_session(session_id)

    agents = {
        "requirements": RequirementsAgent(),
        "prep": DataPrepAgent(),
        "online_search": OnlineSearchAgent(),
        "model_search": ModelSearchAgent(),
        "train": TrainingAgent(),
        "prediction": PredictionAgent(),
        "inverse_design": InverseDesignAgent(),
    }

    if not session.history:
        session.history.append({"role": "assistant", "content": GREETING})

    # Always send greeting + current state on new WS connection so frontend can sync
    await send_ws(websocket, "message", GREETING, session.state)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_msg = payload.get("message", "")
            msg_type = payload.get("type", "message")
            _touch(session)

            # Handle cancel request
            if msg_type == "cancel":
                session.request_cancel()
                continue

            # Handle model selection
            if msg_type == "select_model":
                model_path = payload.get("model_path", "")
                if not model_path or not os.path.exists(model_path):
                    await send_ws(websocket, "message", "Model not found.", session.state)
                    continue
                try:
                    data = joblib.load(model_path)
                    session.model_path = model_path
                    session.targets = data.get("targets", [])
                    session.task_types = [data.get("task_type", "regression")]
                    session.state = "prediction"
                    targets_display = ", ".join(f"**{t}**" for t in session.targets)
                    model_name = data.get("best_model_name", "Unknown")
                    score = data.get("score", 0)
                    msg = (
                        f"Loaded saved model: **{model_name}** (score: {score:.4f})\n"
                        f"Targets: {targets_display}\n\n"
                        "You can now:\n"
                        "1. **Forward prediction** — provide alloy composition to predict properties\n"
                        "2. **Inverse design** — describe desired properties to find optimal compositions\n"
                        "3. **Batch prediction** — upload a CSV for bulk predictions"
                    )
                    session.history.append({"role": "assistant", "content": msg})
                    await send_ws(websocket, "message", msg, "prediction")
                except Exception as e:
                    await send_ws(websocket, "message", f"Failed to load model: {e}", session.state)
                continue

            # Handle reset request
            if msg_type == "reset":
                # Clean up upload files but keep trained models
                for path in (session.data_path, session.clean_data_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                sessions[session_id] = Session()
                session = sessions[session_id]
                session.history.append({"role": "assistant", "content": GREETING})
                await send_ws(websocket, "message", GREETING, session.state)
                continue

            if user_msg:
                session.history.append({"role": "user", "content": user_msg})

            if session.state == "requirements":
                if user_msg.startswith("[SYSTEM]:"):
                    if session.data_summary:
                        summary_msg = (
                            f"{session.data_summary}\n\n"
                            "Which column(s)/property(ies) would you like to predict? "
                            "You can specify **multiple targets** (e.g. 'tensile strength, elongation, and hardness') "
                            "and they'll all be predicted simultaneously by a single multi-output model.\n\n"
                            "Also, is this a **regression** (predicting a number) or "
                            "**classification** (predicting a category) task?"
                        )
                    else:
                        summary_msg = (
                            "Your file has been uploaded. "
                            "Which property(ies) would you like to predict?"
                        )
                    session.awaiting_target = True
                    session.history.append({"role": "assistant", "content": summary_msg})
                    await send_ws(websocket, "message", summary_msg, session.state)

                elif session.awaiting_target:
                    response, targets, task_types, _, ready = await agents[
                        "requirements"
                    ].process(session.history, data_summary=session.data_summary)
                    session.history.append({"role": "assistant", "content": response})
                    await send_ws(websocket, "message", response, session.state)

                    if ready and targets:
                        session.targets = targets
                        session.task_types = task_types
                        session.has_dataset = True
                        session.ready_for_pipeline = True
                        session.awaiting_target = False

                else:
                    response, targets, task_types, has_dataset, ready = await agents[
                        "requirements"
                    ].process(session.history)
                    session.history.append({"role": "assistant", "content": response})
                    await send_ws(websocket, "message", response, session.state)

                # Trigger pipeline when ready
                if session.ready_for_pipeline and session.data_path:
                    await run_pipeline(session, websocket, agents)

            elif session.state == "prediction":
                # Handle batch prediction signal from file upload in prediction state
                if user_msg.startswith("[SYSTEM]:BATCH:"):
                    batch_msg = "Processing batch predictions..."
                    await send_ws(websocket, "status", batch_msg, session.state)
                    summary, output_path = await agents["prediction"].batch_predict(
                        session.model_path, session.data_path
                    )
                    await send_ws(websocket, "message", summary, session.state)

                elif session.pending_inverse_constraints is not None:
                    # Step 2 of inverse design: user is providing element selection
                    await send_ws(websocket, "status", "Running inverse design optimization...", session.state)
                    data_path = session.clean_data_path or session.data_path
                    response = await agents["inverse_design"].parse_elements_and_run(
                        user_msg, session.model_path, data_path,
                        session.pending_inverse_constraints,
                        session.pending_inverse_fixed or {},
                    )
                    session.pending_inverse_constraints = None
                    session.pending_inverse_fixed = None
                    session.history.append({"role": "assistant", "content": response})
                    await send_ws(websocket, "message", response, session.state)

                elif _is_inverse_design_request(user_msg):
                    # Step 1 of inverse design: extract constraints and ask for elements
                    data_path = session.clean_data_path or session.data_path
                    response, constraints, fixed = await agents["inverse_design"].extract_constraints(
                        user_msg, session.model_path, data_path
                    )
                    if constraints:
                        # Store constraints and wait for element selection
                        session.pending_inverse_constraints = constraints
                        session.pending_inverse_fixed = fixed
                    session.history.append({"role": "assistant", "content": response})
                    await send_ws(websocket, "message", response, session.state)

                else:
                    # Forward prediction: user provides composition, get property
                    response = await agents["prediction"].process(
                        user_msg, session.model_path, session.clean_data_path or session.data_path
                    )
                    session.history.append({"role": "assistant", "content": response})
                    await send_ws(websocket, "message", response, session.state)

            else:
                await send_ws(
                    websocket, "message",
                    f"Processing in background... ({session.state})",
                    session.state,
                )

    except WebSocketDisconnect:
        logger.info("Client %s disconnected", session_id)
