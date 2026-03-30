from __future__ import annotations

import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from backend.agents import PredictionAgent
from backend.config import MAX_UPLOAD_BYTES
from backend.routes._helpers import sessions, sanitize_id, sanitize_filename

router = APIRouter()


@router.post("/batch-predict")
async def batch_predict(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a CSV for batch predictions using the trained model."""
    session_id = sanitize_id(session_id)
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
    safe_name = sanitize_filename(file.filename or "batch.csv")
    csv_path = f"uploads/{session_id}_batch_{safe_name}"
    with open(csv_path, "wb") as f:
        f.write(content)

    prediction_agent = PredictionAgent()
    summary, output_path = await prediction_agent.batch_predict(session.model_path, csv_path)

    try:
        os.remove(csv_path)
    except OSError:
        pass

    if output_path and os.path.exists(output_path):
        return {"status": "success", "message": summary, "download_path": output_path}
    return {"status": "error", "message": summary}


@router.get("/download/{filepath:path}")
async def download_file(filepath: str):
    """Download a prediction results file."""
    if not (filepath.startswith("uploads/") or filepath.startswith("models/")):
        raise HTTPException(403, "Access denied")
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")
    return FileResponse(filepath, filename=os.path.basename(filepath))
