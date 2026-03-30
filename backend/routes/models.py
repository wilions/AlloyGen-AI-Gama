from __future__ import annotations

import glob as globmod
import logging
import os

import joblib
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
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
    models.sort(key=lambda m: m["score"], reverse=True)
    return {"models": models}
