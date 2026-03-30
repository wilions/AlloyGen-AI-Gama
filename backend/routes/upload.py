from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS
from backend.routes._helpers import sanitize_id, sanitize_filename, get_or_create_session, sessions, purge_expired_sessions

router = APIRouter()


@router.post("/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    session_id = sanitize_id(session_id)
    if not session_id:
        raise HTTPException(400, "Invalid session ID")

    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_UPLOAD_BYTES // 1024 // 1024} MB")

    purge_expired_sessions()
    session = get_or_create_session(session_id)

    os.makedirs("uploads", exist_ok=True)
    safe_name = sanitize_filename(file.filename or "upload.csv")
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


@router.get("/dataset-stats/{session_id}")
async def dataset_stats(session_id: str):
    """Return column statistics, correlation matrix, and distributions for the session's dataset."""
    session_id = sanitize_id(session_id)
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    data_path = session.clean_data_path or session.data_path
    if not data_path or not os.path.exists(data_path):
        raise HTTPException(404, "No dataset found for this session")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise HTTPException(400, f"Could not read dataset: {e}")

    # Numeric columns only
    numeric_df = df.select_dtypes(include=["number"])
    cols = list(numeric_df.columns)

    # Correlation matrix
    corr = numeric_df.corr().fillna(0)
    corr_matrix = corr.values.tolist()

    # Column statistics
    col_stats = {}
    for col in cols:
        c = numeric_df[col].dropna()
        col_stats[col] = {
            "count": int(c.count()),
            "mean": round(float(c.mean()), 4) if len(c) > 0 else 0,
            "std": round(float(c.std()), 4) if len(c) > 1 else 0,
            "min": round(float(c.min()), 4) if len(c) > 0 else 0,
            "max": round(float(c.max()), 4) if len(c) > 0 else 0,
        }

    # Distribution data (sample values for histograms)
    distributions = []
    for col in cols[:12]:  # limit to 12 columns
        values = numeric_df[col].dropna().tolist()
        if len(values) > 500:
            values = list(np.random.RandomState(42).choice(values, 500, replace=False))
        distributions.append({"column": col, "values": values})

    return {
        "columns": cols,
        "correlation": {"columns": cols, "matrix": corr_matrix},
        "stats": col_stats,
        "distributions": distributions,
        "shape": list(df.shape),
    }
