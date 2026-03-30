from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
FLASH_MODEL = os.environ.get("FLASH_MODEL", "gemini-2.5-flash")
PRO_MODEL = os.environ.get("PRO_MODEL", "gemini-2.5-flash")

# --- Upload ---
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))  # 50 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# --- Session ---
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 3600))  # 1 hour

# --- Training ---
MIN_ROWS = int(os.environ.get("MIN_ROWS", 10))
MAX_NAN_RATIO = float(os.environ.get("MAX_NAN_RATIO", 0.5))
MODEL_TIMEOUT_SECONDS = int(os.environ.get("MODEL_TIMEOUT_SECONDS", 300))  # 5 min per model
CV_FOLDS = int(os.environ.get("CV_FOLDS", 5))
HYPERPARAMETER_ITER = int(os.environ.get("HYPERPARAMETER_ITER", 20))  # RandomizedSearchCV iterations
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.2))

# --- ML model defaults ---
MLP_MAX_ITER = int(os.environ.get("MLP_MAX_ITER", 1000))
LOGISTIC_MAX_ITER = int(os.environ.get("LOGISTIC_MAX_ITER", 1000))
LASSO_MAX_ITER = int(os.environ.get("LASSO_MAX_ITER", 5000))
ELASTICNET_MAX_ITER = int(os.environ.get("ELASTICNET_MAX_ITER", 5000))

if not GEMINI_API_KEY:
    import warnings
    warnings.warn(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key.",
        RuntimeWarning,
    )
