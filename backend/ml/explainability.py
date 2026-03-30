"""SHAP-based model explainability for AlloyGen 2.0.

Computes SHAP values after training and for individual predictions.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.info("SHAP not installed — explainability features will be unavailable")


def compute_shap_values(
    model,
    X_train: pd.DataFrame | np.ndarray,
    feature_names: Optional[list[str]] = None,
    max_samples: int = 100,
) -> Optional[dict]:
    """Compute SHAP values for a trained model.

    Returns dict with:
    - mean_abs_shap: {feature: mean_abs_shap_value} (global importance)
    - base_value: expected model output
    """
    if not HAS_SHAP:
        logger.info("SHAP not available, skipping explainability")
        return None

    try:
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_train.columns)
            X_arr = X_train.values
        else:
            X_arr = np.asarray(X_train)

        # Subsample for speed
        if len(X_arr) > max_samples:
            indices = np.random.RandomState(42).choice(len(X_arr), max_samples, replace=False)
            X_sample = X_arr[indices]
        else:
            X_sample = X_arr

        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            # Try tree explainer (fast)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except Exception:
            # Fall back to KernelExplainer (slower but universal)
            background = shap.kmeans(X_sample, min(10, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=50)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)

        if feature_names and len(feature_names) == len(mean_abs):
            importance = {
                name: round(float(val), 6)
                for name, val in sorted(
                    zip(feature_names, mean_abs),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }
        else:
            importance = {
                f"feature_{i}": round(float(val), 6)
                for i, val in enumerate(sorted(mean_abs, reverse=True))
            }

        base_value = float(explainer.expected_value)
        if hasattr(explainer.expected_value, '__len__'):
            base_value = float(explainer.expected_value[0])

        return {
            "mean_abs_shap": importance,
            "base_value": round(base_value, 4),
        }

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


def explain_prediction(
    model,
    X_input: np.ndarray | pd.DataFrame,
    X_background: np.ndarray | pd.DataFrame,
    feature_names: Optional[list[str]] = None,
    max_background: int = 50,
) -> Optional[dict]:
    """Explain a single prediction using SHAP.

    Returns dict with:
    - contributions: {feature: shap_value} (positive = pushes up, negative = pushes down)
    - base_value: expected output without any features
    - predicted_value: model prediction for this input
    """
    if not HAS_SHAP:
        return None

    try:
        if isinstance(X_input, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_input.columns)
            X_in = X_input.values
        else:
            X_in = np.asarray(X_input)

        if isinstance(X_background, pd.DataFrame):
            X_bg = X_background.values
        else:
            X_bg = np.asarray(X_background)

        if X_in.ndim == 1:
            X_in = X_in.reshape(1, -1)

        # Subsample background
        if len(X_bg) > max_background:
            indices = np.random.RandomState(42).choice(len(X_bg), max_background, replace=False)
            X_bg = X_bg[indices]

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_in)
        except Exception:
            background = shap.kmeans(X_bg, min(10, len(X_bg)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_in, nsamples=50)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        values = shap_values[0] if shap_values.ndim > 1 else shap_values

        if feature_names and len(feature_names) == len(values):
            contributions = {
                name: round(float(val), 6)
                for name, val in sorted(
                    zip(feature_names, values),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:10]  # top 10
            }
        else:
            contributions = {
                f"feature_{i}": round(float(val), 6)
                for i, val in enumerate(values[:10])
            }

        base_value = float(explainer.expected_value)
        if hasattr(explainer.expected_value, '__len__'):
            base_value = float(explainer.expected_value[0])

        predicted = float(model.predict(X_in).ravel()[0])

        return {
            "contributions": contributions,
            "base_value": round(base_value, 4),
            "predicted_value": round(predicted, 4),
        }

    except Exception as e:
        logger.warning("SHAP prediction explanation failed: %s", e)
        return None
