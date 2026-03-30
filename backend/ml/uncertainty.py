"""Uncertainty quantification for AlloyGen 2.0.

Provides:
- Ensemble uncertainty: train N models with different seeds, report mean ± std
- Gaussian Process regression via sklearn (lightweight, no GPyTorch dependency for MVP)
- Every prediction returns (value, lower_bound, upper_bound, confidence)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class PredictionWithUncertainty:
    """A prediction with confidence bounds."""
    value: float
    lower: float
    upper: float
    std: float
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "std": round(self.std, 4),
            "confidence": self.confidence,
        }


def _confidence_level(std: float, value_range: float) -> str:
    """Classify confidence based on std relative to the training range."""
    if value_range <= 0:
        return "low"
    ratio = std / value_range
    if ratio < 0.05:
        return "high"
    elif ratio < 0.15:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Ensemble uncertainty estimator
# ---------------------------------------------------------------------------
class EnsembleUncertaintyEstimator(BaseEstimator, RegressorMixin):
    """Train an ensemble of diverse models and use prediction variance as uncertainty."""

    def __init__(self, n_estimators: int = 5):
        self.n_estimators = n_estimators
        self.models_ = []
        self.scaler_ = StandardScaler()
        self.y_range_ = 1.0

    def fit(self, X, y):
        X_scaled = self.scaler_.fit_transform(X)
        y_arr = np.asarray(y).ravel()
        self.y_range_ = float(np.ptp(y_arr)) if len(y_arr) > 1 else 1.0
        self.models_ = []

        model_classes = [
            lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            lambda seed: ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
            lambda seed: GradientBoostingRegressor(n_estimators=100, random_state=seed),
        ]

        for i in range(self.n_estimators):
            cls = model_classes[i % len(model_classes)]
            model = cls(seed=42 + i)
            model.fit(X_scaled, y_arr)
            self.models_.append(model)

        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models_])
        return np.mean(preds, axis=0)

    def predict_with_uncertainty(self, X) -> list[PredictionWithUncertainty]:
        X_scaled = self.scaler_.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.models_])
        means = np.mean(preds, axis=0)
        stds = np.std(preds, axis=0)

        results = []
        for mean, std in zip(means, stds):
            confidence = _confidence_level(std, self.y_range_)
            results.append(PredictionWithUncertainty(
                value=float(mean),
                lower=float(mean - 2 * std),
                upper=float(mean + 2 * std),
                std=float(std),
                confidence=confidence,
            ))
        return results

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


# ---------------------------------------------------------------------------
# Gaussian Process regressor wrapper
# ---------------------------------------------------------------------------
class GPRegressorWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible GP regressor with uncertainty output.

    Uses Matern 5/2 kernel — standard for materials science.
    Includes automatic scaling and noise estimation.
    """

    def __init__(self, alpha: float = 1e-7, n_restarts: int = 3):
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()
        self.gp_ = None
        self.y_range_ = 1.0

    def fit(self, X, y):
        X_scaled = self.scaler_X_.fit_transform(X)
        y_arr = np.asarray(y).ravel().reshape(-1, 1)
        self.y_range_ = float(np.ptp(y_arr)) if len(y_arr) > 1 else 1.0
        y_scaled = self.scaler_y_.fit_transform(y_arr).ravel()

        kernel = ConstantKernel(1.0) * Matern(
            length_scale=1.0, nu=2.5
        ) + WhiteKernel(noise_level=0.1)

        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
        )
        self.gp_.fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        X_scaled = self.scaler_X_.transform(X)
        y_scaled = self.gp_.predict(X_scaled)
        return self.scaler_y_.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def predict_with_uncertainty(self, X) -> list[PredictionWithUncertainty]:
        X_scaled = self.scaler_X_.transform(X)
        y_scaled_mean, y_scaled_std = self.gp_.predict(X_scaled, return_std=True)

        # Inverse transform mean
        y_mean = self.scaler_y_.inverse_transform(y_scaled_mean.reshape(-1, 1)).ravel()
        # Scale std back (std transforms by scale factor only, not shift)
        y_std = y_scaled_std * self.scaler_y_.scale_[0]

        results = []
        for mean, std in zip(y_mean, y_std):
            confidence = _confidence_level(std, self.y_range_)
            results.append(PredictionWithUncertainty(
                value=float(mean),
                lower=float(mean - 2 * std),
                upper=float(mean + 2 * std),
                std=float(std),
                confidence=confidence,
            ))
        return results

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

    @property
    def feature_importances_(self):
        """Approximate feature importance from GP length scales."""
        if self.gp_ is None or self.gp_.kernel_ is None:
            return None
        try:
            # Get length scales from the Matern kernel
            params = self.gp_.kernel_.get_params()
            for key, val in params.items():
                if 'length_scale' in key and hasattr(val, '__len__'):
                    # Inverse of length scale = importance (shorter = more important)
                    return 1.0 / np.array(val)
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_uncertainty_to_prediction(
    model,
    X: np.ndarray | pd.DataFrame,
    target_stats: Optional[dict] = None,
) -> list[PredictionWithUncertainty]:
    """Get predictions with uncertainty from any model that supports it.

    Falls back to point prediction with no uncertainty bounds.
    """
    if hasattr(model, "predict_with_uncertainty"):
        return model.predict_with_uncertainty(X)

    # Fallback: just predict without uncertainty
    preds = model.predict(X)
    if preds.ndim == 1:
        preds = preds.reshape(-1)
    y_range = 1.0
    if target_stats:
        y_range = target_stats.get("range", 1.0)

    return [
        PredictionWithUncertainty(
            value=float(p), lower=float(p), upper=float(p),
            std=0.0, confidence="unknown",
        )
        for p in preds
    ]
