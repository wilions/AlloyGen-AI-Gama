"""Active learning / sequential experimental design for AlloyGen 2.0.

Provides a campaign-based active learning loop:
1. User defines element parameters, constraints, and objectives
2. System suggests experiments using acquisition functions
3. User submits measured results
4. System retrains and suggests next experiments

Uses sklearn's GP for surrogates — lightweight, no BayBE dependency for MVP.
BayBE can be added later for more advanced acquisition functions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class CampaignParameter:
    """A single parameter (element) with bounds."""
    name: str
    lower: float
    upper: float


@dataclass
class CampaignConfig:
    """Configuration for an active learning campaign."""
    parameters: list[CampaignParameter]
    objectives: list[str]  # target property names
    constraints: Optional[list[dict]] = None  # e.g., sum-to-100%
    batch_size: int = 5
    max_iterations: int = 20


@dataclass
class Iteration:
    """Record of a single active learning iteration."""
    number: int
    suggested: pd.DataFrame
    results: Optional[pd.DataFrame] = None
    best_value: Optional[float] = None
    model_score: Optional[float] = None


@dataclass
class Campaign:
    """An active learning campaign state."""
    config: CampaignConfig
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    iterations: list[Iteration] = field(default_factory=list)
    gp_model: Optional[GaussianProcessRegressor] = None
    scaler_X: Optional[StandardScaler] = None
    scaler_y: Optional[StandardScaler] = None
    status: str = "initialized"  # initialized, suggesting, waiting_results, completed

    @property
    def current_iteration(self) -> int:
        return len(self.iterations)

    @property
    def best_so_far(self) -> Optional[float]:
        if self.data.empty or not self.config.objectives:
            return None
        obj = self.config.objectives[0]
        if obj in self.data.columns:
            return float(self.data[obj].max())
        return None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "current_iteration": self.current_iteration,
            "max_iterations": self.config.max_iterations,
            "best_value": self.best_so_far,
            "n_data_points": len(self.data),
            "parameters": [{"name": p.name, "lower": p.lower, "upper": p.upper} for p in self.config.parameters],
            "objectives": self.config.objectives,
        }


class ActiveLearningEngine:
    """Manages active learning campaigns."""

    def create_campaign(
        self,
        parameters: list[dict],
        objectives: list[str],
        constraints: Optional[list[dict]] = None,
        existing_data: Optional[pd.DataFrame] = None,
        batch_size: int = 5,
        max_iterations: int = 20,
    ) -> Campaign:
        """Create a new active learning campaign."""
        params = [CampaignParameter(**p) for p in parameters]
        config = CampaignConfig(
            parameters=params,
            objectives=objectives,
            constraints=constraints,
            batch_size=batch_size,
            max_iterations=max_iterations,
        )
        campaign = Campaign(config=config)

        if existing_data is not None and not existing_data.empty:
            campaign.data = existing_data.copy()
            self._fit_surrogate(campaign)

        return campaign

    def suggest_experiments(self, campaign: Campaign) -> pd.DataFrame:
        """Suggest next batch of experiments using acquisition function."""
        param_names = [p.name for p in campaign.config.parameters]
        batch_size = campaign.config.batch_size

        if campaign.data.empty or len(campaign.data) < 3:
            # Cold start: use Latin Hypercube Sampling
            suggestions = self._lhs_sample(campaign.config.parameters, batch_size * 2)
        else:
            # Fit surrogate and use Expected Improvement
            self._fit_surrogate(campaign)
            suggestions = self._ei_suggest(campaign, n_candidates=batch_size)

        # Apply composition constraint (sum to 100%) if applicable
        has_sum_constraint = campaign.config.constraints and any(
            c.get("type") == "sum_to_100" for c in campaign.config.constraints
        )
        if has_sum_constraint:
            suggestions = self._normalize_compositions(suggestions, param_names)

        iteration = Iteration(number=campaign.current_iteration, suggested=suggestions)
        campaign.iterations.append(iteration)
        campaign.status = "waiting_results"

        return suggestions

    def submit_results(self, campaign: Campaign, results: pd.DataFrame) -> dict:
        """Submit experimental results and update the campaign."""
        if not campaign.iterations:
            raise ValueError("No pending suggestions to submit results for")

        latest = campaign.iterations[-1]
        latest.results = results

        # Merge into campaign data
        campaign.data = pd.concat([campaign.data, results], ignore_index=True)

        # Refit surrogate
        self._fit_surrogate(campaign)

        # Compute metrics
        obj = campaign.config.objectives[0] if campaign.config.objectives else None
        if obj and obj in campaign.data.columns:
            latest.best_value = float(campaign.data[obj].max())
            if campaign.gp_model:
                param_names = [p.name for p in campaign.config.parameters]
                X = campaign.data[param_names].values
                y = campaign.data[obj].values
                X_scaled = campaign.scaler_X.transform(X)
                latest.model_score = float(campaign.gp_model.score(X_scaled, campaign.scaler_y.transform(y.reshape(-1, 1)).ravel()))

        # Check stopping
        if campaign.current_iteration >= campaign.config.max_iterations:
            campaign.status = "completed"
        else:
            campaign.status = "suggesting"

        return {
            "iteration": latest.number,
            "best_value": latest.best_value,
            "model_score": latest.model_score,
            "n_data_points": len(campaign.data),
            "status": campaign.status,
            "convergence": self._check_convergence(campaign),
        }

    def _fit_surrogate(self, campaign: Campaign) -> None:
        """Fit a GP surrogate to the current data."""
        param_names = [p.name for p in campaign.config.parameters]
        obj = campaign.config.objectives[0] if campaign.config.objectives else None

        if obj is None or obj not in campaign.data.columns:
            return

        X = campaign.data[param_names].values
        y = campaign.data[obj].values.reshape(-1, 1)

        if len(X) < 2:
            return

        campaign.scaler_X = StandardScaler()
        campaign.scaler_y = StandardScaler()
        X_scaled = campaign.scaler_X.fit_transform(X)
        y_scaled = campaign.scaler_y.fit_transform(y).ravel()

        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
        campaign.gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, alpha=1e-6
        )
        campaign.gp_model.fit(X_scaled, y_scaled)

    def _ei_suggest(self, campaign: Campaign, n_candidates: int = 5) -> pd.DataFrame:
        """Generate suggestions using Expected Improvement acquisition function."""
        param_names = [p.name for p in campaign.config.parameters]
        n_random = 5000

        # Generate random candidates
        candidates = self._random_sample(campaign.config.parameters, n_random)
        X_cand = candidates[param_names].values
        X_cand_scaled = campaign.scaler_X.transform(X_cand)

        # Predict mean and std
        mu, sigma = campaign.gp_model.predict(X_cand_scaled, return_std=True)

        # Inverse transform to original scale
        mu_orig = campaign.scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()

        # Current best (in scaled space)
        obj = campaign.config.objectives[0]
        y_best_scaled = campaign.scaler_y.transform(
            np.array([[campaign.data[obj].max()]])
        ).ravel()[0]

        # Expected Improvement
        from scipy.stats import norm
        z = (mu - y_best_scaled) / (sigma + 1e-10)
        ei = (mu - y_best_scaled) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0

        # Select top candidates
        top_indices = np.argsort(ei)[-n_candidates:][::-1]
        suggestions = candidates.iloc[top_indices].copy()
        suggestions["acquisition_value"] = ei[top_indices]
        suggestions["predicted_value"] = mu_orig[top_indices]
        suggestions["uncertainty"] = sigma[top_indices] * campaign.scaler_y.scale_[0]

        return suggestions.reset_index(drop=True)

    def _lhs_sample(self, parameters: list[CampaignParameter], n: int) -> pd.DataFrame:
        """Latin Hypercube Sampling for initial experiments."""
        rng = np.random.RandomState(42)
        d = len(parameters)

        # LHS: divide each dimension into n intervals, sample one from each
        result = np.zeros((n, d))
        for i in range(d):
            perm = rng.permutation(n)
            result[:, i] = (perm + rng.uniform(size=n)) / n

        # Scale to parameter bounds
        data = {}
        for i, p in enumerate(parameters):
            data[p.name] = p.lower + result[:, i] * (p.upper - p.lower)

        return pd.DataFrame(data)

    def _random_sample(self, parameters: list[CampaignParameter], n: int) -> pd.DataFrame:
        """Generate random samples within parameter bounds."""
        rng = np.random.RandomState()
        data = {}
        for p in parameters:
            data[p.name] = rng.uniform(p.lower, p.upper, n)
        return pd.DataFrame(data)

    def _normalize_compositions(self, df: pd.DataFrame, param_names: list[str]) -> pd.DataFrame:
        """Normalize composition columns to sum to 100%."""
        result = df.copy()
        sums = result[param_names].sum(axis=1)
        for col in param_names:
            result[col] = result[col] / sums * 100.0
        return result

    def _check_convergence(self, campaign: Campaign, patience: int = 3) -> dict:
        """Check if the campaign has converged."""
        if len(campaign.iterations) < patience + 1:
            return {"converged": False, "reason": "Not enough iterations"}

        obj = campaign.config.objectives[0] if campaign.config.objectives else None
        if obj is None:
            return {"converged": False, "reason": "No objective"}

        # Check if best value hasn't improved in last `patience` iterations
        recent_bests = [
            it.best_value for it in campaign.iterations[-patience:]
            if it.best_value is not None
        ]
        if len(recent_bests) >= patience:
            improvement = max(recent_bests) - min(recent_bests)
            if improvement < 1e-4 * abs(max(recent_bests)):
                return {"converged": True, "reason": f"Best value unchanged for {patience} iterations"}

        return {"converged": False, "reason": "Still improving"}
