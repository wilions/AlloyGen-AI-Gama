"""Multi-objective inverse design optimization for AlloyGen 2.0.

Upgrades from Gama's Optuna approach:
- Reparameterized composition: optimize N-1 free variables, Nth = remainder
- Multivariate TPE for correlated composition variables
- Increased trials (1000 default)
- Returns full Pareto front for interactive visualization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


@dataclass
class InverseDesignResult:
    """Result from inverse design optimization."""
    candidates: list[dict]
    pareto_front: list[dict]
    n_trials: int
    best_score: float

    def to_dict(self) -> dict:
        return {
            "candidates": self.candidates[:20],  # top 20
            "pareto_front": self.pareto_front,
            "n_trials": self.n_trials,
            "best_score": round(self.best_score, 4),
        }


def _is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal points (minimize all objectives).

    Args:
        costs: (n_points, n_objectives) array where lower is better
    Returns:
        boolean mask of Pareto-optimal points
    """
    is_optimal = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_optimal[i]:
            # A point is dominated if another point is <= in all objectives and < in at least one
            is_optimal[is_optimal] = np.any(costs[is_optimal] < c, axis=1) | np.all(costs[is_optimal] == c, axis=1)
            is_optimal[i] = True
    return is_optimal


def run_multi_objective_optimization(
    model,
    feature_names: list[str],
    element_ranges: dict[str, tuple[float, float]],
    objectives: dict[str, str],  # target_name -> "maximize" | "minimize"
    fixed_features: Optional[dict[str, float]] = None,
    n_trials: int = 1000,
    feature_stats: Optional[dict] = None,
) -> InverseDesignResult:
    """Run multi-objective inverse design with composition constraints.

    Key improvements over Gama:
    - Reparameterized composition (N-1 free vars, Nth as remainder)
    - Multivariate TPE for correlated variables
    - Full Pareto front output

    Args:
        model: trained sklearn model
        feature_names: list of feature names the model expects
        element_ranges: {element: (min%, max%)} for composition variables
        objectives: {target_name: "maximize" or "minimize"}
        fixed_features: {feature_name: value} for non-element features
        n_trials: number of optimization trials
        feature_stats: {feature: {min, max, median}} for filling non-element features
    """
    fixed_features = fixed_features or {}
    feature_stats = feature_stats or {}

    elements = list(element_ranges.keys())
    if not elements:
        raise ValueError("No elements specified for optimization")

    # Identify which elements are in the model features
    element_feature_map = {}
    for elem in elements:
        for feat in feature_names:
            if feat == elem or feat.startswith(elem + "_"):
                element_feature_map[elem] = feat
                break

    # Reparameterization: optimize N-1 elements, last = 100 - sum(others)
    free_elements = elements[:-1]
    remainder_element = elements[-1]
    remainder_min, remainder_max = element_ranges[remainder_element]

    all_candidates = []

    def objective(trial: optuna.Trial) -> tuple:
        # Sample N-1 free composition variables
        composition = {}
        running_sum = 0.0

        for elem in free_elements:
            lo, hi = element_ranges[elem]
            # Constrain upper bound to leave room for remaining elements
            max_possible = min(hi, 100.0 - running_sum - remainder_min)
            if max_possible < lo:
                return tuple(1e10 for _ in objectives)  # infeasible
            val = trial.suggest_float(elem, lo, min(hi, max_possible))
            composition[elem] = val
            running_sum += val

        # Last element = remainder
        remainder = 100.0 - running_sum
        if remainder < remainder_min or remainder > remainder_max:
            return tuple(1e10 for _ in objectives)  # infeasible
        composition[remainder_element] = remainder

        # Build feature vector
        features = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            if fname in fixed_features:
                features[i] = fixed_features[fname]
            elif fname in composition:
                features[i] = composition[fname]
            else:
                # Check element_feature_map
                for elem, mapped_feat in element_feature_map.items():
                    if mapped_feat == fname and elem in composition:
                        features[i] = composition[elem]
                        break
                else:
                    # Use median from training data
                    if fname in feature_stats:
                        features[i] = feature_stats[fname].get("median", 0)

        # Predict
        X = features.reshape(1, -1)
        try:
            preds = model.predict(X)
            if hasattr(preds, 'ravel'):
                preds = preds.ravel()
        except Exception:
            return tuple(1e10 for _ in objectives)

        # Build candidate record
        candidate = {**composition}
        scores = []
        for idx, (target, direction) in enumerate(objectives.items()):
            pred_val = float(preds[idx]) if len(preds) > idx else float(preds[0])
            candidate[target] = pred_val
            # For Optuna: minimize (negate for maximization)
            if direction == "maximize":
                scores.append(-pred_val)
            else:
                scores.append(pred_val)

        candidate["_scores"] = scores
        all_candidates.append(candidate)
        return tuple(scores)

    # Create multi-objective study with multivariate TPE
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=42,
    )

    n_objectives = len(objectives)
    directions = []
    for direction in objectives.values():
        directions.append("minimize")  # we handle max by negation

    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if not all_candidates:
        return InverseDesignResult([], [], n_trials, 0.0)

    # Extract Pareto front
    scores_array = np.array([c["_scores"] for c in all_candidates])
    pareto_mask = _is_pareto_optimal(scores_array)

    # Clean up candidates
    for c in all_candidates:
        del c["_scores"]

    pareto_candidates = [c for c, m in zip(all_candidates, pareto_mask) if m]

    # Sort all by first objective
    first_target = list(objectives.keys())[0]
    first_dir = list(objectives.values())[0]
    reverse = first_dir == "maximize"
    all_candidates.sort(key=lambda c: c.get(first_target, 0), reverse=reverse)

    best_score = all_candidates[0].get(first_target, 0) if all_candidates else 0.0

    return InverseDesignResult(
        candidates=all_candidates,
        pareto_front=pareto_candidates,
        n_trials=n_trials,
        best_score=best_score,
    )
