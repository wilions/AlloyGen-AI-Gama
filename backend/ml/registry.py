"""Model registry and multi-output wrappers for AlloyGen 2.0.

Extracted from training_agent.py for modularity.
Adds correlation-adaptive multi-target strategy (RegressorChain for correlated targets).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier, RegressorChain, ClassifierChain
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, RidgeClassifier, Lasso, ElasticNet, BayesianRidge,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from backend.config import MLP_MAX_ITER, LOGISTIC_MAX_ITER, LASSO_MAX_ITER, ELASTICNET_MAX_ITER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-output support
# ---------------------------------------------------------------------------
NATIVE_MULTI_OUTPUT_REGRESSION = {
    "randomforestregressor", "extratreesregressor", "decisiontreeregressor",
    "kneighborsregressor",
}
NATIVE_MULTI_OUTPUT_CLASSIFICATION = {
    "randomforestclassifier", "extratreesclassifier", "decisiontreeclassifier",
    "kneighborsclassifier",
}

# Tree-based keys (for feature importance extraction)
TREE_BASED_KEYS = {
    "randomforestregressor", "randomforestclassifier",
    "extratreesregressor", "extratreesclassifier",
    "decisiontreeregressor", "decisiontreeclassifier",
    "gradientboostingregressor", "gradientboostingclassifier",
    "histgradientboostingregressor", "histgradientboostingclassifier",
    "adaboostregressor", "adaboostclassifier",
    "xgbregressor", "xgbclassifier",
    "lgbmregressor", "lgbmclassifier",
    "catboostregressor", "catboostclassifier",
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Ensemble — tree-based
    "randomforestregressor": lambda: RandomForestRegressor(random_state=42),
    "randomforestclassifier": lambda: RandomForestClassifier(random_state=42),
    "extratreesregressor": lambda: ExtraTreesRegressor(random_state=42),
    "extratreesclassifier": lambda: ExtraTreesClassifier(random_state=42),
    "decisiontreeregressor": lambda: DecisionTreeRegressor(random_state=42),
    "decisiontreeclassifier": lambda: DecisionTreeClassifier(random_state=42),
    "histgradientboostingregressor": lambda: HistGradientBoostingRegressor(random_state=42),
    "histgradientboostingclassifier": lambda: HistGradientBoostingClassifier(random_state=42),
    # Boosting
    "gradientboostingregressor": lambda: GradientBoostingRegressor(random_state=42),
    "gradientboostingclassifier": lambda: GradientBoostingClassifier(random_state=42),
    "adaboostregressor": lambda: AdaBoostRegressor(random_state=42),
    "adaboostclassifier": lambda: AdaBoostClassifier(random_state=42),
    "xgbregressor": lambda: XGBRegressor(random_state=42, verbosity=0),
    "xgbclassifier": lambda: XGBClassifier(random_state=42, verbosity=0),
    "lgbmregressor": lambda: LGBMRegressor(random_state=42, verbosity=-1),
    "lgbmclassifier": lambda: LGBMClassifier(random_state=42, verbosity=-1),
    "catboostregressor": lambda: CatBoostRegressor(random_state=42, verbose=0),
    "catboostclassifier": lambda: CatBoostClassifier(random_state=42, verbose=0),
    # Linear
    "linearregression": lambda: LinearRegression(),
    "logisticregression": lambda: LogisticRegression(max_iter=LOGISTIC_MAX_ITER),
    "ridgeregressor": lambda: Ridge(),
    "ridgeclassifier": lambda: RidgeClassifier(),
    "lassoregressor": lambda: Lasso(max_iter=LASSO_MAX_ITER),
    "elasticnetregressor": lambda: ElasticNet(max_iter=ELASTICNET_MAX_ITER),
    "bayesianridgeregressor": lambda: BayesianRidge(),
    "bayesianridgeclassifier": lambda: BayesianRidge(),
    # SVM (needs scaling)
    "svrregressor": lambda: Pipeline([("scaler", StandardScaler()), ("svr", SVR())]),
    "svcclassifier": lambda: Pipeline([("scaler", StandardScaler()), ("svc", SVC())]),
    # KNN (needs scaling)
    "kneighborsregressor": lambda: Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor())]),
    "kneighborsclassifier": lambda: Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
    # Neural network (needs scaling)
    "mlpregressor": lambda: Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(max_iter=MLP_MAX_ITER, random_state=42))]),
    "mlpclassifier": lambda: Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(max_iter=MLP_MAX_ITER, random_state=42))]),
}


# ---------------------------------------------------------------------------
# Correlation-adaptive multi-target wrapping
# ---------------------------------------------------------------------------
def compute_target_correlation(df: pd.DataFrame, targets: list[str]) -> float:
    """Compute mean absolute pairwise correlation among target columns.
    Returns 0.0 if single target or insufficient data.
    """
    if len(targets) < 2:
        return 0.0
    target_df = df[targets].dropna()
    if len(target_df) < 5:
        return 0.0
    corr = target_df.corr().abs()
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper_vals = corr.values[mask]
    return float(np.mean(upper_vals)) if len(upper_vals) > 0 else 0.0


def wrap_multi_output(
    model,
    key: str,
    is_classification: bool,
    num_targets: int,
    use_chain: bool = False,
) -> object:
    """Wrap a model for multi-output prediction.

    Args:
        model: sklearn-compatible estimator
        key: registry key (lowercase)
        is_classification: whether this is a classification task
        num_targets: number of target variables
        use_chain: if True, use RegressorChain/ClassifierChain (for correlated targets)
    """
    if num_targets <= 1:
        return model
    native_set = NATIVE_MULTI_OUTPUT_CLASSIFICATION if is_classification else NATIVE_MULTI_OUTPUT_REGRESSION
    if key in native_set:
        return model
    if use_chain:
        if is_classification:
            return ClassifierChain(model, random_state=42)
        return RegressorChain(model, random_state=42)
    if is_classification:
        return MultiOutputClassifier(model)
    return MultiOutputRegressor(model)


def extract_feature_importance(model, features: list[str], key: str) -> Optional[dict]:
    """Extract feature importance from tree-based models."""
    if key not in TREE_BASED_KEYS:
        return None
    try:
        inner = model
        if hasattr(model, "estimators_") and hasattr(model, "estimator"):
            # MultiOutput wrapper — average across estimators
            importances = np.mean(
                [est.feature_importances_ for est in model.estimators_
                 if hasattr(est, "feature_importances_")],
                axis=0,
            )
        elif hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        elif hasattr(inner, "named_steps"):
            # Pipeline — get the estimator step
            for step_name, step in inner.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    importances = step.feature_importances_
                    break
            else:
                return None
        else:
            return None

        # Return top 10 most important
        indices = np.argsort(importances)[::-1][:10]
        return {features[i]: round(float(importances[i]), 4) for i in indices if i < len(features)}
    except Exception as e:
        logger.debug("Could not extract feature importance: %s", e)
        return None
