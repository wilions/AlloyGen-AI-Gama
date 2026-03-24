from __future__ import annotations
import asyncio
import os
import logging
import time
from typing import Optional, Union, Callable, Awaitable

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    Lasso,
    ElasticNet,
    BayesianRidge,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from backend.agents.base_agent import BaseAgent
from backend.config import (
    MIN_ROWS, MAX_NAN_RATIO, MODEL_TIMEOUT_SECONDS, CV_FOLDS,
    TEST_SIZE, MLP_MAX_ITER, LOGISTIC_MAX_ITER, LASSO_MAX_ITER, ELASTICNET_MAX_ITER,
)
from backend.errors import DataValidationError, ModelTrainingError, TargetNotFoundError

logger = logging.getLogger(__name__)

MAX_SAVED_MODELS = 10


def _cleanup_old_models(model_dir: str, keep: int = MAX_SAVED_MODELS) -> None:
    """Delete worst-scoring model files, keeping only the top `keep` by score."""
    import glob as globmod
    files = globmod.glob(os.path.join(model_dir, "model_*.joblib"))
    if len(files) <= keep:
        return
    # Load score from each model file
    scored: list[tuple[str, float]] = []
    for f in files:
        try:
            data = joblib.load(f)
            scored.append((f, data.get("score", 0.0)))
        except Exception:
            scored.append((f, -1.0))  # broken files get lowest priority
    # Sort by score descending — keep the best
    scored.sort(key=lambda x: x[1], reverse=True)
    for path, score in scored[keep:]:
        try:
            os.remove(path)
            logger.info("Removed low-scoring model: %s (score=%.4f)", path, score)
        except OSError:
            pass


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

# ---------------------------------------------------------------------------
# Model registry — uses config values
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Ensemble — tree-based
    "randomforestregressor": lambda: RandomForestRegressor(random_state=42),
    "randomforestclassifier": lambda: RandomForestClassifier(random_state=42),
    "extratreesregressor": lambda: ExtraTreesRegressor(random_state=42),
    "extratreesclassifier": lambda: ExtraTreesClassifier(random_state=42),
    "decisiontreeregressor": lambda: DecisionTreeRegressor(random_state=42),
    "decisiontreeclassifier": lambda: DecisionTreeClassifier(random_state=42),
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

_TREE_BASED_KEYS = {
    "randomforestregressor", "randomforestclassifier",
    "extratreesregressor", "extratreesclassifier",
    "decisiontreeregressor", "decisiontreeclassifier",
    "gradientboostingregressor", "gradientboostingclassifier",
    "adaboostregressor", "adaboostclassifier",
    "xgbregressor", "xgbclassifier",
    "lgbmregressor", "lgbmclassifier",
    "catboostregressor", "catboostclassifier",
}

MIN_FEATURE_COLS = 1

# Type alias for progress callback
ProgressCallback = Optional[Callable[[str, int, int], Awaitable[None]]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap_multi_output(model, key: str, is_classification: bool, num_targets: int):
    """Wrap a model with MultiOutput if it doesn't natively support it."""
    if num_targets <= 1:
        return model
    native_set = NATIVE_MULTI_OUTPUT_CLASSIFICATION if is_classification else NATIVE_MULTI_OUTPUT_REGRESSION
    if key in native_set:
        return model
    if is_classification:
        return MultiOutputClassifier(model)
    return MultiOutputRegressor(model)


def _validate_data(df: pd.DataFrame, target_variables: list[str]) -> list[str]:
    """Pre-training data validation. Returns list of issues (empty = OK)."""
    issues = []
    if len(df) < MIN_ROWS:
        issues.append(f"Dataset has only {len(df)} rows (minimum {MIN_ROWS} required).")

    feature_cols = [c for c in df.columns if c not in target_variables]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_features) < MIN_FEATURE_COLS:
        issues.append(f"No numeric feature columns found (need at least {MIN_FEATURE_COLS}).")

    for tv in target_variables:
        nan_count = df[tv].isna().sum()
        if nan_count == len(df):
            issues.append(f"Target '{tv}' is entirely NaN.")
        elif len(df) > 0 and nan_count / len(df) > MAX_NAN_RATIO:
            issues.append(f"Target '{tv}' has {nan_count}/{len(df)} NaN values ({nan_count/len(df):.0%}).")
        if pd.api.types.is_numeric_dtype(df[tv]) and df[tv].nunique() < 2:
            issues.append(f"Target '{tv}' has only {df[tv].nunique()} unique value(s).")

    constant_cols = [c for c in numeric_features if df[c].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant feature columns (will be dropped): {', '.join(constant_cols[:5])}")

    return issues


def _extract_feature_importance(model, features: list[str], key: str) -> Optional[dict]:
    """Extract feature importance from tree-based models."""
    if key not in _TREE_BASED_KEYS:
        return None
    try:
        inner = model
        if hasattr(model, "estimators_") and hasattr(model, "estimator"):
            # MultiOutput wrapper — average across estimators
            all_imp = []
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    all_imp.append(est.feature_importances_)
            if all_imp:
                importances = np.mean(all_imp, axis=0)
                ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                return {name: float(imp) for name, imp in ranked[:10]}
            return None

        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
            ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            return {name: float(imp) for name, imp in ranked[:10]}
    except Exception as e:
        logger.debug("Feature importance extraction failed for %s: %s", key, e)
    return None


def _run_cv(model, X, y, is_classification: bool) -> Optional[float]:
    """Run k-fold cross-validation synchronously. Returns mean score or None."""
    try:
        scoring = "accuracy" if is_classification else "r2"
        scores = cross_val_score(model, X, y, cv=min(CV_FOLDS, len(X)), scoring=scoring)
        return float(np.mean(scores))
    except Exception as e:
        logger.debug("CV failed: %s", e)
        return None


async def _train_single_model(
    loop, model_name: str, key: str, X_train, y_train, X_test, y_test,
    X_full, y_full,
    is_classification: bool, is_multi: bool, num_targets: int,
    target_variables: list[str], features: list[str],
) -> dict:
    """Train a single model with timeout. Returns results dict."""
    factory = MODEL_REGISTRY.get(key)
    if factory is None:
        return {"name": model_name, "score": None, "cv_score": None,
                "text": f"{model_name}: Skipped (unsupported)", "model": None, "importance": None}

    try:
        base_model = factory()
        model = _wrap_multi_output(base_model, key, is_classification, num_targets)

        # Train with timeout
        await asyncio.wait_for(
            loop.run_in_executor(None, model.fit, X_train, y_train),
            timeout=MODEL_TIMEOUT_SECONDS,
        )
        preds = model.predict(X_test)

        # Score
        if is_classification:
            if is_multi:
                preds_arr = np.array(preds)
                if preds_arr.ndim == 1:
                    preds_arr = preds_arr.reshape(-1, num_targets)
                scores = [accuracy_score(y_test.iloc[:, i], preds_arr[:, i]) for i in range(num_targets)]
                score = float(np.mean(scores))
                per_target = ", ".join(f"{target_variables[i]}={s:.4f}" for i, s in enumerate(scores))
                text = f"{model_name}: Avg Accuracy = {score:.4f} ({per_target})"
            else:
                score = accuracy_score(y_test, preds)
                text = f"{model_name}: Accuracy = {score:.4f}"
        else:
            if is_multi:
                preds_arr = np.array(preds)
                if preds_arr.ndim == 1:
                    preds_arr = preds_arr.reshape(-1, num_targets)
                scores = [r2_score(y_test.iloc[:, i], preds_arr[:, i]) for i in range(num_targets)]
                score = float(np.mean(scores))
                per_target = ", ".join(f"{target_variables[i]}={s:.4f}" for i, s in enumerate(scores))
                text = f"{model_name}: Avg R² = {score:.4f} ({per_target})"
            else:
                score = r2_score(y_test, preds)
                text = f"{model_name}: R² Score = {score:.4f}"

        # Cross-validation (only for single-target; multi-target CV is too slow)
        cv_score = None
        if not is_multi and len(X_full) <= 5000:
            cv_model = factory()
            cv_score = await asyncio.wait_for(
                loop.run_in_executor(None, _run_cv, cv_model, X_full, y_full, is_classification),
                timeout=MODEL_TIMEOUT_SECONDS,
            )
            if cv_score is not None:
                text += f" | CV({CV_FOLDS}-fold) = {cv_score:.4f}"

        importance = _extract_feature_importance(model, features, key)

        logger.info("%s: score=%.4f cv=%s", model_name, score, cv_score)
        return {"name": model_name, "score": score, "cv_score": cv_score,
                "text": text, "model": model, "key": key, "importance": importance}

    except asyncio.TimeoutError:
        logger.warning("Model %s timed out after %ds", model_name, MODEL_TIMEOUT_SECONDS)
        return {"name": model_name, "score": None, "cv_score": None,
                "text": f"{model_name}: Timed out ({MODEL_TIMEOUT_SECONDS}s)", "model": None, "importance": None}
    except Exception as e:
        logger.warning("Model %s failed: %s", model_name, e)
        return {"name": model_name, "score": None, "cv_score": None,
                "text": f"{model_name}: Failed ({e})", "model": None, "importance": None}


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------
class TrainingAgent(BaseAgent):
    async def process(
        self,
        data_path: str,
        target_variables: Union[str, list[str]],
        task_type: str,
        models_to_train: list[str],
        on_progress: ProgressCallback = None,
    ) -> tuple[str, Optional[str]]:
        """Train models in parallel with CV, timeouts, feature importance, and versioned saving.

        Args:
            on_progress: async callback(model_name, completed_count, total_count) for per-model updates.
        """
        if isinstance(target_variables, str):
            target_variables = [target_variables]

        num_targets = len(target_variables)
        is_multi = num_targets > 1

        try:
            df = pd.read_csv(data_path)

            for tv in target_variables:
                if tv not in df.columns:
                    raise TargetNotFoundError(f"Target column '{tv}' not found in dataset.")

            # --- Coerce target columns to consistent types ---
            is_classification = task_type.lower() == "classification"
            for tv in target_variables:
                if is_classification:
                    # For classification: ensure all values are strings
                    df[tv] = df[tv].astype(str)
                    # Remove rows where target became 'nan' from conversion
                    df = df[df[tv] != 'nan']
                else:
                    # For regression: coerce to numeric, strip inequality prefixes
                    if not pd.api.types.is_numeric_dtype(df[tv]):
                        cleaned = df[tv].astype(str).str.replace(r"[\\()]+", "", regex=True).str.strip()
                        cleaned = cleaned.str.replace(r"^[<>≤≥]=?\s*", "", regex=True)
                        df[tv] = pd.to_numeric(cleaned, errors="coerce")
                        logger.info("Coerced target '%s' to numeric", tv)
                    df = df.dropna(subset=[tv])

            if len(df) < MIN_ROWS:
                raise DataValidationError(
                    f"After cleaning target columns, only {len(df)} rows remain "
                    f"(minimum {MIN_ROWS} required). Check your target column for non-numeric values."
                )

            # --- Data validation ---
            issues = _validate_data(df, target_variables)
            critical = [i for i in issues if "entirely NaN" in i or "No numeric feature" in i]
            if critical:
                raise DataValidationError(
                    "Data validation failed:\n" + "\n".join(f"- {i}" for i in issues)
                )

            # Drop constant / high-NaN columns
            feature_cols = [c for c in df.columns if c not in target_variables]
            drop_cols = [
                c for c in feature_cols
                if df[c].nunique() <= 1 or (len(df) > 0 and df[c].isna().sum() / len(df) > MAX_NAN_RATIO)
            ]
            if drop_cols:
                logger.info("Dropping low-quality columns: %s", drop_cols)
                df = df.drop(columns=drop_cols)

            X = df.drop(columns=target_variables)
            y = df[target_variables] if is_multi else df[target_variables[0]]
            features = list(X.columns)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42
            )

            loop = asyncio.get_event_loop()
            total_models = len(models_to_train)

            # --- Train each model with per-model progress ---
            results_list = []
            for idx, model_name in enumerate(models_to_train):
                key = model_name.lower().replace(" ", "").replace("_", "")

                if on_progress:
                    await on_progress(model_name, idx, total_models)

                result = await _train_single_model(
                    loop, model_name, key, X_train, y_train, X_test, y_test,
                    X, y, is_classification, is_multi, num_targets,
                    target_variables, features,
                )
                results_list.append(result)

            if on_progress:
                await on_progress("done", total_models, total_models)

            # Find best
            successful = [r for r in results_list if r["score"] is not None]
            if not successful:
                # Collect unique failure reasons to help the user
                failure_reasons = set()
                for r in results_list:
                    text = r.get("text", "")
                    if "Failed" in text:
                        reason = text.split("Failed (", 1)[-1].rstrip(")")
                        failure_reasons.add(reason)
                reasons_text = "\n".join(f"- {r}" for r in list(failure_reasons)[:3])
                raise ModelTrainingError(
                    f"All models failed to train.\n\n**Possible causes:**\n{reasons_text}\n\n"
                    "Please check your data for mixed types or invalid values in the target column."
                )

            best = max(successful, key=lambda r: r["score"])
            result_texts = [r["text"] for r in results_list]

            # --- Feature importance ---
            importance_text = ""
            if best.get("importance"):
                top_features = list(best["importance"].items())[:8]
                importance_lines = [f"  - {name}: {imp:.4f}" for name, imp in top_features]
                importance_text = (
                    f"\n\n**Top features (by importance in {best['name']}):**\n"
                    + "\n".join(importance_lines)
                )

            # --- Validation warnings ---
            warnings_text = ""
            if issues:
                warnings_text = "\n\n**Data warnings:**\n" + "\n".join(f"- {i}" for i in issues)

            # --- Save with timestamp version ---
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            timestamp = int(time.time())
            model_path = f"{model_dir}/model_{timestamp}.joblib"
            # Compute feature/target stats for use without original data file
            feature_stats = {}
            for col in features:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    c = df[col].dropna()
                    if len(c) > 0:
                        feature_stats[col] = {
                            "min": float(c.min()),
                            "max": float(c.max()),
                            "median": float(c.median()),
                        }
            target_stats = {}
            for col in target_variables:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    c = df[col].dropna()
                    if len(c) > 1:
                        target_stats[col] = {
                            "min": float(c.min()),
                            "max": float(c.max()),
                            "var": float(c.var()),
                            "range": float(c.max() - c.min()),
                        }

            joblib.dump({
                "model": best["model"],
                "features": features,
                "targets": target_variables,
                "best_model_name": best["name"],
                "score": best["score"],
                "cv_score": best.get("cv_score"),
                "task_type": task_type,
                "feature_importance": best.get("importance"),
                "timestamp": timestamp,
                "all_results": [{"name": r["name"], "score": r["score"], "cv_score": r.get("cv_score")} for r in results_list],
                "feature_stats": feature_stats,
                "target_stats": target_stats,
            }, model_path)

            # Keep only the last 10 models
            _cleanup_old_models(model_dir, keep=10)

            targets_str = ", ".join(f"'{t}'" for t in target_variables)
            metric_name = (
                "Avg Accuracy" if is_classification and is_multi
                else "Accuracy" if is_classification
                else "Avg R²" if is_multi
                else "R² Score"
            )
            prompt = (
                f"You are a metallurgical ML expert. You just trained several "
                f"{'multi-output ' if is_multi else ''}models "
                f"for a {task_type} task predicting {targets_str}. "
                f"Here are the test set results (with {CV_FOLDS}-fold cross-validation where available):\n"
                f"{chr(10).join(result_texts)}\n\n"
                f"The best model was {best['name']} with a {metric_name} of "
                f"{best['score']:.4f}."
                f"{importance_text}"
                f"{warnings_text}\n\n"
                f"Write a short, engaging summary reporting these "
                f"results. Include the feature importance insights if available. "
                f"Note the cross-validation scores for robustness assessment. "
                f"Mention any data warnings if relevant. "
                f"Keep it to 1-2 concise paragraphs. End by telling them the model "
                f"is ready to predict new alloy compositions."
            )

            report = await self._chat(
                [{"role": "user", "content": prompt}], model=self.pro_model
            )
            return f"Training Agent: {report}", model_path

        except (TargetNotFoundError, DataValidationError, ModelTrainingError) as e:
            return str(e), None
        except Exception as e:
            logger.error("Training failed: %s", e)
            return f"Error during training: {e}", None
