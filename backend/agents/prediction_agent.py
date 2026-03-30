from __future__ import annotations
import json
import os
import logging
from typing import Optional

import numpy as np
import joblib
import pandas as pd
from backend.agents.base_agent import BaseAgent
from backend.errors import PredictionError
from backend.ml.uncertainty import add_uncertainty_to_prediction
from backend.ml.explainability import explain_prediction

logger = logging.getLogger(__name__)


def _load_model(model_path: str) -> dict:
    """Load and validate saved model."""
    if not model_path or not os.path.exists(model_path):
        raise PredictionError("No trained model found. Please upload data and train a model first.")
    return joblib.load(model_path)


def _align_features(df_input: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Align input DataFrame columns to match training features exactly.
    Missing columns get 0, extra columns are dropped, order is preserved.
    """
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0.0
    return df_input[expected_features]


def _check_extrapolation(
    feature_dict: dict, data_path: str, features: list[str],
) -> list[str]:
    """Check which input features fall outside the training data range.
    Returns list of warning strings for out-of-range features.
    """
    warnings = []
    try:
        df = pd.read_csv(data_path)
        for f in features:
            if f not in df.columns or not pd.api.types.is_numeric_dtype(df[f]):
                continue
            val = feature_dict.get(f, 0.0)
            col = df[f].dropna()
            if len(col) == 0:
                continue
            lo, hi = float(col.min()), float(col.max())
            if val < lo or val > hi:
                warnings.append(
                    f"`{f}` = {val:.4g} (training range: {lo:.4g} – {hi:.4g})"
                )
    except Exception:
        pass
    return warnings


def _check_prediction_plausibility(
    pred_arr: np.ndarray, targets: list[str], data_path: str,
) -> list[str]:
    """Flag predictions that fall far outside training target ranges."""
    warnings = []
    try:
        df = pd.read_csv(data_path)
        for i, t in enumerate(targets):
            if i >= len(pred_arr):
                break
            v = float(pred_arr[i])
            if t not in df.columns or not pd.api.types.is_numeric_dtype(df[t]):
                continue
            col = df[t].dropna()
            if len(col) < 2:
                continue
            lo, hi = float(col.min()), float(col.max())
            data_range = hi - lo
            if data_range < 1e-9:
                continue
            # Flag if prediction is more than 50% of data range beyond bounds
            margin = data_range * 0.5
            if v < lo - margin or v > hi + margin:
                warnings.append(
                    f"`{t}` = {v:.4f} is far outside training range "
                    f"({lo:.4g} – {hi:.4g})"
                )
    except Exception:
        pass
    return warnings


def _fmt_val(v) -> str:
    """Format a prediction value — numeric with 4 decimals, otherwise as-is."""
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return str(v)


def _format_prediction(
    prediction, targets: list[str], best_name: str, feature_dict: dict,
    extrap_warnings: list[str], plausibility_warnings: list[str],
) -> str:
    """Format a single prediction result as markdown."""
    pred_arr = np.array(prediction).flatten()

    if len(targets) > 1:
        pred_values = pred_arr[:len(targets)]
        preds_lines = "\n".join(
            f"  - **{t}**: {_fmt_val(v)}" for t, v in zip(targets, pred_values)
        )
        result = f"**Predictions** (model: {best_name}):\n{preds_lines}\n\n"
    else:
        pred_val = pred_arr[0] if len(pred_arr) > 0 else prediction
        result = f"**Prediction for '{targets[0]}'**: {_fmt_val(pred_val)} (model: {best_name})\n\n"

    # Add warnings if any
    all_warnings = []
    if extrap_warnings:
        all_warnings.append(
            "**⚠️ Extrapolation warning** — these inputs are outside the training data range:\n"
            + "\n".join(f"  - {w}" for w in extrap_warnings[:5])
        )
    if plausibility_warnings:
        all_warnings.append(
            "**⚠️ Plausibility warning** — predictions may be unreliable:\n"
            + "\n".join(f"  - {w}" for w in plausibility_warnings)
        )

    if all_warnings:
        result += "\n".join(all_warnings) + "\n\n"
        result += (
            "*The model is extrapolating beyond its training data, which can produce "
            "physically implausible values. Consider inputs closer to the training distribution.*\n\n"
        )

    result += f"**Input features used:** {feature_dict}"
    return result


class PredictionAgent(BaseAgent):
    async def process(
        self, user_input: str, model_path: str, data_path: str
    ) -> str:
        """Extract features via LLM, predict, format result. Feature-aligned for safety."""
        try:
            model_data = _load_model(model_path)
            model = model_data["model"]
            features = model_data["features"]
            targets = model_data.get("targets", ["target"])
            best_name = model_data.get("best_model_name", "unknown")

            extract_prompt = (
                "You are an AI extracting feature values for a machine learning prediction model. "
                "The user is describing material properties or alloy compositions. "
                "Extract the values corresponding to these exact feature names: "
                f"{features}\n\n"
                "Return a JSON object where keys are feature names and values are float numbers. "
                "If a feature is not mentioned by the user, use a reasonable default (e.g., 0 for "
                "absent elements). Output strictly valid JSON."
            )

            raw = await self._chat(
                [
                    {"role": "system", "content": extract_prompt},
                    {"role": "user", "content": user_input},
                ],
                json_mode=True,
            )

            try:
                feature_dict = json.loads(raw)
            except json.JSONDecodeError:
                return "I couldn't parse the feature values. Please describe the alloy composition more clearly."

            df_input = _align_features(pd.DataFrame([feature_dict]), features)
            prediction = model.predict(df_input)

            # Check for extrapolation and plausibility
            extrap_warnings = _check_extrapolation(feature_dict, data_path, features)
            pred_arr = np.array(prediction).flatten()
            plausibility_warnings = _check_prediction_plausibility(pred_arr, targets, data_path)

            # Uncertainty quantification
            uncertainty_text = ""
            target_stats = model_data.get("target_stats", {})
            if hasattr(model, "predict_with_uncertainty"):
                uq_results = add_uncertainty_to_prediction(model, df_input, target_stats.get(targets[0]) if len(targets) == 1 else None)
                if uq_results and uq_results[0].std > 0:
                    uq = uq_results[0]
                    uncertainty_text = (
                        f"\n**Confidence interval (95%):** [{uq.lower:.4f}, {uq.upper:.4f}] "
                        f"(std: {uq.std:.4f}, confidence: {uq.confidence})"
                    )

            # SHAP explanation for this prediction
            shap_text = ""
            try:
                if data_path and os.path.exists(data_path):
                    bg_df = pd.read_csv(data_path)
                    bg_features = _align_features(bg_df.copy(), features)
                    shap_result = explain_prediction(model, df_input, bg_features, feature_names=features)
                    if shap_result and shap_result.get("contributions"):
                        top_contributions = list(shap_result["contributions"].items())[:5]
                        shap_lines = []
                        for name, val in top_contributions:
                            direction = "+" if val > 0 else ""
                            shap_lines.append(f"  - **{name}**: {direction}{val:.4f}")
                        shap_text = "\n**Feature contributions (SHAP):**\n" + "\n".join(shap_lines)
            except Exception as e:
                logger.debug("SHAP prediction explanation skipped: %s", e)

            result = _format_prediction(
                prediction, targets, best_name, feature_dict,
                extrap_warnings, plausibility_warnings,
            )
            return result + uncertainty_text + shap_text

        except PredictionError as e:
            return str(e)
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return f"Error making prediction: {e}"

    async def batch_predict(
        self, model_path: str, csv_path: str
    ) -> tuple[str, Optional[str]]:
        """Run batch predictions from a CSV file. Returns (summary_message, output_csv_path)."""
        try:
            model_data = _load_model(model_path)
            model = model_data["model"]
            features = model_data["features"]
            targets = model_data.get("targets", ["target"])
            best_name = model_data.get("best_model_name", "unknown")

            df = pd.read_csv(csv_path)
            original_shape = df.shape

            df_input = _align_features(df.copy(), features)
            predictions = model.predict(df_input)

            pred_arr = np.array(predictions)
            if len(targets) > 1:
                if pred_arr.ndim == 1:
                    pred_arr = pred_arr.reshape(-1, len(targets))
                for i, t in enumerate(targets):
                    df[f"predicted_{t}"] = pred_arr[:, i]
            else:
                df[f"predicted_{targets[0]}"] = pred_arr.flatten()

            output_path = csv_path.replace(".csv", "_predictions.csv")
            df.to_csv(output_path, index=False)

            n_rows = original_shape[0]
            targets_str = ", ".join(f"**{t}**" for t in targets)
            return (
                f"Batch prediction complete! Predicted {targets_str} for **{n_rows}** rows "
                f"using **{best_name}**. Results saved to `{os.path.basename(output_path)}`.",
                output_path,
            )

        except PredictionError as e:
            return str(e), None
        except Exception as e:
            logger.error("Batch prediction failed: %s", e)
            return f"Error during batch prediction: {e}", None
