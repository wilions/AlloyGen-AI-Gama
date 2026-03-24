from __future__ import annotations
import asyncio
import json
import os
import logging
from typing import Optional

import numpy as np
import joblib
import optuna
import pandas as pd
from backend.agents.base_agent import BaseAgent
from backend.errors import PredictionError

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 200       # Number of optimization trials
N_CANDIDATES = 5     # Top candidates to return
TOLERANCE = 0.05     # 5% tolerance on target constraints


def _load_model(model_path: str) -> dict:
    if not model_path or not os.path.exists(model_path):
        raise PredictionError("No trained model found. Please train a model first.")
    return joblib.load(model_path)


def _get_feature_ranges(
    data_path: str, features: list[str], model_data: dict | None = None,
) -> dict[str, tuple[float, float]]:
    """Extract min/max ranges for each feature from training data or model stats."""
    # Try reading from data file first
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        ranges = {}
        for f in features:
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                col = df[f].dropna()
                if len(col) > 0:
                    ranges[f] = (float(col.min()), float(col.max()))
                else:
                    ranges[f] = (0.0, 1.0)
            else:
                ranges[f] = (0.0, 1.0)
        return ranges

    # Fall back to stats embedded in model file
    stats = (model_data or {}).get("feature_stats", {})
    ranges = {}
    for f in features:
        if f in stats:
            ranges[f] = (stats[f]["min"], stats[f]["max"])
        else:
            ranges[f] = (0.0, 1.0)
    return ranges


def _get_feature_medians(
    data_path: str, features: list[str], model_data: dict | None = None,
) -> dict[str, float]:
    """Get median values for each feature from training data or model stats."""
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        medians = {}
        for f in features:
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                medians[f] = float(df[f].median())
            else:
                medians[f] = 0.0
        return medians

    stats = (model_data or {}).get("feature_stats", {})
    medians = {}
    for f in features:
        if f in stats:
            medians[f] = stats[f]["median"]
        else:
            medians[f] = 0.0
    return medians


def _identify_element_features(features: list[str]) -> list[str]:
    """Identify which features are likely elemental composition columns (e.g., 'Fe at%', 'C', 'Cr at%')."""
    # Common element symbols
    elements = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu",
    }
    element_features = []
    for f in features:
        # Check if feature name starts with or is an element symbol
        # Handle formats like "Fe at%", "C", "Cr at%", "Fe_at_pct"
        base = f.split()[0].split("_")[0].replace("at%", "").replace("wt%", "").strip()
        if base in elements:
            element_features.append(f)
    return element_features


def _compute_target_weights(
    data_path: str, targets: list[str], constraints: list[dict],
    model_data: dict | None = None,
) -> dict[str, float]:
    """Compute per-target weight for the optimisation penalty.

    Combines two signals:
      1. **Inverse-variance** — targets with lower variance in the training
         data are harder for the model to move, so violations matter more.
         weight_var = 1 / var(target).  Targets not in the data get weight 1.
      2. **Constraint precision** — a tight constraint window (e.g. 49-51)
         signals the user cares more than a loose one (e.g. 5-20).
         weight_prec = data_range / constraint_span.
         Exact targets get the maximum precision weight.
         One-sided constraints (only min or only max) get weight 1 (neutral).

    The final weight is the product: w = w_var * w_prec, then normalised so
    the mean weight across all constrained targets equals 1.0.  This keeps
    the overall penalty magnitude stable while redistributing influence.
    """
    # --- Pre-compute data stats for each target ---
    data_var: dict[str, float] = {}
    data_range: dict[str, float] = {}

    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        for t in targets:
            if t in df.columns and pd.api.types.is_numeric_dtype(df[t]):
                col = df[t].dropna()
                if len(col) > 1:
                    data_var[t] = max(float(col.var()), 1e-12)
                    data_range[t] = max(float(col.max() - col.min()), 1e-9)
    else:
        # Fall back to stats embedded in model file
        target_stats = (model_data or {}).get("target_stats", {})
        for t in targets:
            if t in target_stats:
                data_var[t] = max(target_stats[t].get("var", 1.0), 1e-12)
                data_range[t] = max(target_stats[t].get("range", 1.0), 1e-9)

    # --- Compute raw weight for each constrained target ---
    raw_weights: dict[str, float] = {}
    constrained_targets: list[str] = []
    for c in constraints:
        t = c["target"]
        if t in raw_weights:
            continue
        constrained_targets.append(t)

        # Inverse-variance component
        w_var = 1.0 / data_var[t] if t in data_var else 1.0

        # Constraint-precision component
        lo = c.get("min")
        hi = c.get("max")
        exact = c.get("exact")
        if exact is not None:
            # Exact target → maximum precision (use full data range as span ratio)
            w_prec = data_range.get(t, 1.0)  # numerator; denominator ≈ 0 → large
        elif lo is not None and hi is not None:
            constraint_span = abs(hi - lo)
            if constraint_span > 1e-9:
                w_prec = data_range.get(t, constraint_span) / constraint_span
            else:
                w_prec = data_range.get(t, 1.0)  # effectively exact
        else:
            # One-sided constraint — no precision signal, neutral weight
            w_prec = 1.0

        raw_weights[t] = w_var * w_prec

    if not raw_weights:
        return {}

    # --- Normalise so mean weight = 1.0 ---
    mean_w = sum(raw_weights.values()) / len(raw_weights)
    if mean_w < 1e-15:
        return {t: 1.0 for t in raw_weights}
    return {t: w / mean_w for t, w in raw_weights.items()}


def _run_optimization(
    model,
    features: list[str],
    targets: list[str],
    constraints: list[dict],
    feature_ranges: dict[str, tuple[float, float]],
    fixed_features: dict[str, float],
    active_elements: list[str],
    medians: dict[str, float],
    target_weights: dict[str, float],
    data_ranges: dict[str, float],
) -> list[dict]:
    """Run Optuna optimization constrained to only the specified elements.

    active_elements: features to optimize over (user-selected elements)
    All other features are fixed at 0 (for element features) or median (for non-element features).
    target_weights: per-target weights combining inverse-variance and constraint
                    precision (mean-normalised to 1.0).
    data_ranges: per-target data range for magnitude normalisation.
    """
    # Identify which features are element-based
    all_element_features = set(_identify_element_features(features))

    # Determine if element features use percentage units (at%, wt%)
    # by checking if any element feature name contains "%" or if values sum near 100
    _pct_keywords = {"at%", "wt%", "at_%", "wt_%", "at_pct", "wt_pct", "%"}
    _uses_pct = any(
        any(kw in f.lower() for kw in _pct_keywords) for f in all_element_features
    )

    # Fixed element contributions to the sum
    _fixed_element_sum = sum(
        fixed_features[f] for f in fixed_features if f in all_element_features
    )

    def objective(trial: optuna.Trial) -> float:
        feature_dict = {}
        for f in features:
            if f in fixed_features:
                feature_dict[f] = fixed_features[f]
            elif f in active_elements:
                # User-selected element: optimize
                lo, hi = feature_ranges.get(f, (0.0, 1.0))
                if lo == hi:
                    feature_dict[f] = lo
                else:
                    feature_dict[f] = trial.suggest_float(f, lo, hi)
            elif f in all_element_features:
                # Element NOT selected by user: fix to 0
                feature_dict[f] = 0.0
            else:
                # Non-element feature (e.g., temperature, time): use median
                feature_dict[f] = medians.get(f, 0.0)

        # Enforce sum-to-100 for element percentages: rescale active elements
        if _uses_pct and active_elements:
            active_sum = sum(feature_dict[f] for f in active_elements)
            target_sum = 100.0 - _fixed_element_sum
            if active_sum > 1e-9 and target_sum > 0:
                scale_factor = target_sum / active_sum
                for f in active_elements:
                    feature_dict[f] *= scale_factor

        df_input = pd.DataFrame([feature_dict])[features]
        prediction = model.predict(df_input)
        pred_arr = np.array(prediction).flatten()

        total_penalty = 0.0
        for c in constraints:
            target_name = c["target"]
            if target_name not in targets:
                continue
            idx = targets.index(target_name)
            if idx >= len(pred_arr):
                continue
            pred_val = pred_arr[idx]
            scale = data_ranges.get(target_name, 1.0)
            weight = target_weights.get(target_name, 1.0)

            # Classification target: binary match penalty
            try:
                pred_num = float(pred_val)
            except (ValueError, TypeError):
                # String prediction (classification) — exact match or large penalty
                desired = c.get("exact") or c.get("min") or c.get("max")
                if desired is not None and str(pred_val) != str(desired):
                    total_penalty += weight * 100.0
                continue

            # Normalise violation by data range, then apply weight
            if c.get("exact") is not None:
                try:
                    total_penalty += weight * (abs(pred_num - float(c["exact"])) / scale) ** 2
                except (ValueError, TypeError):
                    if str(pred_val) != str(c["exact"]):
                        total_penalty += weight * 100.0
            else:
                if c.get("min") is not None:
                    try:
                        min_val = float(c["min"])
                        if pred_num < min_val:
                            total_penalty += weight * ((min_val - pred_num) / scale) ** 2
                    except (ValueError, TypeError):
                        pass
                if c.get("max") is not None:
                    try:
                        max_val = float(c["max"])
                        if pred_num > max_val:
                            total_penalty += weight * ((pred_num - max_val) / scale) ** 2
                    except (ValueError, TypeError):
                        pass

        return total_penalty

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Collect top N unique candidates
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    candidates = []
    seen = set()

    for trial in sorted_trials:
        if len(candidates) >= N_CANDIDATES:
            break
        if trial.value is None:
            continue

        composition = {}
        for f in features:
            if f in fixed_features:
                composition[f] = fixed_features[f]
            elif f in active_elements:
                composition[f] = trial.params.get(f, feature_ranges.get(f, (0.0, 1.0))[0])
            elif f in all_element_features:
                composition[f] = 0.0
            else:
                composition[f] = medians.get(f, 0.0)

        # Rescale active elements to sum to 100% (minus fixed elements)
        if _uses_pct and active_elements:
            active_sum = sum(composition[f] for f in active_elements)
            target_sum = 100.0 - _fixed_element_sum
            if active_sum > 1e-9 and target_sum > 0:
                scale_factor = target_sum / active_sum
                for f in active_elements:
                    composition[f] *= scale_factor

        key = tuple(round(v, 4) for v in composition.values())
        if key in seen:
            continue
        seen.add(key)

        df_input = pd.DataFrame([composition])[features]
        pred = model.predict(df_input)
        pred_arr = np.array(pred).flatten()

        pred_dict = {}
        for i, t in enumerate(targets):
            if i < len(pred_arr):
                try:
                    pred_dict[t] = float(pred_arr[i])
                except (ValueError, TypeError):
                    pred_dict[t] = str(pred_arr[i])

        satisfied = True
        for c in constraints:
            t = c["target"]
            if t not in pred_dict:
                continue
            v = pred_dict[t]

            # Classification target: check exact match
            if isinstance(v, str):
                desired = c.get("exact") or c.get("min") or c.get("max")
                if desired is not None and str(v) != str(desired):
                    satisfied = False
                continue

            scale = data_ranges.get(t, max(abs(v), 1.0))
            if c.get("exact") is not None:
                try:
                    if abs(v - float(c["exact"])) / max(scale, 1e-6) > TOLERANCE:
                        satisfied = False
                except (ValueError, TypeError):
                    if str(v) != str(c["exact"]):
                        satisfied = False
            if c.get("min") is not None:
                try:
                    if v < float(c["min"]) - abs(float(c["min"])) * TOLERANCE:
                        satisfied = False
                except (ValueError, TypeError):
                    pass
            if c.get("max") is not None:
                try:
                    if v > float(c["max"]) + abs(float(c["max"])) * TOLERANCE:
                        satisfied = False
                except (ValueError, TypeError):
                    pass

        candidates.append({
            "composition": composition,
            "predictions": pred_dict,
            "penalty": trial.value,
            "satisfies_constraints": satisfied,
        })

    return candidates


class InverseDesignAgent(BaseAgent):

    async def extract_constraints(
        self, user_input: str, model_path: str, data_path: str,
    ) -> tuple[str, Optional[list[dict]], Optional[dict]]:
        """Step 1: Extract constraints from user input and ask which elements to include.

        Returns (message_to_user, constraints, fixed_features).
        If constraints is None, the message is a clarification request.
        """
        try:
            model_data = _load_model(model_path)
            features = model_data["features"]
            targets = model_data.get("targets", ["target"])

            # Identify element features in the dataset
            element_features = _identify_element_features(features)

            extract_prompt = (
                "You are an AI assistant for inverse alloy design. "
                "The user wants to find alloy compositions that achieve certain target properties. "
                f"The model predicts these targets: {targets}\n"
                f"The model uses these input features: {features}\n\n"
                "From the user's message, extract:\n"
                "1. 'constraints': target property requirements. Each should have:\n"
                "   - 'target': exact name from the targets list\n"
                "   - 'min': minimum value (null if not specified)\n"
                "   - 'max': maximum value (null if not specified)\n"
                "   - 'exact': exact desired value (null if range is given)\n"
                "2. 'fixed_features': any input features the user wants to hold constant "
                "(e.g., 'keep C at 0.4%'). Dict of feature_name → value. Empty dict if none.\n\n"
                "Return strictly valid JSON:\n"
                '{"constraints": [{"target": "...", "min": null, "max": null, "exact": null}, ...], '
                '"fixed_features": {}}'
            )

            raw = await self._chat(
                [
                    {"role": "system", "content": extract_prompt},
                    {"role": "user", "content": user_input},
                ],
                json_mode=True,
            )

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return (
                    "I couldn't understand your design requirements. Please specify the desired property "
                    "values (e.g., 'design an alloy with hardness between 45-55').",
                    None, None,
                )

            constraints = parsed.get("constraints", [])
            fixed_features = parsed.get("fixed_features", {})

            if not constraints:
                return (
                    "I couldn't identify any target property constraints. Please specify what you want, e.g.:\n"
                    "- 'Design an alloy with hardness > 50'\n"
                    "- 'Find a composition with tensile strength between 800-1000 MPa'\n"
                    f"- Available targets: {', '.join(targets)}",
                    None, None,
                )

            # Format constraints for display
            constraints_desc = []
            for c in constraints:
                t = c["target"]
                if c.get("exact") is not None:
                    constraints_desc.append(f"**{t}** = {c['exact']}")
                else:
                    parts = []
                    if c.get("min") is not None:
                        parts.append(f"> {c['min']}")
                    if c.get("max") is not None:
                        parts.append(f"< {c['max']}")
                    constraints_desc.append(f"**{t}** {' and '.join(parts)}")

            # Format available elements
            if element_features:
                elements_display = ", ".join(f"`{e}`" for e in element_features)
            else:
                elements_display = "*(no element-specific columns detected)*"

            msg = (
                f"Got it! I'll search for compositions where {', '.join(constraints_desc)}.\n\n"
                f"**Available elements in the dataset:**\n{elements_display}\n\n"
                "**How many elements** do you want in the alloy, and **which ones**?\n"
                "For example: *\"5 elements: Fe, C, Cr, Mn, Ni\"*\n\n"
                "Or say **\"all\"** to optimize over all available elements."
            )

            return msg, constraints, fixed_features

        except PredictionError as e:
            return str(e), None, None
        except Exception as e:
            logger.error("Constraint extraction failed: %s", e)
            return f"Error: {e}", None, None

    async def parse_elements_and_run(
        self,
        user_input: str,
        model_path: str,
        data_path: str,
        constraints: list[dict],
        fixed_features: dict[str, float],
    ) -> str:
        """Step 2: Parse user's element selection, then run optimization."""
        try:
            model_data = _load_model(model_path)
            model = model_data["model"]
            features = model_data["features"]
            targets = model_data.get("targets", ["target"])
            best_name = model_data.get("best_model_name", "unknown")

            element_features = _identify_element_features(features)

            # Use LLM to parse which elements the user selected
            parse_prompt = (
                "The user is selecting which elements to include in an alloy design optimization. "
                f"Available element features: {element_features}\n\n"
                "From the user's message, extract:\n"
                "1. 'selected_elements': list of feature names from the available list that the user wants. "
                "Match element symbols to the exact feature names (e.g., if user says 'Fe' and feature is 'Fe at%', use 'Fe at%').\n"
                "2. If user says 'all', return all available elements.\n\n"
                "Return strictly valid JSON:\n"
                '{"selected_elements": ["Fe at%", "C", ...]}'
            )

            raw = await self._chat(
                [
                    {"role": "system", "content": parse_prompt},
                    {"role": "user", "content": user_input},
                ],
                json_mode=True,
            )

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return "I couldn't understand your element selection. Please list the elements, e.g., 'Fe, C, Cr, Mn, Ni'."

            selected = parsed.get("selected_elements", [])
            if not selected:
                return "No elements selected. Please specify which elements to include, e.g., 'Fe, C, Cr, Mn, Ni' or 'all'."

            # Validate selections exist in features
            valid_elements = [e for e in selected if e in features]
            if not valid_elements:
                return (
                    f"None of the selected elements match the dataset features. "
                    f"Available: {', '.join(element_features)}"
                )

            # Get feature ranges, medians, and target weights for normalisation
            feature_ranges = _get_feature_ranges(data_path, features, model_data)
            medians = _get_feature_medians(data_path, features, model_data)
            target_weights = _compute_target_weights(data_path, targets, constraints, model_data)

            # Data ranges for magnitude normalisation (separate from weights)
            data_ranges: dict[str, float] = {}
            if data_path and os.path.exists(data_path):
                df_tmp = pd.read_csv(data_path)
                for t in targets:
                    if t in df_tmp.columns and pd.api.types.is_numeric_dtype(df_tmp[t]):
                        col = df_tmp[t].dropna()
                        data_ranges[t] = max(float(col.max() - col.min()), 1.0) if len(col) > 1 else 1.0
                    else:
                        data_ranges[t] = 1.0
            else:
                target_stats = model_data.get("target_stats", {})
                for t in targets:
                    if t in target_stats:
                        data_ranges[t] = max(target_stats[t].get("range", 1.0), 1.0)
                    else:
                        data_ranges[t] = 1.0

            # Run optimization
            loop = asyncio.get_event_loop()
            candidates = await loop.run_in_executor(
                None,
                _run_optimization,
                model, features, targets, constraints, feature_ranges,
                fixed_features, valid_elements, medians, target_weights, data_ranges,
            )

            if not candidates:
                return "Optimization couldn't find any candidate compositions. Try different elements or relax your constraints."

            # Format results
            n_satisfied = sum(1 for c in candidates if c["satisfies_constraints"])
            constraints_desc = []
            for c in constraints:
                t = c["target"]
                if c.get("exact") is not None:
                    constraints_desc.append(f"{t} = {c['exact']}")
                else:
                    parts = []
                    if c.get("min") is not None:
                        parts.append(f"> {c['min']}")
                    if c.get("max") is not None:
                        parts.append(f"< {c['max']}")
                    constraints_desc.append(f"{t} {' and '.join(parts)}")

            elements_str = ", ".join(valid_elements)
            result = f"**Inverse Design Results** (model: {best_name})\n\n"
            result += f"**Target constraints:** {', '.join(constraints_desc)}\n"
            result += f"**Active elements:** {elements_str}\n"
            if fixed_features:
                fixed_desc = ", ".join(f"{k}={v}" for k, v in fixed_features.items())
                result += f"**Fixed features:** {fixed_desc}\n"
            if len(target_weights) > 1:
                weight_desc = ", ".join(
                    f"{t}: {w:.2f}x" for t, w in sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
                )
                result += f"**Target weights** (auto): {weight_desc}\n"
            result += f"**Candidates satisfying constraints:** {n_satisfied}/{len(candidates)}\n\n"

            for i, cand in enumerate(candidates):
                status = "✅" if cand["satisfies_constraints"] else "⚠️"
                result += f"### Candidate {i+1} {status}\n"

                # Show predicted values
                pred_lines = [
                    f"  - {t}: **{v:.4f}**" if isinstance(v, (int, float)) else f"  - {t}: **{v}**"
                    for t, v in cand["predictions"].items()
                ]
                result += "**Predicted properties:**\n" + "\n".join(pred_lines) + "\n\n"

                # Show only the active element compositions (sorted by value)
                comp = cand["composition"]
                active_comp = {k: v for k, v in comp.items() if k in valid_elements and abs(v) > 1e-6}
                sorted_comp = sorted(active_comp.items(), key=lambda x: x[1], reverse=True)
                comp_lines = [f"  - {k}: **{v:.4f}**" for k, v in sorted_comp]
                if not comp_lines:
                    comp_lines = ["  - (all selected elements near zero)"]
                result += "**Composition:**\n" + "\n".join(comp_lines) + "\n\n"

            result += (
                "---\n"
                "*Note: These are model-predicted compositions. Candidates marked ✅ satisfy your "
                "constraints within 5% tolerance. Non-selected elements are set to 0, non-element "
                "features use dataset medians. Always validate with domain expertise and experiments.*"
            )

            return result

        except PredictionError as e:
            return str(e)
        except Exception as e:
            logger.error("Inverse design failed: %s", e)
            return f"Error during inverse design: {e}"
