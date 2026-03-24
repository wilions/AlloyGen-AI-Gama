from __future__ import annotations
import re
import os
import logging
from typing import Optional, Union
import pandas as pd
from backend.agents.base_agent import BaseAgent
from backend.utils import find_target_column

logger = logging.getLogger(__name__)


class DataPrepAgent(BaseAgent):
    async def process(
        self, data_path: str, target_variables: Union[str, list[str]]
    ) -> tuple[str, Optional[str]]:
        """Clean dataset for one or multiple targets. Returns (summary_message, clean_path)."""
        # Normalize to list
        if isinstance(target_variables, str):
            target_variables = [target_variables]

        try:
            if not os.path.exists(data_path):
                return "Error: Data file not found.", None

            if data_path.endswith(".csv"):
                df = pd.read_csv(data_path)
            elif data_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(data_path)
            else:
                return "Error: Unsupported file format.", None

            original_shape = df.shape
            df.dropna(axis=1, how="all", inplace=True)

            # Match all target columns
            matched_targets = []
            for tv in target_variables:
                matched_col = find_target_column(df.columns.tolist(), tv)
                if matched_col is None:
                    available = ", ".join(df.columns.tolist())
                    return (
                        f"Error: Could not find a column matching '{tv}' "
                        f"in the dataset. Available columns are: {available}. "
                        f"Please specify the exact target column name."
                    ), None
                if matched_col != tv:
                    logger.info("Matched target '%s' -> '%s'", tv, matched_col)
                matched_targets.append(matched_col)

            # Drop rows where ANY target is missing
            df.dropna(subset=matched_targets, inplace=True)

            # Try to coerce object columns that look numeric (e.g. "207\" or "(207)\" or "< 1")
            for col in df.select_dtypes(include=["object", "category"]).columns:
                # Strip common noise: backslashes, parens, extra whitespace
                cleaned = df[col].astype(str).str.replace(r"[\\()]+", "", regex=True).str.strip()
                # Handle inequality prefixes like "< 1", "> 100", "≤ 5", "≥ 10"
                cleaned = cleaned.str.replace(r"^[<>≤≥]=?\s*", "", regex=True)
                numeric_parsed = pd.to_numeric(cleaned, errors="coerce")
                # If >50% of non-null values are numeric, convert the column
                if numeric_parsed.notna().sum() > 0.5 * df[col].notna().sum():
                    df[col] = numeric_parsed
                    logger.info("Coerced column '%s' from object to numeric", col)

            # Drop columns that are entirely empty after cleaning
            df.dropna(axis=1, how="all", inplace=True)

            # Drop rows where target became NaN after coercion
            df.dropna(subset=matched_targets, inplace=True)

            # Fill numeric missing values
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # One-hot encode remaining categoricals (excluding targets)
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            cat_to_encode = [c for c in categorical_cols if c not in matched_targets]
            # Drop high-cardinality columns (>20 unique values) to avoid feature explosion
            low_card = [c for c in cat_to_encode if df[c].nunique() <= 20]
            high_card = [c for c in cat_to_encode if df[c].nunique() > 20]
            if high_card:
                logger.info("Dropping high-cardinality columns: %s", high_card)
                df.drop(columns=high_card, inplace=True)
            if low_card:
                df = pd.get_dummies(df, columns=low_card, drop_first=True)

            # Sanitize column names for XGBoost compatibility
            df.columns = [re.sub(r"[\[\]<>]", "_", col) for col in df.columns]

            base, ext = os.path.splitext(data_path)
            clean_path = f"{base}_cleaned{ext}"
            df.to_csv(clean_path, index=False)

            targets_display = ", ".join(f"'{t}'" for t in matched_targets)
            prompt = (
                f"You are a metallurgical data scientist. You have just cleaned a dataset "
                f"for predicting {targets_display}. The original dataset had shape "
                f"{original_shape}, and after handling missing values and one-hot encoding "
                f"categorical variables, the cleaned dataset has shape {df.shape} and was saved. "
                "Write a concise, engaging summary of these data preparation steps. Keep it to 1 paragraph."
            )

            report = await self._chat([{"role": "user", "content": prompt}])
            return f"Data Prep Agent: {report}", clean_path

        except Exception as e:
            logger.error("Data preparation failed: %s", e)
            return f"Error during data preparation: {e}", None
