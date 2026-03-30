"""LLM-powered literature data extraction agent for AlloyGen 2.0.

Extracts composition-property data from paper abstracts, tables,
or free-form text into structured JSON format.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd
from backend.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class LiteratureAgent(BaseAgent):
    async def extract_data(self, text: str, existing_columns: Optional[list[str]] = None) -> tuple[str, Optional[pd.DataFrame]]:
        """Extract composition-property data from text (paper abstract, table, etc.).

        Returns (summary_message, extracted_dataframe_or_None).
        """
        columns_hint = ""
        if existing_columns:
            columns_hint = (
                f"\n\nThe existing dataset has these columns: {existing_columns}. "
                "Try to match extracted data to these column names where possible."
            )

        system_prompt = (
            "You are an expert materials scientist data extractor. "
            "Extract alloy composition and property data from the given text. "
            "Return a JSON object with:\n"
            '- "columns": list of column names (element symbols for compositions, property names for measurements)\n'
            '- "data": list of rows, each row is a list of values in the same order as columns\n'
            '- "units": dict mapping column name to unit string\n'
            '- "notes": any relevant notes about data quality or assumptions\n\n'
            "For compositions, use element symbols (Fe, Cr, Ni, etc.). "
            "For properties, use descriptive names (Hardness_HV, Tensile_Strength_MPa, etc.). "
            "If a value is not available, use null. "
            "All numeric values should be numbers, not strings."
            f"{columns_hint}\n\n"
            "Return ONLY valid JSON."
        )

        try:
            raw = await self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                model=self.pro_model,
                json_mode=True,
            )

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{[\s\S]*\}', raw)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    return "Could not parse the extraction result. Please try again with clearer input.", None

            columns = result.get("columns", [])
            data = result.get("data", [])
            units = result.get("units", {})
            notes = result.get("notes", "")

            if not columns or not data:
                return "No data could be extracted from the provided text. Please provide text containing alloy compositions and/or measured properties.", None

            df = pd.DataFrame(data, columns=columns)

            # Build summary
            n_rows = len(df)
            n_cols = len(columns)
            elements = [c for c in columns if len(c) <= 2 and c[0].isupper()]
            properties = [c for c in columns if c not in elements]

            units_text = ""
            if units:
                units_lines = [f"  - {k}: {v}" for k, v in units.items()]
                units_text = "\n**Units:**\n" + "\n".join(units_lines)

            notes_text = f"\n**Notes:** {notes}" if notes else ""

            summary = (
                f"**Literature Extraction Complete**\n\n"
                f"Extracted **{n_rows} data points** with **{n_cols} columns** from the provided text.\n\n"
                f"**Elements detected:** {', '.join(elements) if elements else 'None'}\n"
                f"**Properties detected:** {', '.join(properties) if properties else 'None'}"
                f"{units_text}"
                f"{notes_text}\n\n"
                f"Preview (first 5 rows):\n```\n{df.head().to_string()}\n```\n\n"
                "Would you like to append this data to your current dataset?"
            )

            return summary, df

        except Exception as e:
            logger.error("Literature extraction failed: %s", e)
            return f"Error extracting data: {e}", None

    async def append_to_dataset(
        self, extracted_df: pd.DataFrame, data_path: str
    ) -> tuple[str, str]:
        """Append extracted data to an existing dataset.

        Returns (summary_message, updated_data_path).
        """
        try:
            existing_df = pd.read_csv(data_path)
            original_shape = existing_df.shape

            # Align columns: add missing columns with NaN
            for col in extracted_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            for col in existing_df.columns:
                if col not in extracted_df.columns:
                    extracted_df[col] = None

            # Reorder extracted to match existing
            extracted_df = extracted_df.reindex(columns=existing_df.columns)

            combined = pd.concat([existing_df, extracted_df], ignore_index=True)
            combined.to_csv(data_path, index=False)

            return (
                f"Data appended successfully! Dataset grew from {original_shape[0]} to "
                f"{len(combined)} rows ({len(extracted_df)} new entries added). "
                f"The updated dataset has {combined.shape[1]} columns."
            ), data_path

        except Exception as e:
            logger.error("Failed to append extracted data: %s", e)
            return f"Error appending data: {e}", data_path
