from __future__ import annotations
import re
from typing import Optional


def normalize_column_name(name: str) -> str:
    """Normalize a column/target name for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r"[\s_\-]+", "", name)
    name = re.sub(r"(mpa|gpa|pct|%|°c|hv|hrc)$", "", name)
    return name


def find_target_column(columns: list[str], target: str) -> Optional[str]:
    """Return the best-matching column name for target, or None."""
    if target in columns:
        return target
    norm_target = normalize_column_name(target)
    for col in columns:
        if normalize_column_name(col) == norm_target:
            return col
    for col in columns:
        norm_col = normalize_column_name(col)
        if norm_target in norm_col or norm_col in norm_target:
            return col
    return None
