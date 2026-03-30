"""Alloy-specific auto-featurization for AlloyGen 2.0.

Key differentiator: detects element composition columns automatically,
computes physics-informed alloy parameters (VEC, δ, ΔS_mix, ΔH_mix, Ω),
and performs automatic feature selection for small datasets.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Periodic table data (subset relevant to alloys)
# ---------------------------------------------------------------------------
ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
}

# Atomic weights (g/mol)
ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95,
    "K": 39.10, "Ca": 40.08, "Sc": 44.96, "Ti": 47.87, "V": 50.94,
    "Cr": 52.00, "Mn": 54.94, "Fe": 55.85, "Co": 58.93, "Ni": 58.69,
    "Cu": 63.55, "Zn": 65.38, "Ga": 69.72, "Ge": 72.63, "As": 74.92,
    "Se": 78.97, "Br": 79.90, "Kr": 83.80,
    "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22, "Nb": 92.91,
    "Mo": 95.95, "Tc": 97.0, "Ru": 101.1, "Rh": 102.9, "Pd": 106.4,
    "Ag": 107.9, "Cd": 112.4, "In": 114.8, "Sn": 118.7, "Sb": 121.8,
    "Te": 127.6, "I": 126.9, "Xe": 131.3,
    "Cs": 132.9, "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9,
    "Nd": 144.2, "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3,
    "Tb": 158.9, "Dy": 162.5, "Ho": 164.9, "Er": 167.3, "Tm": 168.9,
    "Yb": 173.0, "Lu": 175.0, "Hf": 178.5, "Ta": 180.9, "W": 183.8,
    "Re": 186.2, "Os": 190.2, "Ir": 192.2, "Pt": 195.1, "Au": 197.0,
    "Hg": 200.6, "Tl": 204.4, "Pb": 207.2, "Bi": 209.0,
}

# Atomic radii (pm) — metallic/empirical
ATOMIC_RADIUS = {
    "Li": 152, "Be": 112, "B": 87, "C": 77, "N": 75,
    "Na": 186, "Mg": 160, "Al": 143, "Si": 117, "P": 110,
    "K": 227, "Ca": 197, "Sc": 162, "Ti": 147, "V": 134,
    "Cr": 128, "Mn": 127, "Fe": 126, "Co": 125, "Ni": 124,
    "Cu": 128, "Zn": 134, "Ga": 135, "Ge": 122,
    "Rb": 248, "Sr": 215, "Y": 180, "Zr": 160, "Nb": 146,
    "Mo": 139, "Tc": 136, "Ru": 134, "Rh": 134, "Pd": 137,
    "Ag": 144, "Cd": 151, "In": 167, "Sn": 140, "Sb": 140,
    "Cs": 265, "Ba": 222, "La": 187, "Ce": 182, "Pr": 182,
    "Nd": 181, "Sm": 180, "Eu": 180, "Gd": 180, "Tb": 177,
    "Dy": 178, "Ho": 176, "Er": 176, "Tm": 176, "Yb": 176,
    "Lu": 174, "Hf": 159, "Ta": 146, "W": 139, "Re": 137,
    "Os": 135, "Ir": 136, "Pt": 139, "Au": 144, "Hg": 151,
    "Tl": 170, "Pb": 175, "Bi": 156,
}

# Valence electron concentration
VEC_VALUES = {
    "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5,
    "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Ga": 3, "Ge": 4,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5,
    "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
    "Ag": 11, "Cd": 12, "In": 3, "Sn": 4, "Sb": 5,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Pr": 5,
    "Nd": 6, "Sm": 8, "Eu": 9, "Gd": 10, "Tb": 11,
    "Dy": 12, "Ho": 13, "Er": 14, "Tm": 15, "Yb": 16,
    "Lu": 3, "Hf": 4, "Ta": 5, "W": 6, "Re": 7,
    "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Hg": 12,
    "Tl": 3, "Pb": 4, "Bi": 5,
}

# Miedema parameters for mixing enthalpy estimation
# (electronegativity φ, electron density n_ws^{1/3}, molar volume V_m)
MIEDEMA = {
    "Al": (4.20, 1.39, 10.0), "Ti": (3.65, 1.47, 10.6), "V": (4.25, 1.64, 8.4),
    "Cr": (4.65, 1.73, 7.2), "Mn": (4.45, 1.61, 7.4), "Fe": (4.93, 1.77, 7.1),
    "Co": (5.10, 1.75, 6.7), "Ni": (5.20, 1.75, 6.6), "Cu": (4.55, 1.47, 7.1),
    "Zn": (4.10, 1.32, 9.2), "Zr": (3.40, 1.39, 14.0), "Nb": (4.05, 1.62, 10.8),
    "Mo": (4.65, 1.77, 9.4), "Pd": (5.45, 1.67, 8.9), "Ag": (4.45, 1.39, 10.3),
    "Hf": (3.55, 1.43, 13.4), "Ta": (4.05, 1.63, 10.8), "W": (4.80, 1.81, 9.5),
    "Re": (5.20, 1.86, 8.9), "Pt": (5.65, 1.78, 9.1), "Au": (5.15, 1.57, 10.2),
}

# Gas constant (J/mol·K)
R = 8.314


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
def _strip_element_suffix(col: str) -> Optional[str]:
    """Extract element symbol from column names like 'Fe_wt%', 'Cr_at%', 'Ni_ppm'."""
    # Try exact match first
    if col in ELEMENTS:
        return col
    # Strip common suffixes
    m = re.match(r"^([A-Z][a-z]?)(?:_(?:wt|at|mol|mass)?%?|_ppm|_frac)?$", col, re.IGNORECASE)
    if m and m.group(1) in ELEMENTS:
        return m.group(1)
    return None


def detect_element_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detect columns that represent element compositions.

    Returns dict mapping column_name -> element_symbol.
    """
    element_cols = {}
    for col in df.columns:
        sym = _strip_element_suffix(col.strip())
        if sym is not None and pd.api.types.is_numeric_dtype(df[col]):
            element_cols[col] = sym
    return element_cols


def detect_composition_format(df: pd.DataFrame, element_cols: dict[str, str]) -> str:
    """Detect if compositions are in atomic fraction, weight%, or ppm.

    Returns: 'atomic_fraction', 'weight_percent', 'atomic_percent', or 'ppm'.
    """
    if not element_cols:
        return "unknown"
    cols = list(element_cols.keys())
    row_sums = df[cols].sum(axis=1).dropna()
    if len(row_sums) == 0:
        return "unknown"
    median_sum = row_sums.median()
    if median_sum < 2.0:
        return "atomic_fraction"
    elif 50 < median_sum < 150:
        return "weight_percent"
    elif median_sum > 10000:
        return "ppm"
    else:
        return "weight_percent"  # default


def to_atomic_fraction(
    df: pd.DataFrame,
    element_cols: dict[str, str],
    fmt: str,
) -> pd.DataFrame:
    """Convert element columns to atomic fractions (summing to ~1.0)."""
    result = df.copy()
    cols = list(element_cols.keys())

    if fmt == "atomic_fraction":
        return result  # already good

    if fmt == "ppm":
        for col in cols:
            result[col] = result[col] / 1e6

    elif fmt in ("weight_percent", "atomic_percent"):
        for col in cols:
            result[col] = result[col] / 100.0

    if fmt == "weight_percent":
        # Convert weight fraction to atomic fraction
        for idx in result.index:
            moles = {}
            for col, sym in element_cols.items():
                wt = result.at[idx, col]
                aw = ATOMIC_WEIGHT.get(sym, 1.0)
                moles[col] = wt / aw if aw > 0 else 0
            total_moles = sum(moles.values())
            if total_moles > 0:
                for col in cols:
                    result.at[idx, col] = moles[col] / total_moles

    # Normalize to sum to 1.0
    for idx in result.index:
        s = sum(result.at[idx, c] for c in cols)
        if s > 0:
            for c in cols:
                result.at[idx, c] = result.at[idx, c] / s

    return result


# ---------------------------------------------------------------------------
# Alloy physics parameters
# ---------------------------------------------------------------------------
def compute_vec(row: pd.Series, element_cols: dict[str, str]) -> float:
    """Valence Electron Concentration — weighted average of VEC values.
    VEC > 8 → FCC, VEC < 6.87 → BCC (Guo et al., 2011).
    """
    vec = 0.0
    for col, sym in element_cols.items():
        x = row.get(col, 0.0) or 0.0
        vec += x * VEC_VALUES.get(sym, 0)
    return vec


def compute_delta(row: pd.Series, element_cols: dict[str, str]) -> float:
    """Atomic size mismatch parameter δ (%).
    δ < 6.6% favors solid solution; δ > 6.6% → intermetallics (Zhang et al., 2008).
    """
    r_avg = 0.0
    for col, sym in element_cols.items():
        x = row.get(col, 0.0) or 0.0
        r_avg += x * ATOMIC_RADIUS.get(sym, 0)
    if r_avg == 0:
        return 0.0
    delta_sq = 0.0
    for col, sym in element_cols.items():
        x = row.get(col, 0.0) or 0.0
        r = ATOMIC_RADIUS.get(sym, 0)
        if r > 0:
            delta_sq += x * (1 - r / r_avg) ** 2
    return 100 * np.sqrt(delta_sq)


def compute_entropy_mixing(row: pd.Series, element_cols: dict[str, str]) -> float:
    """Configurational mixing entropy ΔS_mix (J/mol·K).
    ΔS_mix = -R * Σ(x_i * ln(x_i)).
    HEA definition: ΔS_mix ≥ 1.5R ≈ 12.47 J/mol·K.
    """
    s = 0.0
    for col, _ in element_cols.items():
        x = row.get(col, 0.0) or 0.0
        if x > 1e-10:
            s -= x * np.log(x)
    return R * s


def compute_enthalpy_mixing(row: pd.Series, element_cols: dict[str, str]) -> float:
    """Miedema mixing enthalpy ΔH_mix (kJ/mol).
    Approximation: ΔH_mix = Σ_i Σ_{j>i} 4 * Ω_ij * x_i * x_j
    where Ω_ij is estimated from Miedema electronegativities.
    """
    items = [(col, sym) for col, sym in element_cols.items()
             if sym in MIEDEMA and (row.get(col, 0.0) or 0.0) > 1e-10]
    h_mix = 0.0
    for i, (col_i, sym_i) in enumerate(items):
        x_i = row.get(col_i, 0.0) or 0.0
        phi_i, n_i, v_i = MIEDEMA[sym_i]
        for col_j, sym_j in items[i + 1:]:
            x_j = row.get(col_j, 0.0) or 0.0
            phi_j, n_j, v_j = MIEDEMA[sym_j]
            # Simplified Miedema: ΔH ∝ -(Δφ)² + Q/P*(Δn^{1/3})²
            delta_phi = phi_i - phi_j
            delta_n = n_i - n_j
            omega_ij = -delta_phi ** 2 + 0.115 * delta_n ** 2  # simplified
            h_mix += 4 * omega_ij * x_i * x_j
    return h_mix


def compute_omega(
    row: pd.Series,
    element_cols: dict[str, str],
    t_m: Optional[float] = None,
) -> float:
    """Yang Ω parameter — thermodynamic stability indicator.
    Ω = T_m * ΔS_mix / |ΔH_mix|. Ω > 1.1 and δ < 6.6% → solid solution (Yang & Zhang, 2012).
    """
    if t_m is None:
        # Estimate T_m as composition-weighted average of melting points
        TM_VALUES = {
            "Al": 933, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519,
            "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 693,
            "Zr": 2128, "Nb": 2750, "Mo": 2896, "Pd": 1828, "Ag": 1235,
            "Hf": 2506, "Ta": 3290, "W": 3695, "Re": 3459, "Pt": 2041,
            "Au": 1337, "Sn": 505, "Sb": 904,
        }
        t_m = sum(
            (row.get(col, 0.0) or 0.0) * TM_VALUES.get(sym, 1500)
            for col, sym in element_cols.items()
        )

    ds = compute_entropy_mixing(row, element_cols)
    dh = compute_enthalpy_mixing(row, element_cols)
    if abs(dh) < 1e-10:
        return 100.0  # effectively infinite stability
    return t_m * ds / (abs(dh) * 1000)  # convert dh kJ→J


def compute_alloy_features(
    df: pd.DataFrame,
    element_cols: dict[str, str],
) -> pd.DataFrame:
    """Add alloy physics features to a DataFrame with atomic-fraction composition columns."""
    features = pd.DataFrame(index=df.index)

    features["VEC"] = df.apply(lambda r: compute_vec(r, element_cols), axis=1)
    features["delta_r"] = df.apply(lambda r: compute_delta(r, element_cols), axis=1)
    features["dS_mix"] = df.apply(lambda r: compute_entropy_mixing(r, element_cols), axis=1)
    features["dH_mix"] = df.apply(lambda r: compute_enthalpy_mixing(r, element_cols), axis=1)
    features["Omega"] = df.apply(lambda r: compute_omega(r, element_cols), axis=1)

    # Derived features
    features["n_elements"] = df[list(element_cols.keys())].gt(1e-10).sum(axis=1)

    logger.info(
        "Computed 6 alloy physics features (VEC, δ, ΔS_mix, ΔH_mix, Ω, n_elements) "
        "for %d rows with %d elements", len(df), len(element_cols)
    )
    return features


# ---------------------------------------------------------------------------
# Feature selection for small datasets
# ---------------------------------------------------------------------------
def auto_select_features(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    max_features: Optional[int] = None,
) -> list[str]:
    """Automatic feature selection for small materials datasets.

    Steps:
    1. Remove zero/near-zero variance features
    2. Remove highly correlated features (|r| > 0.95)
    3. Tree-based importance ranking → keep top features
    """
    from sklearn.ensemble import RandomForestRegressor

    if max_features is None:
        max_features = min(30, max(5, len(X) // 5))

    selected = list(X.columns)

    # 1. Remove near-zero variance
    variances = X[selected].var()
    low_var = variances[variances < 1e-10].index.tolist()
    selected = [c for c in selected if c not in low_var]
    if low_var:
        logger.info("Removed %d near-zero variance features", len(low_var))

    if len(selected) <= max_features:
        return selected

    # 2. Remove highly correlated features (keep first of each pair)
    corr_matrix = X[selected].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > 0.95].tolist()
        if correlated:
            to_drop.update(correlated)
    selected = [c for c in selected if c not in to_drop]
    if to_drop:
        logger.info("Removed %d highly correlated features", len(to_drop))

    if len(selected) <= max_features:
        return selected

    # 3. Tree-based importance ranking
    X_sel = X[selected].fillna(0)
    y_flat = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=5)
    rf.fit(X_sel, y_flat)
    importances = pd.Series(rf.feature_importances_, index=selected)
    top = importances.nlargest(max_features).index.tolist()
    logger.info("Selected top %d features by importance (from %d)", len(top), len(selected))
    return top


# ---------------------------------------------------------------------------
# Main featurization pipeline
# ---------------------------------------------------------------------------
def featurize_alloy_dataset(
    df: pd.DataFrame,
    target_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Auto-detect composition columns, compute alloy features, and select features.

    Returns:
        (featurized_df, element_cols, new_feature_names)
    """
    element_cols = detect_element_columns(df)
    if not element_cols:
        logger.info("No element composition columns detected — skipping alloy featurization")
        return df, {}, []

    logger.info(
        "Detected %d element columns: %s",
        len(element_cols),
        ", ".join(f"{col}→{sym}" for col, sym in element_cols.items()),
    )

    # Detect format and convert to atomic fractions
    fmt = detect_composition_format(df, element_cols)
    logger.info("Detected composition format: %s", fmt)

    df_converted = to_atomic_fraction(df, element_cols, fmt)

    # Compute alloy physics features
    alloy_features = compute_alloy_features(df_converted, element_cols)
    new_feature_names = list(alloy_features.columns)

    # Add features to dataset
    result = df.copy()
    for col in alloy_features.columns:
        result[col] = alloy_features[col]

    return result, element_cols, new_feature_names
