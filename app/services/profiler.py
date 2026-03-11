"""
Data profiling. Detects missing %, duplicates, inferred types, cardinality, basic outliers.
Pure functions; no side effects.
"""
from __future__ import annotations

import json
import warnings
from typing import Any

import pandas as pd


# Types we infer for automation
INFERRED_NUMERIC = "numeric"
INFERRED_CATEGORICAL = "categorical"
INFERRED_DATETIME = "datetime"
INFERRED_TEXT = "text"


def _infer_column_type(series: pd.Series) -> str:
    """Infer high-level type: numeric, categorical, datetime, or text."""
    if pd.api.types.is_numeric_dtype(series):
        return INFERRED_NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return INFERRED_DATETIME
    # Object/string
    non_null = series.dropna()
    if len(non_null) == 0:
        return INFERRED_TEXT
    # Try numeric coercion
    coerced = pd.to_numeric(non_null.astype(str).str.replace(",", "").str.replace(" ", ""), errors="coerce")
    if coerced.notna().mean() >= 0.9:
        return INFERRED_NUMERIC
    # Try datetime (element-by-element so mixed formats are handled)
    try:
        from app.services.cleaner import _coerce_series_to_datetime
        dt = _coerce_series_to_datetime(non_null.astype(str))
        if dt.notna().mean() >= 0.8:
            return INFERRED_DATETIME
    except Exception:
        pass
    # Low cardinality -> categorical
    n_unique = non_null.nunique()
    if n_unique <= 50 and len(non_null) >= 10 and n_unique <= len(non_null) // 2:
        return INFERRED_CATEGORICAL
    return INFERRED_TEXT


def _numeric_outliers_iqr(series: pd.Series) -> dict[str, Any]:
    """Basic outlier stats using IQR. Returns low/high bounds and count outside."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"lower_bound": float(q1), "upper_bound": float(q3), "count_below": 0, "count_above": 0}
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    below = (series < low).sum()
    above = (series > high).sum()
    return {
        "lower_bound": float(low),
        "upper_bound": float(high),
        "count_below": int(below),
        "count_above": int(above),
    }


def profile_column(series: pd.Series, name: str) -> dict[str, Any]:
    """Profile a single column: missing %, dtype, inferred type, cardinality, and outliers (if numeric)."""
    total = len(series)
    missing = series.isna().sum()
    missing_pct = round(100.0 * missing / total, 2) if total else 0.0
    non_null = series.dropna()
    cardinality = int(non_null.nunique()) if len(non_null) else 0
    inferred = _infer_column_type(series)
    out: dict[str, Any] = {
        "column": name,
        "dtype": str(series.dtype),
        "inferred_type": inferred,
        "missing_count": int(missing),
        "missing_pct": missing_pct,
        "cardinality": cardinality,
        "non_null_count": int(len(non_null)),
    }
    if inferred == INFERRED_NUMERIC and pd.api.types.is_numeric_dtype(series):
        out["outliers_iqr"] = _numeric_outliers_iqr(series)
    elif inferred == INFERRED_NUMERIC and not pd.api.types.is_numeric_dtype(series):
        # Try to get numeric for IQR
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            out["outliers_iqr"] = _numeric_outliers_iqr(coerced.dropna())
    return out


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Full dataset profile: shape, duplicates, and per-column stats."""
    n_rows, n_cols = df.shape
    duplicate_count = int(df.duplicated().sum())
    columns = []
    for c in df.columns:
        columns.append(profile_column(df[c], str(c)))
    return {
        "rows": n_rows,
        "columns": n_cols,
        "duplicate_rows": duplicate_count,
        "column_profiles": columns,
    }


def profile_to_json(profile: dict[str, Any]) -> str:
    """Serialize profile to JSON string (for report embedding)."""
    return json.dumps(profile, indent=2)
