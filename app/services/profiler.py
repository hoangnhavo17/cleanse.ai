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


def _numeric_outliers_zscore(
    series: pd.Series, z_threshold: float = 3.0
) -> dict[str, Any]:
    """
    Outlier stats using z-score.

    Uses population std (ddof=0) and counts values where |z| > z_threshold.
    Intended for profiling only — never auto-deletes rows.
    """
    if series.empty:
        return {"z_threshold": float(z_threshold), "count": 0, "max_abs_z": 0.0}
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return {"z_threshold": float(z_threshold), "count": 0, "max_abs_z": 0.0}
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return {"z_threshold": float(z_threshold), "count": 0, "max_abs_z": 0.0}
    z = (s - mean) / std
    abs_z = z.abs()
    mask = abs_z > z_threshold
    return {
        "z_threshold": float(z_threshold),
        "count": int(mask.sum()),
        "max_abs_z": float(abs_z.max()) if not abs_z.empty else 0.0,
    }


def profile_column(series: pd.Series, name: str) -> dict[str, Any]:
    """Profile a single column: missing %, dtype, inferred type, cardinality, and outliers (if numeric)."""
    total = len(series)
    missing = series.isna().sum()
    missing_pct = round(100.0 * missing / total, 2) if total else 0.0
    non_null = series.dropna()
    non_null_count = len(non_null)
    cardinality = int(non_null.nunique()) if non_null_count else 0
    inferred = _infer_column_type(series)
    out: dict[str, Any] = {
        "column": name,
        "dtype": str(series.dtype),
        "inferred_type": inferred,
        "missing_count": int(missing),
        "missing_pct": missing_pct,
        "cardinality": cardinality,
        "non_null_count": int(non_null_count),
    }
    if inferred == INFERRED_NUMERIC and pd.api.types.is_numeric_dtype(series):
        out["outliers_iqr"] = _numeric_outliers_iqr(series)
        out["outliers_zscore"] = _numeric_outliers_zscore(series)
    elif inferred == INFERRED_NUMERIC and not pd.api.types.is_numeric_dtype(series):
        # Try to get numeric for IQR / z-score
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            cleaned = coerced.dropna()
            out["outliers_iqr"] = _numeric_outliers_iqr(cleaned)
            out["outliers_zscore"] = _numeric_outliers_zscore(cleaned)
    # Type validation: numeric-looking but high cardinality → likely code/identifier (e.g. zip_code)
    if inferred == INFERRED_NUMERIC and non_null_count >= 20:
        unique_ratio = cardinality / max(1, non_null_count)
        if unique_ratio > 0.5 and cardinality >= 20:
            out["numeric_but_categorical"] = {
                "cardinality": cardinality,
                "non_null_count": int(non_null_count),
                "unique_ratio": round(unique_ratio, 3),
            }
    return out


def _missingness_pattern_for_column(df: pd.DataFrame, col: str) -> dict[str, Any] | None:
    """
    If column has meaningful missingness, find the (other_column, value) for which
    this column is missing most often. Suggests missingness may be meaningful (e.g.
    salary missing when department = 'intern') rather than random.

    Returns None if no strong pattern; otherwise
    {when_column, when_value, missing_rate_when, overall_missing_pct}.
    """
    if col not in df.columns:
        return None
    missing = df[col].isna()
    n_missing = missing.sum()
    n_rows = len(df)
    if n_missing < 10 or n_rows < 20:
        return None
    overall_pct = 100.0 * n_missing / n_rows
    best_other: str | None = None
    best_value: Any = None
    best_rate: float = 0.0
    best_count: int = 0
    for other in df.columns:
        if other == col:
            continue
        # Prefer low-cardinality (categorical) columns for interpretability
        other_series = df[other].dropna().astype(str)
        if other_series.nunique() > 80:
            continue
        for val in other_series.unique():
            mask = (df[other].astype(str) == val)
            if mask.sum() < 10:
                continue
            rate = 100.0 * missing[mask].sum() / mask.sum()
            if rate > best_rate:
                best_rate = rate
                best_other = other
                best_value = val
                best_count = int(mask.sum())
    if best_other is None or best_count < 10:
        return None
    # Only report if missing rate in that segment is much higher than overall
    if best_rate < overall_pct * 1.2 or best_rate < 20.0 or (best_rate - overall_pct) < 15.0:
        return None
    return {
        "when_column": best_other,
        "when_value": str(best_value),
        "missing_rate_when": round(best_rate, 1),
        "overall_missing_pct": round(overall_pct, 1),
    }


def _find_redundant_column_pairs(
    df: pd.DataFrame, min_corr: float = 0.99
) -> list[dict[str, Any]]:
    """
    Find pairs of numeric columns with correlation ≈ 1 or ≈ -1 (redundant or derived).
    Returns list of {col_a, col_b, correlation} for |corr| >= min_corr.
    """
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric) < 2:
        return []
    try:
        corr = df[numeric].corr()
    except Exception:
        return []
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, a in enumerate(numeric):
        for b in numeric[i + 1 :]:
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if abs(r) >= min_corr:
                seen.add(key)
                pairs.append({"col_a": a, "col_b": b, "correlation": round(float(r), 4)})
    return pairs


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Full dataset profile: shape, duplicates, per-column stats, missingness patterns, redundant pairs."""
    n_rows, n_cols = df.shape
    duplicate_count = int(df.duplicated().sum())
    columns = []
    for c in df.columns:
        cp = profile_column(df[c], str(c))
        missing_pct = float(cp.get("missing_pct", 0) or 0)
        if missing_pct >= 5.0:
            pattern = _missingness_pattern_for_column(df, str(c))
            if pattern:
                cp["missingness_pattern"] = pattern
        columns.append(cp)
    redundant_pairs = _find_redundant_column_pairs(df)
    return {
        "rows": n_rows,
        "columns": n_cols,
        "duplicate_rows": duplicate_count,
        "column_profiles": columns,
        "redundant_pairs": redundant_pairs,
    }


def profile_to_json(profile: dict[str, Any]) -> str:
    """Serialize profile to JSON string (for report embedding)."""
    return json.dumps(profile, indent=2)
