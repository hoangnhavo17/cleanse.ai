"""
Profile-based automation: type fixes (coerce inferred numeric), duplicate removal.
Used after standard cleaning (clean()) in the pipeline. No imputation — missing values stay missing.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.profiler import INFERRED_NUMERIC, profile_dataset


def apply_duplicate_removal(df: pd.DataFrame, actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Remove duplicate rows; record count removed."""
    n_before = len(df)
    out = df.drop_duplicates()
    n_removed = n_before - len(out)
    if n_removed > 0:
        actions.append({
            "action": "remove_duplicates",
            "rows_removed": n_removed,
            "safe": True,
        })
    return out


def apply_type_fixes(df: pd.DataFrame, profile: dict[str, Any], actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Coerce object columns that profile says are numeric into numeric type."""
    out = df.copy()
    for cp in profile.get("column_profiles", []):
        col = cp.get("column")
        if col not in out.columns:
            continue
        if cp.get("inferred_type") != INFERRED_NUMERIC:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        try:
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().any():
                from app.services.cleaner import _prefer_int_if_whole
                out[col] = _prefer_int_if_whole(coerced)
                actions.append({
                    "action": "coerce_numeric",
                    "column": col,
                    "safe": True,
                })
        except Exception:
            continue
    return out


def apply_automated_rules(
    df: pd.DataFrame,
    profile: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply profile-based rules: type fixes (object → numeric where inferred), duplicate removal.
    No imputation — missing values are left as-is.
    """
    if profile is None:
        profile = profile_dataset(df)
    actions: list[dict[str, Any]] = []
    out = df.copy()
    out = apply_type_fixes(out, profile, actions)
    out = apply_duplicate_removal(out, actions)
    return out, actions
