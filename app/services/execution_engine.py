"""
Execution engine for applying user-approved fixes to a DataFrame.

Used by the smart app and report flow; pure, testable application of actions.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

import pandas as pd

from app.services.issues import Action, ActionKind, action_to_dict


def _apply_drop_duplicate_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    before = len(df)
    out = df.drop_duplicates()
    removed = before - len(out)
    return out, {
        "action": ActionKind.DROP_DUPLICATE_ROWS.value,
        "rows_removed": int(removed),
        "source": "user_approved",
    }


def _apply_drop_rows_where_missing(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if column not in df.columns:
        return df, {
            "action": ActionKind.DROP_ROWS_WHERE_MISSING.value,
            "column": column,
            "rows_removed": 0,
            "source": "user_approved",
            "error": "column_not_found",
        }
    before = len(df)
    out = df[df[column].notna()]
    removed = before - len(out)
    return out, {
        "action": ActionKind.DROP_ROWS_WHERE_MISSING.value,
        "column": column,
        "rows_removed": int(removed),
        "source": "user_approved",
    }


def _apply_drop_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if column not in df.columns:
        return df, {
            "action": ActionKind.DROP_COLUMN.value,
            "column": column,
            "source": "user_approved",
            "error": "column_not_found",
        }
    out = df.drop(columns=[column])
    return out, {
        "action": ActionKind.DROP_COLUMN.value,
        "column": column,
        "source": "user_approved",
    }


def _apply_fillna_constant(df: pd.DataFrame, column: str, value: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if column not in df.columns:
        return df, {
            "action": ActionKind.FILLNA_CONSTANT.value,
            "column": column,
            "source": "user_approved",
            "error": "column_not_found",
        }
    out = df.copy()
    before_missing = out[column].isna().sum()
    out[column] = out[column].fillna(value)
    after_missing = out[column].isna().sum()
    filled = before_missing - after_missing
    return out, {
        "action": ActionKind.FILLNA_CONSTANT.value,
        "column": column,
        "value": value,
        "filled_count": int(filled),
        "source": "user_approved",
    }


def _apply_cast_type(df: pd.DataFrame, column: str, target_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if column not in df.columns:
        return df, {
            "action": ActionKind.CAST_TYPE.value,
            "column": column,
            "target_type": target_type,
            "source": "user_approved",
            "error": "column_not_found",
        }
    out = df.copy()
    res: Dict[str, Any] = {
        "action": ActionKind.CAST_TYPE.value,
        "column": column,
        "target_type": target_type,
        "source": "user_approved",
    }
    try:
        if target_type == "numeric":
            out[column] = pd.to_numeric(out[column], errors="coerce")
        elif target_type == "datetime":
            out[column] = pd.to_datetime(out[column], errors="coerce", dayfirst=True)
        else:
            # Fallback: keep as string
            out[column] = out[column].astype(str)
    except Exception as exc:  # pragma: no cover - defensive logging
        res["error"] = f"cast_failed: {exc}"
        return df, res
    return out, res


def apply_actions(
    df: pd.DataFrame,
    actions: Iterable[Action],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply a sequence of Actions to a DataFrame.

    Returns (new_df, applied_action_logs) where applied_action_logs are JSON-friendly
    dicts that can be appended to the report's actions list.
    """
    out = df.copy()
    logs: List[Dict[str, Any]] = []
    for action in actions:
        kind = action.kind
        column = action.column or action.params.get("column")  # type: ignore[union-attr]
        params = action.params

        if kind == ActionKind.DROP_DUPLICATE_ROWS.value:
            out, log = _apply_drop_duplicate_rows(out)
        elif kind == ActionKind.DROP_ROWS_WHERE_MISSING.value and column is not None:
            out, log = _apply_drop_rows_where_missing(out, column)
        elif kind == ActionKind.DROP_COLUMN.value and column is not None:
            out, log = _apply_drop_column(out, column)
        elif kind == ActionKind.FILLNA_CONSTANT.value and column is not None:
            value = params.get("value")
            if value is None:
                log = {
                    "action": ActionKind.FILLNA_CONSTANT.value,
                    "column": column,
                    "source": "user_approved",
                    "error": "missing_value_param",
                }
            else:
                out, log = _apply_fillna_constant(out, column, value)
        elif kind == ActionKind.CAST_TYPE.value and column is not None:
            target = params.get("target_type", "numeric")
            out, log = _apply_cast_type(out, column, target)
        else:
            # Unknown or unsupported action; record and skip.
            log = {
                "action": kind,
                "column": column,
                "source": "user_approved",
                "error": "unsupported_action",
                "raw": action_to_dict(action),
            }
        logs.append(log)
    return out, logs

