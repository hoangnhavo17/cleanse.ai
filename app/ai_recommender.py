"""
AI recommendation layer for Cleanse.

Uses Google Gemini API (free tier) to recommend cleaning actions for detected
dataset issues. Each recommendation maps to a concrete Python fix that the
user can preview and accept.

Pipeline: issue detected → AI recommendation → user previews code → accepts → fix applied
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from app.services.cleaner import _coerce_series_to_datetime

import pandas as pd
import requests
import streamlit as st

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent"
)

ALLOWED_ACTIONS = frozenset(
    {"keep", "normalize_values", "coerce_type", "title_case", "flag_for_manual_review"}
)


def _get_api_key() -> str:
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        raise RuntimeError(
            "GEMINI_API_KEY not found in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml or configure it in "
            "Streamlit Community Cloud settings."
        )
    if not key or not key.strip():
        raise RuntimeError("GEMINI_API_KEY is empty.")
    return key.strip()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_recommendation_prompt(issues: List[Dict[str, Any]]) -> str:
    # Minimal payload to stay under free-tier token limits (250k TPM, 10 RPM for 2.5 Flash)
    issues_json = json.dumps(issues, indent=2, default=str)

    return (
        "You are a data-cleaning expert. The dataset has already been preprocessed; "
        "your job is to suggest ONLY last-mile fixes for the specific issues below.\n\n"
        "For each issue, infer the column domain from the provided values "
        "(sample_values, top_value_counts, suspicious_values, unparseable_values) and "
        "choose exactly ONE action.\n\n"
        "Allowed actions: keep, normalize_values, coerce_type, title_case, flag_for_manual_review.\n"
        "- normalize_values: map listed or clearly variant values to a canonical form via fix_params.mappings.\n"
        "- coerce_type: only when almost all non-null values already parse as the target type "
        "(datetime, numeric, or string); set fix_params.target_type accordingly. Do NOT use coerce_type to "
        "fix specific bad strings when unparseable_values are provided.\n"
        "- title_case: apply title-style capitalization to a text column when appropriate, but NOT when "
        "suspicious_values are present.\n\n"
        "If unparseable_values are present (e.g. a datetime column with 2–3 bad strings), you MUST choose "
        "normalize_values and provide explicit fix_params.mappings for EACH unparseable value. Only use "
        "flag_for_manual_review when you truly cannot propose safe, concrete mappings. Do NOT use coerce_type "
        "when unparseable_values is non-empty.\n\n"
        "If suspicious_values are present for a text_quality issue (e.g. encoding glitches like 'La vita B9 bella' "
        "or obvious one-off variants like 'Italy1'), you MUST choose normalize_values and provide explicit "
        "fix_params.mappings for EACH suspicious value. Do NOT choose title_case for issues with suspicious_values.\n\n"
        "When using normalize_values, ONLY include mappings where the source and target are different "
        "(e.g. 'La vita B9 bella' → 'La vita è bella'). Do NOT include identity mappings such as "
        "'12 Angry Men' → '12 Angry Men' or 'Brazil' → 'Brazil'.\n\n"
        "Return ONLY a JSON array. Each element must be: "
        "{\"issue_id\":\"<id>\", \"inferred_domain\":\"<type>\", \"recommended_action\":\"<action>\", "
        "\"reasoning\":\"<very_short>\", \"fix_params\":{}}.\n\n"
        f"ISSUES:\n{issues_json}\n\nJSON array only."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> List[Dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "")

    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    if first_bracket == -1:
        raise ValueError(
            f"No JSON array found in model response. Raw text: {text[:500]}"
        )

    # If the closing ']' is missing or clearly malformed, fall back to using
    # everything from the first '[' onward. The fallback parser below will
    # still try to recover any well-formed objects.
    if last_bracket == -1 or last_bracket <= first_bracket:
        json_str = cleaned[first_bracket:]
    else:
        json_str = cleaned[first_bracket : last_bracket + 1]

    def _normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Each element must be a dict, got {type(item).__name__}."
                )
            if "issue_id" not in item:
                raise ValueError(f"Missing 'issue_id' in recommendation: {item}")
            if item.get("recommended_action", "") not in ALLOWED_ACTIONS:
                item["recommended_action"] = "flag_for_manual_review"
            # Keep reasoning very short; if the model omits it, default to empty.
            item.setdefault("reasoning", "")
            item.setdefault("fix_params", {})
        return items

    # First, try to parse as a proper JSON array.
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array, got {type(data).__name__}.")
        return _normalize_items(data)
    except json.JSONDecodeError:
        # Fallback: extract complete {...} objects by brace-counting so we get
        # full objects even when "fix_params": {} or nested braces exist.
        objs: List[Dict[str, Any]] = []
        i = 0
        while i < len(json_str):
            i = json_str.find("{", i)
            if i == -1:
                break
            depth = 0
            start = i
            in_string = False
            escape = False
            j = i
            while j < len(json_str):
                c = json_str[j]
                if escape:
                    escape = False
                    j += 1
                    continue
                if in_string:
                    if c == '"':
                        in_string = False
                    elif c == "\\":
                        escape = True
                    j += 1
                    continue
                if c == '"':
                    in_string = True
                    j += 1
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        fragment = json_str[start : j + 1]
                        try:
                            obj = json.loads(fragment)
                            if isinstance(obj, dict) and "issue_id" in obj:
                                objs.append(obj)
                        except json.JSONDecodeError:
                            pass
                        i = j + 1
                        break
                j += 1
            else:
                i += 1
        if not objs:
            raise ValueError(
                "Failed to parse JSON from model response (even after fallback). "
                f"Extracted: {json_str[:500]}"
            )
        return _normalize_items(objs)


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def generate_ai_recommendations(
    issues: List[Dict[str, Any]],
    timeout: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """Send all issues in a single Gemini request. Returns dict keyed by issue_id."""
    if not issues:
        return {}

    api_key = _get_api_key()
    prompt = build_recommendation_prompt(issues)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 4096,
        },
    }

    url = f"{GEMINI_ENDPOINT}?key={api_key}"

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Google Gemini API. Check your network connection."
        )
    except requests.Timeout:
        raise TimeoutError(
            f"Gemini API request timed out after {timeout}s. "
            "Try again or reduce the number of issues."
        )

    if resp.status_code == 429:
        raise RuntimeError(
            "Gemini free-tier quota exceeded (limits are per-minute: 10 requests/min and "
            "250k input tokens/min for 2.5 Flash). Wait a minute and try again, or use "
            "fewer issues. Usage: https://ai.dev/rate-limit"
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini API returned status {resp.status_code}: "
            f"{resp.text[:400]}"
        )

    body = resp.json()

    try:
        raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected Gemini response structure: {json.dumps(body)[:400]}"
        ) from exc

    recommendations = extract_json_from_response(raw_text)

    known_ids = {issue["issue_id"] for issue in issues}
    result: Dict[str, Dict[str, Any]] = {}
    for rec in recommendations:
        issue_id = rec["issue_id"]
        if issue_id not in known_ids:
            continue
        result[issue_id] = {
            "recommended_action": rec["recommended_action"],
            "reasoning": rec.get("reasoning", ""),
            "fix_params": rec.get("fix_params", {}),
            "inferred_domain": rec.get("inferred_domain", ""),
        }

    return result


# ---------------------------------------------------------------------------
# Fix code preview
# ---------------------------------------------------------------------------

def get_fix_code(
    action: str,
    column: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a readable Python code string showing what the fix will do."""
    col = column
    if action == "keep":
        return "# No changes — keeping as-is"
    if action == "flag_for_manual_review":
        return "# Flagged for manual review — no automatic changes"
    if action == "normalize_values" and col:
        mappings = (params or {}).get("mappings", {})
        if mappings:
            mapping_str = json.dumps(mappings, indent=2)
            return f'df["{col}"] = df["{col}"].replace({mapping_str})'
        return f'# normalize_values: no mappings provided for "{col}"'
    if action == "coerce_type" and col:
        target = (params or {}).get("target_type", "")
        if not target:
            return (
                "# coerce_type: infer target type from values (numeric vs datetime) and "
                f'apply an appropriate coercion for "{col}" using pandas.'
            )
        if target == "datetime":
            return f'df["{col}"] = pd.to_datetime(df["{col}"], errors="coerce")'
        if target == "numeric":
            return f'df["{col}"] = pd.to_numeric(df["{col}"], errors="coerce")'
        return f'df["{col}"] = df["{col}"].astype(str)'
    if action == "title_case" and col:
        return f'df["{col}"] = df["{col}"].astype(str).str.title()'
    return f"# Unrecognized action: {action}"


# ---------------------------------------------------------------------------
# Fix execution
# ---------------------------------------------------------------------------


def _infer_target_type_from_values(series: pd.Series) -> str:
    """
    Heuristic fallback when the model omits fix_params.target_type for coerce_type.

    - If ~90%+ of non-null values parse as numeric → "numeric"
    - Else if ~80%+ parse as datetime → "datetime"
    - Else → "string"
    """
    non_null = series.dropna()
    if non_null.empty:
        return "string"
    # Try numeric
    as_str = non_null.astype(str).str.replace(",", "").str.replace(" ", "")
    numeric = pd.to_numeric(as_str, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        return "numeric"
    # Try datetime (element-by-element to handle mixed formats)
    dt = _coerce_series_to_datetime(non_null.astype(str))
    if dt.notna().mean() >= 0.8:
        return "datetime"
    return "string"


def _try_safe_coerce_after_normalize(df: pd.DataFrame, col: str) -> None:
    """
    In-place: if column has non-null values, try datetime then numeric coercion.
    Only updates df[col] when no non-null value would become null.
    """
    if df[col].notna().sum() == 0:
        return
    coerced_dt = _coerce_series_to_datetime(df[col])
    if (coerced_dt.notna() == df[col].notna()).all():
        df[col] = coerced_dt
        return
    as_str = df[col].astype(str)
    coerced_num = pd.to_numeric(as_str, errors="coerce")
    non_null = as_str.str.strip() != ""
    if non_null.any() and coerced_num[non_null].notna().all():
        df[col] = coerced_num


def apply_fix(
    df: pd.DataFrame,
    action: str,
    column: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply the recommended fix and return a new DataFrame."""
    df = df.copy()
    col = column

    if action in ("keep", "flag_for_manual_review"):
        return df
    if action == "normalize_values" and col and col in df.columns:
        mappings = (params or {}).get("mappings", {})
        if mappings:
            df[col] = df[col].replace(mappings)
        _try_safe_coerce_after_normalize(df, col)
        return df
    if action == "coerce_type" and col and col in df.columns:
        target = (params or {}).get("target_type") or _infer_target_type_from_values(
            df[col]
        )
        if target == "datetime":
            df[col] = _coerce_series_to_datetime(df[col])
        elif target == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
        return df
    if action == "title_case" and col and col in df.columns:
        # Title-case non-null strings; leave nulls as-is (astype(str) turns NaN to 'nan', then we restore)
        mask = df[col].notna()
        ser = df.loc[mask, col].astype(str).str.strip().str.title()
        df.loc[mask, col] = ser
        return df

    return df
