from __future__ import annotations

import io
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.cleaner import clean, _coerce_series_to_datetime
from app.services.csv_io import load_csv
from app.services.issues import Issue, IssueCategory, compute_quality_score, detect_issues_from_profile
from app.services.profiler import profile_dataset
from app.ai_recommender import apply_fix, generate_ai_recommendations, get_fix_code

# Cap issues sent to Gemini per request (free-tier token limits).
MAX_ISSUES_PER_REQUEST = 8


# --- Helpers -----------------------------------------------------------------

DEMO_DATASETS: Dict[str, Path] = {
    "Warehouse Inventory": Path("data/warehouse_messy_data.csv"),
    "Healthcare Records": Path("data/healthcare_messy_data.csv"),
    "HR Data": Path("data/messy_HR_data.csv"),
    "Movie Ratings": Path("data/messy_IMDB_dataset.csv"),
}


def _available_demo_datasets() -> Dict[str, Path]:
    return {label: path for label, path in DEMO_DATASETS.items() if path.exists()}


def _preview_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with null-like values shown as empty cells."""
    if df.empty:
        return df
    null_like = {"na", "nan", "n/a", "none", "null", "#n/a", "-", "<na>", "nat"}
    out = df.astype(str)
    for col in out.columns:
        mask = out[col].str.strip().str.lower().isin(null_like)
        out[col] = out[col].mask(mask, "")
    return out


def _read_csv_robust(src: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(src)
    except UnicodeDecodeError:
        return pd.read_csv(src, encoding="latin1")


def _build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "non_null", "missing_pct"])
    # Use the same null semantics as preprocessing (standardize_nulls) so
    # missing percentages in steps 1 and 3 are consistent.
    from app.services.cleaner import standardize_nulls

    df_norm = standardize_nulls(df.copy())
    non_null = df_norm.notna().sum()
    missing_pct = (df_norm.isna().sum() / len(df_norm) * 100).round(1)
    return pd.DataFrame({
        "column": df_norm.columns,
        "dtype": df_norm.dtypes.astype(str).values,
        "non_null": non_null.values,
        "missing_pct": missing_pct.values,
    })


def _format_df_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with datetime columns written as YYYY-MM-DD (no time).
    Avoids exporting '2024-02-14 00:00:00'; NaT becomes empty string.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d").astype(str).replace("NaT", "")
    return out


def _summarize_cleaning(df_raw: pd.DataFrame, df_after: pd.DataFrame) -> list[str]:
    """Return a terse summary of preprocessing changes that actually affected the data."""
    summaries: list[str] = []
    rows_before, cols_before = df_raw.shape
    rows_after, cols_after = df_after.shape
    if rows_after != rows_before:
        summaries.append(f"rows {rows_before} → {rows_after}")
    if cols_after != cols_before:
        summaries.append(f"columns {cols_before} → {cols_after}")
    dropped_cols = [c for c in df_raw.columns if c not in df_after.columns]
    if dropped_cols:
        summaries.append(f"dropped {len(dropped_cols)} empty column(s)")
    # Dtype changes (object → numeric/datetime/string)
    type_changes: list[str] = []
    for c in df_after.columns:
        if c in df_raw.columns:
            before = str(df_raw[c].dtype)
            after = str(df_after[c].dtype)
            if before != after:
                type_changes.append(f"{c}: {before}→{after}")
    if type_changes:
        summaries.append("types changed in " + ", ".join(type_changes))
    return summaries


def _is_id_like_column(df: pd.DataFrame, col: str) -> bool:
    """
    True if the column values look like identifiers (any format). Value-based only.
    IDs are typically: single token (no spaces), bounded length, alphanumeric + _-,
    and uniform length across values.
    """
    if col not in df.columns:
        return False
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""]
    if len(series) < 5:
        return False
    s = series.astype(str)
    # Single token, no spaces
    if not s.str.match(r"^\S+$", na=False).all():
        return False
    # Bounded length (IDs are not long free text)
    if (s.str.len() > 64).any():
        return False
    # Common ID charset: letters, digits, underscore, hyphen
    if not s.str.match(r"^[a-zA-Z0-9_-]+$", na=False).all():
        return False
    # Uniform length: at most a few distinct lengths, majority share the same length
    lengths = s.str.len()
    if lengths.nunique() > 3:
        return False
    mode_count = lengths.value_counts().iloc[0]
    if mode_count / len(lengths) < 0.85:
        return False
    return True


def _is_text_column_consistent(df: pd.DataFrame, col: str) -> bool:
    """
    Heuristic: categorical/text column already looks consistent, so we can
    safely skip sending this column to Gemini.

    Definition of "consistent" here:
    - After lowercasing and normalizing whitespace, each normalized value
      maps to exactly one original spelling (no variants like 'usa' vs 'USA').
    """
    if col not in df.columns:
        return True
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return True
    # ID-like columns (ttXXXXXXX, etc.) are not free text — skip title_case.
    if _is_id_like_column(df, col):
        return True
    # Email-like columns (user@domain.tld) are also structured identifiers. We
    # treat them as already consistent and skip sending them for text-quality
    # suggestions.
    if series.str.contains(r"@.+\.", regex=True).mean() >= 0.5:
        return True
    # Always send all-lowercase columns to Gemini so it can decide on title_case
    # (e.g. product names like "gadget y", "widget a").
    if series.str.islower().all():
        return False
    # If most values are pure text but some contain digits, that suggests
    # stray characters (e.g. "Italy1", "La Vita B9 Bella") — send to Gemini.
    has_digit = series.str.contains(r"\d", regex=True)
    digit_share = has_digit.mean()
    if 0 < digit_share < 0.5:
        return False
    # Normalize case and whitespace.
    norm = (
        series.str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # Build mapping from normalized value -> set of raw spellings.
    groups: dict[str, set[str]] = {}
    for raw, key in zip(series.tolist(), norm.tolist()):
        groups.setdefault(key, set()).add(raw)
    # If any normalized key has more than one spelling, there are variants.
    for spellings in groups.values():
        if len(spellings) > 1:
            return False
    return True


_MOJIBAKE_PATTERN = re.compile(r"Ã[\x80-\xbf]")
_HEX_BYTE_IN_TEXT = re.compile(
    # Require at least one digit so we don't accidentally match common
    # two-letter words like "de", "da", "be" as hex-byte artifacts.
    r"(?<=[A-Za-z] )(?:[89][0-9A-Fa-f]|[A-Fa-f][0-9])(?= [A-Za-z])"
)


def _detect_suspicious_values(df: pd.DataFrame, col: str) -> list[str]:
    """
    For a text/categorical column, return a short list of suspicious values
    that likely need manual review or AI-assisted mapping.

    Heuristics (all value-based, no domain vocab):
    1. Encoding artifacts:
       a. Mojibake (UTF-8 → Latin-1 round-trip), e.g. 'LÃ©on'.
       b. Leaked hex bytes from Latin-1/CP1252 embedded between normal
          words, e.g. 'La vita B9 bella' (B9 = è in CP1252).
    2. Stray digits appended/prepended to otherwise alphabetic values
       in a column where most values are purely alphabetic
       (e.g. 'Italy1' in a Country column). Skipped when the column
       naturally contains digits (e.g. movie titles like 'Se7en').
    3. Punctuation-only variants: values that, after stripping
       non-alphanumeric characters, match a more frequent value
       (e.g. 'US.' as a variant of 'US').
    4. Low-frequency values (possible typos / rare variants).
       Apply only for small/medium-cardinality categoricals; skip
       very high-cardinality columns (titles, free text).
    """
    if col not in df.columns:
        return []
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return []
    if _is_id_like_column(df, col):
        return []

    suspicious: list[str] = []
    vc = series.value_counts()

    # 1a. Mojibake: Ã followed by a high byte (UTF-8 → Latin-1 corruption)
    has_mojibake = series.str.contains(_MOJIBAKE_PATTERN)
    if has_mojibake.any():
        suspicious.extend(
            series[has_mojibake].value_counts().head(5).index.tolist()
        )

    # 1b. Leaked hex bytes: an isolated 2-char hex token (80–FF range)
    #     sitting between normal alphabetic words, e.g. "La vita B9 bella".
    has_hex_byte = series.str.contains(_HEX_BYTE_IN_TEXT)
    if has_hex_byte.any():
        suspicious.extend(
            series[has_hex_byte].value_counts().head(5).index.tolist()
        )

    # 2. Stray trailing/leading digits on otherwise-alpha values, but ONLY
    #    when the column is mostly digit-free (avoids flagging legit titles
    #    like 'Se7en', '12 Angry Men', 'Terminator 2').
    has_digit = series.str.contains(r"\d", regex=True)
    digit_share = has_digit.mean()
    if 0 < digit_share < 0.3:
        stray_digit = series.str.match(
            r"^[A-Za-z ]+\d+$|^\d+[A-Za-z ]+$", na=False
        )
        if stray_digit.any():
            suspicious.extend(
                series[stray_digit].value_counts().head(5).index.tolist()
            )

    # 3. Punctuation-only variants and short-code variants:
    #    (a) Within the same stripped key, flag values that are less frequent
    #        than the dominant spelling (e.g. 'US.' when 'US' is more common).
    #    (b) When there is a very common "long" value (e.g. 'USA'), treat
    #        shorter stripped variants (e.g. 'US', 'US.') as suspicious too.
    stripped_to_raw: dict[str, list[tuple[str, int]]] = {}
    for raw_val, count in vc.items():
        key = re.sub(r"[^A-Za-z0-9 ]", "", str(raw_val)).strip().lower()
        if key:
            stripped_to_raw.setdefault(key, []).append((str(raw_val), count))

    # (a) Punctuation-only variants under the same stripped key.
    for key, variants in stripped_to_raw.items():
        if len(variants) <= 1:
            continue
        variants.sort(key=lambda x: -x[1])
        dominant_raw, dominant_count = variants[0]
        for raw_val, count in variants[1:]:
            if raw_val != dominant_raw:
                suspicious.append(raw_val)

    # (b) Short-code variants when a longer base form is clearly dominant.
    #     Example: if 'USA' is much more frequent than 'US'/'US.', then both
    #     'US' and 'US.' should be flagged as suspicious.
    #     We look for short stripped keys (len 2–3) whose total frequency is
    #     much lower than any strictly longer key that starts with them.
    key_totals: dict[str, int] = {
        key: sum(count for _, count in variants)
        for key, variants in stripped_to_raw.items()
    }
    for short_key, short_total in key_totals.items():
        if len(short_key) < 2 or len(short_key) > 3:
            continue
        # Find longer keys that start with the short key, e.g. 'usa' for 'us'.
        longer_candidates = [
            total
            for key, total in key_totals.items()
            if len(key) > len(short_key) and key.startswith(short_key)
        ]
        if not longer_candidates:
            continue
        dominant_long_total = max(longer_candidates)
        # Only treat the short key as suspicious when the long form is clearly
        # dominant in the column.
        if dominant_long_total >= 3 * short_total and short_total > 0:
            for raw_val, _ in stripped_to_raw.get(short_key, []):
                suspicious.append(raw_val)

    # 4. Low-frequency values (possible typos / rare variants).
    #    Apply only for small/medium-cardinality categoricals; skip
    #    very high-cardinality columns (titles, free text).
    if 0 < len(vc) <= 200:
        threshold = max(1, int(0.01 * len(series)))
        rare = vc[vc <= threshold].head(10)
        suspicious.extend(rare.index.tolist())

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in suspicious:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def _issue_priority(issue: Issue) -> int:
    """Lower = higher priority for Gemini. mixed_types first, then text_quality."""
    if issue.category == IssueCategory.MIXED_TYPES:
        return 0
    if issue.category == IssueCategory.TEXT_QUALITY:
        return 1
    return 2


def _get_actionable_issues(issues: List[Issue], df: pd.DataFrame) -> List[Issue]:
    """Return issues worth sending to Gemini, sorted by priority."""
    out = [i for i in issues if _should_send_to_gemini(i, df)]
    out.sort(key=_issue_priority)
    return out


def _get_issues_without_recs(
    actionable_issues: List[Issue],
    ai_recs: Dict[str, Any],
) -> List[Issue]:
    """From prioritized actionable issues, return those not yet sent to Gemini."""
    return [i for i in actionable_issues if i.id not in ai_recs]


def _should_send_to_gemini(issue: Issue, df: pd.DataFrame) -> bool:
    """
    Decide if an issue is worth sending to Gemini (preprocess-first, AI-second).

    - Only send mixed_types and text_quality issues; structural issues (missingness,
      duplicates, constant columns, etc.) are handled by preprocessing and rules.
    - For text_quality, skip columns that already look internally consistent
      (e.g. all-uppercase category codes, single spelling per normalized value).
    """
    # Non-text and non-mixed-type issues do not need Gemini; handled by preprocessing/rules.
    if issue.category not in (IssueCategory.MIXED_TYPES, IssueCategory.TEXT_QUALITY):
        return False

    if issue.category == IssueCategory.TEXT_QUALITY and issue.column:
        if _is_text_column_consistent(df, issue.column):
            return False
    return True


def _run_pipeline(
    df_raw: pd.DataFrame,
    file_name: str,
    apply_standard_clean: bool,
) -> Dict[str, Any]:
    """Run the Cleanse pipeline and return all results in a single dict."""
    df_work = df_raw.copy()
    rows_before, cols_before = df_work.shape

    if apply_standard_clean:
        df_work = clean(df_work)

    profile_before = profile_dataset(df_work)

    # Automated rules (type fixes, duplicate removal) have been folded into the
    # core cleaning stage or delegated to AI recommendations, so we keep df_work
    # as-is here.
    df_after = df_work
    actions: List[Dict[str, Any]] = []

    rows_after, cols_after = df_after.shape
    profile_after = profile_dataset(df_after)
    issues: List[Issue] = detect_issues_from_profile(profile_after)
    quality = compute_quality_score(profile_before, profile_after, issues)

    buf = io.StringIO()
    _format_df_for_csv_export(df_after).to_csv(buf, index=False, na_rep="")

    return {
        "df_raw": df_raw,
        "df_after": df_after,
        "rows_before": rows_before,
        "cols_before": cols_before,
        "rows_after": rows_after,
        "cols_after": cols_after,
        "issues": issues,
        "quality": quality,
        "quality_base": quality,
        "actions": actions,
        "cleaned_csv_bytes": buf.getvalue().encode("utf-8"),
        "file_name": file_name,
    }


def _issues_to_recommender_input(
    issues: List[Issue],
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Convert Issue objects into a lean payload for Gemini (keeps under token limits)."""
    out: List[Dict[str, Any]] = []
    for issue in issues:
        col = issue.column
        entry: Dict[str, Any] = {
            "issue_id": issue.id,
            "issue_type": issue.category.value,
            "column": col,
            "description": issue.message,
        }
        if col and col in df.columns:
            series = df[col]
            entry["dtype"] = str(series.dtype)
            entry["missing_percent"] = round(series.isna().mean() * 100, 1)
            entry["unique_values"] = int(series.nunique())
            entry["sample_values"] = [
                str(v) for v in series.dropna().head(4).tolist()
            ]
            # Small value counts so model can suggest normalize_values; keeps tokens low
            vc = series.dropna().astype(str).value_counts().head(3)
            entry["top_value_counts"] = vc.to_dict()
            # For text_quality issues, include suspicious values so Gemini can
            # propose concrete normalize_values mappings for typos / variants.
            if issue.category == IssueCategory.TEXT_QUALITY:
                sus = _detect_suspicious_values(df, col)
                if sus:
                    entry["suspicious_values"] = sus

            # For mixed_types issues, flag values that fail to parse according to the
            # inferred target type so Gemini can propose precise normalize_values mappings.
            if issue.category == IssueCategory.MIXED_TYPES:
                inferred = str(issue.stats.get("inferred_type", "") or "").lower()
                if inferred == "datetime":
                    coerced_dt = _coerce_series_to_datetime(series)
                    failed_dt = series.notna() & coerced_dt.isna()
                    if failed_dt.any():
                        entry["unparseable_values"] = (
                            series[failed_dt].astype(str).unique().tolist()
                        )
                elif inferred == "numeric":
                    # Use pandas numeric parsing on the already preprocessed values.
                    as_str = series.astype(str)
                    coerced_num = pd.to_numeric(as_str, errors="coerce")
                    non_null = as_str.str.strip() != ""
                    failed_num = non_null & coerced_num.isna()
                    if failed_num.any():
                        entry["unparseable_values"] = (
                            series[failed_num].astype(str).unique().tolist()
                        )
        out.append(entry)
    return out


def _update_run_result(res: Dict[str, Any], df_new: pd.DataFrame) -> None:
    """Recompute profile, issues, quality, CSV bytes and update session_state."""
    buf = io.StringIO()
    _format_df_for_csv_export(df_new).to_csv(buf, index=False, na_rep="")
    profile_after = profile_dataset(df_new)
    new_issues = detect_issues_from_profile(profile_after)
    quality = compute_quality_score(
        profile_dataset(res["df_raw"]),
        profile_after,
        new_issues,
    )
    base_quality = res.get("quality_base", res.get("quality"))
    st.session_state["run_result"] = {
        **res,
        "df_after": df_new,
        "rows_after": len(df_new),
        "cols_after": len(df_new.columns),
        "issues": new_issues,
        "quality": quality,
        "quality_base": base_quality,
        "cleaned_csv_bytes": buf.getvalue().encode("utf-8"),
    }


def _apply_ai_fix(res: Dict[str, Any], df_new: pd.DataFrame) -> None:
    """
    Apply an AI-driven fix:
    - update df_after / shape / download bytes
    - recompute quality based on existing issues
    but **do not** change the issues list, so cards remain visible.
    Store a copy of the dataframe so session state and UI (e.g. schema) refresh.
    """
    df_new = df_new.copy()
    buf = io.StringIO()
    _format_df_for_csv_export(df_new).to_csv(buf, index=False, na_rep="")
    profile_after = profile_dataset(df_new)
    quality = compute_quality_score(
        profile_dataset(res["df_raw"]),
        profile_after,
        res["issues"],
    )
    base_quality = res.get("quality_base", res.get("quality"))
    st.session_state["run_result"] = {
        **res,
        "df_after": df_new,
        "rows_after": len(df_new),
        "cols_after": len(df_new.columns),
        "quality": quality,
        "quality_base": base_quality,
        "cleaned_csv_bytes": buf.getvalue().encode("utf-8"),
    }


def _compute_run_result_with_fixes(
    run_result_base: Dict[str, Any],
    ai_applied_fixes: Dict[str, Dict[str, Any]],
    issues: List[Issue],
) -> Dict[str, Any]:
    """
    Recompute run_result by applying each fix in ai_applied_fixes to the base df,
    in the order issues appear. Used when undoing (must recompute from base).
    """
    df = run_result_base["df_after"].copy()
    for issue in issues:
        if issue.id not in ai_applied_fixes:
            continue
        rec = ai_applied_fixes[issue.id]
        df = apply_fix(
            df,
            rec["action"],
            rec.get("column"),
            rec.get("params"),
        )
    buf = io.StringIO()
    _format_df_for_csv_export(df).to_csv(buf, index=False, na_rep="")
    profile_after = profile_dataset(df)
    quality = compute_quality_score(
        profile_dataset(run_result_base["df_raw"]),
        profile_after,
        run_result_base["issues"],
    )
    base_quality = run_result_base.get("quality_base", run_result_base.get("quality"))
    return {
        **run_result_base,
        "df_after": df,
        "rows_after": len(df),
        "cols_after": len(df.columns),
        "quality": quality,
        "quality_base": base_quality,
        "cleaned_csv_bytes": buf.getvalue().encode("utf-8"),
    }


def _apply_single_fix_and_update_run_result(
    res: Dict[str, Any],
    issue_id: str,
    action: str,
    column: Optional[str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply one fix to res["df_after"] and return updated run_result.
    Fast path: no re-application of other fixes.
    """
    df = apply_fix(res["df_after"].copy(), action, column, params)
    buf = io.StringIO()
    _format_df_for_csv_export(df).to_csv(buf, index=False, na_rep="")
    profile_after = profile_dataset(df)
    quality = compute_quality_score(
        profile_dataset(res["df_raw"]),
        profile_after,
        res["issues"],
    )
    base_quality = res.get("quality_base", res.get("quality"))
    return {
        **res,
        "df_after": df,
        "rows_after": len(df),
        "cols_after": len(df.columns),
        "quality": quality,
        "quality_base": base_quality,
        "cleaned_csv_bytes": buf.getvalue().encode("utf-8"),
    }


# --- Main app ----------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Cleanse – Smart Data Cleaning",
        layout="wide",
        page_icon="✨",
    )

    if "run_result" not in st.session_state:
        st.session_state["run_result"] = None
    if "ai_recommendations" not in st.session_state:
        st.session_state["ai_recommendations"] = None
    if "ai_applied_fixes" not in st.session_state:
        st.session_state["ai_applied_fixes"] = {}
    if "run_result_base" not in st.session_state:
        st.session_state["run_result_base"] = None
    if "dataset_key" not in st.session_state:
        st.session_state["dataset_key"] = None


    # --- Sidebar ---------------------------------------------------------------
    st.sidebar.title("Dataset")

    old_source = st.session_state.get("Source")
    if old_source == "Demo dataset":
        st.session_state["Source"] = "Demo Dataset"

    source = st.sidebar.radio("Source", ["Upload CSV", "Demo Dataset"])

    df_raw: pd.DataFrame | None = None
    file_name: str | None = None

    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV File", type=["csv"], key="cp_upload")
        if uploaded is not None:
            try:
                df_raw = _read_csv_robust(uploaded)
                file_name = uploaded.name
            except Exception as exc:
                st.sidebar.error(f"Failed to read CSV: {exc}")
    else:
        demos = _available_demo_datasets()
        if not demos:
            st.sidebar.warning("No Demo Datasets found in `data/`.")
        else:
            old_label = st.session_state.get("cp_demo")
            if isinstance(old_label, str) and old_label not in demos:
                for label_key in demos:
                    if old_label.startswith(label_key):
                        st.session_state["cp_demo"] = label_key
                        break
            label = st.sidebar.selectbox("Demo Dataset", list(demos.keys()), key="cp_demo")
            if label:
                path = demos[label]
                try:
                    df_raw, _sep_used = load_csv(path)
                    file_name = path.name
                except Exception as exc:
                    st.sidebar.error(f"Failed to load demo dataset: {exc}")

    # If the active dataset changed (new upload or different demo), reset results
    # and mark that we should scroll back to the top on this render so the user
    # reorients from Step 1.
    dataset_key: str | None = None
    if df_raw is not None and file_name is not None:
        if source == "Upload CSV":
            dataset_key = f"upload::{file_name}::{df_raw.shape[0]}::{df_raw.shape[1]}"
        else:
            dataset_key = f"demo::{file_name}"

    prev_key: str | None = st.session_state.get("dataset_key")
    if dataset_key is not None and dataset_key != prev_key:
        st.session_state["dataset_key"] = dataset_key
        st.session_state["run_result"] = None
        st.session_state["ai_recommendations"] = None
        st.session_state["ai_applied_fixes"] = {}
        st.session_state["run_result_base"] = None


    # --- Styles & hero ---------------------------------------------------------
    st.markdown(
        """
        <style>
          [data-testid="stAppViewContainer"] > .main {
            padding-left: 2rem; padding-right: 2rem; padding-top: 0;
          }
          .block-container { padding-top: 0; }
          .cp-hero { padding: 0 0 1rem 0; display: flex; flex-direction: column; gap: 0.75rem; }
          .cp-eyebrow { font-size: 0.9rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #60a5fa; }
          .cp-title { font-size: 3.0rem; font-weight: 700; letter-spacing: -0.04em; margin: 0; }
          .cp-subtitle { max-width: 640px; font-size: 1.02rem; color: #9ca3af; }
          .cp-card {
            border-radius: 1rem; border: 1px solid rgba(148,163,184,0.35);
            background: radial-gradient(circle at top left, #020617, #020617);
            padding: 1.4rem 1.5rem 1.2rem 1.5rem; margin-bottom: 0.75rem;
          }
          .cp-tag { display: inline-flex; align-items: center; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: 500; }
          .cp-tag-error { background: rgba(248,113,113,0.1); color: #fecaca; border: 1px solid rgba(248,113,113,0.4); }
          .cp-tag-warning { background: rgba(250,204,21,0.08); color: #facc15; border: 1px solid rgba(250,204,21,0.35); }
          .cp-tag-info { background: rgba(96,165,250,0.12); color: #bfdbfe; border: 1px solid rgba(96,165,250,0.4); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="cp-hero">
          <h1 class="cp-title">Cleanse.ai</h1>
          <p class="cp-subtitle">
            From messy CSVs to trustworthy datasets. Pick a dataset, preview it,
            run preprocessing, review AI suggestions, then download the cleaned result.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Step 1: Preview -------------------------------------------------------
    st.subheader("1. Preview, Schema & Report")

    if df_raw is not None and file_name is not None:
        tab_prev, tab_schema, tab_report1 = st.tabs(
            ["Preview rows", "Schema", "Report"]
        )
        with tab_prev:
            st.dataframe(_preview_dataframe(df_raw.head(50)), width="stretch")
        with tab_schema:
            st.dataframe(_build_schema_table(df_raw), width="stretch")
        with tab_report1:
            # Step 1 report is based on the raw data only (no preprocessing, no AI).
            profile_raw = profile_dataset(df_raw)
            raw_issues = detect_issues_from_profile(profile_raw)
            quality_raw = compute_quality_score(profile_raw, None, raw_issues)
            st.markdown("**Data quality on raw data**")
            st.metric("Overall score", quality_raw.get("score", 0.0))
            comps = quality_raw.get("components", {}) or {}
            st.write(
                {
                    "completeness": comps.get("completeness"),
                    "consistency": comps.get("consistency"),
                    "validity": comps.get("validity"),
                    "uniqueness": comps.get("uniqueness"),
                }
            )
            patterns = [
                (cp.get("column"), cp.get("missingness_pattern"))
                for cp in (profile_raw.get("column_profiles") or [])
                if cp.get("missingness_pattern")
            ]
            if patterns:
                st.markdown("**Missingness patterns**")
                st.caption(
                    "Columns where missingness is strongly tied to another column's value."
                )
                for col_name, pat in patterns:
                    when_col = pat.get("when_column", "")
                    when_val = pat.get("when_value", "")
                    rate = pat.get("missing_rate_when", 0)
                    overall = pat.get("overall_missing_pct", 0)
                    st.markdown(
                        f"- **{col_name}**: {overall}% missing overall; "
                        f"{rate}% missing when `{when_col}` = \"{when_val}\""
                    )

        # --- Step 2: Core Preprocessing ----------------------------------------
        st.subheader("2. Core Preprocessing")
        st.caption(
            "Standard cleaning",
            help=(
                "Trim whitespace, standardize nulls, clean column names, "
                "drop fully-empty rows/columns, remove exact duplicates, and "
                "optionally capitalize person names. No imputation or "
                "domain-specific normalization; those come from AI suggestions."
            ),
        )

        if st.button("Run Preprocessing and Analyze", type="primary"):
            with st.spinner("Running preprocessing, profiling, and issue detection…"):
                st.session_state["run_result"] = _run_pipeline(
                    df_raw, file_name, apply_standard_clean=True,
                )
                st.session_state["ai_recommendations"] = None
                st.session_state["ai_applied_fixes"] = {}
                st.session_state["run_result_base"] = None
                for key in ("last_ai_snapshot", "last_ai_issue_id"):
                    st.session_state.pop(key, None)

        # If preprocessing has already been run, show a concise summary of what changed.
        res_preview: Dict[str, Any] | None = st.session_state.get("run_result")
        if res_preview is not None:
            clean_summary = _summarize_cleaning(res_preview["df_raw"], res_preview["df_after"])
            if clean_summary:
                st.caption("Preprocessing — " + " · ".join(clean_summary))

        # --- Step 3: Cleaned Dataset -------------------------------------------
        st.subheader("3. Cleaned Dataset & Reports")
        res: Dict[str, Any] | None = st.session_state.get("run_result")

        if res is None:
            st.info("Run preprocessing above to see the cleaned dataset here.")
        else:
            df_after: pd.DataFrame = res["df_after"]
            tab_clean, tab_schema_after, tab_report3 = st.tabs(
                ["Preview rows", "Schema", "Reports"]
            )
            with tab_clean:
                st.dataframe(_preview_dataframe(df_after.head(50)), width="stretch")
                st.caption(
                    f"Showing first 50 rows · {res['rows_after']:,} rows × {res['cols_after']} columns "
                    f"(from {res['rows_before']:,} × {res['cols_before']})."
                )
            with tab_schema_after:
                # Key by id(df_after) so the schema table refreshes when df_after is replaced after apply
                st.dataframe(
                    _build_schema_table(df_after),
                    width="stretch",
                    key=f"schema_after_{id(res['df_after'])}",
                )
            with tab_report3:
                q = res.get("quality")
                if not q:
                    st.info("Quality report is not available.")
                else:
                    st.markdown("**Cleaned data quality**")
                    st.metric("Overall score", q.get("score", 0.0))
                    comps = q.get("components", {}) or {}
                    st.write(
                        {
                            "completeness": comps.get("completeness"),
                            "consistency": comps.get("consistency"),
                            "validity": comps.get("validity"),
                            "uniqueness": comps.get("uniqueness"),
                        }
                    )
                profile_after = profile_dataset(df_after)
                patterns3 = [
                    (cp.get("column"), cp.get("missingness_pattern"))
                    for cp in (profile_after.get("column_profiles") or [])
                    if cp.get("missingness_pattern")
                ]
                if patterns3:
                    st.markdown("**Missingness patterns**")
                    st.caption(
                        "Columns where missingness is strongly tied to another column's value."
                    )
                    for col_name, pat in patterns3:
                        when_col = pat.get("when_column", "")
                        when_val = pat.get("when_value", "")
                        rate = pat.get("missing_rate_when", 0)
                        overall = pat.get("overall_missing_pct", 0)
                        st.markdown(
                            f"- **{col_name}**: {overall}% missing overall; "
                            f"{rate}% missing when `{when_col}` = \"{when_val}\""
                        )

            # --- Step 4: Recommendations ---------------------------------------
            st.subheader("4. Recommendations")

            issues: List[Issue] = res["issues"]
            if not issues:
                st.success("No issues detected. The dataset looks clean.")
            else:
                # Compute actionable issues and the next batch that will be sent to Gemini.
                actionable = _get_actionable_issues(issues, df_after)
                existing_recs = st.session_state.get("ai_recommendations") or {}
                # Send all actionable issues in a single request (up to MAX_ISSUES_PER_REQUEST),
                # even if some already have recommendations. This avoids needing multiple clicks
                # just to cover a small set of issues.
                next_batch = actionable[:MAX_ISSUES_PER_REQUEST]

                if next_batch:
                    with st.expander(
                        "Issues that will be sent to Gemini (next batch)", expanded=False
                    ):
                        st.caption(
                            f"Up to {MAX_ISSUES_PER_REQUEST} highest-priority, fixable issues "
                            "are sent per request."
                        )
                        for iss in next_batch:
                            st.markdown(
                                f"- **{iss.category.name}** · `{iss.column}` · `{iss.id}`  \n"
                                f"  {iss.message}"
                            )
                else:
                    st.caption(
                        "All actionable issues already have recommendations; nothing new will be "
                        "sent in the next request."
                    )

                if st.button("Generate Recommendations", type="primary"):
                    if next_batch:
                        recommender_issues = _issues_to_recommender_input(next_batch, df_after)
                        with st.spinner("Analyzing issues via Gemini…"):
                            try:
                                recs = generate_ai_recommendations(recommender_issues)
                                st.session_state["ai_recommendations"] = {
                                    **existing_recs,
                                    **recs,
                                }
                            except Exception as e:
                                msg = str(e)
                                # Auto-retry once on transient JSON parse/truncation errors.
                                if msg.startswith(
                                    "Failed to parse JSON from model response"
                                ):
                                    try:
                                        recs = generate_ai_recommendations(recommender_issues)
                                        st.session_state["ai_recommendations"] = {
                                            **existing_recs,
                                            **recs,
                                        }
                                    except Exception as e2:
                                        st.error(f"Failed to generate recommendations: {e2}")
                                else:
                                    st.error(f"Failed to generate recommendations: {e}")
                    else:
                        st.info("All issues already have recommendations.")

                ai_recs = st.session_state.get("ai_recommendations")

                if ai_recs is None:
                    st.info(
                        f"**{len(actionable)}** issue(s) may benefit from AI. "
                        "Click **Generate Recommendations** to get suggestions."
                    )
                    if len(actionable) > MAX_ISSUES_PER_REQUEST:
                        st.caption(
                            f"Up to {MAX_ISSUES_PER_REQUEST} issues are sent per request. "
                            "Click again after the first batch to add recommendations for the rest."
                        )
                else:
                    matched = [
                        (iss, ai_recs[iss.id])
                        for iss in issues
                        if iss.id in ai_recs
                    ]
                    actionable = _get_actionable_issues(issues, df_after)
                    issues_without_recs = _get_issues_without_recs(actionable, ai_recs)

                    if issues_without_recs:
                        st.caption(
                            f"{len(issues_without_recs)} issue(s) not yet sent. "
                            "Click **Generate Recommendations** again to add their cards."
                        )

                    if not matched:
                        st.warning(
                            "No recommendations match the current issues. "
                            "Click **Generate Recommendations** to refresh."
                        )
                    else:
                        actionable: list[tuple[Issue, dict[str, Any]]] = []
                        for issue, rec in matched:
                            if rec.get("recommended_action") == "keep":
                                continue
                            actionable.append((issue, rec))

                        if not actionable:
                            st.info("No actionable recommendations.")
                        else:
                            ai_applied_fixes = st.session_state.get(
                                "ai_applied_fixes", {}
                            )
                            run_result_base = st.session_state.get(
                                "run_result_base"
                            )

                            for issue, rec in actionable:
                                action = rec["recommended_action"]
                                with st.container(border=True):
                                    st.markdown(f"**{issue.title}** · `{action}`")
                                    st.caption(f"**Problem:** {issue.message}")
                                    st.caption(
                                        f"**Inferred domain:** "
                                        f"{rec.get('inferred_domain') or '—'}"
                                    )

                                    # For normalize_values, allow per-mapping toggles so the user
                                    # can accept or reject each individual mapping. If there is
                                    # only a single mapping, the extra toggles are unnecessary and
                                    # the recommendation can be applied as a whole. Also drop
                                    # identity mappings (src == tgt) since they are no-ops.
                                    selected_mappings: dict[str, str] | None = None
                                    if (
                                        action == "normalize_values"
                                        and issue.column
                                        and isinstance(
                                            rec.get("fix_params", {}).get("mappings"), dict
                                        )
                                    ):
                                        raw_mappings: dict[str, str] = rec["fix_params"][
                                            "mappings"
                                        ]
                                        # Drop identity mappings like "Brazil" -> "Brazil".
                                        mappings: dict[str, str] = {
                                            str(src): tgt
                                            for src, tgt in raw_mappings.items()
                                            if str(src) != str(tgt)
                                        }
                                        if mappings and len(mappings) > 1:
                                            st.caption("Proposed value mappings:")
                                            selected_mappings = {}
                                            for i, (src, tgt) in enumerate(
                                                mappings.items()
                                            ):
                                                checked = st.checkbox(
                                                    f"`{src}` → `{tgt}`",
                                                    key=f"mapping_{issue.id}_{i}",
                                                    value=True,
                                                )
                                                if checked:
                                                    selected_mappings[src] = tgt

                                    fix_code = get_fix_code(
                                        action, issue.column, rec.get("fix_params", {}),
                                    )
                                    st.caption("**Python recommendation:**")
                                    st.code(fix_code, language="python")

                                    if action == "flag_for_manual_review":
                                        st.info(
                                            "No automatic fix for this recommendation."
                                        )
                                    else:
                                        is_applied = issue.id in ai_applied_fixes
                                        toggled = st.checkbox(
                                            "Apply recommendation",
                                            key=f"apply_toggle_{issue.id}",
                                            value=is_applied,
                                        )
                                        if toggled and not is_applied:
                                            # Snapshot base once (no deep copy of large df)
                                            if run_result_base is None:
                                                st.session_state["run_result_base"] = {
                                                    **res,
                                                    "df_after": res["df_after"].copy(),
                                                }
                                            # Use only the selected mappings, if applicable,
                                            # and always drop identity mappings.
                                            params = rec.get("fix_params", {}) or {}
                                            if action == "normalize_values":
                                                base_mappings = {
                                                    str(src): tgt
                                                    for src, tgt in (
                                                        params.get("mappings", {}) or {}
                                                    ).items()
                                                    if str(src) != str(tgt)
                                                }
                                                if selected_mappings is not None:
                                                    base_mappings = selected_mappings
                                                params = {
                                                    **params,
                                                    "mappings": base_mappings,
                                                }
                                            applied = {
                                                **ai_applied_fixes,
                                                issue.id: {
                                                    "action": action,
                                                    "column": issue.column,
                                                    "params": params,
                                                },
                                            }
                                            st.session_state["ai_applied_fixes"] = applied
                                            # Fast path: apply only this fix to current df
                                            new_res = _apply_single_fix_and_update_run_result(
                                                res,
                                                issue.id,
                                                action,
                                                issue.column,
                                                params,
                                            )
                                            st.session_state["run_result"] = new_res
                                            st.rerun()
                                        elif not toggled and is_applied:
                                            base = st.session_state.get("run_result_base")
                                            if base is None:
                                                # Should not happen if apply ran first; use res as fallback
                                                base = {
                                                    **res,
                                                    "df_after": res["df_after"].copy(),
                                                }
                                            applied = {
                                                k: v
                                                for k, v in ai_applied_fixes.items()
                                                if k != issue.id
                                            }
                                            st.session_state["ai_applied_fixes"] = applied
                                            new_res = _compute_run_result_with_fixes(
                                                base, applied, issues
                                            )
                                            st.session_state["run_result"] = new_res
                                            if not applied:
                                                st.session_state["run_result_base"] = None
                                            st.rerun()

            # --- Step 5: Download ----------------------------------------------
            st.subheader("5. Download Final Dataset")
            st.download_button(
                label="Download cleaned CSV",
                data=res["cleaned_csv_bytes"],
                file_name=f"cleaned_{res['file_name']}",
                mime="text/csv",
            )
    else:
        st.info("Choose a dataset from the sidebar to see a preview and schema.")


if __name__ == "__main__":
    main()
