"""
Generic preprocessing for any dataset. Composed in clean().

Steps: trim whitespace, standardize nulls, normalize column names, drop empty columns,
optional person-name capitalization, int-like and numeric string normalization,
mixed-type coercion (numeric/datetime from values only), drop empty rows, remove duplicates.

All behavior is value-based (patterns, shares, parsing); no column names or
dataset-specific logic. Domain fixes (country, title, ratings, etc.) are left to Gemini.
"""
import re
import unicodedata
import warnings
from datetime import datetime
from difflib import get_close_matches
from typing import Any

import numpy as np
import pandas as pd

from app.config import CAPITALIZE_PERSON_NAMES, PERSON_NAME_CASE

# Date formats to try (strptime-style). Order: unambiguous first, then DD/MM, then MM/DD,
# then digit-only variants used in messy exports.
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %b %y",
    "%b %d %Y",
    "%b %d %y",
    "%B %d %Y",
    "%B %d, %Y",  # "April 5, 2018"
    "%d %B %Y",
    "%m %d %Y",
    "%d %m %Y",
    "%d-%m-%y",
    "%m-%d-%y",
    "%Y%m%d",     # 20200220
    "%m%d%Y",     # 01152020
    "%Y",
    "%y",
)


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace on object (text) columns only. Preserves NA (no astype(str) that would turn NA into 'nan')."""
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].apply(lambda v: pd.NA if pd.isna(v) else str(v).strip())
    return out


_NULL_SENTINELS = {
    "",
    "na",
    "nan",
    "n/a",
    "n.a.",
    "n.a",
    "none",
    "null",
    "#n/a",
    "-",
    "not applicable",
    "not appl.",
    "n.applicable",
}


def standardize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize null-like values.

    - Blank / whitespace-only cells become null.
    - Common string sentinels like 'NA', 'N/A', 'None', 'NULL', '#N/A', '-' also become null.
    This keeps the CSV output consistent: blank in the source → blank cell in the cleaned file.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        def _to_null(v):
            if v is None or pd.isna(v):
                return pd.NA
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return pd.NA
                if s.lower() in _NULL_SENTINELS:
                    return pd.NA
            return v

        out[col] = out[col].apply(_to_null)
    return out


def _normalize_header_generic(s: str) -> str:
    """Adaptable header cleanup: no vocabulary. Strip, fold accents to ASCII, collapse spaces, title-case."""
    if not s:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lower().title()
    return s


# Domain-neutral words that often appear in column names. Used only to fix spelling/truncation per token.
# Covers business, retail, surveys, CRM, etc. — not biased toward one domain.
COMMON_HEADER_WORDS: tuple[str, ...] = (
    "name", "title", "id", "code", "type", "category", "status", "date", "year", "time",
    "amount", "price", "cost", "total", "value", "quantity", "number", "count", "score", "rating",
    "email", "address", "phone", "city", "state", "country", "region", "zip", "postal",
    "customer", "order", "product", "item", "description", "note", "comment", "reference",
    "first", "last", "full", "company", "department", "unit", "currency", "percent", "percentage",
    "original", "release", "content", "duration", "votes", "income", "revenue", "runtime", "minutes",
    "genre", "gender",
)
HEADER_WORD_FUZZY_CUTOFF: float = 0.72
# Only fix tokens this short or shorter (likely truncations). Longer tokens (e.g. "Gender") left as-is.
HEADER_SPELLING_MAX_TOKEN_LEN: int = 5


def _fix_header_spelling(header: str) -> str:
    """Fuzzy-match short tokens only to fix truncation (titl→title, genr→genre). Long tokens (e.g. Gender) unchanged."""
    if not header:
        return header
    words_lower = list(COMMON_HEADER_WORDS)
    tokens = header.split()
    fixed = []
    for t in tokens:
        tl = t.lower()
        if len(tl) > HEADER_SPELLING_MAX_TOKEN_LEN:
            fixed.append(t)
            continue
        matches = get_close_matches(tl, words_lower, n=1, cutoff=HEADER_WORD_FUZZY_CUTOFF)
        if matches:
            fixed.append(matches[0].title())
        else:
            fixed.append(t)
    return " ".join(fixed)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and generic cleanup (encoding, spaces, case), then spelling fix (fuzzy per-token). No config renames."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    renames: dict[str, str] = {}
    used = set(out.columns)
    for col in out.columns:
        if col in renames:
            continue
        cleaned = _normalize_header_generic(col)
        cleaned = _fix_header_spelling(cleaned)
        if cleaned and cleaned != col and cleaned not in used:
            renames[col] = cleaned
            used.add(cleaned)
    if renames:
        out = out.rename(columns=renames)
    return out


def _preprocess_date_string(s: str) -> str:
    """Normalize date string for parsing: strip, collapse spaces, remove 'of', strip ordinals."""
    if not s or pd.isna(s) or not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" of ", " ")
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _resolve_person_name_columns(df: pd.DataFrame) -> list[str]:
    """Columns to capitalize as person names. Value-based only: columns where most non-empty
    values look like multi-token alphabetic strings (e.g. "david lee"). Config unused.
    """
    name_pattern = re.compile(
        r"^[A-Za-z][A-Za-z\.'\-]*(\s+[A-Za-z][A-Za-z\.'\-]*){0,3}$"
    )
    person_like: set[str] = set()
    for c in df.select_dtypes(include=["object"]).columns:
        series = df[c].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 20:
            continue
        lower_vals = non_empty.str.lower()
        if (lower_vals.str.contains("@")).mean() >= 0.2:
            continue
        if (non_empty.str.contains(r"\d", regex=True)).mean() >= 0.3:
            continue

        share_name_like = non_empty.str.match(name_pattern).mean()
        if share_name_like < 0.6:
            continue

        # Multi-token values (contain a space) distinguish names from codes/categories.
        # "david lee" / "emily davis" have spaces; "USA" / "Approved" / "PG-13" do not.
        share_multi_token = non_empty.str.contains(r"\s").mean()
        if share_multi_token >= 0.5:
            person_like.add(c)
            continue

        # Single-token dominant — apply code-like exclusions so we don't title-case
        # categorical values (e.g. "USA" → "Usa", "METFORMIN" → "Metformin").
        short_vals = non_empty[non_empty.str.len() <= 12]
        if len(short_vals) / len(non_empty) >= 0.5 and short_vals.nunique() <= 30:
            continue
        n_unique = non_empty.nunique()
        if n_unique <= 50:
            mode_count = non_empty.value_counts().iloc[0]
            if mode_count / len(non_empty) >= 0.15:
                continue
        person_like.add(c)

    return list(person_like)


def _apply_name_case(s: str, style: str) -> str:
    """Apply casing style: 'title' (each word) or 'sentence' (first letter only)."""
    if style == "sentence":
        return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper() if s else s
    return s.title()


def capitalize_person_names(df: pd.DataFrame) -> pd.DataFrame:
    """Case values in person-name columns (value-based detection). CAPITALIZE_PERSON_NAMES / PERSON_NAME_CASE are on/off and style only."""
    if not CAPITALIZE_PERSON_NAMES:
        return df.copy()
    out = df.copy()
    cols = _resolve_person_name_columns(df)
    case_style = (PERSON_NAME_CASE or "title").strip().lower() or "title"
    if case_style not in ("title", "sentence"):
        case_style = "title"

    for col in cols:
        if col not in out.select_dtypes(include=["object"]).columns:
            continue

        def _case_if_present(v):
            if pd.isna(v) or v is None:
                return v
            s = str(v).strip()
            if not s:
                return v
            return _apply_name_case(s, case_style)

        out[col] = out[col].apply(_case_if_present)
    return out


def _parse_with_fmt(s: str, fmt: str) -> bool:
    """Return True if s parses with the given format."""
    try:
        datetime.strptime(s, fmt)
        return True
    except (ValueError, TypeError):
        return False


def _find_date_columns(df: pd.DataFrame) -> list[str]:
    """Columns to consider as dates: value-based detection only. A column is a date column if
    a large share of its values parse as dates with our known formats. Config unused.
    """
    cols: list[str] = []
    # Value-based: columns where a
    # high share of non-empty values parse with one of our known date formats.
    for c in df.columns:
        if c in cols:
            continue
        series = df[c]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.astype(str).apply(_preprocess_date_string)
        non_null = non_null[non_null != ""]
        non_null = non_null[non_null.str.lower() != "nan"]
        n = len(non_null)
        if n < 10:
            continue
        # Use same parsing as coercion (DATE_FORMATS + dateutil) so we don't miss
        # single-format date columns (e.g. "Jan 15, 2024") and so we skip them from
        # numeric normalization and coerce when 100% parse.
        parsed = sum(1 for s in non_null if _parse_single_date(s) is not None)
        if parsed == n:
            cols.append(c)
    return cols


# Word form → number (for Salary, Age, Quantity, etc.). Uppercase key; applied when value matches.
WORD_TO_NUMBER: dict[str, int | float] = {
    "TWENTY": 20, "THIRTY": 30, "FORTY": 40, "FIFTY": 50, "SIXTY": 60,
    "SEVENTY": 70, "EIGHTY": 80, "NINETY": 90,
    "ONE HUNDRED": 100, "TWO HUNDRED": 200, "THREE HUNDRED": 300, "FOUR HUNDRED": 400,
    "FIVE HUNDRED": 500, "SIX HUNDRED": 600, "SEVEN HUNDRED": 700, "EIGHT HUNDRED": 800, "NINE HUNDRED": 900,
    "ONE THOUSAND": 1000, "TWO THOUSAND": 2000, "THREE THOUSAND": 3000, "FOUR THOUSAND": 4000,
    "FIVE THOUSAND": 5000, "SIX THOUSAND": 6000, "SEVEN THOUSAND": 7000, "EIGHT THOUSAND": 8000, "NINE THOUSAND": 9000,
    "FIFTY THOUSAND": 50000, "SIXTY THOUSAND": 60000, "SEVENTY THOUSAND": 70000,
}


def _normalize_word_numbers_series(series: pd.Series) -> pd.Series:
    """
    Convert pure word-form numbers (e.g. 'forty', 'SIXTY THOUSAND') to digits using
    WORD_TO_NUMBER. Values with digits or punctuation are left unchanged so we
    don't accidentally touch IDs or mixed strings.
    """
    def _convert(v: object) -> object:
        if pd.isna(v):
            return v
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return v
        # Skip anything that already contains a digit or obvious punctuation.
        if any(ch.isdigit() for ch in s) or any(ch in "/-:,." for ch in s):
            return v
        key = s.upper()
        if key in WORD_TO_NUMBER:
            return str(WORD_TO_NUMBER[key])
        return v

    return series.apply(_convert)


def _normalize_numeric_string(val: str) -> str:
    """Strip $ and spaces, keep only digits/dot/comma; comma→dot; multiple dots→thousand sep (remove).
    Replaces word numbers (e.g. thirty→30, SIXTY THOUSAND→60000). Preserves titles like 'Se7en'.
    For $-prefixed strings, replace digit lookalikes (o/O→0, l/I→1).
    Cleans float-like typos: trailing 'f', exponent e-0/e+0, colon as decimal, double dot, trailing dot.
    """
    if not isinstance(val, str) or pd.isna(val):
        return val
    s = val.strip()
    if not s:
        return val
    # Float-like cleanup before alpha check: trailing f/F, exponent typo e-0/e+0
    s = s.rstrip("fF")
    s = re.sub(r"e[+-]?0$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if not s:
        return val
    # Word numbers (e.g. "thirty", "SIXTY THOUSAND")
    key = s.upper().strip()
    if key in WORD_TO_NUMBER:
        return str(WORD_TO_NUMBER[key])
    # Allow letters only when $ prefix and rest is numeric; otherwise skip (e.g. "Se7en", "La vita B9 bella")
    if s.startswith("$"):
        rest = s.lstrip("$").strip()
        # Income typo: letter o/O as zero, l/I as one (only in numeric-looking part)
        rest = rest.replace("o", "0").replace("O", "0").replace("l", "1").replace("I", "1")
        if any(c.isalpha() for c in rest):
            return val
        s = rest
    else:
        if any(c.isalpha() for c in s):
            return val
    # Locale-aware handling when both comma and dot appear (e.g. 1,234.56 vs 1.234,56).
    if "," in s and "." in s:
        us_pattern = re.compile(r"^\s*[-+]?\d{1,3}(,\d{3})*(\.\d+)?\s*$")
        eu_pattern = re.compile(r"^\s*[-+]?\d{1,3}(\.\d{3})*(,\d+)?\s*$")
        if us_pattern.match(s):
            # US style: commas are thousands, dot is decimal. Remove commas only.
            numeric = s.replace(",", "")
            numeric = "".join(c for c in numeric if c in "0123456789.")
            numeric = numeric.rstrip(".")
            if numeric:
                return numeric
        if eu_pattern.match(s):
            # European style: dots are thousands, comma is decimal. Remove dots, comma→dot.
            numeric = s.replace(".", "").replace(",", ".")
            numeric = "".join(c for c in numeric if c in "0123456789.")
            numeric = numeric.rstrip(".")
            if numeric:
                return numeric
    s = s.replace(":", ".")
    cleaned = "".join(c for c in s if c in "0123456789.,")
    if not cleaned:
        return val
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"\.+", ".", cleaned)
    # Strip trailing dot/comma so "8.7." stays 8.7, not 87
    cleaned = cleaned.rstrip(".,")
    if cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    # One dot + exactly 3 digits after = European thousands (e.g. 668.473 → 668473). Votes/counts become integer.
    elif cleaned.count(".") == 1 and re.match(r"^\d+\.\d{3}$", cleaned):
        cleaned = cleaned.replace(".", "")
    if not cleaned:
        return val
    # Strip leading zeros: 08.9 → 8.9, 009 → 9; keep 0 and 0.xxx
    if "." in cleaned:
        before, _, after = cleaned.partition(".")
        before = before.lstrip("0") or "0"
        cleaned = f"{before}.{after}" if after else before
    else:
        cleaned = cleaned.lstrip("0") or "0"
    # For currency values we now normalize to a pure numeric string so the column
    # can be coerced to a numeric dtype; currency formatting is handled at the UI
    # / presentation layer instead of being stored in the cleaned dataset.
    return cleaned


# Skip coercion when values actually have "$" (source included it); don't add $ or skip by column name.
_CURRENCY_IN_VALUE_MIN_SHARE = 0.2  # if >= 20% of non-null values start with $, keep column as text
# Only coerce when most values convert; mixed columns (many text) stay text so we don't lose content.
_COERCE_MIN_NUMERIC_SHARE = 0.9  # coerce only if >= 90% of non-null values are numeric
_RATIO_LIKE_PATTERN = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")  # e.g. 120/80, 140/90
_RATIO_COLUMN_MIN_SHARE = 0.2  # if >= 20% of non-null values match num/num, treat column as ratio (keep as text)


def _columns_with_email_like_values(df: pd.DataFrame, object_columns: list[str]) -> set[str]:
    """Columns where a large share of values look like emails (user@domain)."""
    out: set[str] = set()
    for col in object_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 10:
            continue
        share_email = non_empty.str.contains(r"@.+\.", regex=True).mean()
        if share_email >= 0.6:
            out.add(col)
    return out


def normalize_email_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase email-like columns so accidental capitalisation (e.g. 'User@Domain.com')
    does not create spurious distinct values. Detection is value-based only.
    """
    out = df.copy()
    object_cols = [c for c in out.select_dtypes(include=["object"]).columns if c in out.columns]
    email_cols = _columns_with_email_like_values(out, object_cols)
    for col in email_cols:
        series = out[col]
        mask = series.notna()
        out.loc[mask, col] = series.loc[mask].astype(str).str.strip().str.lower()
    return out


def _columns_with_phone_like_values(df: pd.DataFrame, object_columns: list[str]) -> set[str]:
    """Columns where a large share of values look like phone numbers."""
    out: set[str] = set()
    pattern = re.compile(r"^[\d\+\-\(\)\s]{7,}$")
    for col in object_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 10:
            continue
        share_phone = non_empty.apply(lambda v: bool(pattern.match(v))).mean()
        if share_phone >= 0.6:
            out.add(col)
    return out


# Integer-like: digits with optional single letter prefix/suffix (e.g. 90, 178c, m90) or "inf". Inferred from values.
_INT_LIKE_PATTERN = re.compile(
    r"^(?:inf|infinity|\d+[a-zA-Z]?|[a-zA-Z]?\d+)$",
    re.IGNORECASE,
)


def _normalize_int_like_string(val: object) -> object:
    """Normalize int-like values: inf→null, strip single leading/trailing letter from digit strings (e.g. 178c→178)."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    if not s:
        return pd.NA
    if s.lower() in ("inf", "infinity"):
        return pd.NA
    # Digits with optional single letter at end: 178c → 178, 90m → 90
    m = re.match(r"^(\d+)([a-zA-Z])?$", s)
    if m:
        return m.group(1)
    # Single letter + digits: m90 → 90
    m = re.match(r"^([a-zA-Z])(\d+)$", s)
    if m:
        return m.group(2)
    return val


def _columns_with_int_like_values(df: pd.DataFrame, object_columns: list[str]) -> set[str]:
    """Columns where most values look like integers (digits, optional single letter, or inf). Value-based."""
    out: set[str] = set()
    for col in object_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 5:
            continue
        match_count = non_empty.apply(lambda v: bool(_INT_LIKE_PATTERN.match(v))).sum()
        if match_count >= len(non_empty) * 0.8:
            out.add(col)
    return out


def normalize_int_like_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize int-like columns: inf→null, 178c→178, m90→90. Inferred from values only."""
    out = df.copy()
    object_cols = [c for c in out.select_dtypes(include=["object"]).columns if c in out.columns]
    int_like_cols = _columns_with_int_like_values(out, object_cols)
    for col in int_like_cols:
        out[col] = out[col].apply(_normalize_int_like_string)
    return out


def _columns_with_ratio_like_values(df: pd.DataFrame, object_columns: list[str]) -> set[str]:
    """Columns where a significant share of values look like 'systolic/diastolic' (e.g. 120/80). Adaptable, no config."""
    out: set[str] = set()
    for col in object_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        match_count = non_empty.apply(lambda v: bool(_RATIO_LIKE_PATTERN.match(v))).sum()
        if match_count >= len(non_empty) * _RATIO_COLUMN_MIN_SHARE:
            out.add(col)
    return out


def _columns_to_skip_numeric(df: pd.DataFrame) -> set[str]:
    """Columns to skip for numeric normalization/coercion. Value-based only: dates, ratio-like,
    email-like, phone-like, person-name-like. No header-based hints.
    """
    skip: set[str] = set()
    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c in df.columns]
    # Skip strongly date-like columns for numeric normalization even if a few values
    # fail parsing; we still require 100% parse later before coercing to datetime.
    for col in object_cols:
        series = df[col]
        as_str = series.astype(str)
        non_empty = as_str.str.strip() != ""
        if not non_empty.any():
            continue
        coerced = _coerce_series_to_datetime(as_str)
        share_dt = (coerced.notna() & non_empty).sum() / non_empty.sum()
        if share_dt >= _COERCE_DATETIME_THRESHOLD:
            skip.add(col)
    # Also skip columns that are 100% parseable dates according to strict detector.
    skip |= set(_find_date_columns(df))
    skip |= _columns_with_ratio_like_values(df, object_cols)
    skip |= _columns_with_email_like_values(df, object_cols)
    skip |= _columns_with_phone_like_values(df, object_cols)
    # Person-name-like detection shares logic with capitalization helper; reuse it by calling resolver
    for c in _resolve_person_name_columns(df):
        skip.add(c)
    return skip


def normalize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric-looking strings. Skips dates, text-like columns, and ratio-like columns (e.g. 120/80) from data."""
    out = df.copy()
    skip_cols = _columns_to_skip_numeric(out)
    for col in out.select_dtypes(include=["object"]).columns:
        if col in skip_cols:
            continue
        out[col] = out[col].apply(_normalize_numeric_string)
    return out


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely null or have no name."""
    out = df.copy()
    out = out.dropna(axis=1, how="all")
    out = out[[c for c in out.columns if str(c).strip() != ""]]
    if not out.columns.is_unique:
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def _prefer_int_if_whole(series: pd.Series) -> pd.Series:
    """If numeric series has only whole numbers (and maybe NaN/inf), use nullable Int64; inf/NaN become NA."""
    if not pd.api.types.is_float_dtype(series):
        return series
    finite = series.copy()
    finite = finite.replace([float("inf"), float("-inf")], pd.NA)
    non_null = finite.dropna()
    if len(non_null) == 0:
        return finite
    arr = non_null.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or not (arr == np.trunc(arr)).all():
        return series
    if (arr < np.iinfo(np.int64).min).any() or (arr > np.iinfo(np.int64).max).any():
        return series
    try:
        return finite.astype("Int64")
    except (ValueError, TypeError, pd.errors.IntCastingNaNError):
        return series


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce only when most values are numeric; mixed/text columns stay as-is so output stays adaptable."""
    out = df.copy()
    skip_cols = _columns_to_skip_numeric(out)
    object_cols = [c for c in out.select_dtypes(include=["object"]).columns if c in out.columns]
    skip_cols |= _columns_with_currency_values(out, object_cols)
    for col in object_cols:
        if col in skip_cols:
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        non_empty = out[col].notna() & (out[col].astype(str).str.strip() != "")
        n = non_empty.sum()
        if n == 0:
            continue
        # Share of non-empty values that converted to numeric (mixed columns stay text)
        numeric_share = (coerced.notna() & non_empty).sum() / n
        if numeric_share >= _COERCE_MIN_NUMERIC_SHARE and coerced.notna().any():
            out[col] = _prefer_int_if_whole(coerced)
    return out


def _row_is_empty(row: pd.Series) -> bool:
    """True if every value is null or blank/nan-like (so all-semicolon rows are dropped)."""
    for v in row:
        if pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, str) and v.strip().lower() == "nan":
            continue
        return False
    return True


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all values are null or blank or the literal 'nan'. Addresses empty rows in messy data."""
    mask = df.apply(_row_is_empty, axis=1)
    return df.loc[~mask].copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows. Optionally sort for deterministic order."""
    return df.drop_duplicates()


# Minimum share of non-null values that must parse for adaptable coercion.
_COERCE_NUMERIC_THRESHOLD = 0.9
_COERCE_DATETIME_THRESHOLD = 0.8


def _parse_single_date(val) -> pd.Timestamp | None:
    """Try to parse a single value as a date using DATE_FORMATS, then dateutil fallback."""
    if pd.isna(val):
        return None
    s = _preprocess_date_string(str(val).strip())
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except (ValueError, TypeError):
            continue
    # Do not try to "correct" invalid dates (e.g. 1984-02-34, 1976-13-24) via
    # fuzzy parsers like dateutil. If a value doesn't match one of our known
    # formats with a valid calendar date, treat it as invalid (NaT) so it
    # becomes null instead of being silently adjusted.
    return None


def _coerce_series_to_datetime(series: pd.Series) -> pd.Series:
    """Parse each value individually using known DATE_FORMATS + dateutil fallback."""
    return series.apply(lambda v: _parse_single_date(v) if not pd.isna(v) else pd.NaT)


def _infer_coercion_type(series: pd.Series) -> str | None:
    """
    Infer target type for an object column from values only.
    Returns "numeric", "datetime", or None. Adaptable: no config, pure value-based.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    # First, normalize obvious word-form numbers so they count toward the numeric share.
    non_null = _normalize_word_numbers_series(non_null)
    as_str = non_null.astype(str).str.strip()
    as_str = as_str[as_str.str.len() > 0]
    if len(as_str) == 0:
        return None

    # Use the same numeric string normalizer as the coercion step so inference
    # and coercion agree (e.g. '9,.0', '8,9f', '08.9', '8..8', '++8.7' all
    # become clean float-like strings before we measure the numeric share).
    #
    # IMPORTANT: values that clearly look like dates (contain '/' or '-') should
    # not count toward the numeric share, otherwise columns like "Last Restocked"
    # with values such as "3/1/2024" can be (incorrectly) inferred as numeric.
    date_punct = as_str.str.contains(r"[/-]", regex=True)
    norm_for_numeric = as_str.copy()
    norm_for_numeric[date_punct] = pd.NA
    norm_for_numeric = norm_for_numeric.astype(str).apply(_normalize_numeric_string)
    numeric = pd.to_numeric(norm_for_numeric, errors="coerce")
    dt = _coerce_series_to_datetime(as_str)
    share_n = numeric.notna().mean()
    share_d = dt.notna().mean()

    # For small columns, be strict: only coerce when 100% of non-null values parse.
    if len(as_str) < 10:
        if numeric.notna().all():
            return "numeric"
        if dt.notna().all():
            return "datetime"
        return None

    # For larger columns, use adaptable thresholds.
    if share_n >= _COERCE_NUMERIC_THRESHOLD and share_n >= share_d:
        return "numeric"
    if share_d >= _COERCE_DATETIME_THRESHOLD and share_d > share_n:
        return "datetime"
    return None


def coerce_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce object columns to numeric or datetime when a large share of values
    parse as that type. Fully adaptable (value-based); no config.
    """
    out = df.copy()
    date_cols = set(_find_date_columns(out))

    for col in date_cols:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        coerced = _coerce_series_to_datetime(series)
        new_nats = (series.notna() & coerced.isna()).sum()
        if new_nats > 0:
            continue
        out[col] = coerced

    for col in out.select_dtypes(include=["object"]).columns:
        if col in date_cols:
            continue
        target = _infer_coercion_type(out[col])
        if target == "numeric":
            raw_str = out[col].astype(str)
            series_norm = _normalize_word_numbers_series(out[col])
            series_norm = series_norm.astype(str).apply(_normalize_numeric_string)
            coerced = pd.to_numeric(series_norm, errors="coerce")
            actually_present = out[col].notna()
            new_nans = (actually_present & coerced.isna()).sum()
            if new_nans > 0:
                continue
            # Use original values to detect decimal (e.g. "9." → normalized "9" has no dot; raw had "9.")
            has_decimal = raw_str.str.contains(r"\.", regex=True, na=False).any()
            if has_decimal:
                out[col] = coerced.astype("float64")
            else:
                out[col] = _prefer_int_if_whole(coerced)
        elif target == "datetime":
            coerced = _coerce_series_to_datetime(out[col])
            new_nats = (out[col].notna() & coerced.isna()).sum()
            if new_nats > 0:
                continue
            out[col] = coerced
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compose core, domain-neutral cleaning steps in fixed order.

    - Whitespace and null normalization, header cleanup
    - Drop empty rows/columns
    - Adaptable mixed-type coercion (numeric/datetime inferred from values)
    - Drop duplicates

    All title-casing (people, products, titles) is handled by the AI layer.
    Domain-specific normalization (countries, content ratings, etc.) also remains in Gemini.
    """
    steps_applied: list[str] = []
    metadata: dict[str, Any] = {}

    before = df
    df = trim_whitespace(df)
    if not df.equals(before):
        steps_applied.append("trim_whitespace")

    before = df
    df = standardize_nulls(df)
    if not df.equals(before):
        steps_applied.append("standardize_nulls")

    before = df
    df = normalize_column_names(df)
    if not df.equals(before):
        steps_applied.append("normalize_column_names")

    before = df
    df = drop_empty_columns(df)
    if not df.equals(before):
        steps_applied.append("drop_empty_columns")

    if CAPITALIZE_PERSON_NAMES:
        before = df
        df = capitalize_person_names(df)
        if not df.equals(before):
            steps_applied.append("capitalize_person_names")

    before = df
    df = normalize_int_like_strings(df)
    if not df.equals(before):
        steps_applied.append("normalize_int_like_strings")

    before = df
    df = normalize_numeric_strings(df)
    if not df.equals(before):
        steps_applied.append("normalize_numeric_strings")

    before = df
    df = normalize_email_like_columns(df)
    if not df.equals(before):
        steps_applied.append("normalize_email_like_columns")

    before = df
    df = coerce_mixed_types(df)
    if not df.equals(before):
        steps_applied.append("coerce_mixed_types")

    before = df
    df = drop_empty_rows(df)
    if not df.equals(before):
        steps_applied.append("drop_empty_rows")

    before = df
    df = remove_duplicates(df)
    if not df.equals(before):
        steps_applied.append("remove_duplicates")

    # Detect currency-like columns based on original string values so the UI can
    # render them with $ formatting even though we coerce them to numeric.
    currency_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_empty = series.dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]
        if len(non_empty) == 0:
            continue
        with_dollar = non_empty.str.startswith("$").sum()
        if with_dollar >= len(non_empty) * _CURRENCY_IN_VALUE_MIN_SHARE:
            currency_cols.append(col)

    df = df.reset_index(drop=True)
    metadata["steps_applied"] = steps_applied
    metadata["currency_columns"] = currency_cols
    return df, metadata
