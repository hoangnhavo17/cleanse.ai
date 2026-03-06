"""
Cleaning steps: trim, standardize nulls, normalize column names, fix country/title typos,
capitalize person names, normalize release dates, normalize numeric strings, coerce_numeric,
drop empty rows/columns, remove_duplicates. Composed in clean(). All pure.
"""
import re
import unicodedata
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd

from app.config import (
    CANONICAL_COUNTRIES,
    CAPITALIZE_PERSON_NAMES,
    COLUMN_RENAMES,
    COLUMN_RENAME_PREFIXES,
    COUNTRY_COLUMNS,
    COUNTRY_FUZZY_CUTOFF,
    COUNTRY_TYPO_MAP,
    DURATION_COLUMNS,
    PERSON_NAME_CASE,
    PERSON_NAME_COLUMNS,
    RELEASE_DATE_COLUMN,
    TITLE_COLUMNS,
    TITLE_PATTERN_REPLACEMENTS,
    TITLE_TYPO_MAP,
)

# Date formats to try (strptime-style). Order: unambiguous first, then DD/MM, then MM/DD.
RELEASE_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
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
    "none",
    "null",
    "#n/a",
    "-",
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


def _resolve_columns(
    df: pd.DataFrame,
    config_columns: tuple[str, ...],
    fallback_substrings: tuple[str, ...],
) -> list[str]:
    """Columns to apply a step to: config list (if non-empty) else those whose name contains any of fallback_substrings."""
    if config_columns:
        return [c for c in config_columns if c in df.columns]
    return [
        c for c in df.columns
        if any(sub in str(c).lower() for sub in fallback_substrings)
    ]


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
    """Strip, config overrides, generic cleanup (encoding, spaces, case), then spelling fix (fuzzy per-token)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    renames: dict[str, str] = {}
    # 1) Optional config overrides (exact + prefix)
    for old, new in COLUMN_RENAMES.items():
        if old in out.columns:
            renames[old] = new
    for col in list(out.columns):
        if col in renames:
            continue
        for prefix, new_name in COLUMN_RENAME_PREFIXES.items():
            if col.startswith(prefix) and col != new_name:
                renames[col] = new_name
                break
    # 2) Generic cleanup then spelling fix (word-level fuzzy match; no full-header list)
    used = set(out.columns) | set(renames.values())
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


def _parse_release_date(val: str, preferred_fmt: str | None = None):
    """Parse a single date value; return YYYY-MM-DD string or pd.NA. Try preferred_fmt first if given."""
    s = _preprocess_date_string(val)
    if not s:
        return pd.NA
    formats_to_try = (
        [preferred_fmt] + [f for f in RELEASE_DATE_FORMATS if f != preferred_fmt]
        if preferred_fmt
        else list(RELEASE_DATE_FORMATS)
    )
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    try:
        from dateutil.parser import parse as dateutil_parse

        dt = dateutil_parse(s, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, ImportError):
        return pd.NA


def _infer_date_format(series: pd.Series) -> str | None:
    """Return the format that parses a majority of non-null values, or None."""
    non_null = series.dropna().astype(str).apply(_preprocess_date_string)
    non_null = non_null[non_null != ""]
    n = len(non_null)
    if n == 0:
        return None
    best_fmt: str | None = None
    best_count = 0
    for fmt in RELEASE_DATE_FORMATS:
        count = 0
        for s in non_null:
            try:
                datetime.strptime(s, fmt)
                count += 1
            except (ValueError, TypeError):
                pass
        if count > best_count:
            best_count = count
            best_fmt = fmt
    if best_fmt is not None and best_count >= n * 0.5:
        return best_fmt
    return None


def _normalize_country_raw(s: str) -> str:
    """Strip and remove trailing digits/punctuation for matching."""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"[\d\.\,\;\s]+$", "", s).strip()  # trailing digits/punct
    return s


def _country_to_canonical(val: str) -> str:
    """Map a country value to canonical form: typo map → normalize → exact match → fuzzy match."""
    if pd.isna(val) or str(val).strip() == "":
        return val
    s = str(val).strip()
    if s in COUNTRY_TYPO_MAP:
        return COUNTRY_TYPO_MAP[s]
    norm = _normalize_country_raw(s)
    if not norm:
        return val
    if norm in CANONICAL_COUNTRIES:
        return norm
    matches = get_close_matches(norm, list(CANONICAL_COUNTRIES), n=1, cutoff=COUNTRY_FUZZY_CUTOFF)
    return matches[0] if matches else val


def fix_country_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Replace country-like columns with canonical names. Columns from config or auto-detect ('country')."""
    out = df.copy()
    cols = _resolve_columns(df, COUNTRY_COLUMNS, ("country",))
    for col in cols:
        out[col] = out[col].apply(_country_to_canonical)
    return out


def _apply_title_fixes(val: str) -> str:
    """Apply full-string typo map, then pattern (substring) replacements."""
    if pd.isna(val):
        return val
    s = str(val)
    if s in TITLE_TYPO_MAP:
        s = TITLE_TYPO_MAP[s]
    for pattern, replacement in TITLE_PATTERN_REPLACEMENTS.items():
        s = s.replace(pattern, replacement)
    return s


def fix_title_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Replace title-like columns: exact map, then pattern replacements. Columns from config or auto-detect ('title','name')."""
    out = df.copy()
    cols = _resolve_columns(df, TITLE_COLUMNS, ("title", "name"))
    for col in cols:
        out[col] = out[col].apply(_apply_title_fixes)
    return out


_CONTENT_RATING_CANONICAL: dict[str, str] = {
    # Film/TV style content ratings; keys are uppercased variants.
    "G": "G",
    "PG": "PG",
    "PG-13": "PG-13",
    "PG13": "PG-13",
    "R": "R",
    "NC-17": "NC-17",
    "NC17": "NC-17",
    "NOT RATED": "Not Rated",
    "UNRATED": "Unrated",
    "APPROVED": "Approved",
}


def _normalize_content_rating_value(val: object) -> object:
    """Normalize content rating codes without losing semantics (e.g. keep 'PG-13')."""
    if pd.isna(val) or val is None:
        return val
    s = str(val).strip()
    if not s:
        return val
    key = s.upper()
    return _CONTENT_RATING_CANONICAL.get(key, s)


def normalize_content_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize content-rating-like columns to canonical forms, based on values not headers.

    We look for object columns where a substantial share of non-null values look like
    known rating codes (PG, PG-13, R, etc.), and only normalize those.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        series = out[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        # Share of values that map to a known content rating code
        upper_vals = non_empty.str.upper()
        share_rating_like = upper_vals.isin(_CONTENT_RATING_CANONICAL.keys()).mean()
        if share_rating_like < 0.4:
            continue
        out[col] = out[col].apply(_normalize_content_rating_value)
    return out


def _resolve_person_name_columns(df: pd.DataFrame) -> list[str]:
    """Columns to capitalize as person names.

    Priority:
    1. Explicit config PERSON_NAME_COLUMNS (if non-empty).
    2. Otherwise, combine:
       - Header-based hints: column name contains 'name' but not 'title'.
       - Value-based detection: columns where most non-empty values look like person names
         (alphabetic, 1–4 tokens, not emails/phones, few digits/punctuation).
    """
    if PERSON_NAME_COLUMNS:
        return [c for c in PERSON_NAME_COLUMNS if c in df.columns]

    header_hits: list[str] = [
        c
        for c in df.columns
        if "name" in str(c).strip().lower() and "title" not in str(c).strip().lower()
    ]

    person_like: set[str] = set()
    for c in df.select_dtypes(include=["object"]).columns:
        if c in header_hits:
            continue
        series = df[c].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 20:
            continue
        lower_vals = non_empty.str.lower()
        # Exclude obvious emails/URLs and heavy-digit fields
        if (lower_vals.str.contains("@")).mean() >= 0.2:
            continue
        if (non_empty.str.contains(r"\d", regex=True)).mean() >= 0.3:
            continue
        # Exclude columns that look like content ratings (PG, PG-13, R, etc.)
        upper_vals = non_empty.str.upper()
        rating_like_share = upper_vals.isin(_CONTENT_RATING_CANONICAL.keys()).mean()
        if rating_like_share >= 0.4:
            continue
        # Exclude code-like columns: one value repeats a lot (e.g. USA, Approved).
        # The norm is that repeated value; variants should be corrected by domain steps
        # (country, content rating), not by person-name title-casing (which would turn USA → Usa).
        n_unique = non_empty.nunique()
        if n_unique <= 50:
            mode_count = non_empty.value_counts().iloc[0]
            if mode_count / len(non_empty) >= 0.15:
                continue
        # Candidate person-name pattern: 1–4 tokens, alphabetic with limited punctuation
        pattern = r"^[A-Za-z][A-Za-z\.'\-]*(\s+[A-Za-z][A-Za-z\.'\-]*){0,3}$"
        share_name_like = non_empty.str.match(pattern).mean()
        if share_name_like >= 0.6:
            person_like.add(c)

    return list(dict.fromkeys(header_hits + list(person_like)))


def _apply_name_case(s: str, style: str) -> str:
    """Apply casing style: 'title' (each word) or 'sentence' (first letter only)."""
    if style == "sentence":
        return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper() if s else s
    return s.title()


def capitalize_person_names(df: pd.DataFrame) -> pd.DataFrame:
    """Case values in person-name columns. Config: columns, on/off (CAPITALIZE_PERSON_NAMES), style (PERSON_NAME_CASE)."""
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


def normalize_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Extract leading number from duration-like columns (e.g. '178c' → 178). Columns from config or auto-detect."""
    out = df.copy()
    cols = _resolve_columns(
        df, DURATION_COLUMNS,
        ("duration", "length", "runtime", "minutes", "mins", "time"),
    )

    def _extract_leading_number(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if not s:
            return pd.NA
        m = re.match(r"^(\d+(?:\.\d+)?)", s)
        return m.group(1) if m else val

    for col in cols:
        out[col] = out[col].apply(_extract_leading_number)
    return out


def _find_release_date_column(df: pd.DataFrame) -> str | None:
    """Return column name that matches RELEASE_DATE_COLUMN (case-insensitive), or None."""
    if RELEASE_DATE_COLUMN is None:
        return None
    key = RELEASE_DATE_COLUMN.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == key:
            return c
    return None


# Min share of non-null date values that one format must parse to "keep existing pattern" (no normalize).
_DATE_MAJORITY_THRESHOLD = 0.85


def _date_column_has_clear_majority_format(series: pd.Series, threshold: float = _DATE_MAJORITY_THRESHOLD) -> bool:
    """True if one date format parses at least `threshold` of date-like values (then we keep existing pattern)."""
    non_null = series.astype(str).apply(_preprocess_date_string)
    non_null = non_null[non_null != ""]
    non_null = non_null[non_null.str.lower() != "nan"]
    n = len(non_null)
    if n == 0:
        return True
    best_count = 0
    for fmt in RELEASE_DATE_FORMATS:
        count = sum(
            1
            for s in non_null
            if _parse_with_fmt(s, fmt)
        )
        if count > best_count:
            best_count = count
    return best_count >= n * threshold


def _parse_with_fmt(s: str, fmt: str) -> bool:
    """Return True if s parses with the given format."""
    try:
        datetime.strptime(s, fmt)
        return True
    except (ValueError, TypeError):
        return False


def _find_date_columns(df: pd.DataFrame) -> list[str]:
    """Columns to consider as dates: config release + header hints + value-based detection.

    Header hints: name contains 'date' or 'restocked'.
    Value-based: for object/string columns that aren't obviously text-like, if a large share of values
    parse as dates with one of our known formats, treat them as date columns even if the header
    doesn't mention 'date'.
    """
    cols: list[str] = []
    release = _find_release_date_column(df)
    if release is not None:
        cols.append(release)
    # 1) Header-based hints
    for c in df.columns:
        if c in cols:
            continue
        lower = str(c).lower()
        if "date" in lower or "restocked" in lower:
            cols.append(c)
    # 2) Value-based detection: avoid obviously text-like columns; look for columns where a
    # high share of non-empty values parse with one of our known date formats.
    for c in df.columns:
        if c in cols:
            continue
        series = df[c]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        lower_name = str(c).lower()
        if any(sub in lower_name for sub in _TEXT_LIKE_COLUMN_SUBSTRINGS):
            continue
        non_null = series.astype(str).apply(_preprocess_date_string)
        non_null = non_null[non_null != ""]
        non_null = non_null[non_null.str.lower() != "nan"]
        n = len(non_null)
        if n < 10:
            continue
        best_count = 0
        for fmt in RELEASE_DATE_FORMATS:
            count = 0
            for s in non_null:
                if _parse_with_fmt(s, fmt):
                    count += 1
            if count > best_count:
                best_count = count
        if best_count >= n * 0.8:
            cols.append(c)
    return cols


def normalize_release_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize date columns to YYYY-MM-DD only when there is no clear majority pattern.
    If one format parses a large majority of values (e.g. DD/MM/YYYY), keep the existing pattern.
    """
    out = df.copy()
    for col in _find_date_columns(out):
        series = out[col]
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
            def _from_yyyymmdd(val):
                if pd.isna(val):
                    return pd.NA
                s = str(int(val))
                if len(s) == 8:
                    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                if len(s) == 6:
                    y = int(s[4:6])
                    year = (2000 + y) if 0 <= y <= 68 else (1900 + y) if 69 <= y <= 99 else y
                    return f"{year}-{s[:2]}-{s[2:4]}"
                return pd.NA
            out[col] = series.apply(_from_yyyymmdd)
            continue
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            continue
        if _date_column_has_clear_majority_format(out[col]):
            continue
        inferred_fmt = _infer_date_format(out[col])
        out[col] = out[col].apply(
            lambda v: pd.NA if pd.isna(v) else _parse_release_date(str(v).strip(), preferred_fmt=inferred_fmt)
        )
    return out


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


def _normalize_numeric_string(val: str) -> str:
    """Strip $ and spaces, keep only digits/dot/comma; comma→dot; multiple dots→thousand sep (remove).
    Replaces word numbers (e.g. thirty→30, SIXTY THOUSAND→60000). Preserves titles like 'Se7en'.
    For $-prefixed strings, replace digit lookalikes (o/O→0, l/I→1).
    """
    if not isinstance(val, str) or pd.isna(val):
        return val
    s = val.strip()
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
    # Keep currency symbol when source had $ (e.g. "$ 28815245" → "$28815245")
    if isinstance(val, str) and val.strip().startswith("$"):
        return "$" + cleaned
    return cleaned


# Substrings in column names that suggest "keep as text" (no numeric normalize/coerce). Data-driven ratio detection
# (e.g. 120/80) is used in addition, so no per-dataset config is needed.
_TEXT_LIKE_COLUMN_SUBSTRINGS: tuple[str, ...] = (
    "title", "titl", "name", "director", "country", "genre", "rating", "content",
    "date", "year", "id", "code", "category", "type", "description", "note",
    "phone", "email",
)
# Only skip coercion when values actually have "$" (source included it); don't add $ or skip by column name.
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


def _columns_with_currency_values(df: pd.DataFrame, object_columns: list[str]) -> set[str]:
    """Columns where a significant share of values start with '$' (source had currency). Don't coerce those."""
    out: set[str] = set()
    for col in object_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        with_dollar = non_empty.str.startswith("$").sum()
        if with_dollar >= len(non_empty) * _CURRENCY_IN_VALUE_MIN_SHARE:
            out.add(col)
    return out


def _columns_to_skip_numeric(df: pd.DataFrame) -> set[str]:
    """Columns to skip for numeric normalization/coercion.

    - Dates (including restocked).
    - Text-like headers (name, title, id, etc.).
    - Value-based text: ratio-like (120/80), email-like, phone-like, and person-name-like columns.
    """
    skip = set(_find_date_columns(df))
    for c in df.columns:
        if "restocked" in str(c).lower():
            skip.add(c)
    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c in df.columns]
    for col in object_cols:
        if any(sub in str(col).lower() for sub in _TEXT_LIKE_COLUMN_SUBSTRINGS):
            skip.add(col)
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


def _columns_already_domain_normalized(df: pd.DataFrame) -> set[str]:
    """Columns that are handled by a domain step (country, content rating). Skip these in generic categorical normalize."""
    out: set[str] = set()
    out.update(_resolve_columns(df, COUNTRY_COLUMNS, ("country",)))
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) == 0:
            continue
        upper_vals = non_empty.str.upper()
        if upper_vals.isin(_CONTENT_RATING_CANONICAL.keys()).mean() >= 0.4:
            out.add(col)
    return out


def normalize_categorical_to_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no domain step exists: for code-like columns (low cardinality, one dominant value),
    normalize variants to the most common value (e.g. usa, Usa → USA if USA is the mode).
    Skips columns already handled by country or content-rating domain steps.
    """
    out = df.copy()
    skip = _columns_already_domain_normalized(out)
    for col in out.select_dtypes(include=["object"]).columns:
        if col in skip:
            continue
        series = out[col].dropna().astype(str).str.strip()
        non_empty = series[series != ""]
        if len(non_empty) < 10:
            continue
        n_unique = non_empty.nunique()
        if n_unique > 50:
            continue
        mode_val = non_empty.mode()
        if len(mode_val) == 0:
            continue
        norm = str(mode_val.iloc[0]).strip()
        if not norm:
            continue
        mode_share = (non_empty == norm).mean()
        if mode_share < 0.15:
            continue

        def _to_norm(v: object) -> object:
            if pd.isna(v) or v is None:
                return v
            s = str(v).strip()
            if not s:
                return v
            if s.lower() == norm.lower():
                return norm
            return v

        out[col] = out[col].apply(_to_norm)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Compose cleaning steps in fixed order. Dataset-agnostic: column choices are config-driven
    or auto-detected by column name (e.g. any column with 'country' gets country canonicalization).
    """
    df = trim_whitespace(df)
    df = standardize_nulls(df)
    df = normalize_column_names(df)
    df = drop_empty_columns(df)
    df = fix_country_typos(df)
    df = fix_title_typos(df)
    if CAPITALIZE_PERSON_NAMES:
        df = capitalize_person_names(df)
    df = normalize_content_ratings(df)
    df = normalize_categorical_to_mode(df)
    df = normalize_release_dates(df)
    df = normalize_duration(df)
    df = normalize_numeric_strings(df)
    df = coerce_numeric(df)
    df = drop_empty_rows(df)
    df = remove_duplicates(df)
    return df.reset_index(drop=True)
