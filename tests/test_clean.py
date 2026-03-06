"""Unit tests for cleaning steps."""
import pandas as pd
import pytest

from app.services.cleaner import (
    clean,
    coerce_numeric,
    drop_empty_columns,
    drop_empty_rows,
    fix_country_typos,
    fix_title_typos,
    normalize_column_names,
    normalize_duration,
    normalize_numeric_strings,
    normalize_release_dates,
    remove_duplicates,
    standardize_nulls,
    trim_whitespace,
)


def test_trim_whitespace():
    df = pd.DataFrame({"a": ["  x  ", "y"], "b": [1, 2]})
    out = trim_whitespace(df)
    assert out["a"].tolist() == ["x", "y"]
    assert out["b"].tolist() == [1, 2]


def test_standardize_nulls():
    # Blank/whitespace and common sentinels like None/N/A become null
    df = pd.DataFrame({"a": ["", "  ", "None", "N/A", "ok"]})
    out = standardize_nulls(df)
    assert pd.isna(out.loc[0, "a"])
    assert pd.isna(out.loc[1, "a"])
    assert pd.isna(out.loc[2, "a"])
    assert pd.isna(out.loc[3, "a"])
    assert out.loc[4, "a"] == "ok"


def test_drop_empty_rows():
    df = pd.DataFrame({"a": [1, pd.NA, pd.NA], "b": [2, pd.NA, pd.NA]})
    out = drop_empty_rows(df)
    assert len(out) == 1
    assert out.loc[0, "a"] == 1
    # Row of literal "nan" (as from empty CSV row) is also dropped
    df2 = pd.DataFrame({"x": ["nan", "ok"], "y": ["nan", "ok"]})
    out2 = drop_empty_rows(df2)
    assert len(out2) == 1
    assert out2.loc[1, "x"] == "ok"


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    out = remove_duplicates(df)
    assert len(out) == 2


def test_normalize_column_names():
    # Generic cleanup + per-token spelling fix: "Original titl" → "Original Title", "Genr" → "Genre"
    df = pd.DataFrame({"Original titl": [1], "Genr": [2]})
    out = normalize_column_names(df)
    assert "Original Title" in out.columns
    assert "Genre" in out.columns


def test_normalize_numeric_strings_preserves_titles():
    df = pd.DataFrame({"a": ["Se7en", " 42 "], "b": ["$ 1.234,5", "x"]})
    out = normalize_numeric_strings(df)
    assert out.loc[0, "a"] == "Se7en"
    assert out.loc[1, "a"] == "42"
    assert out.loc[0, "b"] == "1234.5"
    assert out.loc[1, "b"] == "x"


def test_normalize_numeric_strings_income_digit_lookalikes():
    """Income typo: letter o as zero (e.g. $ 4o8,035,783 -> 408035783)."""
    df = pd.DataFrame({"Income": ["$ 4o8,035,783", "$ 123"]})
    out = normalize_numeric_strings(df)
    assert out.loc[0, "Income"] == "408035783"
    assert out.loc[1, "Income"] == "123"


def test_fix_country_typos():
    # Explicit map (US., US), normalize (Italy1 → Italy), fuzzy (New Zesland → New Zealand)
    df = pd.DataFrame({
        "Country": ["New Zesland", "New Zeland", "US.", "Italy1", "USA"],
    })
    out = fix_country_typos(df)
    assert out["Country"].tolist() == ["New Zealand", "New Zealand", "USA", "Italy", "USA"]


def test_fix_country_typos_robust():
    """Unseen typos still get fixed via normalize or fuzzy (no explicit map entry)."""
    df = pd.DataFrame({"Country": ["Italy2", "New Zealnd", "  Germany  "]})
    out = fix_country_typos(df)
    assert out.loc[0, "Country"] == "Italy"   # normalize strips trailing digits
    assert out.loc[1, "Country"] == "New Zealand"  # fuzzy match
    assert out.loc[2, "Country"] == "Germany"     # exact after strip


def test_fix_title_typos():
    # Pattern replacement B9 → è (no need for full-string map)
    df = pd.DataFrame({"Original title": ["La vita B9 bella", "Other Film"]})
    out = fix_title_typos(df)
    assert out.loc[0, "Original title"] == "La vita è bella"
    assert out.loc[1, "Original title"] == "Other Film"


def test_normalize_duration():
    df = pd.DataFrame({"Duration": ["178c", "96", " 120 ", pd.NA]})
    out = normalize_duration(df)
    assert out["Duration"].tolist() == ["178", "96", "120", pd.NA]


def test_normalize_release_dates():
    df = pd.DataFrame({
        "Release year": [
            "1995-02-10",
            "09 21 1972",
            "22 Feb 04",
            "23rd December of 1966",
            "18/11/1976",
            "10-29-99",
        ],
        "other": [1, 2, 3, 4, 5, 6],
    })
    out = normalize_release_dates(df)
    assert out["Release year"].tolist() == [
        "1995-02-10",
        "1972-09-21",
        "2004-02-22",
        "1966-12-23",
        "1976-11-18",
        "1999-10-29",
    ]


def test_drop_empty_columns():
    df = pd.DataFrame({"a": [1, 2], "": [pd.NA, pd.NA], "b": [3, 4]})
    out = drop_empty_columns(df)
    assert list(out.columns) == ["a", "b"]


def test_clean_end_to_end():
    df = pd.DataFrame({
        "name": ["  Alice  ", "  Bob  ", "  Alice  "],
        "value": [" 42 ", " 43 ", " 42 "],
    })
    out = clean(df)
    assert out["name"].str.strip().tolist() == ["Alice", "Bob"]
    assert len(out) == 2
