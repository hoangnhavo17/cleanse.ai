"""Unit tests for cleaning steps."""
import pandas as pd
import pytest

from app.services.cleaner import (
    clean,
    coerce_numeric,
    drop_empty_columns,
    drop_empty_rows,
    normalize_column_names,
    normalize_numeric_strings,
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


def test_drop_empty_columns():
    df = pd.DataFrame({"a": [1, 2], "": [pd.NA, pd.NA], "b": [3, 4]})
    out = drop_empty_columns(df)
    assert list(out.columns) == ["a", "b"]


def test_clean_end_to_end():
    df = pd.DataFrame({
        "name": ["  Alice  ", "  Bob  ", "  Alice  "],
        "value": [" 42 ", " 43 ", " 42 "],
    })
    out, meta = clean(df)
    assert out["name"].str.strip().tolist() == ["Alice", "Bob"]
    assert len(out) == 2
    assert "steps_applied" in meta


*** End of File
