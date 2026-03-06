"""Integration tests: run pipeline on a temp messy.csv and assert cleaned.csv content."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.config import CSV_SEP
from app.main import run


def test_pipeline_produces_cleaned_csv():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "messy.csv"
        out_path = Path(tmp) / "cleaned.csv"
        pd.DataFrame({
            "a": ["  x  ", "  x  ", "y"],
            "b": ["1", "1", "2"],
        }).to_csv(in_path, index=False, sep=CSV_SEP)
        run(input_path=in_path, output_path=out_path, report=False)
        assert out_path.exists()
        df = pd.read_csv(out_path, sep=CSV_SEP)
        assert len(df) == 2
        assert df["a"].str.strip().tolist() == ["x", "y"]


def test_pipeline_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "messy.csv"
        out1 = Path(tmp) / "out1.csv"
        out2 = Path(tmp) / "out2.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(in_path, index=False, sep=CSV_SEP)
        run(input_path=in_path, output_path=out1, report=False)
        run(input_path=in_path, output_path=out2, report=False)
        assert out1.read_text() == out2.read_text()
