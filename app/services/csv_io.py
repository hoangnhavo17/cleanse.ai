"""
CSV I/O: load_csv, save_csv.
Auto-detects delimiter when sep=None so comma- and semicolon-separated files both work.
"""
from pathlib import Path

import pandas as pd

from app.config import ENCODING, CSV_SEP


def _detect_sep(path: str | Path) -> str:
    """Sniff first line: use comma or semicolon, whichever yields more columns (min 2)."""
    path = Path(path)
    with open(path, "r", encoding=ENCODING, errors="replace") as f:
        first = f.readline()
    n_semi = len(first.split(";"))
    n_comma = len(first.split(","))
    if n_semi >= 2 and n_semi >= n_comma:
        return ";"
    if n_comma >= 2:
        return ","
    return ";"


def _header_and_max_cols(path: str | Path, sep: str) -> tuple[list[str], int]:
    """Return (header column names, max number of columns in any data row)."""
    path = Path(path)
    with open(path, "r", encoding=ENCODING, errors="replace") as f:
        first = f.readline()
        header = [c.strip() for c in first.rstrip("\n\r").split(sep)]
        max_cols = len(header)
        for line in f:
            n = len(line.rstrip("\n\r").split(sep))
            if n > max_cols:
                max_cols = n
    return header, max_cols


def _looks_thousands_fragment(left: object, right: object) -> bool:
    """True if left is 1–3 digits and right is exactly 3 digits (e.g. '1' and '000').
    Right may be coerced to number (e.g. 0 or 0.0 from '000'), so we accept that too.
    """
    ls, rs = str(left).strip(), str(right).strip()
    if not ls or not ls.isdigit() or len(ls) < 1 or len(ls) > 3:
        return False
    if rs.isdigit() and len(rs) == 3:
        return True
    # Pandas may coerce "000" to 0 or 0.0
    if rs in ("0", "0.0", "0.00"):
        return True
    return False


def _repair_comma_thousands(df: pd.DataFrame, sep_used: str) -> pd.DataFrame:
    """If CSV is comma-separated and a numeric field contained a thousands comma (e.g. 1,000),
    the row can have an extra column. Find the consecutive pair (i, i+1) that looks like '1'+'000',
    merge for those rows, shift left, and drop the extra column.
    """
    if sep_used != ",":
        return df
    cols = list(df.columns)
    n = len(cols)
    if n < 2:
        return df
    # Find index i such that (col i, col i+1) has the most rows matching thousands pattern
    best_i = -1
    best_count = 0
    for i in range(n - 1):
        left_name, right_name = cols[i], cols[i + 1]
        count = df.apply(
            lambda row: _looks_thousands_fragment(row[left_name], row[right_name]),
            axis=1,
        ).sum()
        if count > best_count:
            best_count = count
            best_i = i
    if best_i < 0 or best_count == 0:
        return df
    prev_name = cols[best_i]
    frag_name = cols[best_i + 1]
    mask = df.apply(
        lambda row: _looks_thousands_fragment(row[prev_name], row[frag_name]),
        axis=1,
    )
    # Merge (e.g. Quantity "1" + Price "000" → Quantity "1000"). Right may be 0/0.0 from "000"
    def _right_frag(x: object) -> str:
        s = str(x).strip()
        return "000" if s in ("0", "0.0", "0.00") else s

    merged = (
        df.loc[mask, prev_name].astype(str).str.strip()
        + df.loc[mask, frag_name].apply(_right_frag)
    )
    df.loc[mask, prev_name] = merged
    # For merged rows only: shift left so col j+1..end-1 get col j+2..end
    j = best_i + 1
    for c in range(j, n - 1):
        df.loc[mask, cols[c]] = df.loc[mask, cols[c + 1]].values
    # Drop the extra column (last one, which is typically Unnamed)
    df = df.drop(columns=[cols[n - 1]])
    return df


def load_csv(path: str | Path, sep: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load CSV. When sep is None, auto-detect (comma vs semicolon). Returns (df, sep_used).
    Uses keep_default_na=False so source literals like 'None' and 'NONE' stay as-is; only blank cells become null.
    Repairs comma-separated rows where a numeric field contained a thousands comma (e.g. 1,000) so it stays one column.
    """
    path = Path(path)
    sep_used = sep if sep is not None else _detect_sep(path)
    kwargs = dict(
        sep=sep_used,
        encoding=ENCODING,
        encoding_errors="replace",
        on_bad_lines="skip",
        keep_default_na=False,
    )
    if sep_used == ",":
        header, max_cols = _header_and_max_cols(path, sep_used)
        if max_cols > len(header):
            # Force enough columns so no value is dropped; repair will merge and drop the extra.
            names = header + [f"Unnamed_{i}" for i in range(max_cols - len(header))]
            try:
                df = pd.read_csv(path, **kwargs, names=names, header=0)
                df = _repair_comma_thousands(df, sep_used)
            except pd.errors.ParserError:
                # Some rows may have fewer fields than the header-derived width; fall back to
                # a simpler read that lets pandas handle ragged rows (with on_bad_lines="skip").
                df = pd.read_csv(path, **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    return df, sep_used


def save_csv(df: pd.DataFrame, path: str | Path, sep: str | None = None) -> None:
    """Save DataFrame to CSV. Uses given sep or config CSV_SEP."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep_used = sep if sep is not None else CSV_SEP
    df.to_csv(path, index=False, sep=sep_used, encoding=ENCODING)
