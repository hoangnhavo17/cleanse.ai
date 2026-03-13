"""
Config: input/output paths, encoding, CSV format, column renames.
"""
from pathlib import Path

# Default paths (project root relative to this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Centralized data directories
DATA_DIR: Path = _PROJECT_ROOT / "data"
CLEANED_DATA_DIR: Path = DATA_DIR / "cleaned"
REPORTS_DIR: Path = DATA_DIR / "reports"

# Default input/output (can be overridden)
INPUT_PATH: Path = DATA_DIR / "messy.csv"
OUTPUT_PATH: Path = CLEANED_DATA_DIR / "cleaned.csv"

# Report outputs (in data/reports by default)
REPORT_JSON_PATH: Path = REPORTS_DIR / "cleaning_report.json"
REPORT_HTML_PATH: Path = REPORTS_DIR / "cleaning_report.html"

ENCODING: str = "utf-8"

# CSV delimiter. This is a generic default; the loader can still sniff
# separators per-file when needed.
CSV_SEP: str = ","

# Person-name capitalization (value-based columns only). Disabled by default so that
# all title-casing (people, products, titles) is handled by the AI layer instead of
# standard cleaning.
CAPITALIZE_PERSON_NAMES: bool = False
# How to case: "title" (Grace, Mary Jane) or "sentence" (Grace, Mary jane).
PERSON_NAME_CASE: str = "title"
