"""
Config: input/output paths, encoding, CSV format, column renames.
"""
from pathlib import Path

# Default paths (project root relative to this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Centralized data directories
DATA_DIR: Path = _PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
CLEANED_DATA_DIR: Path = DATA_DIR / "cleaned"
REPORTS_DIR: Path = DATA_DIR / "reports"

# Default input/output within data/ (can be overridden)
INPUT_PATH: Path = RAW_DATA_DIR / "messy.csv"
OUTPUT_PATH: Path = CLEANED_DATA_DIR / "cleaned.csv"

# Report outputs (in data/reports by default)
REPORT_JSON_PATH: Path = REPORTS_DIR / "cleaning_report.json"
REPORT_HTML_PATH: Path = REPORTS_DIR / "cleaning_report.html"

ENCODING: str = "utf-8"

# CSV delimiter (e.g. ";" for European / IMDb-style files)
CSV_SEP: str = ";"

# Optional per-dataset overrides. Leave empty to rely on built-in header vocabulary (fuzzy match).
# Only needed when you want to force a specific rename the vocabulary wouldn't pick.
COLUMN_RENAMES: dict[str, str] = {}
COLUMN_RENAME_PREFIXES: dict[str, str] = {}

# Column to standardize as release date (output format YYYY-MM-DD). None or missing column = skip step.
RELEASE_DATE_COLUMN: str | None = "Release year"

# Which columns to run domain steps on. Empty = auto-detect by name so any dataset works.
# E.g. empty COUNTRY_COLUMNS → any column whose name contains "country" gets canonicalization.
COUNTRY_COLUMNS: tuple[str, ...] = ()
TITLE_COLUMNS: tuple[str, ...] = ()
PERSON_NAME_COLUMNS: tuple[str, ...] = ()  # Empty = any column with "name" in name, excluding "title".
# Set False to skip person-name capitalization entirely (e.g. preserve original casing).
CAPITALIZE_PERSON_NAMES: bool = True
# How to case: "title" (Grace, Mary Jane) or "sentence" (Grace, Mary jane).
PERSON_NAME_CASE: str = "title"
DURATION_COLUMNS: tuple[str, ...] = ()

# Country: explicit overrides (checked first), then normalize + fuzzy match against canonical list.
COUNTRY_TYPO_MAP: dict[str, str] = {
    "US.": "USA",
    "US": "USA",
}
# Canonical country names (output forms). Used for exact match after normalize and for fuzzy match.
CANONICAL_COUNTRIES: tuple[str, ...] = (
    "USA", "UK", "New Zealand", "Italy", "Germany", "France", "Japan", "Brazil",
    "South Korea", "India", "Denmark", "Iran", "West Germany", "Spain", "Mexico",
    "Canada", "Australia", "China", "Sweden", "Norway", "Ireland", "Poland",
    "Argentina", "Russia", "Hong Kong", "Netherlands", "Belgium", "Austria",
    "Switzerland", "Greece", "Portugal", "Finland", "Czech Republic", "Hungary",
    "Romania", "Turkey", "Israel", "Egypt", "South Africa", "Thailand", "Indonesia",
    "Philippines", "Taiwan",
)
# Minimum similarity (0–1) to accept a fuzzy match. Higher = stricter.
COUNTRY_FUZZY_CUTOFF: float = 0.82

# Title: full-string overrides (exact match), then pattern replacements (substring).
TITLE_TYPO_MAP: dict[str, str] = {}
# Substring → replacement; applied so e.g. "B9" → "è" in any title (robust to similar typos).
TITLE_PATTERN_REPLACEMENTS: dict[str, str] = {
    "B9": "è",
}
