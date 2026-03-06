# Cleanse

One pipeline: load CSV → rule-based clean → profile + automated rules → save cleaned CSV + report.

## What it does

- **Input:** `data/raw/messy.csv` (or any CSV path)
- **Output:** `data/cleaned/cleaned.csv` + reports in `data/reports/` (use `--no-report` to skip reports)

**Pipeline:** load CSV → **clean** (trim, nulls, renames, numeric, coerce, dedupe) → **profile** → **apply automated rules** (type fixes, dedupe) → save CSV + report.

### Features (rule-based clean + automated rules)

1. **Delimiter** — configurable `CSV_SEP` (e.g. `;` for European/IMDb-style files).
2. **Bad variable names** — strip whitespace, rename via `COLUMN_RENAMES`, fix common corruptions (e.g. "Original titl" → "Original title", "Genr" → "Genre").
3. **Trim whitespace** — strip leading/trailing whitespace on object (text) columns.
4. **Standardize nulls** — replace sentinels (e.g. `""`, `"NA"`, `"N/A"`, `#N/A`, `-`, `Nan`, `Inf`) with a single null representation.
5. **Empty rows and columns** — drop rows that are entirely null; drop columns that are entirely null or have no name.
6. **Numeric columns with symbols/units/separators** — strip `$` and spaces, normalize thousand separators (dots) and decimal separators (comma → dot) so values coerce to numeric; titles like "Se7en" are left unchanged.
7. **Basic numeric coercion** — coerce object columns to numeric when at least one value converts; leave the rest as object.
8. **Remove exact duplicates** — identical rows become one.

### Type and column strategy

- **Respect pandas' default inference** — On load, `read_csv` does not override dtypes; pandas infers types (e.g. numeric columns become int/float). Only encoding is set for reproducibility.
- **Normalize obvious text columns** — For columns that are object/string, apply trim and standardize nulls. Object dtype = treat as text.
- **Numeric coercion** — Coerce object columns to numeric when most values convert; mixed columns stay text.

## How to run

From the project root:

**Recommended: use the project venv** (avoids NumPy 2 / pandas conflicts with system or Anaconda):

```bash
# One-time: create venv and install (already done if .venv exists)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place your input file as data/raw/messy.csv (or set paths in app/config.py / .env)
# Run the pipeline (reads data/raw/messy.csv, writes data/cleaned/cleaned.csv)
python -m app.main
```

Or without activating: `/.venv/bin/python -m app.main` (Unix) or `.venv\Scripts\python -m app.main` (Windows).

Override paths programmatically by calling `run(input_path="...", output_path="...")` from code, or adjust paths in `app/config.py`.

To skip writing report files:

```bash
python -m app.main data/raw/messy.csv -o data/cleaned/out.csv --no-report
```

Report paths default to `data/reports/` (e.g. `data/reports/out_report.json`); or set `REPORT_JSON_PATH` / `REPORT_HTML_PATH` in `app/config.py`.

## Project structure

```
Cleanse/
├── app/
│   ├── __init__.py
│   ├── main.py          # entrypoint: run pipeline
│   ├── config.py        # paths, encoding, CSV_SEP, COLUMN_RENAMES, null sentinels
│   ├── services/
│   │   ├── csv_io.py           # load_csv (sep), save_csv
│   │   ├── cleaner.py           # rule-based: trim, nulls, renames, numeric, coerce, dedupe
│   │   ├── profiler.py          # profile_dataset (missing %, types, cardinality, outliers)
│   │   ├── automated_cleaner.py # profile-based: type fixes, dedupe
│   │   └── report_generator.py  # build_report, write_report_json/html
│   ├── models/
│   ├── routes/
│   └── utils/
├── data/
│   ├── raw/              # input CSVs (e.g. messy_*.csv)
│   ├── cleaned/          # cleaned outputs (e.g. cleaned_*.csv)
│   └── reports/          # JSON/HTML reports
├── tests/
│   ├── test_clean.py    # unit tests for cleaning steps
│   └── test_pipeline.py # integration: messy.csv → cleaned.csv
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Stage 3 – Smart Cleaning Assistant (human-in-the-loop)

Stage 3 adds a review layer on top of the automated cleaning:

- **Safe auto-clean** still runs end-to-end (trim, nulls, renames, numeric normalization, duplicates, type coercion).
- The system then **profiles** the dataset and detects potential issues (missingness, outliers, mixed types, high cardinality, constant columns, duplicates).
- Issues and a **data quality score** (0–100 with component breakdown) are written into the JSON report.
- A small **web review app** lets you inspect issues and selectively apply additional fixes.

### Generate reports with Stage 3 metadata

From the project root:

```bash
# Run the pipeline (as before) – now reports include issues + quality score
python -m app.main data/raw/messy.csv -o data/cleaned/cleaned.csv

# Or explicitly enable review mode (same cleaning, richer report metadata)
python -m app.main --review data/raw/messy.csv -o data/cleaned/cleaned.csv
```

This produces:

- Cleaned CSV under `data/cleaned/`
- JSON + HTML reports under `data/reports/`, including:
  - `summary` (rows/cols before/after, duplicates, actions)
  - `issues` (detected problems and suggested actions)
  - `quality` (overall score + completeness/consistency/validity/uniqueness)

### Launch the web review UI

In a terminal with the venv activated:

```bash
streamlit run app/review_app.py
```

Then:

- Select a JSON report from `data/reports/`.
- Inspect the data quality score and component breakdown.
- Expand issues and choose suggested fixes (drop column, drop rows where missing, etc.).
- Apply selected fixes and **download a reviewed CSV**.

## Tests

```bash
pytest
```

## What this proves

- **Pipeline thinking** — load → clean → profile → apply rules → report in one flow.
- **Automation mindset** — system analyzes data and chooses cleaning rules (impute, coerce, dedupe).
- **Data quality awareness** — missing %, cardinality, inferred types, basic outliers.
- **System design** — modular profiler, rule engine, and report generator; easy to extend.
