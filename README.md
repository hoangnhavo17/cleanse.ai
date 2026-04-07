# Cleanse.ai

**Try it:** [cleanseai.streamlit.app](https://cleanseai.streamlit.app)

Cleanse.ai loads messy CSVs, runs **standard cleaning** (trim, nulls, column names, dedupe, type hints), profiles the data, scores quality, and suggests **AI-assisted fixes** (Gemini) for the last mile. You stay in control: review suggestions, apply what you want, then download the cleaned CSV.

No install required — use the link above.

## What you can do in the app

1. **Preview, schema & report** — Upload a CSV or pick a demo dataset; inspect rows, schema, and raw data quality.
2. **Core preprocessing** — Run **standard cleaning** (always on when you run preprocessing): whitespace, nulls, headers, empty rows/columns, duplicates. Domain-specific fixes come from AI in the next step.
3. **Cleaned data & reports** — See the cleaned preview, schema, and quality score (completeness, consistency, validity, uniqueness).
4. **Recommendations** — Get Gemini suggestions for issues (e.g. normalize values, types, title case); toggle and apply selectively.
5. **Download** — Export the final cleaned CSV.

**AI recommendations** need a Gemini API key in [Streamlit Community Cloud](https://streamlit.io/cloud) app secrets (`GEMINI_API_KEY`) for the hosted app.

## What the pipeline does (conceptually)

- **Standard cleaning:** delimiter handling, header cleanup, trim, standardize nulls, drop empty rows/columns, numeric/datetime coercion where values support it, remove exact duplicates.
- **Profiling & issues:** missingness, types, cardinality, outliers, mixed types, text quality, etc.
- **Reports (batch pipeline):** JSON/HTML with summary, issues, and quality — the web app focuses on interactive review and download.

## Project layout (high level)

```
app/
  main.py           # batch / CLI pipeline
  smart_app.py      # Streamlit UI
  services/         # cleaning, profiling, issues, reports
data/               # demo CSVs in-repo; generated outputs gitignored
tests/
```

## Deploy your own copy (Streamlit Community Cloud)

1. Connect this repo on [Streamlit Community Cloud](https://streamlit.io/cloud).
2. **Main file:** `app/smart_app.py`
3. **Python:** 3.11 or 3.12 in Advanced settings (avoid 3.14 for dependency wheels).
4. **Secrets:** `GEMINI_API_KEY` for AI recommendations.

## What this project demonstrates

- End-to-end **load → clean → profile → report** thinking.
- **Human-in-the-loop** AI for fixes you approve, not blind automation.
- **Data quality** signals: scores, issues, and selective application of changes.
