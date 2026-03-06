"""
Entrypoint: one pipeline — load → clean (rule-based) → profile + automated rules → save + report.

Usage:
  python -m app.main                                      # use config defaults (INPUT_PATH, OUTPUT_PATH in data/)
  python -m app.main data/raw/messy_IMDB_dataset.csv      # input CSV, output: cleaned_<name>.csv + report
  python -m app.main data/raw/messy.csv -o data/cleaned/out.csv   # custom output (report in data/reports/)
  python -m app.main data/raw/messy.csv --no-report       # skip writing report files
"""
from pathlib import Path

from app.config import (
    INPUT_PATH,
    OUTPUT_PATH,
    REPORT_HTML_PATH,
    REPORT_JSON_PATH,
    REPORTS_DIR,
    CLEANED_DATA_DIR,
)


def run(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    report: bool = True,
    report_json_path: str | Path | None = None,
    report_html_path: str | Path | None = None,
    review: bool = False,
) -> None:
    """Run full pipeline: load → clean → profile + automated rules → save; write report if report=True."""
    from app.services.csv_io import load_csv, save_csv
    from app.services.cleaner import clean
    from app.services.profiler import profile_dataset
    from app.services.automated_cleaner import apply_automated_rules
    from app.services.report_generator import (
        build_report,
        write_report_json,
        write_report_html,
    )
    from app.services.issues import (
        compute_quality_score,
        detect_issues_from_profile,
        issues_to_dicts,
    )

    in_path = Path(input_path) if input_path is not None else INPUT_PATH
    out_path = Path(output_path) if output_path is not None else OUTPUT_PATH

    df, sep = load_csv(in_path)
    df = clean(df)

    profile_before = profile_dataset(df)
    rows_before, cols_before = df.shape[0], df.shape[1]
    df, actions = apply_automated_rules(df, profile_before)
    rows_after, cols_after = df.shape[0], df.shape[1]
    profile_after = profile_dataset(df)

    # Stage 3: derive issues and a simple data quality score from profiles.
    issues = detect_issues_from_profile(profile_before)
    quality = compute_quality_score(profile_before, profile_after, issues)

    save_csv(df, out_path, sep=sep)

    if report:
        rep = build_report(
            profile_before,
            profile_after,
            actions,
            rows_before,
            rows_after,
            cols_before,
            cols_after,
            issues=issues_to_dicts(issues),
            quality=quality,
            input_path=str(in_path),
            output_path=str(out_path),
        )
        r_json = Path(report_json_path) if report_json_path is not None else REPORT_JSON_PATH
        r_html = Path(report_html_path) if report_html_path is not None else REPORT_HTML_PATH
        write_report_json(rep, r_json)
        write_report_html(rep, r_html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean a CSV: load → clean → automated rules → save + report.",
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=None,
        help="Input CSV path (default: from config INPUT_PATH)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: data/cleaned/cleaned_<input_stem>.csv or config OUTPUT_PATH)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing report JSON/HTML",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Enable Stage 3 review mode (generates richer report for the web review app).",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv) if args.input_csv else None
    out_path = Path(args.output) if args.output else None

    if in_path is not None and out_path is None:
        out_path = CLEANED_DATA_DIR / f"cleaned_{in_path.name}"

    report_json = report_html = None
    if not args.no_report and out_path is not None:
        report_json = REPORTS_DIR / f"{out_path.stem}_report.json"
        report_html = REPORTS_DIR / f"{out_path.stem}_report.html"

    run(
        input_path=in_path,
        output_path=out_path,
        report=not args.no_report,
        report_json_path=report_json,
        report_html_path=report_html,
        review=args.review,
    )
