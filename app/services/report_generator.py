"""
Cleaning report output (JSON and HTML).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_report(
    profile_before: dict[str, Any],
    profile_after: dict[str, Any] | None,
    actions: list[dict[str, Any]],
    rows_before: int,
    rows_after: int,
    columns_before: int,
    columns_after: int,
    *,
    issues: list[dict[str, Any]] | None = None,
    quality: dict[str, Any] | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Build the cleaning report structure."""
    issues = issues or []
    quality = quality or {}
    summary: dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "columns_before": columns_before,
        "columns_after": columns_after,
        "duplicate_rows_before": profile_before.get("duplicate_rows", 0),
        "actions_count": len(actions),
        "issues_count": len(issues),
    }
    if quality:
        summary["quality_score"] = quality.get("score")
    if input_path is not None:
        summary["input_path"] = input_path
    if output_path is not None:
        summary["output_path"] = output_path

    return {
        "summary": summary,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "actions": actions,
        "issues": issues,
        "quality": quality,
    }


def _generate_ai_commentary(report: dict[str, Any]) -> str | None:
    """
    Use Gemini API to generate a human-friendly summary of the cleaning run.

    Best-effort: returns None if the API key is missing or the call fails,
    so the rest of the report still renders normally.
    """
    import requests

    try:
        import streamlit as st
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None
    if not api_key or not api_key.strip():
        return None

    summary = report.get("summary", {})
    quality = report.get("quality", {}) or {}
    issues = report.get("issues", []) or []
    actions = report.get("actions", []) or []

    issues_serialized = issues[:20]
    actions_serialized = actions[:40]

    prompt = (
        "You are a data cleaning and data quality expert. "
        "A business stakeholder needs to understand the result of a data "
        "cleaning run. Based on the structured details below, write 2-4 short "
        "paragraphs explaining:\n"
        "- what was cleaned and why it mattered\n"
        "- the main issues that were found\n"
        "- what was fixed automatically vs. what may still need human attention.\n\n"
        "Avoid code snippets; use clear, plain language.\n\n"
        f"Summary:\n{json.dumps(summary, indent=2)}\n\n"
        f"Quality:\n{json.dumps(quality, indent=2)}\n\n"
        f"Issues (truncated):\n{json.dumps(issues_serialized, indent=2)}\n\n"
        f"Actions (truncated):\n{json.dumps(actions_serialized, indent=2)}"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.5-flash:generateContent?key={api_key.strip()}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code != 200:
            return None
        body = resp.json()
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def write_report_json(report: dict[str, Any], path: str | Path) -> None:
    """Write report as JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_report_html(report: dict[str, Any], path: str | Path) -> None:
    """Write human-readable HTML report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report.get("summary", {})
    actions = report.get("actions", [])
    profile_before = report.get("profile_before", {})
    profile_after = report.get("profile_after") or {}
    issues = report.get("issues", []) or []
    quality = report.get("quality", {}) or {}

    ai_commentary = _generate_ai_commentary(report)
    ai_commentary_html = (
        "<br>".join(ai_commentary.splitlines()) if ai_commentary else ""
    )

    rows_before = summary.get("rows_before", 0)
    rows_after = summary.get("rows_after", 0)
    cols_before = summary.get("columns_before", 0)
    cols_after = summary.get("columns_after", 0)
    dup_before = summary.get("duplicate_rows_before", 0)
    quality_score = quality.get("score")
    quality_components = quality.get("components", {})
    issues_count = summary.get("issues_count", len(issues))

    actions_rows = "".join(
        f"<tr><td>{a.get('action', '')}</td><td>{a.get('column', '-')}</td>"
        f"<td>{a.get('value', a.get('rows_removed', a.get('filled_count', '-')))}</td></tr>"
        for a in actions
    )

    columns_summary_before = "".join(
        f"<tr><td>{p.get('column')}</td><td>{p.get('inferred_type')}</td>"
        f"<td>{p.get('missing_pct', 0)}%</td><td>{p.get('cardinality', '-')}</td></tr>"
        for p in profile_before.get("column_profiles", [])
    )
    columns_summary_after = "".join(
        f"<tr><td>{p.get('column')}</td><td>{p.get('inferred_type')}</td>"
        f"<td>{p.get('missing_pct', 0)}%</td><td>{p.get('cardinality', '-')}</td></tr>"
        for p in profile_after.get("column_profiles", [])
    ) if profile_after else "<tr><td colspan='4'>—</td></tr>"

    issues_rows = "".join(
        f"<tr><td>{i.get('column', '-')}</td>"
        f"<td>{i.get('category')}</td>"
        f"<td>{i.get('severity')}</td>"
        f"<td>{i.get('title')}</td></tr>"
        for i in issues
    ) or "<tr><td colspan='4'>No issues recorded.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cleanse – Cleaning Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.5rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.5rem; margin: 0.5rem 0; }}
    .summary span {{ background: #f0f0f0; padding: 0.4rem 0.6rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Cleanse – Cleaning Report</h1>
  <h2>Summary</h2>
  <div class="summary">
    <span><strong>Rows:</strong> {rows_before} → {rows_after}</span>
    <span><strong>Columns:</strong> {cols_before} → {cols_after}</span>
    <span><strong>Duplicates (before):</strong> {dup_before}</span>
    <span><strong>Actions applied:</strong> {len(actions)}</span>
    <span><strong>Issues detected:</strong> {issues_count}</span>"""
    if quality_score is not None:
        html += f"""
    <span><strong>Data quality score:</strong> {quality_score} / 100</span>"""
    html += """
  </div>
  <h2>Actions applied</h2>"""
    if ai_commentary_html:
        html += f"""
  <h2>AI summary of this cleaning run</h2>
  <p>{ai_commentary_html}</p>"""
    html += """
  <table>
    <thead><tr><th>Action</th><th>Column</th><th>Value / Count</th></tr></thead>
    <tbody>{actions_rows}</tbody>
  </table>
  <h2>Issues detected</h2>
  <table>
    <thead><tr><th>Column</th><th>Category</th><th>Severity</th><th>Title</th></tr></thead>
    <tbody>""" + issues_rows + """</tbody>
  </table>
  <h2>Profile before (sample)</h2>
  <table>
    <thead><tr><th>Column</th><th>Inferred type</th><th>Missing %</th><th>Cardinality</th></tr></thead>
    <tbody>{columns_summary_before}</tbody>
  </table>
  <h2>Profile after (sample)</h2>
  <table>
    <thead><tr><th>Column</th><th>Inferred type</th><th>Missing %</th><th>Cardinality</th></tr></thead>
    <tbody>{columns_summary_after}</tbody>
  </table>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
