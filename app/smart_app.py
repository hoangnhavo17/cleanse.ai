from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# Ensure project root is importable when run via `streamlit run app/smart_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.automated_cleaner import apply_automated_rules
from app.services.cleaner import clean
from app.services.csv_io import load_csv
from app.services.execution_engine import apply_actions
from app.services.issues import Action, Issue, SuggestedAction, compute_quality_score, detect_issues_from_profile
from app.services.profiler import profile_dataset


# --- Helpers -----------------------------------------------------------------

DEMO_DATASETS: Dict[str, Path] = {
    "Warehouse Inventory": Path("data/raw/warehouse_messy_data.csv"),
    "Healthcare Records": Path("data/raw/healthcare_messy_data.csv"),
    "HR Data": Path("data/raw/messy_HR_data.csv"),
    "Movie Ratings": Path("data/raw/messy_IMDB_dataset.csv"),
}


def _available_demo_datasets() -> Dict[str, Path]:
    return {label: path for label, path in DEMO_DATASETS.items() if path.exists()}


def _read_csv_robust(src: Any) -> pd.DataFrame:
    """
    Read a CSV handling common encoding issues.

    Used for uploaded files (file-like objects). Tries UTF-8 first,
    then falls back to latin-1 so that files with extended characters
    still load instead of raising a UnicodeDecodeError.
    """
    try:
        return pd.read_csv(src)
    except UnicodeDecodeError:
        return pd.read_csv(src, encoding="latin1")


def _build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "non_null", "missing_pct"])

    non_null = df.notna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(1)

    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null": non_null.values,
            "missing_pct": missing_pct.values,
        }
    )
    return schema


def _run_pipeline_on_dataframe(
    df_raw: pd.DataFrame,
    file_name: str,
    apply_standard_clean: bool,
    apply_automated_rules_flag: bool,
) -> Dict[str, Any]:
    """Run CleanPilot pipeline with configurable preprocessing steps."""
    df_work = df_raw.copy()
    rows_before, cols_before = df_work.shape

    if apply_standard_clean:
        df_work = clean(df_work)

    profile_before = profile_dataset(df_work)

    actions: List[Dict[str, Any]] = []
    if apply_automated_rules_flag:
        df_after, actions = apply_automated_rules(df_work, profile_before)
    else:
        df_after = df_work

    rows_after, cols_after = df_after.shape
    profile_after = profile_dataset(df_after)

    issues: List[Issue] = detect_issues_from_profile(profile_after)
    quality = compute_quality_score(profile_before, profile_after, issues)

    buf = io.StringIO()
    df_after.to_csv(buf, index=False, na_rep="")
    cleaned_csv_bytes = buf.getvalue().encode("utf-8")

    return {
        "df_raw": df_raw,
        "df_after": df_after,
        "rows_before": rows_before,
        "cols_before": cols_before,
        "rows_after": rows_after,
        "cols_after": cols_after,
        "issues": issues,
        "quality": quality,
        "actions": actions,
        "cleaned_csv_bytes": cleaned_csv_bytes,
        "file_name": file_name,
        "apply_standard_clean": apply_standard_clean,
        "apply_automated_rules": apply_automated_rules_flag,
    }


def _extract_human_in_loop_suggestions(issues: List[Issue]) -> List[Tuple[Issue, SuggestedAction]]:
    needing_decision: List[Tuple[Issue, SuggestedAction]] = []
    for issue in issues:
        for suggestion in issue.suggestions:
            if not suggestion.safe:
                needing_decision.append((issue, suggestion))
    return needing_decision


# --- Main app ----------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="CleanPilot – Smart Data Cleaning",
        layout="wide",
        page_icon="✨",
    )

    if "run_result" not in st.session_state:
        st.session_state["run_result"] = None

    # --- Sidebar: dataset selection -------------------------------------------
    st.sidebar.title("Dataset")

    # Backwards-compatible: earlier versions used the option label
    # "Demo dataset" (lowercase "d"). If that value is still stored in
    # session_state from a previous run, normalize it to "Demo Dataset"
    # so the radio stays on the correct choice.
    old_source = st.session_state.get("Source")
    if old_source == "Demo dataset":
        st.session_state["Source"] = "Demo Dataset"

    source = st.sidebar.radio(
        "Source",
        ["Upload CSV", "Demo Dataset"],
    )

    df_raw: pd.DataFrame | None = None
    file_name: str | None = None

    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV File", type=["csv"], key="cp_upload")
        if uploaded is not None:
            try:
                df_raw = _read_csv_robust(uploaded)
                file_name = uploaded.name
            except Exception as exc:  # pragma: no cover - defensive
                st.sidebar.error(f"Failed to read CSV: {exc}")
    else:
        demos = _available_demo_datasets()
        if not demos:
            st.sidebar.warning("No Demo Datasets found in `data/raw/`.")
        else:
            # Migrate any older demo labels that included a suffix like " (messy)"
            # so the selectbox shows the clean labels without parentheses.
            old_label = st.session_state.get("cp_demo")
            if isinstance(old_label, str) and old_label not in demos:
                for label_key in demos.keys():
                    if old_label.startswith(label_key):
                        st.session_state["cp_demo"] = label_key
                        break

            label = st.sidebar.selectbox("Demo Dataset", list(demos.keys()), key="cp_demo")
            if label:
                path = demos[label]
                try:
                    # Use the pipeline's robust CSV loader for demo datasets so
                    # we benefit from delimiter sniffing and numeric repairs.
                    df_raw, _sep_used = load_csv(path)
                    file_name = path.name
                except Exception as exc:  # pragma: no cover - defensive
                    st.sidebar.error(f"Failed to load demo dataset: {exc}")

    # --- Global styles & hero -------------------------------------------------
    st.markdown(
        """
        <style>
          [data-testid="stAppViewContainer"] > .main {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 0;
          }
          .block-container {
            padding-top: 0;
          }
          .cp-hero {
            padding: 0 0 1rem 0;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
          }
          .cp-eyebrow {
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #60a5fa;
          }
          .cp-title {
            font-size: 3.0rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin: 0;
          }
          .cp-subtitle {
            max-width: 640px;
            font-size: 1.02rem;
            color: #9ca3af;
          }
          .cp-card {
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: radial-gradient(circle at top left, #020617, #020617);
            padding: 1.4rem 1.5rem 1.2rem 1.5rem;
          }
          .cp-tag {
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 500;
          }
          .cp-tag-error {
            background: rgba(248, 113, 113, 0.1);
            color: #fecaca;
            border: 1px solid rgba(248, 113, 113, 0.4);
          }
          .cp-tag-warning {
            background: rgba(250, 204, 21, 0.08);
            color: #facc15;
            border: 1px solid rgba(250, 204, 21, 0.35);
          }
          .cp-tag-info {
            background: rgba(96, 165, 250, 0.12);
            color: #bfdbfe;
            border: 1px solid rgba(96, 165, 250, 0.4);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            """
            <div class="cp-hero">
              <div class="cp-eyebrow">CleanPilot</div>
              <h1 class="cp-title">From messy CSVs to trustworthy datasets.</h1>
              <p class="cp-subtitle">
                Use the sidebar to pick a dataset, preview its structure, run generic preprocessing steps,
                then review issues that still need a human in the loop before you ship the data.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Single vertical flow -------------------------------------------------

    st.subheader("1. Preview & Schema")

    if df_raw is not None and file_name is not None:
        tabs = st.tabs(["Preview rows", "Schema"])

        with tabs[0]:
            st.dataframe(df_raw.head(50), use_container_width=True)

        with tabs[1]:
            schema = _build_schema_table(df_raw)
            st.dataframe(schema, use_container_width=True)

        st.subheader("2. Generic Preprocessing Steps")
        st.markdown("Uncheck any steps you **do not** want to run.")

        col_steps = st.columns(2)
        with col_steps[0]:
            apply_standard_clean = st.checkbox(
                "Standard Cleaning",
                value=True,
                key="cp_step_clean",
                help=(
                    "Run the core, opinionated cleaning pipeline:\n"
                    "- normalize null-like values (\"\", \"N/A\", \"nan\", etc.)\n"
                    "- trim whitespace\n"
                    "- standardize booleans and categories\n"
                    "- clean and parse dates where possible\n"
                    "- drop fully-empty rows and obvious junk\n"
                    "Missing values stay missing; no automatic imputation."
                ),
            )
        with col_steps[1]:
            apply_auto_rules = st.checkbox(
                "Apply Automated Rules",
                value=True,
                key="cp_step_auto",
                help=(
                    "Profile-based safe extras applied after standard cleaning:\n"
                    "- coerce columns that look numeric but are stored as text\n"
                    "- remove exact duplicate rows\n"
                    "Does not fill in missing values or apply risky fixes."
                ),
            )

        run_clicked = st.button(
            "Run Preprocessing And Analyze",
            type="primary",
        )

        if run_clicked:
            with st.spinner("Running preprocessing, profiling, and issue detection…"):
                st.session_state["run_result"] = _run_pipeline_on_dataframe(
                    df_raw=df_raw,
                    file_name=file_name,
                    apply_standard_clean=apply_standard_clean,
                    apply_automated_rules_flag=apply_auto_rules,
                )

        st.subheader("3. Cleaned Dataset")

        res: Dict[str, Any] | None = st.session_state.get("run_result")

        if res is None:
            st.info("Run preprocessing above to see the cleaned dataset here.")
        else:
            df_after: pd.DataFrame = res["df_after"]
            tabs_cleaned = st.tabs(["Preview rows", "Schema"])
            with tabs_cleaned[0]:
                st.dataframe(df_after.head(50), use_container_width=True)
                st.caption(
                    f"Showing first 50 rows • {res['rows_after']:,} rows × {res['cols_after']} columns "
                    f"(started from {res['rows_before']:,} rows × {res['cols_before']} columns)."
                )
            with tabs_cleaned[1]:
                schema_after = _build_schema_table(df_after)
                st.dataframe(schema_after, use_container_width=True)

            st.subheader("4. Suggestions")

        if res is None:
            st.info("Run preprocessing to see issues and download options.")
        else:
            issues: List[Issue] = res["issues"]
            needs_decision = _extract_human_in_loop_suggestions(issues)

            # If the previous run requested a reset, clear any stored
            # checkbox values *before* we render the widgets again.
            if st.session_state.get("cp_reset_accept", False):
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("cp_accept_")]
                for k in keys_to_clear:
                    del st.session_state[k]
                st.session_state["cp_reset_accept"] = False

            if not needs_decision:
                st.success(
                    "No suggestions requiring a decision. "
                    "You can still inspect the data before using it downstream."
                )
            else:
                st.markdown("Check **Accept** for Suggestions you want to apply, then click **Apply Selected**.")
                accept_keys: List[str] = []
                for idx, (issue, suggestion) in enumerate(needs_decision):
                    col_name = issue.column or "dataset"
                    sev_tag = "cp-tag-info"
                    sev_label = "Info"
                    if issue.severity.value == "error":
                        sev_tag = "cp-tag-error"
                        sev_label = "High impact"
                    elif issue.severity.value == "warning":
                        sev_tag = "cp-tag-warning"
                        sev_label = "Warning"

                    cb_key = f"cp_accept_{issue.id}_{idx}"
                    accept_keys.append(cb_key)

                    st.markdown(
                        f"""
                        <div class="cp-card">
                          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                            <div style="font-weight:600;">{issue.title}</div>
                            <span class="cp-tag {sev_tag}">{sev_label}</span>
                          </div>
                          <div style="font-size:0.9rem;color:#e5e7eb;margin-bottom:0.2rem;">
                            Column: <code>{col_name}</code>
                          </div>
                          <div style="font-size:0.9rem;color:#9ca3af;margin-bottom:0.3rem;">
                            {issue.message}
                          </div>
                          <div style="font-size:0.9rem;color:#e5e7eb;">
                            <strong>Suggested action:</strong> {suggestion.description}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.checkbox("Accept", key=cb_key, value=False)

                apply_clicked = st.button("Apply Selected Suggestions")
                if apply_clicked:
                    actions_to_apply: List[Action] = []
                    for idx, (issue, suggestion) in enumerate(needs_decision):
                        if st.session_state.get(accept_keys[idx], False):
                            actions_to_apply.append(
                                Action(
                                    issue_id=issue.id,
                                    kind=suggestion.kind,
                                    column=issue.column or suggestion.params.get("column"),
                                    params=dict(suggestion.params),
                                )
                            )
                    if actions_to_apply:
                        try:
                            df_current = res["df_after"]
                            new_df, _logs = apply_actions(df_current, actions_to_apply)
                            buf = io.StringIO()
                            new_df.to_csv(buf, index=False, na_rep="")
                            new_csv_bytes = buf.getvalue().encode("utf-8")
                            profile_after = profile_dataset(new_df)
                            new_issues = detect_issues_from_profile(profile_after)
                            quality = compute_quality_score(
                                profile_dataset(res["df_raw"]),
                                profile_after,
                                new_issues,
                            )
                            st.session_state["run_result"] = {
                                **res,
                                "df_after": new_df,
                                "rows_after": len(new_df),
                                "cols_after": len(new_df.columns),
                                "issues": new_issues,
                                "quality": quality,
                                "cleaned_csv_bytes": new_csv_bytes,
                            }
                            # Ask the next run to clear any stored checkbox
                            # values before re-rendering the suggestion list.
                            st.session_state["cp_reset_accept"] = True
                            st.rerun()
                        except Exception as e:
                            import traceback
                            st.error(f"Failed to apply suggestions: {e}")
                            st.code(traceback.format_exc(), language="text")
                    else:
                        st.warning("No suggestions selected. Check **Accept** on at least one suggestion.")

            st.subheader("5. Download Final Dataset")
            st.download_button(
                label="Download cleaned CSV",
                data=res["cleaned_csv_bytes"],
                file_name=f"cleaned_{res['file_name']}",
                mime="text/csv",
                disabled=res["cleaned_csv_bytes"] is None,
            )
    else:
        st.info("Choose a dataset from the sidebar to see a preview and schema.")


if __name__ == "__main__":
    main()
