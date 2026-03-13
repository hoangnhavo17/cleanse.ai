"""
Issue model and helpers for data quality (used by profiler, reports, and smart app).

Defines:
- Issue / SuggestedAction / Action data structures
- JSON-friendly serialization
- Issue detection from profiler output
- Data quality scoring
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IssueCategory(str, Enum):
    """High-level buckets for detected data quality issues."""

    MISSINGNESS = "missingness"
    OUTLIERS = "outliers"
    MIXED_TYPES = "mixed_types"
    CARDINALITY = "cardinality"
    CONSTANT = "constant"
    DUPLICATES = "duplicates"
    REDUNDANT = "redundant"
    DISTRIBUTION = "distribution"
    TEXT_QUALITY = "text_quality"
    OTHER = "other"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ActionKind(str, Enum):
    """Actions that can be applied to address issues."""

    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    DROP_ROWS_WHERE_MISSING = "drop_rows_where_missing"
    DROP_COLUMN = "drop_column"
    FILLNA_CONSTANT = "fillna_constant"
    CAST_TYPE = "cast_type"


@dataclass
class SuggestedAction:
    """Suggested fix for an issue, shown in the review UI."""

    kind: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    # If true, the action is considered low-risk and can be auto-applied
    # when the user chooses a "apply all safe fixes" option.
    safe: bool = False


@dataclass
class Issue:
    """Detected data quality issue."""

    id: str
    category: IssueCategory
    severity: IssueSeverity
    title: str
    message: str
    column: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[SuggestedAction] = field(default_factory=list)


@dataclass
class Action:
    """Concrete, user-approved action to apply to a dataset."""

    issue_id: str
    kind: str
    column: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


def issue_to_dict(issue: Issue) -> Dict[str, Any]:
    """Serialize Issue to a JSON-friendly dict."""
    data = asdict(issue)
    data["category"] = issue.category.value
    data["severity"] = issue.severity.value
    data["suggestions"] = [
        {
            "kind": s.kind,
            "description": s.description,
            "params": s.params,
            "safe": s.safe,
        }
        for s in issue.suggestions
    ]
    return data


def action_to_dict(action: Action) -> Dict[str, Any]:
    """Serialize Action to dict."""
    return asdict(action)


def detect_issues_from_profile(profile: Dict[str, Any]) -> List[Issue]:
    """
    Build a list of Issue objects from a profiler profile.

    Uses simple heuristics for:
    - high missingness
    - constant columns
    - high-cardinality categoricals
    - mixed types (numeric inferred but non-numeric dtype)
    - numeric outliers (presence only)
    - duplicate rows
    """
    from app.services.profiler import (
        INFERRED_CATEGORICAL,
        INFERRED_DATETIME,
        INFERRED_NUMERIC,
        INFERRED_TEXT,
    )

    issues: List[Issue] = []
    col_profiles = profile.get("column_profiles", []) or []
    n_rows = max(1, int(profile.get("rows", 0)) or 1)

    for cp in col_profiles:
        col = str(cp.get("column"))
        inferred = cp.get("inferred_type")
        missing_pct = float(cp.get("missing_pct", 0.0) or 0.0)
        cardinality = int(cp.get("cardinality", 0) or 0)
        non_null_count = int(cp.get("non_null_count", 0) or 0)
        dtype = str(cp.get("dtype", ""))
        outliers = cp.get("outliers_iqr")

        # Missingness
        if missing_pct >= 30.0:
            sev = IssueSeverity.ERROR if missing_pct >= 60.0 else IssueSeverity.WARNING

            suggestions: List[SuggestedAction] = []
            if missing_pct >= 85.0:
                suggestions.append(
                    SuggestedAction(
                        kind=ActionKind.DROP_COLUMN.value,
                        description=f"Drop column '{col}' entirely (very high missingness).",
                        params={"column": col},
                        safe=False,
                    )
                )

            issues.append(
                Issue(
                    id=f"missing::{col}",
                    category=IssueCategory.MISSINGNESS,
                    severity=sev,
                    title=f"High missingness in '{col}'",
                    message=f"Column '{col}' has {missing_pct:.1f}% missing values.",
                    column=col,
                    stats={
                        "missing_pct": missing_pct,
                        "missing_count": cp.get("missing_count", 0),
                        "rows": n_rows,
                    },
                    suggestions=suggestions,
                )
            )

        # Constant columns
        if non_null_count > 0 and cardinality == 1:
            issues.append(
                Issue(
                    id=f"constant::{col}",
                    category=IssueCategory.CONSTANT,
                    severity=IssueSeverity.INFO,
                    title=f"Constant column '{col}'",
                    message=f"Column '{col}' has a single non-null value across {non_null_count} rows.",
                    column=col,
                    stats={
                        "non_null_count": non_null_count,
                        "cardinality": cardinality,
                    },
                    suggestions=[
                        SuggestedAction(
                            kind=ActionKind.DROP_COLUMN.value,
                            description=f"Drop constant column '{col}'.",
                            params={"column": col},
                            safe=False,
                        )
                    ],
                )
            )

        # Numeric-looking but categorical (e.g. zip_code, IDs): mean/median meaningless
        if inferred == INFERRED_NUMERIC and cp.get("numeric_but_categorical"):
            nb = cp["numeric_but_categorical"]
            issues.append(
                Issue(
                    id=f"type_validation::{col}",
                    category=IssueCategory.OTHER,
                    severity=IssueSeverity.INFO,
                    title=f"Numeric-looking but categorical: '{col}'",
                    message=(
                        f"Column '{col}' looks numeric but has high cardinality ({cardinality} distinct values). "
                        "It may be a code or identifier (e.g. zip_code, ID). "
                        "Summary statistics like mean or median are not meaningful."
                    ),
                    column=col,
                    stats=dict(nb) if isinstance(nb, dict) else {},
                    suggestions=[],
                )
            )

        # High-cardinality categoricals
        if inferred == INFERRED_CATEGORICAL and non_null_count > 0:
            ratio = cardinality / max(1, non_null_count)
            if cardinality > 100 and ratio > 0.5:
                issues.append(
                    Issue(
                        id=f"cardinality::{col}",
                        category=IssueCategory.CARDINALITY,
                        severity=IssueSeverity.INFO,
                        title=f"High-cardinality categorical '{col}'",
                        message=(
                            f"Column '{col}' has {cardinality} distinct values "
                            f"out of {non_null_count} non-null rows."
                        ),
                        column=col,
                        stats={
                            "cardinality": cardinality,
                            "non_null_count": non_null_count,
                        },
                        suggestions=[],
                    )
                )

        # Mixed types: inferred numeric/datetime but dtype suggests otherwise.
        dtype_lower = dtype.lower()
        if inferred in (INFERRED_NUMERIC, INFERRED_DATETIME) and not dtype_lower.startswith(
            ("int", "float", "uint", "datetime64")
        ):
            issues.append(
                Issue(
                    id=f"mixed::{col}",
                    category=IssueCategory.MIXED_TYPES,
                    severity=IssueSeverity.WARNING,
                    title=f"Mixed types in '{col}'",
                    message=(
                        f"Column '{col}' is inferred as {inferred} but stored as dtype '{dtype}', "
                        "which may indicate mixed or invalid values."
                    ),
                    column=col,
                    stats={
                        "dtype": dtype,
                        "inferred_type": inferred,
                    },
                    suggestions=[
                        SuggestedAction(
                            kind=ActionKind.CAST_TYPE.value,
                            description=f"Attempt to cast '{col}' to {inferred}.",
                            params={"column": col, "target_type": inferred},
                            safe=False,
                        )
                    ],
                )
            )

        # Outliers present
        if outliers and (
            outliers.get("count_below", 0) or outliers.get("count_above", 0)
        ):
            issues.append(
                Issue(
                    id=f"outliers::{col}",
                    category=IssueCategory.OUTLIERS,
                    severity=IssueSeverity.INFO,
                    title=f"Numeric outliers in '{col}'",
                    message=(
                        f"Column '{col}' has potential outliers outside "
                        f"[{outliers.get('lower_bound')}, {outliers.get('upper_bound')}]."
                    ),
                    column=col,
                    stats=outliers,
                    suggestions=[],
                )
            )

        # Text quality: for text / categorical columns, create a low-severity issue so the AI
        # layer can reason about spelling, casing, and stray characters (e.g. values like 'Italy1').
        # This is intentionally generic; the recommender sees sample_values and top_value_counts
        # and can propose normalize_values or title_case where appropriate.
        if inferred in (INFERRED_TEXT, INFERRED_CATEGORICAL) and non_null_count >= 10:
            issues.append(
                Issue(
                    id=f"text_quality::{col}",
                    category=IssueCategory.TEXT_QUALITY,
                    severity=IssueSeverity.INFO,
                    title=f"Text quality in '{col}'",
                    message=(
                        f"Column '{col}' contains textual or categorical values. "
                        "Review spelling, casing, and stray characters."
                    ),
                    column=col,
                    stats={
                        "inferred_type": inferred,
                        "non_null_count": non_null_count,
                        "cardinality": cardinality,
                    },
                    suggestions=[],
                )
            )

    # Redundant / derived column pairs (correlation ≈ 1 or ≈ -1)
    for pair in profile.get("redundant_pairs") or []:
        col_a = pair.get("col_a")
        col_b = pair.get("col_b")
        corr = pair.get("correlation")
        if col_a is None or col_b is None:
            continue
        col_a, col_b = str(col_a), str(col_b)
        issues.append(
            Issue(
                id=f"redundant::{col_a}::{col_b}",
                category=IssueCategory.REDUNDANT,
                severity=IssueSeverity.INFO,
                title=f"Redundant or derived columns: '{col_a}' and '{col_b}'",
                message=(
                    f"Columns '{col_a}' and '{col_b}' have correlation {corr}. "
                    "One may be derived from the other (e.g. height_cm vs height_m). "
                    "Consider dropping one to avoid redundancy."
                ),
                column=col_a,
                stats={"col_a": col_a, "col_b": col_b, "correlation": corr},
                suggestions=[
                    SuggestedAction(
                        kind=ActionKind.DROP_COLUMN.value,
                        description=f"Drop column '{col_b}' (keep '{col_a}').",
                        params={"column": col_b},
                        safe=False,
                    ),
                    SuggestedAction(
                        kind=ActionKind.DROP_COLUMN.value,
                        description=f"Drop column '{col_a}' (keep '{col_b}').",
                        params={"column": col_a},
                        safe=False,
                    ),
                ],
            )
        )

    # Dataset-level duplicates
    dup_rows = int(profile.get("duplicate_rows", 0) or 0)
    if dup_rows > 0:
        issues.append(
            Issue(
                id="duplicates::dataset",
                category=IssueCategory.DUPLICATES,
                severity=IssueSeverity.WARNING,
                title="Duplicate rows detected",
                message=f"{dup_rows} duplicate rows detected in the dataset.",
                column=None,
                stats={"duplicate_rows": dup_rows, "rows": n_rows},
                suggestions=[
                    SuggestedAction(
                        kind=ActionKind.DROP_DUPLICATE_ROWS.value,
                        description="Drop duplicate rows (keep first occurrence).",
                        params={},
                        safe=True,
                    )
                ],
            )
        )

    return issues


def compute_quality_score(
    profile_before: Dict[str, Any],
    profile_after: Optional[Dict[str, Any]],
    issues: List[Issue],
) -> Dict[str, Any]:
    """
    Compute a simple 0–100 data quality score with component breakdown.

    Components:
    - completeness
    - consistency
    - validity
    - uniqueness
    """
    profile = profile_after or profile_before
    cols = profile.get("column_profiles", []) or []
    if not cols:
        # Degenerate case: empty profile, treat as neutral quality.
        base = {"score": 80.0}
        return {
            "score": 80.0,
            "components": {
                "completeness": 80.0,
                "consistency": 80.0,
                "validity": 80.0,
                "uniqueness": 80.0,
            },
        }

    # Completeness: 100 - average missing pct across columns.
    missing_pcts = [float(c.get("missing_pct", 0.0) or 0.0) for c in cols]
    avg_missing = sum(missing_pcts) / max(1, len(missing_pcts))
    completeness = max(0.0, min(100.0, 100.0 - avg_missing))

    # Consistency: penalize mixed-type issues relative to column count.
    n_cols = len(cols)
    n_mixed = sum(1 for i in issues if i.category == IssueCategory.MIXED_TYPES)
    consistency_penalty = 100.0 * (n_mixed / max(1, n_cols))
    consistency = max(0.0, 100.0 - consistency_penalty)

    # Validity: penalize severe missingness and outliers.
    n_missing_issues = sum(1 for i in issues if i.category == IssueCategory.MISSINGNESS)
    n_outlier_issues = sum(1 for i in issues if i.category == IssueCategory.OUTLIERS)
    validity_penalty = 5.0 * n_missing_issues + 2.0 * n_outlier_issues
    validity = max(0.0, 100.0 - validity_penalty)

    # Uniqueness: based on duplicate rows after cleaning.
    rows = max(1, int(profile.get("rows", 0)) or 1)
    dup_after = int(profile.get("duplicate_rows", 0) or 0)
    frac_dups = dup_after / max(1, rows)
    uniqueness = max(0.0, 100.0 - 100.0 * frac_dups)

    overall = (completeness + consistency + validity + uniqueness) / 4.0
    return {
        "score": round(overall, 1),
        "components": {
            "completeness": round(completeness, 1),
            "consistency": round(consistency, 1),
            "validity": round(validity, 1),
            "uniqueness": round(uniqueness, 1),
        },
    }


def issues_to_dicts(issues: List[Issue]) -> List[Dict[str, Any]]:
    """Serialize list of Issue objects to list of dicts."""
    return [issue_to_dict(i) for i in issues]

