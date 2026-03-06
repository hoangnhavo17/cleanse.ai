"""
Stage 3 issue model and helpers.

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
    DISTRIBUTION = "distribution"
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

    def _next_id(prefix: str, idx: int) -> str:
        return f"{prefix}_{idx}"

    # Column-level issues
    for idx, cp in enumerate(col_profiles):
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

            # Only suggest dropping the column when missingness is extremely high.
            # Below this threshold we surface the issue but do not propose a
            # destructive action like dropping the column.
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
                    id=_next_id("missing", idx),
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

        # Constant columns (non-null but only one unique value)
        if non_null_count > 0 and cardinality == 1:
            issues.append(
                Issue(
                    id=_next_id("constant", idx),
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

        # High-cardinality categoricals: many unique values compared to rows
        if inferred == INFERRED_CATEGORICAL and non_null_count > 0:
            ratio = cardinality / max(1, non_null_count)
            if cardinality > 100 and ratio > 0.5:
                issues.append(
                    Issue(
                        id=_next_id("cardinality", idx),
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
        # Use a case-insensitive check so nullable dtypes like 'Int64'/'Float64'
        # are treated as numeric, not mixed.
        dtype_lower = dtype.lower()
        if inferred in (INFERRED_NUMERIC, INFERRED_DATETIME) and not dtype_lower.startswith(
            ("int", "float", "uint", "datetime64")
        ):
            issues.append(
                Issue(
                    id=_next_id("mixed", idx),
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
                    id=_next_id("outliers", idx),
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

    # Dataset-level duplicates
    dup_rows = int(profile.get("duplicate_rows", 0) or 0)
    if dup_rows > 0:
        issues.append(
            Issue(
                id="duplicates_dataset",
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

