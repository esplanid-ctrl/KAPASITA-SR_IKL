"""
Real Data Validation layer for KAPASITA-MVP.

MASTER_PROMPT.md non-negotiable rules enforced here:
  #5  Never invent data, regulation, source, threshold, or policy rule.
  #6  Every indicator must be traceable:
      indicator -> source -> year -> definition -> denominator -> transformation.

docs/09_AI_EXECUTION_SPEC.md requires: "inspect real source files rather
than inventing columns" and "validate schema, level, year, denominator,
missingness and duplicates".

This module sits between app/data/drive_adapter.py (raw file access) and
app/analytics/scoring.py (deterministic scoring). Nothing may reach the
scoring engine without passing through build_feature_matrix() here.

Design decision (flagged for sign-off, not a silent contract change):
An indicator that fails validation, or was never ingested, is EXCLUDED
from the feature matrix rather than filled with any guessed value. The
scoring engine (app/analytics/scoring.py) is responsible for treating an
absent column as missing evidence, never as neutral/average evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "DATA_MAPPING_REAL.csv"


@dataclass
class FieldIssue:
    indicator_id: str
    severity: str  # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    indicator_id: str
    ok: bool = True
    row_count: int = 0
    value_column: Optional[str] = None
    missing_rate: float = 1.0
    duplicate_region_count: int = 0
    year_confirmed: bool = False
    issues: List[FieldIssue] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.issues.append(FieldIssue(self.indicator_id, "error", message))
        self.ok = False

    def add_warning(self, message: str) -> None:
        self.issues.append(FieldIssue(self.indicator_id, "warning", message))


def load_indicator_dictionary(path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the indicator -> source -> year -> transformation contract.

    Raises if the dictionary itself is missing: validation cannot proceed
    without a traceability contract to validate against (MASTER_PROMPT #6).
    """
    path = path or MAPPING_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Indicator dictionary not found at {path}. Cannot validate any "
            "data without data/DATA_MAPPING_REAL.csv as the source of truth."
        )
    required_cols = {
        "indicator_id", "domain", "source_key", "file_pattern",
        "field_pattern", "level", "year", "direction", "transform", "component",
    }
    df = pd.read_csv(path)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Indicator dictionary is missing required columns: {sorted(missing)}")
    return df


def _find_field(columns, field_pattern: str) -> List[str]:
    pattern = str(field_pattern).lower()
    return [c for c in columns if pattern in str(c).lower()]


def validate_indicator_file(
    indicator_row: "pd.Series",
    df: pd.DataFrame,
    region_col: str = "region_code",
) -> ValidationReport:
    """Validates one ingested file against its dictionary row.

    Never coerces or guesses: a file that doesn't match its declared
    field_pattern, has duplicate join keys, or is entirely non-numeric is
    rejected (ok=False), not silently patched.
    """
    indicator_id = indicator_row["indicator_id"]
    report = ValidationReport(indicator_id=indicator_id, row_count=len(df))

    if df is None or len(df) == 0:
        report.add_error("File is empty or was not provided.")
        return report

    if region_col not in df.columns:
        report.add_error(f"Missing required join key column '{region_col}'.")
        return report

    region_dtype = df[region_col].dtype
    if region_dtype.kind in "fc":  # float/complex -- region codes must never be numeric
        report.add_error(
            f"'{region_col}' was read as a numeric dtype ({region_dtype}). Dot-separated "
            "region codes (e.g. '11.10') are numeric-looking and get silently corrupted by "
            "pandas (e.g. '11.10' -> 11.1) unless read as a string. Re-read this file with "
            f"dtype={{'{region_col}': str}} -- refusing to validate a possibly-corrupted join key."
        )
        return report

    field_matches = _find_field(df.columns, indicator_row["field_pattern"])
    if not field_matches:
        report.add_error(
            f"No column matches declared field_pattern="
            f"'{indicator_row['field_pattern']}'. Refusing to invent the column."
        )
        return report
    if len(field_matches) > 1:
        report.add_warning(
            f"Multiple columns matched field_pattern "
            f"'{indicator_row['field_pattern']}': {field_matches}. Using '{field_matches[0]}'."
        )
    value_col = field_matches[0]
    report.value_column = value_col

    dup_mask = df[region_col].duplicated(keep=False)
    report.duplicate_region_count = int(dup_mask.sum())
    if report.duplicate_region_count > 0:
        report.add_error(
            f"{report.duplicate_region_count} duplicate '{region_col}' rows found; "
            "denominator/join is not one-to-one (AT-02)."
        )

    numeric = pd.to_numeric(df[value_col], errors="coerce")
    report.missing_rate = float(numeric.isna().mean()) if len(numeric) else 1.0
    if report.missing_rate >= 1.0:
        report.add_error("All values in the value column are missing or non-numeric.")
    elif report.missing_rate > 0.5:
        report.add_warning(
            f"High missingness ({report.missing_rate:.0%}) in '{value_col}'; "
            "this will strongly lower confidence for affected rows."
        )

    declared_year = str(indicator_row["year"])
    if "year" in df.columns:
        years_present = set(df["year"].dropna().astype(str))
        report.year_confirmed = declared_year in years_present
        if not report.year_confirmed:
            report.add_warning(
                f"Declared year {declared_year} not found in file's own 'year' "
                f"column (found: {sorted(years_present)}). Year is unverified."
            )
    else:
        report.add_warning(
            "File has no 'year' column; the dictionary's declared year "
            "cannot be independently confirmed against the source file."
        )

    return report


def validate_region_join(
    dfs_by_indicator: Dict[str, pd.DataFrame], region_col: str = "region_code"
) -> Dict[str, object]:
    """Cross-indicator join coverage check (AT-02).

    Reports, per indicator, which region codes present in the union of all
    indicators are absent from that indicator's file. It does NOT fabricate
    those missing rows; downstream this shows up as missing evidence.
    """
    codes_by_indicator: Dict[str, set] = {}
    for indicator_id, df in dfs_by_indicator.items():
        if df is not None and region_col in df.columns:
            codes_by_indicator[indicator_id] = set(df[region_col].dropna().astype(str))

    if not codes_by_indicator:
        return {"union_region_count": 0, "gap_by_indicator": {}}

    union = set().union(*codes_by_indicator.values())
    gap_by_indicator = {
        indicator_id: sorted(union - codes) for indicator_id, codes in codes_by_indicator.items()
    }
    return {"union_region_count": len(union), "gap_by_indicator": gap_by_indicator}


def build_feature_matrix(
    dfs_by_indicator: Dict[str, pd.DataFrame],
    mapping_df: Optional[pd.DataFrame] = None,
    region_col: str = "region_code",
) -> Tuple[pd.DataFrame, Dict[str, ValidationReport]]:
    """Validates every ingested file against the dictionary and assembles a
    wide feature matrix keyed by region_col.

    Indicators that fail validation, or were never supplied, are EXCLUDED
    from the resulting matrix (the column is simply absent) rather than
    filled with any placeholder. app/analytics/scoring.py must treat an
    absent indicator column as missing evidence, not as a neutral value.

    Returns (feature_df, reports) where reports maps indicator_id ->
    ValidationReport, so the dashboard's data-quality panel (docs/07 feature
    #7) can show exactly why an indicator is or isn't in the score.
    """
    mapping_df = mapping_df if mapping_df is not None else load_indicator_dictionary()
    reports: Dict[str, ValidationReport] = {}
    frames = []

    for _, row in mapping_df.iterrows():
        indicator_id = row["indicator_id"]
        df = dfs_by_indicator.get(indicator_id)
        if df is None:
            report = ValidationReport(indicator_id=indicator_id, row_count=0)
            report.add_error("No ingested file was supplied for this indicator.")
            reports[indicator_id] = report
            continue

        report = validate_indicator_file(row, df, region_col=region_col)
        reports[indicator_id] = report
        if report.ok:
            frame = (
                df[[region_col, report.value_column]]
                .rename(columns={report.value_column: indicator_id})
                .drop_duplicates(subset=region_col)
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[region_col]), reports

    feature_df = frames[0]
    for frame in frames[1:]:
        feature_df = feature_df.merge(frame, on=region_col, how="outer")

    return feature_df, reports


def summarize_reports(reports: Dict[str, ValidationReport]) -> pd.DataFrame:
    """Flat summary table for the dashboard's evidence/data-quality panel."""
    rows = []
    for indicator_id, r in reports.items():
        rows.append(
            {
                "indicator_id": indicator_id,
                "ok": r.ok,
                "row_count": r.row_count,
                "missing_rate": r.missing_rate,
                "duplicate_region_count": r.duplicate_region_count,
                "year_confirmed": r.year_confirmed,
                "issues": "; ".join(f"[{i.severity}] {i.message}" for i in r.issues),
            }
        )
    return pd.DataFrame(rows)
