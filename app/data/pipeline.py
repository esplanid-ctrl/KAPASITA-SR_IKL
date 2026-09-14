"""
Gate 1 pipeline orchestrator.

Real Data Validation -> Region Join Validation -> Indicator Need Transform
-> Domain Aggregation -> Deterministic Scoring -> Policy Rule Evaluation.

This is the ONLY place that should be called by the dashboard or copilot
to get a scored, audited dataframe -- region-master joins, domain-coverage
logic, and policy evaluation must not be re-implemented ad hoc elsewhere.

Per project-owner decision, Gate 1 is not considered "passed" until this
report is produced and inspected, and it must expose:
  - region master schema summary
  - join match rate / unmatched count / duplicate & ambiguous counts
  - geometry availability
  - domain coverage
  - full vs partial_renormalized vs unavailable score counts
  - recommendation rule activation count
  - policy-grounded (candidate/verified/pending_policy_verification) vs
    pending/unlinked recommendation counts
This module never claims "Gate 1 passed" itself -- it only reports the
metrics; interpreting pass/fail is left to the person reading the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import pandas as pd

from . import region_master as rm
from . import validation as v
from ..analytics import scoring as sc
from ..analytics import transform as tr
from ..policy import engine as pe


@dataclass
class Gate1Report:
    indicator_reports: Dict[str, "v.ValidationReport"]
    region_join: Optional["rm.RegionJoinReport"]  # join quality of the MERGED feature matrix
    region_join_by_indicator: Dict[str, "rm.RegionJoinReport"]  # P0-3/4: per real analytical dataset
    region_master_schema: Dict[str, object]
    geometry_quality: pd.DataFrame  # P0-5/P1: geometry-to-region-code consistency + quality metadata
    domain_coverage: pd.DataFrame
    score_mode_counts: Dict[str, int]
    scored_df: pd.DataFrame
    policy_result: Dict[str, list] = field(default_factory=lambda: {"recommendations": [], "unlinked_signals": []})

    def rule_status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.policy_result.get("recommendations", []) + self.policy_result.get("unlinked_signals", []):
            counts[outcome.rule_status] = counts.get(outcome.rule_status, 0) + 1
        return counts

    def summary(self) -> str:
        lines = ["=== Gate 1: Real Data Validation ==="]
        for indicator_id, r in self.indicator_reports.items():
            status = "OK" if r.ok else "FAILED"
            lines.append(
                f"  [{status}] {indicator_id}: missing_rate={r.missing_rate:.0%}, "
                f"duplicates={r.duplicate_region_count}, year_confirmed={r.year_confirmed}"
            )

        lines.append("=== Region Master Schema ===")
        if self.region_master_schema:
            lines.append(f"  rows={self.region_master_schema.get('row_count')}, "
                         f"with_geometry={self.region_master_schema.get('rows_with_geometry')} "
                         f"({self.region_master_schema.get('geometry_coverage', 0):.1%})")
        else:
            lines.append("  NOT LOADED -- no canonical region master was supplied to Gate 1.")

        lines.append("=== Region Join Quality (merged feature matrix) ===")
        if self.region_join is not None:
            lines.append(f"  {self.region_join.summary()}")
        else:
            lines.append("  NOT RUN -- no canonical region master was supplied to Gate 1.")

        lines.append("=== Region Join Quality (per real analytical dataset) ===")
        if self.region_join_by_indicator:
            for indicator_id, join_report in self.region_join_by_indicator.items():
                lines.append(f"  {indicator_id}: {join_report.summary()}")
        else:
            lines.append("  NOT RUN -- no canonical region master was supplied to Gate 1.")

        lines.append("=== Geometry Quality / CRS ===")
        if not self.geometry_quality.empty:
            flag_counts = self.geometry_quality["consistency_flag"].value_counts().to_dict()
            lines.append(f"  consistency check: {flag_counts}")
            lines.append(f"  {rm.CRS_ASSUMPTION_NOTE}")
            lines.append(f"  {rm.GEOMETRY_SIMPLIFICATION_NOTE}")
        else:
            lines.append("  NOT RUN -- no canonical region master was supplied to Gate 1.")

        lines.append("=== Domain Coverage (mean across regions) ===")
        if not self.domain_coverage.empty:
            domain_cols = [c for c in self.domain_coverage.columns if c != "region_code"]
            for col in domain_cols:
                lines.append(f"  {col}: {self.domain_coverage[col].mean():.0%}")
        else:
            lines.append("  (no data)")

        lines.append("=== Score Availability ===")
        for mode in ("full", "partial_renormalized", "unavailable"):
            lines.append(f"  {mode}: {self.score_mode_counts.get(mode, 0)}")

        lines.append("=== Recommendation Rule Activation ===")
        n_rec = len(self.policy_result.get("recommendations", []))
        n_sig = len(self.policy_result.get("unlinked_signals", []))
        lines.append(f"  total rule activations: {n_rec + n_sig} (recommendations={n_rec}, unlinked_signals={n_sig})")
        status_counts = self.rule_status_counts()
        grounded = sum(status_counts.get(s, 0) for s in ("candidate", "verified", "pending_policy_verification"))
        pending_or_unlinked = sum(status_counts.get(s, 0) for s in ("unlinked_policy", "rejected", "policy_inactive"))
        lines.append(f"  policy-grounded (candidate/verified/pending_policy_verification): {grounded}")
        lines.append(f"  pending/unlinked (unlinked_policy/rejected/policy_inactive): {pending_or_unlinked}")
        for status, count in sorted(status_counts.items()):
            lines.append(f"    {status}: {count}")

        return "\n".join(lines)


def run_gate1(
    dfs_by_indicator: Dict[str, pd.DataFrame],
    region_master: Optional[pd.DataFrame] = None,
    region_col: str = "region_code",
    min_domain_coverage: float = tr.DEFAULT_MIN_DOMAIN_COVERAGE,
    critical_domains: Iterable[str] = sc.DEFAULT_CRITICAL_DOMAINS,
    run_policy: bool = True,
) -> Gate1Report:
    mapping_df = v.load_indicator_dictionary()

    feature_df, indicator_reports = v.build_feature_matrix(dfs_by_indicator, mapping_df, region_col)

    region_join_report = None
    region_join_by_indicator: Dict[str, "rm.RegionJoinReport"] = {}
    join_quality_series = None
    region_master_schema: Dict[str, object] = {}
    geometry_quality_df = pd.DataFrame()
    if region_master is not None:
        row_count = len(region_master)
        rows_with_geometry = int(region_master["has_geometry"].sum()) if "has_geometry" in region_master.columns else 0
        region_master_schema = {
            "row_count": row_count,
            "rows_with_geometry": rows_with_geometry,
            "geometry_coverage": (rows_with_geometry / row_count) if row_count else 0.0,
            "crs_assumption": rm.CRS_ASSUMPTION_NOTE,
            "geometry_note": rm.GEOMETRY_SIMPLIFICATION_NOTE,
        }
        if "geometry_quality_json" in region_master.columns:
            geometry_quality_df = rm.geometry_quality_summary(region_master)

        # P0-3/4: validate region join against EVERY real analytical
        # dataset individually, not just the merged feature matrix -- a
        # single indicator source can have its own bad codes that a merge
        # alone would not surface.
        for indicator_id, indicator_df in dfs_by_indicator.items():
            if region_col in indicator_df.columns:
                region_join_by_indicator[indicator_id] = rm.validate_region_join(
                    indicator_df, region_master, region_col=region_col
                )

        if len(feature_df) > 0:
            region_join_report = rm.validate_region_join(feature_df, region_master, region_col=region_col)
            matched_ids = set(region_join_report.matched[region_col]) if len(region_join_report.matched) else set()
            join_quality_series = feature_df[region_col].isin(matched_ids).astype(float)
            join_quality_series.index = feature_df.index

    need_df = tr.indicator_need_scores(feature_df, mapping_df, region_col=region_col)
    domains_df, coverage_df = tr.build_domains(
        need_df, region_col=region_col, min_domain_coverage=min_domain_coverage
    )

    scored_df = sc.add_score(domains_df, critical_domains=critical_domains, join_quality=join_quality_series)

    score_mode_counts = scored_df["score_mode"].value_counts().to_dict() if len(scored_df) else {}

    policy_result = {"recommendations": [], "unlinked_signals": []}
    if run_policy and len(scored_df) > 0:
        policy_result = pe.evaluate(scored_df, region_col=region_col)

    return Gate1Report(
        indicator_reports=indicator_reports,
        region_join=region_join_report,
        region_join_by_indicator=region_join_by_indicator,
        region_master_schema=region_master_schema,
        geometry_quality=geometry_quality_df,
        domain_coverage=coverage_df,
        score_mode_counts=score_mode_counts,
        scored_df=scored_df,
        policy_result=policy_result,
    )
