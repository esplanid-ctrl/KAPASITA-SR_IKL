"""
Indicator -> domain-need transform layer (docs/04, docs/10).

Step 1: convert each validated raw indicator into a 0-100 "need" score:
    bad-is-high indicator:  need = 100 * percentile_rank(x)
    good-is-high indicator: need = 100 * (1 - percentile_rank(x))
(docs/04 "Normalization"; docs/10 "Transform").

Step 2: aggregate indicator-level need into the five domain sub-scores
using the EXACT sub-weights in docs/04 "Subscores":
    INC = 0.60*INC_SCHOOL_COVERAGE_need + 0.40*INC_INFRA_COVERAGE_need
    CAP = 0.70*CAP_TEACHER_EQUITY_need  + 0.30*CAP_GTK_need
    SOC = SOC_P0_need
    DIG = DIG_INTERNET_need
    POL = POL_SIGNAL_need

Missing-data policy (project-owner decision, recorded verbatim):
  - A missing indicator value is NEVER converted to zero.
  - `min_domain_coverage` (default 0.50) is an explicit MVP threshold, not
    derived from any cited specification -- flagged here, not hidden. A
    domain whose real-data coverage (sum of sub-weights backed by a
    present value, divided by the domain's total declared sub-weight) is
    BELOW this threshold is UNAVAILABLE for that region: its value is NaN.

ASSUMPTIONS FLAGGED (traced against the actual spec text, not silently
carried over as approved):
  - docs/10 section "Transform" states, verbatim, only:
        "Percentile-based need normalization, with direction determined
        by the indicator dictionary."
    That is the ONLY normalization method documented anywhere in docs/04
    or docs/10. Neither document defines a distinct formula named
    "capacity_gap" or "evidence_coded" -- those two transform names exist
    only as string values in data/DATA_MAPPING_REAL.csv's `transform`
    column, with no corresponding specification text.
  - `capacity_gap` (CAP_GTK, direction=bad): treated identically to
    `percentile` (bad_high=True) as the closest documented behavior. This
    is an MVP assumption, NOT a specified formula -- it has not been
    promoted into any policy rule and must not be cited as if docs/04/10
    specify it.
  - `evidence_coded` (POL_SIGNAL, direction=bad, level=national): treated
    as an ALREADY 0-100-scaled need value with no percentile ranking,
    because POL_SIGNAL is a single national reading, not a per-region
    distribution a percentile rank could be computed over. This too is an
    MVP assumption with no supporting text in docs/04 or docs/10, and the
    same national value is broadcast to every region in the feature
    matrix -- also unconfirmed.
  Both assumptions remain exactly what they are (MVP placeholders for a
  gap in the spec) and are not implemented as, or referenced by, any
  app.policy rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

DEFAULT_MIN_DOMAIN_COVERAGE = 0.50  # MVP assumption -- see module docstring

# indicator_id -> (domain, sub_weight within domain) -- docs/04 "Subscores"
DOMAIN_COMPOSITION: Dict[str, Dict[str, float]] = {
    "INC": {"INC_SCHOOL_COVERAGE": 0.60, "INC_INFRA_COVERAGE": 0.40},
    "CAP": {"CAP_TEACHER_EQUITY": 0.70, "CAP_GTK": 0.30},
    "SOC": {"SOC_P0": 1.00},
    "DIG": {"DIG_INTERNET": 1.00},
    "POL": {"POL_SIGNAL": 1.00},
}

# transform -> bad_high (True: need = 100*percentile; False: need = 100*(1-percentile))
PERCENTILE_TRANSFORMS = {
    "percentile": True,
    "percentile_invert": False,
    "capacity_gap": True,  # ASSUMPTION -- see module docstring
}
PASSTHROUGH_TRANSFORMS = {"evidence_coded"}  # ASSUMPTION -- already 0-100, no ranking applied


def indicator_need_scores(
    feature_df: pd.DataFrame, mapping_df: pd.DataFrame, region_col: str = "region_code"
) -> pd.DataFrame:
    """Converts each raw indicator column present in feature_df into a
    0-100 need score, per its declared transform/direction. An indicator
    absent from feature_df (failed validation or never ingested) is simply
    absent from the result -- never fabricated."""
    out = pd.DataFrame({region_col: feature_df[region_col]})
    for _, row in mapping_df.iterrows():
        indicator_id = row["indicator_id"]
        if indicator_id not in feature_df.columns:
            continue
        raw = pd.to_numeric(feature_df[indicator_id], errors="coerce")
        transform = row["transform"]
        if transform in PERCENTILE_TRANSFORMS:
            bad_high = PERCENTILE_TRANSFORMS[transform]
            rank = raw.rank(pct=True, method="average")
            need = 100 * rank if bad_high else 100 * (1 - rank)
        elif transform in PASSTHROUGH_TRANSFORMS:
            need = raw.clip(0, 100)
        else:
            raise ValueError(
                f"Unknown transform '{transform}' for indicator {indicator_id}. "
                "Refusing to invent a normalization method for it."
            )
        out[indicator_id] = need
    return out


def build_domains(
    need_df: pd.DataFrame,
    region_col: str = "region_code",
    min_domain_coverage: float = DEFAULT_MIN_DOMAIN_COVERAGE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates indicator-level need scores into domain sub-scores.

    Returns (domains_df, coverage_df), both indexed by region_col with one
    column per domain (INC/CAP/SOC/DIG/POL). `coverage_df` holds, per
    region and domain, the fraction of that domain's declared sub-weight
    actually backed by real data -- this is exactly the "domain coverage"
    that Gate 1 must report before scoring is finalized.

    A domain below `min_domain_coverage` is UNAVAILABLE for that region:
    value = NaN. Never zero, never an average fallback.
    """
    domains = pd.DataFrame({region_col: need_df[region_col]})
    coverage = pd.DataFrame({region_col: need_df[region_col]})

    for domain, composition in DOMAIN_COMPOSITION.items():
        weight_sum = pd.Series(0.0, index=need_df.index)
        weighted_val = pd.Series(0.0, index=need_df.index)
        declared_weight_total = sum(composition.values())

        for indicator_id, sub_weight in composition.items():
            if indicator_id in need_df.columns:
                present = need_df[indicator_id].notna()
                weight_sum = weight_sum + present.astype(float) * sub_weight
                weighted_val = weighted_val + need_df[indicator_id].fillna(0) * sub_weight * present.astype(float)

        domain_coverage = weight_sum / declared_weight_total
        domain_value = (weighted_val / weight_sum.replace(0, pd.NA)).clip(0, 100)
        unavailable = domain_coverage < min_domain_coverage
        domain_value = domain_value.mask(unavailable)

        domains[domain] = domain_value
        coverage[domain] = domain_coverage

    return domains, coverage
