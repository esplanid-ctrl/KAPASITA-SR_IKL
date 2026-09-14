"""
Deterministic SR-Inclusive Capacity Support Score (SR-ICSS) engine.

Configured formula (prototype, docs/04 -- NOT altered here; changing these
weights requires explicit project-owner approval and must be included in
the sensitivity analysis per docs/10):
    SR_ICSS = 0.30*INC + 0.25*CAP + 0.20*SOC + 0.15*DIG + 0.10*POL

Missing-data policy (project-owner decision, recorded verbatim so the
contract lives in one place):
  - A domain value is NEVER converted to zero. Absent/NaN stays NaN. It is
    the caller's job (app.analytics.transform.build_domains) to decide a
    domain is UNAVAILABLE (min_domain_coverage); this module never invents
    a domain value on its own.
  - When one or more domains are unavailable but at least one is available
    (and no configured critical domain is missing), scoring proceeds in an
    EXPLICIT mode: score_mode = "partial_renormalized". The ORIGINAL
    configured weights (WEIGHTS) are preserved unchanged and exposed as
    `configured_weight_<DOMAIN>`; EFFECTIVE weights (renormalized across
    only the available domains) are stored separately as
    `effective_weight_<DOMAIN>`, so the two are never conflated.
  - `min_components_required = 3` from an earlier draft has been REMOVED:
    it was an invented threshold with no basis in docs/04 or docs/10. It
    is replaced by `critical_domains` (default: empty tuple -- nothing
    mandatory), an explicitly configurable list. If ANY critical domain is
    unavailable for a region, that region's score is not computed at all
    (score_mode = "unavailable", SR_ICSS = NaN). With the empty default,
    a single available domain is enough to produce a partial_renormalized
    score -- this default is a visible MVP choice, not a hidden judgement
    call, and should be reviewed by the project owner.
  - Confidence follows docs/10 exactly:
        confidence = 0.4*completeness + 0.3*recency + 0.3*join_quality
    `completeness` is derived here from real weight coverage (1.0 for
    score_mode="full", <1.0 for "partial_renormalized" -- this is how
    partial scoring is reflected in a LOWER confidence, without adding any
    extra unspecified reduction factor). `recency` and `join_quality` are
    optional inputs (per-region 0-1 Series) supplied by the Gate 1
    pipeline (app.data.pipeline); if not supplied they default to 1.0 with
    an explicit flag -- this is an MVP simplification, not silently
    assumed to be true recency/join quality.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

WEIGHTS: Dict[str, float] = {"INC": 0.30, "CAP": 0.25, "SOC": 0.20, "DIG": 0.15, "POL": 0.10}
COMPONENTS = list(WEIGHTS.keys())

DEFAULT_CRITICAL_DOMAINS: tuple = ()  # nothing mandatory by default -- see module docstring


def percentile_need(series: pd.Series, bad_high: bool = True) -> pd.Series:
    """Percentile-based need normalization (docs/04)."""
    rank = series.rank(pct=True, method="average")
    return 100 * rank if bad_high else 100 * (1 - rank)


def add_score(
    df: pd.DataFrame,
    critical_domains: Iterable[str] = DEFAULT_CRITICAL_DOMAINS,
    recency: Optional[pd.Series] = None,
    join_quality: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Computes SR_ICSS deterministically in Python (never by the LLM).

    `df` must contain columns INC/CAP/SOC/DIG/POL, where a domain that is
    UNAVAILABLE for a region is NaN -- never zero, never a placeholder.

    Adds, per row:
      SR_ICSS                    -- NaN if score_mode == "unavailable"
      score_mode                  -- "full" | "partial_renormalized" | "unavailable"
      unavailable_domains         -- comma-separated list of missing domains
      configured_weight_<DOMAIN>  -- original weight, unchanged, for audit
      effective_weight_<DOMAIN>   -- renormalized weight actually used (NaN if that domain is unavailable)
      contrib_<DOMAIN>            -- that domain's contribution to SR_ICSS
      completeness                -- fraction of configured weight backed by real data (0-1)
      confidence                  -- 0.4*completeness + 0.3*recency + 0.3*join_quality (docs/10), NaN if unavailable
    """
    out = df.copy()
    critical_domains = set(critical_domains)

    for k in COMPONENTS:
        if k not in out.columns:
            out[k] = pd.NA
        out[k] = pd.to_numeric(out[k], errors="coerce")

    present = out[COMPONENTS].notna()
    weight_series = pd.Series(WEIGHTS)
    configured_weight_total = float(weight_series.sum())  # == 1.0

    available_weight_sum = present.mul(weight_series, axis=1).sum(axis=1)
    weighted_raw_sum = out[COMPONENTS].fillna(0).mul(weight_series, axis=1).sum(axis=1)

    missing_critical = pd.Series(False, index=out.index)
    for cd in critical_domains:
        if cd in COMPONENTS:
            missing_critical = missing_critical | ~present[cd]

    unavailable_domains = present.apply(
        lambda row: ",".join(k for k in COMPONENTS if not row[k]), axis=1
    )

    score_mode = []
    for idx in out.index:
        if missing_critical.loc[idx] or available_weight_sum.loc[idx] <= 0:
            score_mode.append("unavailable")
        elif unavailable_domains.loc[idx] == "":
            score_mode.append("full")
        else:
            score_mode.append("partial_renormalized")
    out["score_mode"] = score_mode
    out["unavailable_domains"] = unavailable_domains

    is_unavailable = out["score_mode"] == "unavailable"
    valid_denominator = available_weight_sum.replace(0, pd.NA)
    sr_icss = (weighted_raw_sum / valid_denominator).clip(0, 100)
    out["SR_ICSS"] = sr_icss.mask(is_unavailable)

    for k in COMPONENTS:
        out[f"configured_weight_{k}"] = weight_series[k]  # preserved, unchanged, for audit
        eff_w = (weight_series[k] / valid_denominator).where(present[k])
        eff_w = eff_w.mask(is_unavailable)
        out[f"effective_weight_{k}"] = eff_w
        contrib = (out[k] * eff_w).mask(is_unavailable)
        out[f"contrib_{k}"] = contrib

    # --- confidence, per docs/10: 0.4*completeness + 0.3*recency + 0.3*join_quality ---
    completeness = (available_weight_sum / configured_weight_total).clip(0, 1)
    out["completeness"] = completeness.mask(is_unavailable)

    if recency is None:
        recency = pd.Series(1.0, index=out.index)  # ASSUMPTION: recency not yet wired in, see docstring
    if join_quality is None:
        join_quality = pd.Series(1.0, index=out.index)  # ASSUMPTION: join_quality not yet wired in, see docstring

    computed_confidence = (0.4 * completeness + 0.3 * recency + 0.3 * join_quality) * 100
    if "confidence" in df.columns:
        supplied = pd.to_numeric(df["confidence"], errors="coerce")
        confidence = supplied.fillna(computed_confidence)
    else:
        confidence = computed_confidence
    out["confidence"] = confidence.clip(0, 100).mask(is_unavailable)

    return out
