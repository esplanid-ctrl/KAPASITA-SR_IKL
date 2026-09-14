import pandas as pd
from app.analytics.scoring import add_score, WEIGHTS


def test_weights_sum():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_full_mode_when_all_domains_present():
    d = pd.DataFrame({k: [70] for k in WEIGHTS})
    o = add_score(d)
    assert o.loc[0, "score_mode"] == "full"
    assert abs(o.loc[0, "SR_ICSS"] - 70) < 1e-6
    assert o.loc[0, "unavailable_domains"] == ""


def test_missing_domain_never_becomes_zero_or_fifty():
    d = pd.DataFrame({"INC": [80], "CAP": [80], "SOC": [80], "DIG": [80]})  # POL missing
    o = add_score(d)
    assert pd.isna(o.loc[0, "POL"])  # never fabricated as 0 or 50
    assert o.loc[0, "score_mode"] == "partial_renormalized"
    assert o.loc[0, "unavailable_domains"] == "POL"
    # weights re-normalize across the remaining 4 -> still resolves to 80
    assert abs(o.loc[0, "SR_ICSS"] - 80) < 1e-6


def test_configured_weight_preserved_and_effective_weight_renormalized():
    d = pd.DataFrame({"INC": [50], "CAP": [50], "SOC": [50], "DIG": [50]})  # POL missing
    o = add_score(d)
    # original configured weights untouched
    assert abs(o.loc[0, "configured_weight_INC"] - 0.30) < 1e-9
    assert abs(o.loc[0, "configured_weight_POL"] - 0.10) < 1e-9
    # effective weight for INC renormalized across the 4 available domains
    assert abs(o.loc[0, "effective_weight_INC"] - (0.30 / 0.90)) < 1e-9
    assert pd.isna(o.loc[0, "effective_weight_POL"])  # POL itself unavailable


def test_all_domains_missing_is_unavailable_not_zero():
    d = pd.DataFrame({"region_code": ["A"]})
    o = add_score(d)
    assert o.loc[0, "score_mode"] == "unavailable"
    assert pd.isna(o.loc[0, "SR_ICSS"])
    assert pd.isna(o.loc[0, "confidence"])


def test_no_hardcoded_minimum_component_count():
    # Only ONE domain present -- with the default empty critical_domains,
    # this must still score (partial_renormalized), since there is no
    # invented "at least N components" rule anymore.
    d = pd.DataFrame({"INC": [90]})
    o = add_score(d)
    assert o.loc[0, "score_mode"] == "partial_renormalized"
    assert abs(o.loc[0, "SR_ICSS"] - 90) < 1e-6


def test_critical_domain_missing_forces_unavailable():
    d = pd.DataFrame({"INC": [90], "CAP": [90], "SOC": [90], "DIG": [90]})  # POL missing
    o = add_score(d, critical_domains=["POL"])
    assert o.loc[0, "score_mode"] == "unavailable"
    assert pd.isna(o.loc[0, "SR_ICSS"])


def test_confidence_formula_matches_docs10_when_full():
    # completeness=1.0, recency default=1.0, join_quality default=1.0
    # confidence = 0.4*1 + 0.3*1 + 0.3*1 = 1.0 -> 100
    d = pd.DataFrame({k: [70] for k in WEIGHTS})
    o = add_score(d)
    assert abs(o.loc[0, "confidence"] - 100.0) < 1e-6


def test_partial_scoring_yields_lower_confidence_than_full():
    full = pd.DataFrame({k: [70] for k in WEIGHTS})
    partial = pd.DataFrame({"INC": [70], "CAP": [70], "SOC": [70], "DIG": [70]})  # POL missing
    o_full = add_score(full)
    o_partial = add_score(partial)
    assert o_partial.loc[0, "confidence"] < o_full.loc[0, "confidence"]


def test_join_quality_and_recency_feed_into_confidence():
    d = pd.DataFrame({k: [70] for k in WEIGHTS})
    o_low_join = add_score(d, join_quality=pd.Series([0.0]))
    o_high_join = add_score(d, join_quality=pd.Series([1.0]))
    assert o_low_join.loc[0, "confidence"] < o_high_join.loc[0, "confidence"]


def test_contributions_sum_to_total_when_full():
    d = pd.DataFrame({k: [70] for k in WEIGHTS})
    o = add_score(d)
    contrib_cols = [f"contrib_{k}" for k in WEIGHTS]
    assert abs(o[contrib_cols].sum(axis=1).iloc[0] - o["SR_ICSS"].iloc[0]) < 1e-6
