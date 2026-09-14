import pandas as pd

from app.analytics.transform import indicator_need_scores, build_domains
from app.data.validation import load_indicator_dictionary


def test_percentile_invert_direction():
    mapping = load_indicator_dictionary()
    feature_df = pd.DataFrame(
        {"region_code": ["A", "B", "C"], "INC_SCHOOL_COVERAGE": [10, 50, 90]}
    )
    need_df = indicator_need_scores(feature_df, mapping)
    # good-is-high (coverage) -> need = 100*(1-percentile): highest coverage = lowest need
    assert need_df.loc[need_df["region_code"] == "C", "INC_SCHOOL_COVERAGE"].iloc[0] < \
           need_df.loc[need_df["region_code"] == "A", "INC_SCHOOL_COVERAGE"].iloc[0]


def test_percentile_direction_bad_is_high():
    mapping = load_indicator_dictionary()
    feature_df = pd.DataFrame(
        {"region_code": ["A", "B", "C"], "CAP_TEACHER_EQUITY": [10, 50, 90]}
    )
    need_df = indicator_need_scores(feature_df, mapping)
    # bad-is-high -> need = 100*percentile: highest raw value = highest need
    assert need_df.loc[need_df["region_code"] == "C", "CAP_TEACHER_EQUITY"].iloc[0] > \
           need_df.loc[need_df["region_code"] == "A", "CAP_TEACHER_EQUITY"].iloc[0]


def test_missing_indicator_is_absent_not_fabricated():
    mapping = load_indicator_dictionary()
    feature_df = pd.DataFrame({"region_code": ["A", "B"], "SOC_P0": [10, 90]})
    need_df = indicator_need_scores(feature_df, mapping)
    assert "INC_SCHOOL_COVERAGE" not in need_df.columns


def test_domain_full_coverage_when_both_sub_indicators_present():
    need_df = pd.DataFrame(
        {"region_code": ["A"], "INC_SCHOOL_COVERAGE": [80.0], "INC_INFRA_COVERAGE": [60.0]}
    )
    domains, coverage = build_domains(need_df)
    assert abs(coverage.loc[0, "INC"] - 1.0) < 1e-9
    assert abs(domains.loc[0, "INC"] - (0.6 * 80 + 0.4 * 60)) < 1e-9


def test_domain_below_min_coverage_is_unavailable_not_zero():
    # Only INC_INFRA_COVERAGE (weight 0.40) present -> coverage=0.40 < default 0.50
    need_df = pd.DataFrame({"region_code": ["A"], "INC_INFRA_COVERAGE": [60.0]})
    domains, coverage = build_domains(need_df, min_domain_coverage=0.50)
    assert abs(coverage.loc[0, "INC"] - 0.40) < 1e-9
    assert pd.isna(domains.loc[0, "INC"])  # unavailable, never zero


def test_domain_above_custom_min_coverage_is_available():
    need_df = pd.DataFrame({"region_code": ["A"], "INC_INFRA_COVERAGE": [60.0]})
    domains, coverage = build_domains(need_df, min_domain_coverage=0.30)
    assert not pd.isna(domains.loc[0, "INC"])
    assert abs(domains.loc[0, "INC"] - 60.0) < 1e-9
