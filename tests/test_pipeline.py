import pandas as pd

from app.data.pipeline import run_gate1


def _region_master():
    return pd.DataFrame({
        "region_id": ["32.01", "32.02"],
        "region_name": ["KAB. BOGOR", "KAB. SUKABUMI"],
        "province_id": ["32", "32"],
        "province_name": ["JAWA BARAT", "JAWA BARAT"],
        "level": ["kabupaten_kota", "kabupaten_kota"],
        "has_geometry": [True, False],
        "geometry_json": ['[[[[0,0],[1,0],[1,1]]]]', None],
        "geometry_quality_json": [
            '{"ring_count": 1, "simplified_no_holes": true, "centroid_check_distance_deg": 0.1, "consistency_flag": "ok"}',
            '{"ring_count": 0, "simplified_no_holes": true, "centroid_check_distance_deg": null, "consistency_flag": "no_geometry"}',
        ],
    })


def test_gate1_runs_without_region_master():
    dfs = {"INC_SCHOOL_COVERAGE": pd.DataFrame({"region_code": ["32.01", "32.02"], "coverage_value": [40, 80]})}
    report = run_gate1(dfs)
    assert report.region_join is None
    assert report.region_master_schema == {}
    assert "NOT LOADED" in report.summary()
    assert "NOT RUN" in report.summary()


def test_gate1_reports_region_master_schema_and_join_quality():
    master = _region_master()
    dfs = {"INC_SCHOOL_COVERAGE": pd.DataFrame({"region_code": ["32.01", "99.99"], "coverage_value": [40, 80]})}
    report = run_gate1(dfs, region_master=master)
    assert report.region_master_schema["row_count"] == 2
    assert report.region_master_schema["rows_with_geometry"] == 1
    assert "crs_assumption" in report.region_master_schema
    assert len(report.region_join.matched) == 1
    assert len(report.region_join.unmatched) == 1


def test_gate1_validates_join_per_indicator_dataset_not_just_merged():
    master = _region_master()
    dfs = {
        "INC_SCHOOL_COVERAGE": pd.DataFrame({"region_code": ["32.01", "32.02"], "coverage_value": [40, 80]}),
        "CAP_TEACHER_EQUITY": pd.DataFrame({"region_code": ["32.01", "99.99"], "index_value": [10, 20]}),
    }
    report = run_gate1(dfs, region_master=master)
    assert set(report.region_join_by_indicator.keys()) == {"INC_SCHOOL_COVERAGE", "CAP_TEACHER_EQUITY"}
    # INC_SCHOOL_COVERAGE: both codes match
    assert len(report.region_join_by_indicator["INC_SCHOOL_COVERAGE"].matched) == 2
    # CAP_TEACHER_EQUITY: one bad code that the merged-matrix check alone wouldn't necessarily isolate
    assert len(report.region_join_by_indicator["CAP_TEACHER_EQUITY"].unmatched) == 1
    summary_text = report.summary()
    assert "per real analytical dataset" in summary_text


def test_gate1_reports_geometry_quality_and_crs_caveat():
    master = _region_master()
    dfs = {"INC_SCHOOL_COVERAGE": pd.DataFrame({"region_code": ["32.01", "32.02"], "coverage_value": [40, 80]})}
    report = run_gate1(dfs, region_master=master)
    assert set(report.geometry_quality["consistency_flag"]) == {"ok", "no_geometry"}
    summary_text = report.summary()
    assert "UNVERIFIED MVP ASSUMPTION" in summary_text
    assert "VISUALIZATION ONLY" in summary_text


def test_gate1_reports_domain_coverage_and_score_availability():
    dfs = {
        "INC_SCHOOL_COVERAGE": pd.DataFrame({"region_code": ["32.01", "32.02"], "coverage_value": [40, 80]}),
        "INC_INFRA_COVERAGE": pd.DataFrame({"region_code": ["32.01", "32.02"], "coverage_value": [30, 90]}),
    }
    report = run_gate1(dfs)
    assert "INC" in report.domain_coverage.columns
    assert report.domain_coverage["INC"].iloc[0] == 1.0
    assert sum(report.score_mode_counts.values()) == 2


def test_gate1_reports_policy_rule_activation_counts():
    dfs = {
        "CAP_TEACHER_EQUITY": pd.DataFrame({"region_code": ["32.01", "32.02"], "index_value": [95, 5]}),
    }
    report = run_gate1(dfs)
    counts = report.rule_status_counts()
    # CAP-SR-01 should fire (candidate) for the high-value region
    assert counts.get("candidate", 0) >= 1
    summary_text = report.summary()
    assert "Recommendation Rule Activation" in summary_text
    assert "policy-grounded" in summary_text


def test_gate1_can_skip_policy_evaluation():
    dfs = {"CAP_TEACHER_EQUITY": pd.DataFrame({"region_code": ["32.01", "32.02"], "index_value": [95, 5]})}
    report = run_gate1(dfs, run_policy=False)
    assert report.policy_result == {"recommendations": [], "unlinked_signals": []}
