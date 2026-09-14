import pandas as pd

from app.data.validation import (
    load_indicator_dictionary,
    validate_indicator_file,
    validate_region_join,
    build_feature_matrix,
)


def test_dictionary_loads_and_has_contract_columns():
    mapping = load_indicator_dictionary()
    required = {
        "indicator_id", "domain", "source_key", "file_pattern",
        "field_pattern", "level", "year", "direction", "transform", "component",
    }
    assert required.issubset(set(mapping.columns))
    assert len(mapping) > 0


def test_valid_file_passes():
    mapping = load_indicator_dictionary()
    row = mapping.iloc[0]
    df = pd.DataFrame(
        {
            "region_code": ["3201", "3202", "3203"],
            row["field_pattern"] + "_value": [10, 20, 30],
            "year": [row["year"]] * 3,
        }
    )
    report = validate_indicator_file(row, df)
    assert report.ok
    assert report.missing_rate == 0.0
    assert report.duplicate_region_count == 0
    assert report.year_confirmed


def test_missing_field_column_is_rejected_not_invented():
    mapping = load_indicator_dictionary()
    row = mapping.iloc[0]
    df = pd.DataFrame({"region_code": ["3201", "3202"], "unrelated_column": [1, 2]})
    report = validate_indicator_file(row, df)
    assert not report.ok
    assert any("Refusing to invent" in i.message for i in report.issues)


def test_duplicate_region_code_is_rejected():
    mapping = load_indicator_dictionary()
    row = mapping.iloc[0]
    df = pd.DataFrame(
        {
            "region_code": ["3201", "3201"],
            row["field_pattern"] + "_value": [10, 20],
        }
    )
    report = validate_indicator_file(row, df)
    assert not report.ok
    assert report.duplicate_region_count == 2


def test_all_missing_values_rejected():
    mapping = load_indicator_dictionary()
    row = mapping.iloc[0]
    df = pd.DataFrame(
        {
            "region_code": ["3201", "3202"],
            row["field_pattern"] + "_value": [None, None],
        }
    )
    report = validate_indicator_file(row, df)
    assert not report.ok


def test_numeric_region_code_dtype_is_rejected_not_silently_corrupted():
    mapping = load_indicator_dictionary()
    row = mapping.iloc[0]
    # simulate pandas having parsed dot-separated codes as float (data corruption risk)
    df = pd.DataFrame(
        {
            "region_code": pd.to_numeric(pd.Series(["11.1", "11.2"])),  # becomes float64
            row["field_pattern"] + "_value": [10, 20],
        }
    )
    report = validate_indicator_file(row, df)
    assert not report.ok
    assert any("numeric dtype" in i.message for i in report.issues)


def test_region_join_gap_is_reported_not_fabricated():
    dfs = {
        "A": pd.DataFrame({"region_code": ["1", "2", "3"]}),
        "B": pd.DataFrame({"region_code": ["1", "2"]}),
    }
    result = validate_region_join(dfs)
    assert result["union_region_count"] == 3
    assert result["gap_by_indicator"]["B"] == ["3"]
    assert result["gap_by_indicator"]["A"] == []


def test_build_feature_matrix_excludes_failed_and_missing_indicators():
    mapping = load_indicator_dictionary()
    first_id = mapping.iloc[0]["indicator_id"]
    first_field = mapping.iloc[0]["field_pattern"]

    dfs = {
        first_id: pd.DataFrame(
            {
                "region_code": ["3201", "3202"],
                first_field + "_value": [40, 60],
            }
        )
        # every other indicator intentionally not supplied
    }
    feature_df, reports = build_feature_matrix(dfs, mapping_df=mapping)

    assert first_id in feature_df.columns
    for indicator_id, report in reports.items():
        if indicator_id != first_id:
            assert not report.ok  # never silently invented

    other_ids = [i for i in mapping["indicator_id"] if i != first_id]
    for oid in other_ids:
        assert oid not in feature_df.columns
