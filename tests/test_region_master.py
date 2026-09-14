import pandas as pd
import pytest

import json

from app.data.region_master import (
    build_canonical_region_master,
    save_canonical_artifact,
    load_canonical_artifact,
    get_geometry,
    validate_region_join,
    geometry_quality_summary,
    RegionMasterError,
)

WILAYAH_SQL = """
INSERT INTO `wilayah`(`kode`,`nama`) VALUES
('11','Aceh'),
('11.01','Kabupaten Aceh Selatan'),
('11.02','Kabupaten Aceh Tenggara'),
('11.01.01','Kecamatan Contoh'),
('11.01.01.2001','Desa Contoh'),
('12','Sumatera Utara'),
('12.01','Kabupaten Contoh Sumut');
"""

WILAYAH_L12_SQL = """
INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) VALUES
('11','Aceh','Banda Aceh', 5.0, 95.0, 10, 7, 100.0, 1000, '[[2.0,97.0],[2.1,97.1],[2.2,97.0]]', 1);
INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) VALUES
('11.01','Kabupaten Aceh Selatan','Tapak Tuan', 3.0, 97.0, 19, 7, 50.0, 500, '[[3.0,97.0],[3.1,97.1],[3.2,97.0]]', 1);
INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) VALUES
('11.02','Kabupaten Aceh Tenggara','Kutacane', 3.5, 97.8, 168, 7, 40.0, 400, '', 1);
INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) VALUES
('12','Sumatera Utara','Medan', 2.5, 99.0, 20, 7, 200.0, 2000, '[[1,99],[1.1,99.1],[1.2,99]]', 1);
INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) VALUES
('12.01','Kabupaten Contoh Sumut','Contoh', 1.5, 99.5, 5, 7, 30.0, 300, '[[1.5,99.5],[1.6,99.6],[1.7,99.5]]', 1);
"""


@pytest.fixture
def sql_files(tmp_path):
    w_path = tmp_path / "wilayah.sql"
    w_path.write_text(WILAYAH_SQL, encoding="utf-8")
    l12_path = tmp_path / "wilayah_level_1_2.sql"
    l12_path.write_text(WILAYAH_L12_SQL, encoding="utf-8")
    return str(w_path), str(l12_path)


def test_build_canonical_region_master_extracts_only_kab_kota(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    # only the 2-segment (kabupaten/kota) rows -- not province, kecamatan, or desa
    assert set(master["region_id"]) == {"11.01", "11.02", "12.01"}


def test_province_parent_derived_correctly(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    row = master[master["region_id"] == "11.01"].iloc[0]
    assert row["province_id"] == "11"
    assert row["province_name"] == "Aceh"


def test_geometry_present_and_coordinate_order_swapped_to_lng_lat(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    row = master[master["region_id"] == "11.01"].iloc[0]
    assert bool(row["has_geometry"]) is True
    geom = get_geometry(master, "11.01")
    # source path had [3.0,97.0] as [lat,lng] -> normalized geometry point is [lng,lat]
    assert geom[0][0][0] == [97.0, 3.0]


def test_missing_geometry_is_explicit_not_fabricated(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    row = master[master["region_id"] == "11.02"].iloc[0]
    assert bool(row["has_geometry"]) is False
    assert get_geometry(master, "11.02") is None


def test_save_and_load_canonical_artifact_roundtrip(sql_files, tmp_path):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    artifact_path = tmp_path / "region_master.csv"
    save_canonical_artifact(master, str(artifact_path))
    reloaded = load_canonical_artifact(str(artifact_path))
    assert set(reloaded["region_id"]) == set(master["region_id"])
    assert get_geometry(reloaded, "11.01") is not None


def test_load_canonical_artifact_missing_file_raises_clearly(tmp_path):
    with pytest.raises(RegionMasterError):
        load_canonical_artifact(str(tmp_path / "does_not_exist.csv"))


def test_validate_region_join_matched_unmatched(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    source = pd.DataFrame({"region_code": ["11.01", "99.99"]})
    report = validate_region_join(source, master)
    assert len(report.matched) == 1
    assert len(report.unmatched) == 1


def test_validate_region_join_flags_invalid_level(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    # '11' is a province code (1 segment), '11.01.01' is a kecamatan code (3 segments)
    source = pd.DataFrame({"region_code": ["11", "11.01.01", "11.01"]})
    report = validate_region_join(source, master)
    assert len(report.invalid_level) == 2
    assert len(report.matched) == 1


def test_validate_region_join_flags_duplicates(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    source = pd.DataFrame({"region_code": ["11.01", "11.01", "11.02"]})
    report = validate_region_join(source, master)
    assert len(report.duplicates) == 2
    assert len(report.matched) == 1


def test_geometry_quality_flags_consistent_geometry_as_ok(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    quality = json.loads(master[master.region_id == "11.01"].iloc[0]["geometry_quality_json"])
    assert quality["consistency_flag"] == "ok"
    assert quality["simplified_no_holes"] is True


def test_geometry_quality_flags_missing_geometry(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    quality = json.loads(master[master.region_id == "11.02"].iloc[0]["geometry_quality_json"])
    assert quality["consistency_flag"] == "no_geometry"


def test_geometry_quality_summary_covers_all_regions(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    summary = geometry_quality_summary(master)
    assert set(summary["region_id"]) == set(master["region_id"])
    assert "consistency_flag" in summary.columns


def test_validate_region_join_never_joins_by_name_alone_but_flags_mismatch(sql_files):
    w_path, l12_path = sql_files
    master = build_canonical_region_master(w_path, l12_path)
    source = pd.DataFrame({"region_code": ["11.01"], "region_name": ["TOTALLY WRONG NAME"]})
    report = validate_region_join(source, master, region_name_col="region_name")
    assert len(report.matched) == 1  # matched by code regardless of the name
    assert len(report.name_mismatch) == 1  # but the mismatch is surfaced
