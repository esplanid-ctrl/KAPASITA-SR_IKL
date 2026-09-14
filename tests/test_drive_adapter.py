import json
import pandas as pd
import pytest

import app.data.drive_adapter as da
from app.data.region_master import RegionMasterError


def _fake_region_master_csv(tmp_path):
    df = pd.DataFrame({
        "region_id": ["11.01"], "region_name": ["Kab Aceh Selatan"],
        "province_id": ["11"], "province_name": ["Aceh"],
        "level": ["kabupaten_kota"], "has_geometry": [True],
        "geometry_json": [json.dumps([[[[97.0, 3.0], [97.1, 3.1], [97.0, 3.2]]]])],
        "declared_lat": [3.0], "declared_lng": [97.0],
        "geometry_quality_json": [json.dumps({"ring_count": 1, "simplified_no_holes": True,
                                               "centroid_check_distance_deg": 0.05, "consistency_flag": "ok"})],
    })
    path = tmp_path / "region_master_canonical.csv"
    df.to_csv(path, index=False)
    return path


def test_uses_local_cache_without_any_drive_call(tmp_path, monkeypatch):
    cache_path = _fake_region_master_csv(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("list_files should not be called when a local cache exists")

    monkeypatch.setattr(da, "list_files", _boom)
    result = da.load_region_master_from_drive(cache_path=str(cache_path))
    assert list(result["region_id"]) == ["11.01"]


def test_prefers_prebuilt_artifact_in_drive_folder(tmp_path, monkeypatch):
    artifact_src = _fake_region_master_csv(tmp_path / "src")
    (tmp_path / "src").mkdir(exist_ok=True)

    monkeypatch.setattr(da, "list_files", lambda folder_id: [
        {"id": "file1", "name": "region_master_canonical_2026.csv"},
        {"id": "file2", "name": "wilayah.sql"},
        {"id": "file3", "name": "wilayah_level_1_2.sql"},
    ])
    monkeypatch.setattr(da, "download_file", lambda file_id, target_dir=None: artifact_src)

    cache_path = str(tmp_path / "cache.csv")
    result = da.load_region_master_from_drive(folder_id="fake-folder", cache_path=cache_path)
    assert list(result["region_id"]) == ["11.01"]


def test_falls_back_to_raw_sql_pair_and_caches(tmp_path, monkeypatch):
    wilayah_sql = tmp_path / "wilayah.sql"
    wilayah_sql.write_text(
        "INSERT INTO `wilayah`(`kode`,`nama`) VALUES ('11','Aceh'),('11.01','Kabupaten Aceh Selatan');\n",
        encoding="utf-8",
    )
    level12_sql = tmp_path / "wilayah_level_1_2.sql"
    level12_sql.write_text(
        "INSERT INTO `wilayah_level_1_2`(`kode`,`nama`,`ibukota`,`lat`,`lng`,`elv`,`tz`,`luas`,`penduduk`,`path`,`status`) "
        "VALUES ('11.01','Kabupaten Aceh Selatan','Tapak Tuan', 3.0, 97.0, 19, 7, 50.0, 500, "
        "'[[3.0,97.0],[3.1,97.1],[3.2,97.0]]', 1);\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(da, "list_files", lambda folder_id: [
        {"id": "f1", "name": "wilayah.sql"},
        {"id": "f2", "name": "wilayah_level_1_2.sql"},
    ])
    monkeypatch.setattr(
        da, "download_file",
        lambda file_id, target_dir=None: wilayah_sql if file_id == "f1" else level12_sql,
    )

    cache_path = str(tmp_path / "cache.csv")
    result = da.load_region_master_from_drive(folder_id="fake-folder", cache_path=cache_path)
    assert list(result["region_id"]) == ["11.01"]
    # subsequent call must hit the cache, not Drive again
    monkeypatch.setattr(da, "list_files", lambda folder_id: (_ for _ in ()).throw(AssertionError("should not re-list")))
    result2 = da.load_region_master_from_drive(folder_id="fake-folder", cache_path=cache_path)
    assert list(result2["region_id"]) == ["11.01"]


def test_raises_when_folder_has_neither_artifact_nor_raw_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "list_files", lambda folder_id: [{"id": "x", "name": "unrelated_file.txt"}])
    with pytest.raises(RegionMasterError):
        da.load_region_master_from_drive(folder_id="fake-folder", cache_path=str(tmp_path / "no_cache.csv"))
