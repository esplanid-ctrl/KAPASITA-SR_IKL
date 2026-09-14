"""
Canonical administrative region master (docs/03 D08: cahyadsn/wilayah).

Schema verified DIRECTLY against the ZIP the project owner provided (see
docs/generated/REGION_MASTER_MAPPING_SPEC.json for the full inspection
report) -- nothing here is inferred from prior knowledge of the dataset.

Confirmed facts used below:
  - Real tables are `wilayah` (db/wilayah.sql: kode, nama -- all 4 levels)
    and `wilayah_level_1_2` (db/wilayah_level_1_2.sql: adds ibukota, lat,
    lng, elv, tz, luas, penduduk, path, status -- province + kab/kota ONLY).
  - `kode` is dot-segmented: 1 segment=province, 2=kabupaten/kota (the
    PRIMARY ANALYTICAL UNIT), 3=kecamatan, 4=desa/kelurahan. NOT a fixed
    digit-length code (an earlier draft of this module wrongly assumed
    that; this version replaces it).
  - Geometry (`path`) exists ONLY for province + kabupaten/kota, exactly
    the level KAPASITA needs. It is coordinate order [lat, lng] (verified
    against the row's own separate lat/lng columns), NOT GeoJSON-standard
    [lng, lat], and its nesting depth is INCONSISTENT across rows (2, 3,
    or 4 levels deep) -- this module normalizes every row to a single
    MultiPolygon-of-rings-of-[lng,lat] shape before use.
  - CRS is not declared anywhere in the source; WGS84/EPSG:4326 is assumed
    from the coordinate ranges, flagged here as an assumption, not a fact.

This module deliberately does NOT re-parse the SQL dump on every call.
`build_canonical_region_master()` is an offline/preprocessing step; its
output should be saved once via `save_canonical_artifact()` and loaded
via `load_canonical_artifact()` at runtime (project-owner decision: "do
not download/reparse the region ZIP on every Streamlit page load").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .region_sql_parser import parse_insert_rows

LEVEL_BY_SEGMENT_COUNT = {1: "province", 2: "kabupaten_kota", 3: "kecamatan", 4: "desa_kelurahan"}
CANONICAL_ARTIFACT_COLUMNS = [
    "region_id", "region_name", "province_id", "province_name",
    "level", "has_geometry", "geometry_json",
    "declared_lat", "declared_lng", "geometry_quality_json",
]

# P1 decision: `status` in wilayah_level_1_2 has ONE constant observed value
# (1) across the whole snapshot and no documentation anywhere in the
# cahyadsn/wilayah repo explains what it means. It is intentionally NOT
# carried into the canonical artifact and NEVER used analytically (e.g.
# never used to filter/include/exclude regions). If a future snapshot
# shows varying values, its meaning must be confirmed with the upstream
# source before it is used for anything.
UNDOCUMENTED_FIELDS_NOT_USED_ANALYTICALLY = ["status"]

# P1 decision: CRS is NOT declared anywhere in the source repository (no
# EPSG/SRID/WGS84 string appears in db/*.sql, README.md, or apps/inc/*.php).
# WGS84 / EPSG:4326 is only an MVP ASSUMPTION based on the coordinate
# ranges matching Indonesia's real geographic extent in decimal degrees --
# it must never be presented as a verified fact. Every consumer of
# `geometry_json` (dashboard, docs) must surface this exact caveat string.
CRS_ASSUMPTION_NOTE = (
    "CRS is NOT declared in the cahyadsn/wilayah source (no EPSG/SRID found "
    "in db/*.sql, README.md, or apps/inc/*.php). WGS84/EPSG:4326 is an "
    "UNVERIFIED MVP ASSUMPTION based on coordinate ranges only."
)

# P1 decision: the geometry stored in `geometry_json` is a SIMPLIFIED
# representation -- multi-part rings are flattened to independent
# single-ring polygons (see `_find_rings`), which discards any true
# polygon/hole grouping the source may have intended for the 3 rows that
# used 4-level nesting. This simplification is for VISUALIZATION
# (choropleth) ONLY. It must NOT be used for spatial analysis (area
# calculation, point-in-polygon, spatial joins, adjacency) -- use the
# source's own declared `luas` (area) field for area-based analysis
# instead, and treat any geometry-based spatial analysis need as requiring
# a re-derivation from the raw, un-simplified source.
GEOMETRY_SIMPLIFICATION_NOTE = (
    "Geometry is simplified (rings flattened, no polygon/hole grouping) "
    "for CHOROPLETH VISUALIZATION ONLY -- do not use for spatial analysis."
)


class RegionMasterError(RuntimeError):
    pass


def _level_of(kode: str) -> Optional[str]:
    segments = kode.split(".")
    return LEVEL_BY_SEGMENT_COUNT.get(len(segments))


def _is_coord_pair(x) -> bool:
    return (
        isinstance(x, list) and len(x) == 2
        and isinstance(x[0], (int, float)) and isinstance(x[1], (int, float))
    )


def _find_rings(x) -> list:
    """Recursively finds every 'ring' (a list of [num, num] coordinate
    pairs) anywhere inside an arbitrarily/inconsistently nested structure,
    and returns them as a flat list of rings.

    This dump's `path` field mixes THREE different nesting depths across
    rows (see docs/generated/REGION_MASTER_MAPPING_SPEC.json), and even
    within a single row the nesting can be irregular. Rather than trust a
    single guessed depth for the whole row (which broke on real data),
    this walks the structure and treats each set of coordinate pairs it
    finds as one independent ring -- which matches how the source repo's
    own renderer actually uses this field (as loose polygon parts, not a
    strict GeoJSON MultiPolygon with exterior/hole semantics).
    """
    if isinstance(x, list) and x and _is_coord_pair(x[0]):
        return [x]  # x itself is a ring: a list of coordinate pairs
    rings = []
    if isinstance(x, list):
        for child in x:
            rings.extend(_find_rings(child))
    return rings


def _normalize_geometry(raw_path_str: str) -> Optional[list]:
    """Normalizes the free-form `path` field into a MultiPolygon-of-rings
    structure (one polygon per ring found, coordinates as [lng, lat]) --
    see `_find_rings` for why a flat ring-scan is used instead of trusting
    a single nesting-depth guess. Returns None if unparseable/empty --
    geometry is simply absent for that row, never fabricated.
    """
    if not raw_path_str or not raw_path_str.strip():
        return None
    try:
        parsed = json.loads(raw_path_str)
    except (json.JSONDecodeError, TypeError):
        return None

    rings = _find_rings(parsed)
    if not rings:
        return None

    def swap_ring(ring):
        return [[pt[1], pt[0]] for pt in ring]

    return [[swap_ring(ring)] for ring in rings]  # one polygon per ring found


@dataclass
class ParsedRegionSource:
    names_df: pd.DataFrame     # from wilayah.sql: region_id (raw kode), region_name -- ALL levels
    geometry_df: pd.DataFrame  # from wilayah_level_1_2.sql: region_id, geometry_json, has_geometry


def parse_wilayah_sql(sql_path: str) -> pd.DataFrame:
    """Parses db/wilayah.sql (kode, nama -- all 4 levels) into a dataframe
    with columns [kode, nama]."""
    text = Path(sql_path).read_text(encoding="utf-8", errors="replace")
    rows = parse_insert_rows(text, expected_field_count=2)
    return pd.DataFrame(rows, columns=["kode", "nama"])


def _geometry_centroid(geometry: list) -> Optional[tuple]:
    """Simple (unweighted) mean of every point across every polygon/ring in
    a normalized MultiPolygon-of-single-ring geometry (as produced by
    `_normalize_geometry`: a list of polygons, each `[ring]`, each ring a
    list of [lng,lat] points) -- a rough centroid used ONLY for the P0-5
    geometry-to-region-code consistency sanity check, never for real
    spatial analysis (see GEOMETRY_SIMPLIFICATION_NOTE)."""
    all_points = [pt for polygon in geometry for ring in polygon for pt in ring]
    if not all_points:
        return None
    lngs = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    return (sum(lngs) / len(lngs), sum(lats) / len(lats))


CONSISTENCY_DISTANCE_THRESHOLD_DEG = 2.0  # MVP assumption -- see docstring below


def _compute_geometry_quality(rings: Optional[list], declared_lat, declared_lng) -> dict:
    """Builds the P1 `geometry_quality` metadata for one region row, and
    runs the P0-5 geometry-to-region-code consistency check: does the
    (rough, unweighted) centroid of the parsed polygon fall near the
    row's OWN separately-declared lat/lng centroid column? A large
    mismatch would indicate the geometry got attached to the wrong region
    code somewhere in parsing/joining.

    CONSISTENCY_DISTANCE_THRESHOLD_DEG=2.0 degrees (~220km at the equator)
    is an MVP threshold, not derived from any spec -- flagged, not hidden.
    Some genuinely large kabupaten (e.g. in Papua) may legitimately exceed
    it; `consistency_flag` should be read as "worth a manual look", not
    proof of a real error.
    """
    if not rings:
        return {
            "ring_count": 0,
            "simplified_no_holes": True,
            "centroid_check_distance_deg": None,
            "consistency_flag": "no_geometry",
        }
    centroid = _geometry_centroid(rings)
    distance = None
    flag = "not_checked"
    if centroid is not None and pd.notna(declared_lat) and pd.notna(declared_lng):
        geom_lng, geom_lat = centroid
        distance = ((geom_lat - float(declared_lat)) ** 2 + (geom_lng - float(declared_lng)) ** 2) ** 0.5
        flag = "ok" if distance <= CONSISTENCY_DISTANCE_THRESHOLD_DEG else "mismatch"
    return {
        "ring_count": len(rings),
        "simplified_no_holes": True,
        "centroid_check_distance_deg": distance,
        "consistency_flag": flag,
    }


def parse_wilayah_level_1_2_sql(sql_path: str) -> pd.DataFrame:
    """Parses db/wilayah_level_1_2.sql into a dataframe with the raw 11
    columns, geometry normalized into `geometry_json` (a JSON string of
    the standardized [lng,lat] MultiPolygon-of-rings structure -- see
    GEOMETRY_SIMPLIFICATION_NOTE), plus a `geometry_quality_json` column
    (P1 metadata + P0-5 consistency check against the row's own declared
    lat/lng). `status` is parsed but deliberately dropped before returning
    -- see UNDOCUMENTED_FIELDS_NOT_USED_ANALYTICALLY."""
    text = Path(sql_path).read_text(encoding="utf-8", errors="replace")
    rows = parse_insert_rows(text, expected_field_count=11)
    cols = ["kode", "nama", "ibukota", "lat", "lng", "elv", "tz", "luas", "penduduk", "path", "status"]
    df = pd.DataFrame(rows, columns=cols)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    normalized = df["path"].map(_normalize_geometry)
    df["geometry_json"] = normalized.map(lambda g: json.dumps(g) if g is not None else None)
    df["has_geometry"] = df["geometry_json"].notna()
    df["geometry_quality_json"] = [
        json.dumps(_compute_geometry_quality(g, row["lat"], row["lng"]))
        for g, (_, row) in zip(normalized, df.iterrows())
    ]
    # `status` is intentionally not exposed further -- see
    # UNDOCUMENTED_FIELDS_NOT_USED_ANALYTICALLY at module level.
    return df.drop(columns=["status"])


def build_canonical_region_master(wilayah_sql_path: str, level_1_2_sql_path: str) -> pd.DataFrame:
    """Builds the canonical kabupaten/kota-level region master (the
    PRIMARY ANALYTICAL UNIT) by combining both real source files.

    This is the ONE place schema knowledge of the raw cahyadsn/wilayah SQL
    dump lives. Everything downstream (join validation, the dashboard
    choropleth) consumes only the canonical artifact this produces.
    """
    names_df = parse_wilayah_sql(wilayah_sql_path)
    names_df["level"] = names_df["kode"].map(_level_of)

    kab_kota = names_df[names_df["level"] == "kabupaten_kota"].copy()
    if kab_kota.empty:
        raise RegionMasterError(
            "No kabupaten/kota-level (2 dot-segments) rows found in "
            f"{wilayah_sql_path}. The code-schema assumption in this module "
            "no longer matches the source -- do not proceed."
        )
    kab_kota["region_id"] = kab_kota["kode"]
    kab_kota["province_id"] = kab_kota["region_id"].str.split(".").str[0]
    kab_kota["region_name"] = kab_kota["nama"]

    provinces = names_df[names_df["level"] == "province"].copy()
    provinces["province_id"] = provinces["kode"]
    provinces["province_name"] = provinces["nama"]

    merged = kab_kota.merge(provinces[["province_id", "province_name"]], on="province_id", how="left")
    missing_parent = merged["province_name"].isna().sum()
    if missing_parent:
        raise RegionMasterError(
            f"{missing_parent} kabupaten/kota row(s) have no matching province "
            "parent in the source data itself -- refusing to build a canonical "
            "master with orphaned records. Inspect the raw wilayah.sql."
        )

    geometry_df = parse_wilayah_level_1_2_sql(level_1_2_sql_path)
    geometry_df = geometry_df[geometry_df["kode"].map(_level_of) == "kabupaten_kota"]
    merged = merged.merge(
        geometry_df[["kode", "geometry_json", "has_geometry", "lat", "lng", "geometry_quality_json"]]
        .rename(columns={"lat": "declared_lat", "lng": "declared_lng"}),
        left_on="region_id", right_on="kode", how="left", suffixes=("", "_geom"),
    )
    merged["has_geometry"] = merged["has_geometry"].fillna(False)

    return merged[CANONICAL_ARTIFACT_COLUMNS].reset_index(drop=True)


def geometry_quality_summary(region_master: pd.DataFrame) -> pd.DataFrame:
    """P0-5 + P1: summarizes the geometry-to-region-code consistency check
    and geometry quality metadata across the whole master. Returns one row
    per region with its consistency_flag, for the Gate 1 report / dashboard
    to surface -- flagged rows ("mismatch") deserve a manual look before
    being trusted in the choropleth."""
    rows = []
    for _, r in region_master.iterrows():
        quality = json.loads(r["geometry_quality_json"]) if pd.notna(r.get("geometry_quality_json")) else {}
        rows.append({
            "region_id": r["region_id"],
            "region_name": r["region_name"],
            "has_geometry": r["has_geometry"],
            "ring_count": quality.get("ring_count"),
            "centroid_check_distance_deg": quality.get("centroid_check_distance_deg"),
            "consistency_flag": quality.get("consistency_flag", "not_checked"),
        })
    return pd.DataFrame(rows)


def save_canonical_artifact(region_master: pd.DataFrame, path: str) -> None:
    """Saves the canonical region master as CSV (geometry pre-serialized
    to JSON text in `geometry_json`) -- the artifact the dashboard should
    load at runtime instead of ever re-parsing the raw SQL dump."""
    region_master.to_csv(path, index=False)


def load_canonical_artifact(path: str) -> pd.DataFrame:
    """Loads a previously built canonical artifact. Raises clearly if it
    doesn't exist yet -- callers must run the preprocessing step first,
    this function never silently falls back to re-parsing the raw ZIP."""
    p = Path(path)
    if not p.exists():
        raise RegionMasterError(
            f"Canonical region master artifact not found at {path}. Run "
            "app.data.region_master.build_canonical_region_master() once "
            "(offline/preprocessing) and save it with save_canonical_artifact() "
            "before starting the dashboard."
        )
    df = pd.read_csv(p, dtype={"region_id": str, "province_id": str})
    df["has_geometry"] = df["has_geometry"].astype(bool)
    return df


def get_geometry(region_master: pd.DataFrame, region_id: str) -> Optional[list]:
    """Returns the normalized [lng,lat] MultiPolygon-of-rings geometry for
    one region_id, or None if unavailable."""
    row = region_master[region_master["region_id"] == region_id]
    if row.empty or not bool(row.iloc[0]["has_geometry"]):
        return None
    return json.loads(row.iloc[0]["geometry_json"])


# ---------------------------------------------------------------------------
# Join validation
# ---------------------------------------------------------------------------

@dataclass
class RegionJoinReport:
    matched: pd.DataFrame
    unmatched: pd.DataFrame          # region_code not found in the master at all
    duplicates: pd.DataFrame         # region_code appears more than once WITHIN the source data
    ambiguous: pd.DataFrame          # region_code resolves to more than one master row
    name_mismatch: pd.DataFrame      # supplied region_name disagrees with the master's name for that code
    invalid_level: pd.DataFrame      # region_code's segment count doesn't correspond to kabupaten_kota
    missing_province_parent: pd.DataFrame  # matched region whose province_id has no province_name in the master
    match_rate: float
    total_rows: int = 0

    def summary(self) -> str:
        return (
            f"total={self.total_rows}, matched={len(self.matched)} "
            f"({self.match_rate:.1%}), unmatched={len(self.unmatched)}, "
            f"duplicates={len(self.duplicates)}, ambiguous={len(self.ambiguous)}, "
            f"name_mismatch={len(self.name_mismatch)}, invalid_level={len(self.invalid_level)}, "
            f"missing_province_parent={len(self.missing_province_parent)}"
        )


def validate_region_join(
    source_df: pd.DataFrame,
    region_master: pd.DataFrame,
    region_col: str = "region_code",
    region_name_col: Optional[str] = None,
) -> RegionJoinReport:
    """Validates source_df's region codes against the canonical kabupaten/
    kota master. NEVER joins by name alone: `region_name_col`, if given, is
    used only to produce a `name_mismatch` diagnostic, never to establish
    or repair a match.
    """
    df = source_df.copy()
    df["_kode"] = df[region_col].astype(str)
    df["_level"] = df["_kode"].map(_level_of)

    invalid_level = df[df["_level"] != "kabupaten_kota"].drop(columns=["_kode", "_level"])
    df = df[df["_level"] == "kabupaten_kota"]

    dup_mask = df["_kode"].duplicated(keep=False)
    duplicates = df[dup_mask].drop(columns=["_kode", "_level"])

    master_counts = region_master.groupby("region_id").size()
    ambiguous_ids = set(master_counts[master_counts > 1].index)
    ambiguous = df[df["_kode"].isin(ambiguous_ids) & ~dup_mask].drop(columns=["_kode", "_level"])

    known_ids = set(region_master["region_id"])
    matched_mask = df["_kode"].isin(known_ids) & ~dup_mask & ~df["_kode"].isin(ambiguous_ids)
    matched = df[matched_mask].merge(
        region_master, left_on="_kode", right_on="region_id", how="left", suffixes=("_input", "")
    ).drop(columns=["_kode", "_level"])
    unmatched = df[~matched_mask & ~dup_mask & ~df["_kode"].isin(ambiguous_ids)].drop(columns=["_kode", "_level"])

    missing_province_parent = pd.DataFrame()
    if not matched.empty:
        missing_province_parent = matched[matched["province_name"].isna()]

    name_mismatch = pd.DataFrame()
    if region_name_col and not matched.empty:
        # if the source's column name collided with the master's own
        # "region_name" column, pandas suffixed the source's copy
        input_name_col = f"{region_name_col}_input" if f"{region_name_col}_input" in matched.columns else region_name_col
        if input_name_col in matched.columns:
            mismatch_mask = (
                matched[input_name_col].astype(str).str.strip().str.lower()
                != matched["region_name"].astype(str).str.strip().str.lower()
            )
            name_mismatch = matched[mismatch_mask]

    total_rows = len(source_df)
    match_rate = len(matched) / total_rows if total_rows else 0.0

    return RegionJoinReport(
        matched=matched,
        unmatched=unmatched,
        duplicates=duplicates,
        ambiguous=ambiguous,
        name_mismatch=name_mismatch,
        invalid_level=invalid_level,
        missing_province_parent=missing_province_parent,
        match_rate=match_rate,
        total_rows=total_rows,
    )
