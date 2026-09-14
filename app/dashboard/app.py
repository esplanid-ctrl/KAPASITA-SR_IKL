"""
KAPASITA-MVP Dashboard (docs/07 PRD features 1-7).

This dashboard is a thin presentation layer over app.data.pipeline.run_gate1()
and app.policy.engine.evaluate() -- it must never re-implement validation,
scoring, or region-join logic itself (project-owner decision: "region
master is a validation artifact/source, not an ad-hoc lookup embedded in
dashboard code" -- the same principle is applied to validation and scoring
here).

Ingestion mode: this MVP demo accepts one CSV per indicator_id (matching
the exact contract in data/DATA_MAPPING_REAL.csv). Region master can be
loaded from Google Drive (GOOGLE_DRIVE_REGION_FOLDER_ID, wired through
app.data.drive_adapter.load_region_master_from_drive) or from a manual
upload for offline/demo use -- either way it goes through the same
canonical-artifact cache, never re-parsed from the raw ZIP on every page
load.

Copilot grounding note (Stage 5): the Copilot tab is grounded strictly on
this same Gate 1 + policy-engine output via app.copilot.grounding, with
every rule_status spelled out honestly and a post-hoc citation check that
flags any rule_id/source_id/region_id the model mentions that isn't
actually in the evidence it was given.
"""
import json

import pandas as pd
import streamlit as st

from app.data.validation import load_indicator_dictionary
from app.data.region_master import (
    build_canonical_region_master,
    save_canonical_artifact,
    load_canonical_artifact,
    RegionMasterError,
)
from app.data.drive_adapter import load_region_master_from_drive
from app.data.pipeline import run_gate1
from app.policy.engine import evaluate, recommendations_to_frame, unlinked_signals_to_frame, static_rule_status_summary
from app.policy.evidence import load_policy_register
from app.copilot.service import ask
from config.settings import load_region_source_config, ConfigError

st.set_page_config(page_title="KAPASITA-MVP", layout="wide")
st.title("KAPASITA — AI Policy Intelligence")
st.caption("Sekolah Rakyat & Pendidikan Inklusif | MVP LAN Datathon 2026")
st.info(
    "Fokus MVP: layanan inklusif, intervensi kebijakan, dan kapasitas. "
    "Bukan penentuan lokasi, lahan, konstruksi, atau penjadwalan operasional Sekolah Rakyat."
)

REGION_COL = "region_code"


@st.cache_data
def _mapping():
    return load_indicator_dictionary()


mapping_df = _mapping()

# ---------------------------------------------------------------------------
# Sidebar: per-indicator ingestion (mirrors the real Gate 1 contract exactly)
# ---------------------------------------------------------------------------
st.sidebar.header("1. Ingest data (Gate 1 contract)")
st.sidebar.caption(
    "Setiap indikator divalidasi terhadap data/DATA_MAPPING_REAL.csv. "
    "Mode produksi menarik file yang sama dari Google Drive (docs/11); "
    "mode demo ini menerima upload manual per indikator."
)

dfs_by_indicator = {}
for _, row in mapping_df.iterrows():
    indicator_id = row["indicator_id"]
    uploaded = st.sidebar.file_uploader(
        f"{indicator_id}",
        type=["csv"],
        help=f"domain={row['domain']} | file_pattern~'{row['file_pattern']}' | "
             f"field_pattern~'{row['field_pattern']}' | year={row['year']}",
        key=f"upload_{indicator_id}",
    )
    if uploaded is not None:
        # region_code must stay a string (dot-separated codes like "11.01"
        # are numeric-looking and pandas will silently corrupt them to
        # float64, e.g. "11.10" -> 11.1, if not forced to str here).
        dfs_by_indicator[indicator_id] = pd.read_csv(uploaded, dtype={REGION_COL: str})

st.sidebar.markdown("---")
st.sidebar.caption(
    "Master wilayah kanonik (docs/03 D08, cahyadsn/wilayah). "
    "Preferensi: unggah artifact CSV kanonik yang sudah di-preprocess "
    "(lihat scripts/build_region_master.py) -- ini TIDAK di-parse ulang "
    "dari ZIP mentah setiap kali halaman dimuat (keputusan pemilik proyek)."
)
region_master_options = ["Google Drive (GOOGLE_DRIVE_REGION_FOLDER_ID)", "Artifact kanonik (.csv)", "Bangun dari SQL mentah (sekali, untuk demo)"]
try:
    _region_cfg = load_region_source_config(required=False)
    default_index = 0 if _region_cfg.region_folder_id else 1
except ConfigError:
    default_index = 1
region_master_mode = st.sidebar.radio("Sumber master wilayah", region_master_options, index=default_index)
region_master = None
geometry_quality_summary_df = None

if region_master_mode == "Google Drive (GOOGLE_DRIVE_REGION_FOLDER_ID)":
    try:
        region_cfg = load_region_source_config(required=True)
        with st.spinner("Memuat master wilayah dari Google Drive (atau cache lokal jika sudah ada)..."):
            region_master = load_region_master_from_drive(folder_id=region_cfg.region_folder_id)
        st.sidebar.success(f"Master wilayah dimuat dari Drive/cache: {len(region_master)} kab/kota.")
    except ConfigError as e:
        st.sidebar.error(f"GOOGLE_DRIVE_REGION_FOLDER_ID belum dikonfigurasi: {e}")
    except RegionMasterError as e:
        st.sidebar.error(str(e))
elif region_master_mode == "Artifact kanonik (.csv)":
    artifact_upload = st.sidebar.file_uploader("region_master_canonical.csv", type=["csv"], key="upload_artifact")
    if artifact_upload is not None:
        try:
            artifact_upload.seek(0)
            pd.read_csv(artifact_upload).to_csv("/tmp/_region_master_artifact.csv", index=False)
            region_master = load_canonical_artifact("/tmp/_region_master_artifact.csv")
            st.sidebar.success(f"Master wilayah dimuat: {len(region_master)} kab/kota.")
        except RegionMasterError as e:
            st.sidebar.error(str(e))
else:
    wilayah_sql_upload = st.sidebar.file_uploader("wilayah.sql", type=["sql"], key="upload_wilayah_sql")
    l12_sql_upload = st.sidebar.file_uploader("wilayah_level_1_2.sql", type=["sql"], key="upload_l12_sql")

    @st.cache_data
    def _build_from_sql(wilayah_bytes: bytes, l12_bytes: bytes):
        with open("/tmp/_wilayah.sql", "wb") as f:
            f.write(wilayah_bytes)
        with open("/tmp/_wilayah_l12.sql", "wb") as f:
            f.write(l12_bytes)
        built = build_canonical_region_master("/tmp/_wilayah.sql", "/tmp/_wilayah_l12.sql")
        save_canonical_artifact(built, "/tmp/_region_master_artifact.csv")
        return built

    if wilayah_sql_upload is not None and l12_sql_upload is not None:
        try:
            region_master = _build_from_sql(wilayah_sql_upload.getvalue(), l12_sql_upload.getvalue())
            st.sidebar.success(f"Master wilayah dibangun & di-cache: {len(region_master)} kab/kota.")
            st.sidebar.download_button(
                "Unduh artifact kanonik (untuk dipakai ulang)",
                data=open("/tmp/_region_master_artifact.csv", "rb").read(),
                file_name="region_master_canonical.csv",
            )
        except RegionMasterError as e:
            st.sidebar.error(str(e))

min_domain_coverage = st.sidebar.slider(
    "min_domain_coverage (MVP assumption, default 0.50)", 0.0, 1.0, 0.50, 0.05
)
critical_domains = st.sidebar.multiselect(
    "critical_domains (opsional — kosong = tidak ada domain wajib)",
    ["INC", "CAP", "SOC", "DIG", "POL"],
    default=[],
)

if not dfs_by_indicator:
    st.write("Unggah minimal satu file indikator di sidebar untuk menjalankan Gate 1.")
    st.stop()

report = run_gate1(
    dfs_by_indicator,
    region_master=region_master,
    region_col=REGION_COL,
    min_domain_coverage=min_domain_coverage,
    critical_domains=critical_domains,
)
scored_df = report.scored_df

policy_result = evaluate(scored_df, region_col=REGION_COL)
rec_df = recommendations_to_frame(policy_result)
signal_df = unlinked_signals_to_frame(policy_result)

tabs = st.tabs([
    "1. Overview", "2. Support Priority", "3. Why Analysis",
    "4. Intervention & Capacity", "5. Evidence/Lineage",
    "6. AI Copilot", "7. Data Quality/Confidence",
])

# --- 1. Overview -----------------------------------------------------------
with tabs[0]:
    st.subheader("Ringkasan Gate 1")
    st.text(report.summary())
    st.subheader("Skor SR-ICSS")
    st.dataframe(
        scored_df[[REGION_COL, "SR_ICSS", "score_mode", "confidence", "unavailable_domains"]]
        .sort_values("SR_ICSS", ascending=False, na_position="last"),
        use_container_width=True,
    )

# --- 2. Support Priority ----------------------------------------------------
with tabs[1]:
    st.subheader("Prioritas dukungan (bukan lokasi)")
    scoreable = scored_df[scored_df["score_mode"] != "unavailable"].sort_values("SR_ICSS", ascending=False)

    geometry_available = region_master is not None and bool(region_master["has_geometry"].any())
    if geometry_available:
        geo_cols = ["region_id", "region_name", "has_geometry", "geometry_json"]
        if "geometry_quality_json" in region_master.columns:
            geo_cols.append("geometry_quality_json")
        geo_df = scoreable.merge(
            region_master[geo_cols], left_on=REGION_COL, right_on="region_id", how="inner",
        )
        geo_df = geo_df[geo_df["has_geometry"]]
        missing_geo_count = len(scoreable) - len(geo_df)
        if geo_df.empty:
            st.warning(
                "Master wilayah dimuat, tetapi tidak ada wilayah yang cocok dan punya geometri. "
                "Menampilkan peringkat sebagai gantinya."
            )
            st.bar_chart(scoreable.set_index(REGION_COL)["SR_ICSS"])
        else:
            if missing_geo_count > 0:
                st.caption(f"{missing_geo_count} wilayah tidak punya geometri di master wilayah dan tidak tampil di peta (lihat tab Data Quality).")

            # P0-5: surface geometry-to-region-code consistency flags rather
            # than silently trusting every polygon.
            if "geometry_quality_json" in geo_df.columns:
                flags = geo_df["geometry_quality_json"].map(
                    lambda s: json.loads(s).get("consistency_flag") if isinstance(s, str) else "not_checked"
                )
                mismatched = geo_df.loc[flags == "mismatch", "region_name"].tolist()
                if mismatched:
                    st.warning(
                        f"{len(mismatched)} wilayah punya geometri yang jaraknya jauh dari titik pusat "
                        f"resmi (worth a manual look, bukan bukti pasti error): {', '.join(mismatched)}"
                    )

            st.caption(
                "CRS TIDAK dinyatakan eksplisit di sumber cahyadsn/wilayah — WGS84/EPSG:4326 di sini "
                "adalah ASUMSI MVP belum terverifikasi. Geometri disederhanakan (ring diratakan, tanpa "
                "lubang poligon) HANYA untuk visualisasi peta ini, bukan untuk analisis spasial."
            )
            features = []
            for _, r in geo_df.iterrows():
                geom_coords = json.loads(r["geometry_json"])
                features.append({
                    "type": "Feature",
                    "properties": {"region_id": r[REGION_COL]},
                    "geometry": {"type": "MultiPolygon", "coordinates": geom_coords},
                })
            geojson = {"type": "FeatureCollection", "features": features}
            try:
                import plotly.express as px
                fig = px.choropleth(
                    geo_df, geojson=geojson, locations=REGION_COL, featureidkey="properties.region_id",
                    color="SR_ICSS", color_continuous_scale="OrRd", hover_name="region_name",
                    projection="mercator",
                )
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.error("Paket 'plotly' tidak tersedia di environment ini untuk merender choropleth.")
    else:
        st.info(
            "Peta choropleth memerlukan master wilayah dengan geometri (wilayah_level_1_2.path). "
            "Belum tersedia di sesi ini -- menampilkan peringkat, BUKAN peta, dan TIDAK mengunduh "
            "sumber batas wilayah alternatif secara diam-diam (keputusan pemilik proyek)."
        )
        st.bar_chart(scoreable.set_index(REGION_COL)["SR_ICSS"])

    st.dataframe(scoreable[[REGION_COL, "SR_ICSS", "score_mode"]], use_container_width=True)

# --- 3. Why analysis ---------------------------------------------------------
with tabs[2]:
    st.subheader("Kontribusi per domain")
    region_pick = st.selectbox("Pilih wilayah", scored_df[REGION_COL].tolist())
    row = scored_df[scored_df[REGION_COL] == region_pick].iloc[0]
    contrib_cols = [c for c in scored_df.columns if c.startswith("contrib_")]
    st.write(f"score_mode: **{row['score_mode']}** | unavailable_domains: **{row['unavailable_domains'] or '(none)'}**")
    contrib_df = pd.DataFrame({
        "domain": [c.replace("contrib_", "") for c in contrib_cols],
        "contribution": [row[c] for c in contrib_cols],
        "configured_weight": [row[f"configured_weight_{c.replace('contrib_', '')}"] for c in contrib_cols],
        "effective_weight": [row[f"effective_weight_{c.replace('contrib_', '')}"] for c in contrib_cols],
    })
    st.dataframe(contrib_df, use_container_width=True)
    st.caption(
        "'configured_weight' adalah bobot resmi (30/25/20/15/10, tidak diubah). "
        "'effective_weight' adalah bobot ternormalisasi ulang saat score_mode='partial_renormalized'."
    )

# --- 4. Intervention & capacity ----------------------------------------------
with tabs[3]:
    st.subheader("Rekomendasi (evidence-linked)")
    if rec_df.empty:
        st.write("Belum ada rekomendasi yang trigger pada data saat ini.")
    else:
        st.dataframe(rec_df, use_container_width=True)
    st.subheader("Sinyal tanpa evidence kebijakan (unlinked_policy)")
    st.caption("Trigger numerik terpenuhi, tetapi belum ada sumber kebijakan terdaftar — tidak disajikan sebagai rekomendasi resmi.")
    if signal_df.empty:
        st.write("(tidak ada)")
    else:
        st.dataframe(signal_df, use_container_width=True)

# --- 5. Evidence / lineage ---------------------------------------------------
with tabs[4]:
    st.subheader("Status seluruh rule kebijakan (statis, tidak tergantung data)")
    st.dataframe(static_rule_status_summary(), use_container_width=True)
    st.subheader("Registry sumber kebijakan")
    st.dataframe(load_policy_register(), use_container_width=True)
    st.subheader("Kamus indikator (kontrak traceability)")
    st.dataframe(mapping_df, use_container_width=True)

# --- 6. AI Copilot ------------------------------------------------------------
with tabs[5]:
    st.subheader("Tanya Copilot (Gemini, grounded pada hasil di atas)")
    q = st.text_input("Pertanyaan")
    if q:
        with st.spinner("Meminta jawaban dari Copilot..."):
            answer = ask(q, scored_df, rec_df, signal_df, region_col=REGION_COL)
        if "PERINGATAN GROUNDING" in answer:
            st.error("Model menyebut ID yang tidak ada di evidence -- lihat peringatan di bawah.")
        st.write(answer)
    st.caption(
        "Copilot hanya boleh menjawab dari evidence Gate 1 + Policy Engine di atas (rule_status "
        "disebutkan apa adanya: kandidat/verified/pending_policy_verification/unlinked_policy/dst). "
        "Setiap respons diperiksa otomatis untuk rule_id/source_id/kode wilayah yang tidak ada di "
        "evidence -- jika ditemukan, itu ditandai eksplisit sebagai kemungkinan halusinasi, bukan "
        "disembunyikan."
    )

# --- 7. Data quality / confidence --------------------------------------------
with tabs[6]:
    st.subheader("Cakupan domain per wilayah")
    st.dataframe(report.domain_coverage, use_container_width=True)
    st.subheader("Laporan validasi per indikator (Gate 1)")
    indicator_rows = []
    for indicator_id, r in report.indicator_reports.items():
        indicator_rows.append({
            "indicator_id": indicator_id,
            "ok": r.ok,
            "row_count": r.row_count,
            "missing_rate": r.missing_rate,
            "duplicate_region_count": r.duplicate_region_count,
            "year_confirmed": r.year_confirmed,
            "issues": "; ".join(f"[{i.severity}] {i.message}" for i in r.issues),
        })
    st.dataframe(pd.DataFrame(indicator_rows), use_container_width=True)
    if report.region_master_schema:
        st.subheader("Skema master wilayah")
        st.json(report.region_master_schema)
    if report.region_join is not None:
        st.subheader("Kualitas join wilayah (feature matrix gabungan)")
        st.write(report.region_join.summary())
        st.write("Unmatched:")
        st.dataframe(report.region_join.unmatched, use_container_width=True)
        st.write("Ambiguous:")
        st.dataframe(report.region_join.ambiguous, use_container_width=True)
        st.write("Invalid level (bukan kode kab/kota):")
        st.dataframe(report.region_join.invalid_level, use_container_width=True)
        st.write("Missing province parent:")
        st.dataframe(report.region_join.missing_province_parent, use_container_width=True)

        st.subheader("Kualitas join per dataset analitik nyata")
        st.caption("Setiap indikator divalidasi TERPISAH terhadap master wilayah -- bukan hanya di level gabungan.")
        per_indicator_rows = [
            {"indicator_id": iid, **{
                "matched": len(r.matched), "unmatched": len(r.unmatched),
                "duplicates": len(r.duplicates), "ambiguous": len(r.ambiguous),
                "invalid_level": len(r.invalid_level), "match_rate": r.match_rate,
            }}
            for iid, r in report.region_join_by_indicator.items()
        ]
        st.dataframe(pd.DataFrame(per_indicator_rows), use_container_width=True)

        if not report.geometry_quality.empty:
            st.subheader("Kualitas geometri (P0-5: konsistensi geometri vs kode wilayah)")
            st.caption(report.region_master_schema.get("crs_assumption", ""))
            st.caption(report.region_master_schema.get("geometry_note", ""))
            st.dataframe(report.geometry_quality, use_container_width=True)
    else:
        st.info("Master wilayah belum dimuat — validasi join belum dijalankan.")
