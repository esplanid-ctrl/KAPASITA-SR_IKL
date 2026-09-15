"""
KAPASITA Policy Copilot -- Stage 5 (hardened per review, same architecture).

Per project-owner decision: "The LLM must not compensate for missing or
unverified evidence," and this review explicitly says: do not redesign
this architecture, do not add autonomous analytics or new aggregations.
Deterministic Python remains the only source of numerical truth. This
module is exactly the same three-step design as before, with three
guard rails hardened:

  1. GROUNDING: context is built ONLY from already Python-validated
     Gate 1 + policy-engine output (app.copilot.grounding).
  2. UNSUPPORTED-COMPUTATION GUARD: if the question asks for cross-region
     aggregation/comparison/statistics Python hasn't precomputed, `ask()`
     returns a fixed "not available" answer WITHOUT calling Gemini --
     the model is never asked to infer or calculate a number Python
     didn't produce.
  3. CITATION VALIDATION: an IDENTIFIER EXISTENCE CHECK (not a semantic
     check) against the canonical registries + the specific evidence
     given, with fail-safe handling -- if the check itself breaks, that
     is surfaced as "could not be confirmed," never silently treated as
     "no problem found."

Gemini failures (missing key, network/API error, malformed response) are
caught here and degrade to a plain, clearly-labeled Indonesian message --
never an unhandled exception reaching the dashboard.
"""
from __future__ import annotations

from typing import Optional, Set

import pandas as pd

from .gemini_client import generate
from .grounding import build_grounded_context, validate_citations, detect_unsupported_computation

SYSTEM = """Anda adalah KAPASITA Policy Copilot.

ATURAN KETAT (wajib dipatuhi tanpa pengecualian):
1. Gunakan HANYA evidence yang diberikan di bawah ini. Jangan mengarang angka, regulasi, sumber, rule_id, source_id, atau kode wilayah yang tidak tercantum eksplisit di EVIDENCE.
2. Setiap kali menyebut sebuah rule/rekomendasi, sebutkan rule_status-nya PERSIS seperti tertulis di EVIDENCE (mis. "kandidat, belum diverifikasi", "verifikasi registry tertunda"). JANGAN menyebut sesuatu "resmi", "terverifikasi", atau "pasti berlaku" kecuali rule_status-nya benar-benar TERVERIFIKASI di EVIDENCE.
3. Butir berstatus unlinked_policy BUKAN rekomendasi kebijakan -- sebut sebagai "sinyal data tanpa dasar kebijakan", jangan pernah sebagai rekomendasi resmi.
4. Jika evidence tidak cukup untuk menjawab pertanyaan, katakan terus terang bahwa evidence tidak cukup. JANGAN mengisi kekosongan itu dengan asumsi, pengetahuan umum, atau tebakan Anda sendiri.
5. JANGAN menghitung agregat, rata-rata, korelasi, tren, atau ranking lintas-wilayah sendiri -- semua angka yang Anda sebut harus persis berasal dari EVIDENCE, bukan hasil hitung Anda.
6. Jangan menentukan lokasi, lahan, konstruksi, atau penjadwalan operasional Sekolah Rakyat.
7. Bedakan jelas antara fakta dari EVIDENCE, interpretasi Anda, dan opsi kebijakan yang tersedia."""

UNSUPPORTED_COMPUTATION_RESPONSE_TEMPLATE = (
    "Belum bisa menjawab ini: {reason}\n\n"
    "Yang tersedia dari sistem saat ini hanya skor dan rekomendasi PER WILAYAH kabupaten/kota "
    "(lihat tab Overview, Support Priority, dan Intervention & Capacity). Analisis gabungan/"
    "perbandingan/tren lintas-wilayah tidak dihitung otomatis oleh Copilot ini agar tidak ada "
    "angka yang dihasilkan model bahasa tanpa dasar perhitungan Python yang terverifikasi."
)

GEMINI_FAILURE_MESSAGE = (
    "Copilot sedang tidak bisa menjawab (gagal menghubungi Gemini). Ini bukan berarti evidence "
    "tidak ada -- silakan cek tab Intervention & Capacity dan Data Quality secara langsung, atau "
    "coba lagi sebentar lagi."
)


def ask(
    question: str,
    scored_df: pd.DataFrame,
    recommendations_df: Optional[pd.DataFrame] = None,
    unlinked_signals_df: Optional[pd.DataFrame] = None,
    region_col: str = "region_code",
    region_master_ids: Optional[Set[str]] = None,
) -> str:
    """Answers `question` grounded strictly in already-validated Gate 1 +
    policy-engine output.

    `region_master_ids`, if supplied (e.g. the full canonical region
    master's region_id set), lets the citation check distinguish a
    genuinely unknown region code from one that is real but simply wasn't
    part of this query's evidence -- see app.copilot.grounding.CitationCheck.
    """
    # --- Guard 1: unsupported cross-region computation -> never call Gemini for this ---
    unsupported_reason = detect_unsupported_computation(question)
    if unsupported_reason:
        return UNSUPPORTED_COMPUTATION_RESPONSE_TEMPLATE.format(reason=unsupported_reason)

    context = build_grounded_context(scored_df, recommendations_df, unlinked_signals_df, region_col)
    prompt = f"{SYSTEM}\n\nEVIDENCE:\n{context.text}\n\nPERTANYAAN:\n{question}\n\nJawab ringkas dalam Bahasa Indonesia."

    # --- Guard 2: Gemini failures degrade gracefully, never raise to the caller ---
    try:
        response = generate(prompt)
    except Exception:
        return GEMINI_FAILURE_MESSAGE
    if not isinstance(response, str) or not response.strip():
        return GEMINI_FAILURE_MESSAGE

    # --- Guard 3: citation validation (identifier existence check, fail-safe) ---
    check = validate_citations(response, context, region_master_ids=region_master_ids)

    if check.check_failed:
        return (
            "\u26a0\ufe0f Pemeriksaan sitasi (identifier existence check) gagal dijalankan "
            f"({check.check_failure_reason}) -- ID pada jawaban di bawah TIDAK bisa dikonfirmasi "
            "otomatis. Verifikasi manual dianjurkan sebelum menggunakan jawaban ini.\n\n" + response
        )

    if check.has_invented_citations:
        parts = []
        if check.unknown_to_system_rule_ids:
            parts.append(f"rule_id tidak dikenal sistem: {', '.join(check.unknown_to_system_rule_ids)}")
        if check.unknown_to_system_source_ids:
            parts.append(f"source_id tidak dikenal sistem: {', '.join(check.unknown_to_system_source_ids)}")
        if check.unknown_to_system_region_ids:
            parts.append(f"kode wilayah tidak dikenal sistem: {', '.join(check.unknown_to_system_region_ids)}")
        response = (
            "\u26a0\ufe0f PERINGATAN GROUNDING (pemeriksaan keberadaan ID, BUKAN pemeriksaan makna/isi klaim): "
            "respons di bawah menyebut ID yang TIDAK terdaftar di sistem mana pun -- " + "; ".join(parts)
            + ".\n\n" + response
        )
    elif check.has_out_of_context_citations:
        parts = []
        if check.out_of_context_rule_ids:
            parts.append(f"rule_id: {', '.join(check.out_of_context_rule_ids)}")
        if check.out_of_context_source_ids:
            parts.append(f"source_id: {', '.join(check.out_of_context_source_ids)}")
        if check.out_of_context_region_ids:
            parts.append(f"kode wilayah: {', '.join(check.out_of_context_region_ids)}")
        response = (
            "\u2139\ufe0f CATATAN GROUNDING (pemeriksaan keberadaan ID, BUKAN pemeriksaan makna/isi klaim): "
            "ID berikut memang terdaftar di sistem tapi TIDAK ada di evidence yang diberikan untuk "
            "pertanyaan ini -- " + "; ".join(parts) + ".\n\n" + response
        )

    return response
