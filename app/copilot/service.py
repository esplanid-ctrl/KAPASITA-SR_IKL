"""
KAPASITA Policy Copilot -- Stage 5: strict evidence grounding + citation
validation.

Per project-owner decision: "The LLM must not compensate for missing or
unverified evidence." This module:
  1. Builds the ONLY context the model may reason over, from ALREADY
     Python-validated Gate 1 + policy-engine output (never raw indicators,
     never an unstructured dataframe dump).
  2. Instructs the model to state each rule's real status
     (candidate/verified/pending_policy_verification/unlinked_policy/etc.)
     using the SAME wording as the data, never a stronger claim.
  3. After generation, validates every rule_id/source_id/region_id-shaped
     token in the response against what was actually in the context, and
     prepends a visible warning if the model cited anything that wasn't
     there. This is never silently hidden or "fixed" on the model's
     behalf -- the point is to make a hallucinated citation visible to
     the person asking, not to paper over it.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .gemini_client import generate
from .grounding import build_grounded_context, validate_citations

SYSTEM = """Anda adalah KAPASITA Policy Copilot.

ATURAN KETAT (wajib dipatuhi tanpa pengecualian):
1. Gunakan HANYA evidence yang diberikan di bawah ini. Jangan mengarang angka, regulasi, sumber, rule_id, source_id, atau kode wilayah yang tidak tercantum eksplisit di EVIDENCE.
2. Setiap kali menyebut sebuah rule/rekomendasi, sebutkan rule_status-nya PERSIS seperti tertulis di EVIDENCE (mis. "kandidat, belum diverifikasi", "verifikasi registry tertunda"). JANGAN menyebut sesuatu "resmi", "terverifikasi", atau "pasti berlaku" kecuali rule_status-nya benar-benar TERVERIFIKASI di EVIDENCE.
3. Butir berstatus unlinked_policy BUKAN rekomendasi kebijakan -- sebut sebagai "sinyal data tanpa dasar kebijakan", jangan pernah sebagai rekomendasi resmi.
4. Jika evidence tidak cukup untuk menjawab pertanyaan, katakan terus terang bahwa evidence tidak cukup. JANGAN mengisi kekosongan itu dengan asumsi, pengetahuan umum, atau tebakan Anda sendiri.
5. Jangan menentukan lokasi, lahan, konstruksi, atau penjadwalan operasional Sekolah Rakyat.
6. Bedakan jelas antara fakta dari EVIDENCE, interpretasi Anda, dan opsi kebijakan yang tersedia."""


def ask(
    question: str,
    scored_df: pd.DataFrame,
    recommendations_df: Optional[pd.DataFrame] = None,
    unlinked_signals_df: Optional[pd.DataFrame] = None,
    region_col: str = "region_code",
) -> str:
    """Answers `question` grounded strictly in already-validated Gate 1 +
    policy-engine output, then checks the response for invented citations.
    """
    context = build_grounded_context(scored_df, recommendations_df, unlinked_signals_df, region_col)
    prompt = f"{SYSTEM}\n\nEVIDENCE:\n{context.text}\n\nPERTANYAAN:\n{question}\n\nJawab ringkas dalam Bahasa Indonesia."
    response = generate(prompt)

    check = validate_citations(response, context)
    if check.has_invented_citations:
        parts = []
        if check.invented_rule_ids:
            parts.append(f"rule_id tidak dikenal: {', '.join(check.invented_rule_ids)}")
        if check.invented_source_ids:
            parts.append(f"source_id tidak dikenal: {', '.join(check.invented_source_ids)}")
        if check.invented_region_ids:
            parts.append(f"kode wilayah tidak dikenal: {', '.join(check.invented_region_ids)}")
        warning = (
            "\u26a0\ufe0f PERINGATAN GROUNDING: respons di bawah menyebut ID yang TIDAK ada di evidence "
            "yang diberikan (kemungkinan halusinasi model) -- " + "; ".join(parts) + ".\n\n"
        )
        response = warning + response

    return response
