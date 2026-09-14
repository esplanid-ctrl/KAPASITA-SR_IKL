"""
Copilot grounding & citation validation (Stage 5).

Per project-owner decision: "The LLM must not compensate for missing or
unverified evidence." This module builds a structured, ID-labeled context
from ALREADY-VALIDATED Gate 1 / policy-engine output (never raw
indicators, never a free-text dataframe dump), and validates the model's
response afterwards to catch any rule_id / source_id / region_id it
references that is NOT actually present in that context -- i.e. any
citation the model may have hallucinated.

Deterministic Python owns truth here, exactly as it does for scoring: the
LLM is only ever allowed to talk about what it was given, never to invent
or strengthen a fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

RULE_ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-SR-\d{2})\b")
SOURCE_ID_PATTERN = re.compile(r"\b(POL-SR-\d{2})\b")
REGION_ID_PATTERN = re.compile(r"\b(\d{2}\.\d{2})\b")

# Exact wording the model is told to use for each rule_status -- kept as a
# single source of truth so the system prompt and the grounded context
# always describe statuses identically (no room for the model to "round
# up" a candidate mapping into something more confident).
RULE_STATUS_GLOSS = {
    "candidate": "KANDIDAT -- mengutip evidence terdaftar, TAPI BELUM diverifikasi manusia terhadap pasal/halaman sumber",
    "verified": "TERVERIFIKASI manusia",
    "rejected": "DITOLAK -- tidak berlaku, jangan direkomendasikan",
    "pending_policy_verification": "status registry masih 'verify' -- verifikasi TERTUNDA (BUKAN berarti tidak efektif)",
    "data_unavailable": "data indikator TIDAK TERSEDIA -- rule tidak aktif",
    "unlinked_policy": "TIDAK ADA sumber kebijakan terdaftar -- ini SINYAL DATA SAJA, bukan rekomendasi kebijakan",
    "policy_inactive": "sumber kebijakan sudah TIDAK BERLAKU",
}


@dataclass
class GroundedContext:
    text: str
    allowed_rule_ids: Set[str] = field(default_factory=set)
    allowed_source_ids: Set[str] = field(default_factory=set)
    allowed_region_ids: Set[str] = field(default_factory=set)


def build_grounded_context(
    scored_df: pd.DataFrame,
    recommendations_df: Optional[pd.DataFrame],
    unlinked_signals_df: Optional[pd.DataFrame],
    region_col: str = "region_code",
) -> GroundedContext:
    """Builds the ONLY evidence text the Copilot is allowed to reason over.

    Every rule's rule_status is spelled out via RULE_STATUS_GLOSS so the
    model has no excuse to present a candidate mapping as verified, or a
    pending-verification source as settled -- the honest wording is
    already sitting right there in the evidence it was given.
    """
    lines = ["=== WILAYAH & SKOR (dihitung Python, deterministic) ==="]
    for _, row in scored_df.iterrows():
        lines.append(
            f"- {row.get(region_col)}: SR_ICSS={row.get('SR_ICSS')}, "
            f"score_mode={row.get('score_mode')}, confidence={row.get('confidence')}"
        )

    lines.append("\n=== REKOMENDASI (evidence-linked; rule_status WAJIB disebutkan apa adanya) ===")
    if recommendations_df is None or recommendations_df.empty:
        lines.append("(tidak ada rekomendasi yang trigger untuk data ini)")
    else:
        for _, r in recommendations_df.iterrows():
            gloss = RULE_STATUS_GLOSS.get(r.get("rule_status"), str(r.get("rule_status")))
            lines.append(
                f"- rule_id={r.get('rule_id')} | wilayah={r.get('region_code')} | "
                f"rule_status=({gloss}) | source_id={r.get('source_id')} | "
                f"policy_effective_status={r.get('policy_effective_status')} | "
                f"data_confidence={r.get('data_confidence')} | "
                f"intervensi={r.get('intervention')} | kebutuhan_kapasitas={r.get('capacity_need')} | "
                f"alasan={r.get('reason')}"
            )

    lines.append(
        f"\n=== SINYAL TANPA EVIDENCE KEBIJAKAN ({RULE_STATUS_GLOSS['unlinked_policy']}) ==="
    )
    if unlinked_signals_df is None or unlinked_signals_df.empty:
        lines.append("(tidak ada)")
    else:
        for _, r in unlinked_signals_df.iterrows():
            lines.append(
                f"- rule_id={r.get('rule_id')} | wilayah={r.get('region_code')} | "
                f"trigger_value={r.get('trigger_value')} | alasan={r.get('reason')}"
            )

    text = "\n".join(lines)

    allowed_region_ids = set(scored_df[region_col].astype(str)) if region_col in scored_df.columns else set()
    allowed_rule_ids: Set[str] = set()
    allowed_source_ids: Set[str] = set()
    for df in (recommendations_df, unlinked_signals_df):
        if df is not None and not df.empty:
            if "rule_id" in df.columns:
                allowed_rule_ids |= set(df["rule_id"].dropna().astype(str))
            if "region_code" in df.columns:
                allowed_region_ids |= set(df["region_code"].dropna().astype(str))
            if "source_id" in df.columns:
                for cell in df["source_id"].dropna().astype(str):
                    allowed_source_ids |= {s.strip() for s in cell.split(",") if s.strip() and s.strip() != "(none)"}

    return GroundedContext(
        text=text,
        allowed_rule_ids=allowed_rule_ids,
        allowed_source_ids=allowed_source_ids,
        allowed_region_ids=allowed_region_ids,
    )


@dataclass
class CitationCheck:
    invented_rule_ids: List[str]
    invented_source_ids: List[str]
    invented_region_ids: List[str]

    @property
    def has_invented_citations(self) -> bool:
        return bool(self.invented_rule_ids or self.invented_source_ids or self.invented_region_ids)


def validate_citations(response_text: str, context: GroundedContext) -> CitationCheck:
    """Scans the model's response for rule_id/source_id/region_id-shaped
    tokens and flags any that are NOT in the context's allowed sets.

    Since deterministic Python -- not the LLM -- is the source of truth
    for every real ID, any ID-shaped token in the response that isn't in
    the allowed sets can only be the model inventing or misremembering a
    citation; it is never silently corrected, only flagged.
    """
    found_rule_ids = set(RULE_ID_PATTERN.findall(response_text))
    found_source_ids = set(SOURCE_ID_PATTERN.findall(response_text))
    found_region_ids = set(REGION_ID_PATTERN.findall(response_text))

    # "POL-SR-xx" also matches the generic rule_id shape; only flag it as
    # an invented RULE id if it isn't a legitimate source_id either.
    invented_rule_ids = sorted(
        rid for rid in found_rule_ids
        if rid not in context.allowed_rule_ids and rid not in context.allowed_source_ids
    )
    invented_source_ids = sorted(sid for sid in found_source_ids if sid not in context.allowed_source_ids)
    invented_region_ids = sorted(rid for rid in found_region_ids if rid not in context.allowed_region_ids)

    return CitationCheck(
        invented_rule_ids=invented_rule_ids,
        invented_source_ids=invented_source_ids,
        invented_region_ids=invented_region_ids,
    )
