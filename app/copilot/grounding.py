"""
Copilot grounding & citation validation (Stage 5, hardened per review).

Per project-owner decision: "The LLM must not compensate for missing or
unverified evidence." This module builds a structured, ID-labeled context
from ALREADY-VALIDATED Gate 1 / policy-engine output (never raw
indicators, never a free-text dataframe dump).

IMPORTANT SCOPE NOTE (do not redesign this architecture further without
re-reading this note): deterministic Python remains the ONLY source of
numerical truth. This module adds exactly two guard rails around the
existing Gemini call -- it does not add any new analytics, aggregation,
or inference of its own:

  1. CITATION VALIDATION -- an IDENTIFIER EXISTENCE CHECK, NOT a semantic
     verification. `validate_citations()` only checks whether a
     rule_id/source_id/region_id-shaped token the model wrote matches a
     real, registered ID -- exact string lookup against
     app.policy.rules.RULES and app.policy.evidence.load_policy_register()
     (the canonical registries), plus the specific evidence the model was
     actually given for this query. It says NOTHING about whether the
     model's surrounding claim about that ID is true, complete, or
     correctly characterizes the ID's status. Do not read a "no invented
     IDs" result as "the answer is correct" -- it only means every ID
     token used is a real, registered one.
  2. UNSUPPORTED-COMPUTATION GUARD -- `detect_unsupported_computation()`
     is a conservative keyword heuristic that intercepts questions asking
     for cross-region aggregation/comparison/statistics (averages,
     rankings, correlations, trends, "nationwide", etc.) that Python has
     not precomputed anywhere in this codebase. When it matches, `ask()`
     returns a fixed "not available" response WITHOUT calling Gemini at
     all for that turn -- the LLM is never asked to infer or calculate a
     number Python didn't produce. This is intentionally conservative
     (it will over-block some legitimate simple questions that happen to
     contain a matched keyword) because under-blocking risks a fabricated
     number; it is a keyword heuristic, not real intent understanding,
     and should be reviewed/tuned by a human rather than "improved" by
     making it smarter/more autonomous.

Deterministic Python owns truth here, exactly as it does for scoring: the
LLM is only ever allowed to talk about what it was given, never to invent
or strengthen a fact, and never to compute something Python didn't.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

RULE_ID_PATTERN = re.compile(r"\b([A-Z]{2,6}-SR-\d{2})\b")
SOURCE_ID_PATTERN = re.compile(r"\b(POL-SR-\d{2})\b")
REGION_ID_PATTERN = re.compile(r"\b(\d{2}\.\d{2})\b")

# Conservative, bilingual keyword heuristic for "this needs computation
# Python hasn't done" -- see UNSUPPORTED-COMPUTATION GUARD note above.
# MVP assumption: a fixed keyword list, not intent detection. Extend this
# list only when a real gap is found; do not replace it with an
# LLM-based intent classifier without express approval (that would be new
# autonomous inference, not a guard rail).
UNSUPPORTED_COMPUTATION_KEYWORDS = [
    "rata-rata", "rata2", "rerata", "average", "mean",
    "total seluruh", "jumlah seluruh", "jumlah semua",
    "korelasi", "correlation", "tren", "trend", "pertumbuhan", "growth",
    "proyeksi", "forecast", "prediksi", "predict",
    "provinsi mana", "wilayah mana yang paling", "daerah mana yang paling",
    "paling tinggi di seluruh", "paling rendah di seluruh",
    "rank ", "ranking", "urutkan", "peringkat semua",
    "bandingkan semua", "bandingkan antar provinsi", "compare all",
    "across all", "across provinces", "nationwide", "se-indonesia", "seluruh indonesia",
]

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


def detect_unsupported_computation(question: str) -> Optional[str]:
    """Returns a human-readable reason string if `question` appears to ask
    for cross-region aggregation/comparison/statistics that Python has not
    precomputed anywhere (no province-level rollups, correlations, trends,
    or rankings exist in this codebase). Returns None otherwise.

    This is a conservative keyword heuristic (see module docstring) -- it
    is deliberately biased toward over-blocking rather than letting
    Gemini attempt the computation itself.
    """
    q = question.lower()
    for kw in UNSUPPORTED_COMPUTATION_KEYWORDS:
        if kw in q:
            return (
                f"Pertanyaan tampaknya meminta analisis/komputasi lintas-wilayah "
                f"(terdeteksi frasa '{kw}') yang BELUM dihitung oleh Python di sistem ini "
                "(tidak ada agregasi provinsi, korelasi, tren, atau ranking yang tersedia)."
            )
    return None


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
    """Result of an IDENTIFIER EXISTENCE CHECK -- not a semantic/factual
    verification of anything the model said about these IDs.

    `unknown_to_system_*`: the ID-shaped token doesn't match ANY real,
    registered ID anywhere in the system (app.policy.rules.RULES /
    POLICY_SOURCE_REGISTER.csv / the region master given, if any) -- this
    is the strongest signal of outright fabrication.

    `out_of_context_*`: the ID IS a real, registered ID somewhere in the
    system, but was NOT part of the specific evidence given to the model
    for THIS query -- the model still should not have cited it (it had no
    basis to), but this is a weaker signal than outright invention.
    """
    unknown_to_system_rule_ids: List[str] = field(default_factory=list)
    unknown_to_system_source_ids: List[str] = field(default_factory=list)
    unknown_to_system_region_ids: List[str] = field(default_factory=list)
    out_of_context_rule_ids: List[str] = field(default_factory=list)
    out_of_context_source_ids: List[str] = field(default_factory=list)
    out_of_context_region_ids: List[str] = field(default_factory=list)
    check_failed: bool = False
    check_failure_reason: Optional[str] = None

    @property
    def has_invented_citations(self) -> bool:
        return bool(
            self.unknown_to_system_rule_ids or self.unknown_to_system_source_ids
            or self.unknown_to_system_region_ids
        )

    @property
    def has_out_of_context_citations(self) -> bool:
        return bool(
            self.out_of_context_rule_ids or self.out_of_context_source_ids
            or self.out_of_context_region_ids
        )

    # Backward-compatible aliases (Stage 5 field names) -- kept so any
    # existing caller reading .invented_rule_ids etc. keeps working.
    @property
    def invented_rule_ids(self) -> List[str]:
        return self.unknown_to_system_rule_ids

    @property
    def invented_source_ids(self) -> List[str]:
        return self.unknown_to_system_source_ids

    @property
    def invented_region_ids(self) -> List[str]:
        return self.unknown_to_system_region_ids


def _canonical_rule_ids() -> Set[str]:
    """Exact registered rule IDs -- the canonical truth, independent of
    what was actually triggered/shown for a given query."""
    from app.policy.rules import RULES  # local import: avoid a hard
    # dependency from app.copilot on app.policy at module-import time for
    # environments that only need one of the two.
    return {r.rule_id for r in RULES}


def _canonical_source_ids() -> Set[str]:
    """Exact registered policy source IDs from POLICY_SOURCE_REGISTER.csv
    -- the canonical truth, independent of what was cited for a given
    query. Returns an empty set (fail-safe: treat as "cannot confirm
    anything is known") if the registry can't be loaded, rather than
    raising and taking down the whole citation check."""
    try:
        from app.policy.evidence import load_policy_register
        return set(load_policy_register()["source_id"].astype(str))
    except Exception:
        return set()


def validate_citations(
    response_text: str,
    context: GroundedContext,
    region_master_ids: Optional[Set[str]] = None,
) -> CitationCheck:
    """IDENTIFIER EXISTENCE CHECK (not semantic verification -- see
    CitationCheck docstring). Scans `response_text` for rule_id/source_id/
    region_id-shaped tokens and classifies each one against:
      1. the canonical registries (app.policy.rules.RULES,
         POLICY_SOURCE_REGISTER.csv, and `region_master_ids` if supplied)
      2. the specific evidence actually given for this query (`context`)

    Fail-safe: if anything here raises unexpectedly (malformed input,
    registry load failure, etc.), this returns a CitationCheck with
    check_failed=True rather than propagating the exception -- callers
    must treat check_failed as "citation safety could not be confirmed"
    and warn accordingly, never as "no problem found".
    """
    try:
        response_text = response_text or ""
        found_rule_ids = set(RULE_ID_PATTERN.findall(response_text))
        found_source_ids = set(SOURCE_ID_PATTERN.findall(response_text))
        found_region_ids = set(REGION_ID_PATTERN.findall(response_text))

        canonical_rule_ids = _canonical_rule_ids()
        canonical_source_ids = _canonical_source_ids()

        # "POL-SR-xx" also matches the generic rule_id shape; only classify
        # it under rule_ids if it isn't a legitimate source_id anywhere.
        candidate_rule_ids = found_rule_ids - found_source_ids

        unknown_rule_ids = sorted(
            rid for rid in candidate_rule_ids
            if rid not in canonical_rule_ids and rid not in canonical_source_ids
        )
        out_of_context_rule_ids = sorted(
            rid for rid in candidate_rule_ids
            if rid in canonical_rule_ids and rid not in context.allowed_rule_ids
        )

        unknown_source_ids = sorted(sid for sid in found_source_ids if sid not in canonical_source_ids)
        out_of_context_source_ids = sorted(
            sid for sid in found_source_ids
            if sid in canonical_source_ids and sid not in context.allowed_source_ids
        )

        if region_master_ids is not None:
            unknown_region_ids = sorted(rid for rid in found_region_ids if rid not in region_master_ids)
            out_of_context_region_ids = sorted(
                rid for rid in found_region_ids
                if rid in region_master_ids and rid not in context.allowed_region_ids
            )
        else:
            # No canonical region master supplied to the check -- fall back
            # to the weaker "not in this query's context" signal only,
            # reported as unknown_to_system since we cannot distinguish
            # further (fail-safe: treat unconfirmable as unknown, not as safe).
            unknown_region_ids = sorted(rid for rid in found_region_ids if rid not in context.allowed_region_ids)
            out_of_context_region_ids = []

        return CitationCheck(
            unknown_to_system_rule_ids=unknown_rule_ids,
            unknown_to_system_source_ids=unknown_source_ids,
            unknown_to_system_region_ids=unknown_region_ids,
            out_of_context_rule_ids=out_of_context_rule_ids,
            out_of_context_source_ids=out_of_context_source_ids,
            out_of_context_region_ids=out_of_context_region_ids,
        )
    except Exception as e:  # fail-safe: never let a broken check look like a clean one
        return CitationCheck(check_failed=True, check_failure_reason=str(e))
