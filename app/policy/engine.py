"""
Policy Rule Engine.

Evaluates the PolicyRule definitions in app/policy/rules.py against a
scored dataframe (output of app.data.pipeline.run_gate1 /
app.analytics.scoring.add_score) and the evidence registry
(app/policy/evidence.py).

Evidence taxonomy (project-owner decision) -- every RuleOutcome exposes:
    rule_id, rule_status, source_id(s), source_verified,
    policy_effective_status, data_available, data_confidence, reason

rule_status is one of:
    candidate                     -- cites real registered evidence, not yet
                                      human-verified against the source's
                                      section/page.
    verified                      -- a human has confirmed the citation
                                      (source_verified=True) and the cited
                                      source's registry status is resolved.
    rejected                      -- a human reviewed the candidate mapping
                                      and determined it does NOT apply
                                      (mechanism exists; no rule is marked
                                      this way yet).
    pending_policy_verification   -- any cited source's registry status is
                                      "verify" (registry verification
                                      pending) -- NEVER treated as
                                      "ineffective".
    data_unavailable              -- the rule's required indicator does not
                                      exist in the data contract at all.
    unlinked_policy                -- the rule has no registered policy
                                      source to cite.
    policy_inactive                -- reserved for when a cited source's
                                      registry status indicates it is no
                                      longer in force (not present in the
                                      current registry; mechanism exists
                                      for future data).

Only rule_status in {candidate, verified, pending_policy_verification} is
surfaced as a `recommendation`. {data_unavailable, unlinked_policy,
rejected, policy_inactive} are either never fired at all or surfaced only
as `unlinked_signals` -- never dressed up as a policy-grounded recommendation.

Nothing here determines Sekolah Rakyat locations, land, construction, or
operational scheduling (MASTER_PROMPT.md #1-#4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .evidence import EvidenceError, get_sources, load_policy_register
from .rules import RULES, HIGH_NEED_THRESHOLD, PolicyRule

VALID_RULE_STATUSES = {
    "candidate",
    "verified",
    "rejected",
    "pending_policy_verification",
    "data_unavailable",
    "unlinked_policy",
    "policy_inactive",
}


@dataclass
class RuleOutcome:
    region_code: Optional[str]
    rule_id: str
    domain: str
    trigger_description: str
    trigger_value: Optional[float]
    trigger_source: Optional[str]  # "granular indicator" | "aggregated component" | None
    intervention: str
    capacity_need: str
    data_available: bool
    source_ids: List[str] = field(default_factory=list)
    source_verified: bool = False
    policy_effective_status: str = "not_applicable"
    rule_status: str = "candidate"
    data_confidence: Optional[float] = None
    reason: str = ""


def _determine_rule_status(
    rule: PolicyRule, evidence_records: List[Dict]
) -> Tuple[str, str, str]:
    """Returns (rule_status, policy_effective_status, reason).

    Registry verification status ("official"/"secondary"/"verify") is kept
    strictly separate from `rule_status`: a "verify" registry status is
    reported as pending_policy_verification / policy_effective_status=
    "pending_verification", NEVER as the rule or the policy being
    "ineffective" (project-owner decision).
    """
    if not rule.data_available:
        reason = rule.note or "The indicator required to evaluate this rule is not currently available."
        return "data_unavailable", "not_applicable", reason

    if not rule.required_policy_source_ids:
        reason = rule.note or "No registered policy source currently supports this rule."
        return "unlinked_policy", "not_applicable", reason

    if not evidence_records:
        return "unlinked_policy", "not_applicable", "No resolvable policy source for this rule."

    statuses = [str(r.get("status", "")).strip().lower() for r in evidence_records]

    if any(s == "verify" for s in statuses):
        reason = (
            "One or more cited policy sources are marked 'verify' in the "
            "registry (registry verification pending). This does not mean "
            "the policy is legally ineffective -- only that verification is "
            "not yet complete."
        )
        if rule.note:
            reason = f"{rule.note} {reason}"
        return "pending_policy_verification", "pending_verification", reason

    if rule.source_verified:
        effective = "effective" if all(s == "official" for s in statuses) else "supporting_context"
        reason = rule.note or "Source mapping and citation have been manually verified."
        return "verified", effective, reason

    effective = "effective" if all(s == "official" for s in statuses) else "supporting_context"
    reason = (
        "Candidate mapping: cites real registered evidence but has not yet "
        "been manually verified against the source's section/page."
    )
    if rule.note:
        reason = f"{rule.note} {reason}"
    return "candidate", effective, reason


def _trigger_value(row, rule: PolicyRule, raw_indicators: Optional[pd.DataFrame], region_col: str):
    """Prefers the granular pre-aggregation indicator when actually
    available; falls back to the aggregated domain column from the scored
    dataframe otherwise, and reports explicitly which one was used."""
    if raw_indicators is not None and rule.trigger_indicator in raw_indicators.columns:
        match = raw_indicators.loc[raw_indicators[region_col] == row[region_col], rule.trigger_indicator]
        if not match.empty and pd.notna(match.iloc[0]):
            return float(match.iloc[0]), "granular indicator"
    if rule.trigger_component and rule.trigger_component in row.index and pd.notna(row[rule.trigger_component]):
        return float(row[rule.trigger_component]), "aggregated component"
    return None, None


def evaluate(
    scored_df: pd.DataFrame,
    raw_indicators: Optional[pd.DataFrame] = None,
    region_col: str = "region_code",
    register: Optional[pd.DataFrame] = None,
    threshold: float = HIGH_NEED_THRESHOLD,
) -> Dict[str, List[RuleOutcome]]:
    """Returns {"recommendations": [...], "unlinked_signals": [...]}.

    `recommendations` only ever contains rule_status in
    {candidate, verified, pending_policy_verification}: every one of them
    cites at least one real, registered policy source, satisfying
    "Recommendations require policy evidence" (AT-07) even though most are
    not yet human-verified.
    `unlinked_signals` contains rule_status == "unlinked_policy" outcomes:
    the numeric trigger fired but no policy evidence exists to cite.
    rule_status == "data_unavailable" never produces any outcome at all.
    """
    register = register if register is not None else load_policy_register()
    recommendations: List[RuleOutcome] = []
    unlinked_signals: List[RuleOutcome] = []

    for rule in RULES:
        try:
            evidence_records = get_sources(list(rule.required_policy_source_ids), register)
        except EvidenceError:
            continue  # a rule citing a source absent from the registry must never fire

        rule_status, policy_effective_status, reason = _determine_rule_status(rule, evidence_records)
        assert rule_status in VALID_RULE_STATUSES  # guard against typos in the taxonomy

        if rule_status == "data_unavailable":
            continue  # never surfaced as any kind of outcome, per project-owner decision

        source_ids = [r["source_id"] for r in evidence_records]

        if rule.trigger_type == "evidence_presence":
            if rule_status == "unlinked_policy":
                continue  # nothing to trigger on without evidence
            for _, row in scored_df.iterrows():
                data_confidence = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else None
                recommendations.append(
                    RuleOutcome(
                        region_code=row.get(region_col),
                        rule_id=rule.rule_id,
                        domain=rule.domain,
                        trigger_description=rule.trigger_description,
                        trigger_value=None,
                        trigger_source=None,
                        intervention=rule.intervention,
                        capacity_need=rule.capacity_need,
                        data_available=rule.data_available,
                        source_ids=source_ids,
                        source_verified=rule.source_verified,
                        policy_effective_status=policy_effective_status,
                        rule_status=rule_status,
                        data_confidence=data_confidence,
                        reason=reason,
                    )
                )
            continue

        # component_threshold rules
        for _, row in scored_df.iterrows():
            value, trigger_source = _trigger_value(row, rule, raw_indicators, region_col)
            if value is None or value < threshold:
                continue

            data_confidence = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else None
            row_reason = reason
            if trigger_source == "aggregated component":
                addendum = (
                    f"Triggered from the aggregated '{rule.trigger_component}' domain; "
                    f"the granular indicator '{rule.trigger_indicator}' was not separately available."
                )
                row_reason = f"{reason} {addendum}"

            outcome = RuleOutcome(
                region_code=row.get(region_col),
                rule_id=rule.rule_id,
                domain=rule.domain,
                trigger_description=rule.trigger_description,
                trigger_value=value,
                trigger_source=trigger_source,
                intervention=rule.intervention,
                capacity_need=rule.capacity_need,
                data_available=rule.data_available,
                source_ids=source_ids,
                source_verified=rule.source_verified,
                policy_effective_status=policy_effective_status,
                rule_status=rule_status,
                data_confidence=data_confidence,
                reason=row_reason,
            )
            if rule_status == "unlinked_policy":
                unlinked_signals.append(outcome)
            else:
                recommendations.append(outcome)

    return {"recommendations": recommendations, "unlinked_signals": unlinked_signals}


def static_rule_status_summary(register: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rule-level status independent of any scored data -- for a dashboard/
    audit panel to show e.g. "6 rules defined: 4 candidate, 1
    data_unavailable, 1 unlinked_policy" without needing a scoring run."""
    register = register if register is not None else load_policy_register()
    rows = []
    for rule in RULES:
        try:
            evidence_records = get_sources(list(rule.required_policy_source_ids), register)
        except EvidenceError:
            evidence_records = []
        rule_status, policy_effective_status, reason = _determine_rule_status(rule, evidence_records)
        rows.append(
            {
                "rule_id": rule.rule_id,
                "domain": rule.domain,
                "rule_status": rule_status,
                "policy_effective_status": policy_effective_status,
                "data_available": rule.data_available,
                "source_id": ", ".join(r["source_id"] for r in evidence_records) or "(none)",
                "source_verified": rule.source_verified,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _to_frame(outcomes: List[RuleOutcome]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        rows.append(
            {
                "region_code": o.region_code,
                "rule_id": o.rule_id,
                "domain": o.domain,
                "trigger": o.trigger_description,
                "trigger_value": o.trigger_value,
                "trigger_source": o.trigger_source,
                "source_id": ", ".join(o.source_ids) or "(none)",
                "source_verified": o.source_verified,
                "policy_effective_status": o.policy_effective_status,
                "rule_status": o.rule_status,
                "data_available": o.data_available,
                "data_confidence": o.data_confidence,
                "intervention": o.intervention,
                "capacity_need": o.capacity_need,
                "reason": o.reason,
            }
        )
    return pd.DataFrame(rows)


def recommendations_to_frame(result: Dict[str, List[RuleOutcome]]) -> pd.DataFrame:
    return _to_frame(result["recommendations"])


def unlinked_signals_to_frame(result: Dict[str, List[RuleOutcome]]) -> pd.DataFrame:
    return _to_frame(result["unlinked_signals"])
