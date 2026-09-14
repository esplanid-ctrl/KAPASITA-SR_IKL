"""
Policy Rules definitions (docs/06_POLICY_RULES_AND_EVIDENCE.md).

Every rule is expressed so the engine (app/policy/engine.py) can produce
the full traceability chain required by MASTER_PROMPT.md rule #7 and the
evidence taxonomy decided by the project owner:
    rule -> trigger -> indicator evidence -> policy evidence
         -> rule_status / policy_effective_status -> recommendation -> capacity need

No rule may generate a location/construction/operational recommendation
for Sekolah Rakyat (MASTER_PROMPT.md non-negotiable scope #1-#4).

REVIEW DECISION (project owner): all rule -> source_id mappings below are
CANDIDATE mappings, not verified mappings. `source_verified` defaults to
False for every rule and must only be flipped to True once a human has
checked the cited source's actual section/page against the rule's claim.
Until then, no rule may be presented as "policy-grounded" -- the engine
enforces this by never assigning rule_status="verified" while
source_verified is False.

Per-rule notes on data/evidence availability:
  - CAP-SR-02: data_available=False. data/DATA_MAPPING_REAL.csv has NO
    indicator_id for `special_needs_signal` (docs/04 names the field, but
    it was never mapped to a real source). No substitute indicator is
    invented. This rule stays inactive with reason="data_unavailable"
    until the project owner adds a mapped source for it.
  - DIG-SR-01: required_policy_source_ids left EMPTY on purpose. No source
    in POLICY_SOURCE_REGISTER.csv is scoped to digital-learning
    intervention. Per project-owner decision, no regulation is added
    merely to force this rule to trigger -- it stays unlinked_policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

HIGH_NEED_THRESHOLD = 60.0  # matches docs/04 "high" interpretation band (60-79); MVP band, not statutory


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    domain: str
    trigger_description: str
    trigger_type: str  # "component_threshold" | "evidence_presence"
    trigger_component: Optional[str] = None   # aggregated column in the scored df, e.g. "INC"
    trigger_indicator: Optional[str] = None   # granular pre-aggregation indicator_id, if distinct
    required_policy_source_ids: List[str] = ()
    intervention: str = ""
    capacity_need: str = ""
    data_available: bool = True
    source_verified: bool = False  # CANDIDATE by default -- see module docstring
    note: Optional[str] = None


RULES: List[PolicyRule] = [
    PolicyRule(
        rule_id="INC-SR-01",
        domain="INC",
        trigger_description="High inclusive-service gap",
        trigger_type="component_threshold",
        trigger_component="INC",
        trigger_indicator="INC_SCHOOL_COVERAGE",
        required_policy_source_ids=["POL-SR-01"],
        intervention="Strengthen inclusive service planning",
        capacity_need="Competency in inclusive education planning",
        note="Candidate mapping to POL-SR-01 (general SR challenge evidence); section/page not yet verified.",
    ),
    PolicyRule(
        rule_id="INC-SR-02",
        domain="INC",
        trigger_description="High inclusive infrastructure/support gap",
        trigger_type="component_threshold",
        trigger_component="INC",
        trigger_indicator="INC_INFRA_COVERAGE",
        required_policy_source_ids=["POL-SR-04"],
        intervention="Strengthen accessibility/service support planning",
        capacity_need="Needs assessment and service design",
        note=(
            "Candidate mapping to POL-SR-04 (Permendikbudristek 48/2023). "
            "Registry status for POL-SR-04 is 'verify' -- legal effectiveness "
            "is not yet confirmed. This does NOT mean the policy is "
            "ineffective; it means registry verification is pending."
        ),
    ),
    PolicyRule(
        rule_id="CAP-SR-01",
        domain="CAP",
        trigger_description="High teacher equity/capacity gap",
        trigger_type="component_threshold",
        trigger_component="CAP",
        trigger_indicator="CAP_TEACHER_EQUITY",
        required_policy_source_ids=["POL-SR-02"],
        intervention="Targeted competency development",
        capacity_need="Inclusive pedagogy / differentiated learning",
        note="Candidate mapping to POL-SR-02 (teacher readiness/selection); section/page not yet verified.",
    ),
    PolicyRule(
        rule_id="CAP-SR-02",
        domain="CAP",
        trigger_description="Heterogeneous learner need combined with capacity gap",
        trigger_type="component_threshold",
        trigger_component="CAP",
        trigger_indicator="special_needs_signal",
        required_policy_source_ids=["POL-SR-02"],
        intervention="Coaching / learning support",
        capacity_need="Learner assessment and classroom adaptation",
        data_available=False,
        note=(
            "The required indicator ('special_needs_signal') is not currently "
            "available: no indicator_id for it exists in data/DATA_MAPPING_REAL.csv. "
            "This rule is defined per docs/06 but stays inactive "
            "(rule_status='data_unavailable') until a real, mapped source is added. "
            "No substitute indicator is used in its place."
        ),
    ),
    PolicyRule(
        rule_id="PROT-SR-01",
        domain="POL",
        trigger_description="Relevant child-protection evidence is in effect",
        trigger_type="evidence_presence",
        required_policy_source_ids=["POL-SR-03", "POL-SR-06"],
        intervention="Strengthen safeguarding capability",
        capacity_need="Prevention / referral / child protection",
        note=(
            "Evidence-driven, not indicator-threshold-driven: recommends "
            "baseline safeguarding capacity wherever this evidence is cited, "
            "independent of any region's numeric score. Candidate mapping; "
            "section/page not yet verified."
        ),
    ),
    PolicyRule(
        rule_id="DIG-SR-01",
        domain="DIG",
        trigger_description="High digital support gap",
        trigger_type="component_threshold",
        trigger_component="DIG",
        trigger_indicator="DIG_INTERNET",
        required_policy_source_ids=[],
        intervention="Targeted digital learning support",
        capacity_need="Technology-enabled learning",
        note=(
            "No source in POLICY_SOURCE_REGISTER.csv is scoped to digital "
            "learning. Kept unlinked_policy until a verified policy source "
            "is registered -- no regulation is added merely to force this "
            "rule to trigger."
        ),
    ),
]

