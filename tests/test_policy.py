import pandas as pd

from app.policy.evidence import EvidenceError, get_sources, load_policy_register
from app.policy.engine import evaluate, recommendations_to_frame, unlinked_signals_to_frame, static_rule_status_summary
from app.policy.rules import RULES


def _scored(**domains):
    base = {"INC": 10, "CAP": 10, "SOC": 10, "DIG": 10}
    base.update(domains)
    base["region_code"] = "A"
    base["confidence"] = 90.0
    return pd.DataFrame([base])


def test_policy_register_loads():
    register = load_policy_register()
    assert len(register) > 0


def test_get_sources_rejects_invented_id():
    register = load_policy_register()
    try:
        get_sources(["POL-SR-99-FAKE"], register)
        assert False, "should have raised"
    except EvidenceError:
        pass


def test_every_rule_source_id_exists_in_register():
    register = load_policy_register()
    for rule in RULES:
        if rule.required_policy_source_ids:
            get_sources(list(rule.required_policy_source_ids), register)  # raises if invented


def test_cap_sr_02_never_produces_any_outcome_data_unavailable():
    scored = _scored(CAP=100, INC=100, SOC=100, DIG=100)
    result = evaluate(scored)
    all_ids = set(recommendations_to_frame(result).get("rule_id", [])) | set(
        unlinked_signals_to_frame(result).get("rule_id", [])
    )
    assert "CAP-SR-02" not in all_ids


def test_dig_sr_01_is_unlinked_policy_signal_not_recommendation():
    scored = _scored(DIG=90)
    result = evaluate(scored)
    rec_df = recommendations_to_frame(result)
    sig_df = unlinked_signals_to_frame(result)
    assert "DIG-SR-01" not in set(rec_df.get("rule_id", []))
    row = sig_df[sig_df["rule_id"] == "DIG-SR-01"]
    assert len(row) == 1
    assert row.iloc[0]["rule_status"] == "unlinked_policy"


def test_cap_sr_01_is_candidate_not_verified():
    scored = _scored(CAP=90)
    result = evaluate(scored)
    row = recommendations_to_frame(result)
    row = row[row["rule_id"] == "CAP-SR-01"]
    assert len(row) == 1
    assert row.iloc[0]["rule_status"] == "candidate"
    assert row.iloc[0]["source_verified"] == False
    assert row.iloc[0]["policy_effective_status"] == "effective"  # POL-SR-02 registry status is 'official'


def test_inc_sr_02_is_pending_policy_verification_not_ineffective():
    scored = _scored(INC=90)
    result = evaluate(scored)
    row = recommendations_to_frame(result)
    row = row[row["rule_id"] == "INC-SR-02"]
    assert len(row) == 1
    assert row.iloc[0]["rule_status"] == "pending_policy_verification"
    # must NOT be collapsed into "not effective" / False -- it's its own state
    assert row.iloc[0]["policy_effective_status"] == "pending_verification"


def test_prot_sr_01_fires_for_every_region_as_candidate():
    scored = pd.concat([_scored(), _scored()], ignore_index=True)
    scored["region_code"] = ["A", "B"]
    result = evaluate(scored)
    rec_df = recommendations_to_frame(result)
    prot_rows = rec_df[rec_df["rule_id"] == "PROT-SR-01"]
    assert set(prot_rows["region_code"]) == {"A", "B"}
    assert (prot_rows["rule_status"] == "candidate").all()
    assert (prot_rows["policy_effective_status"] == "effective").all()  # both POL-SR-03/06 are 'official'


def test_recommendation_exposes_all_required_fields():
    scored = _scored(CAP=90)
    result = evaluate(scored)
    row = recommendations_to_frame(result)
    row = row[row["rule_id"] == "CAP-SR-01"].iloc[0]
    for field in [
        "rule_id", "rule_status", "source_id", "source_verified",
        "policy_effective_status", "data_available", "data_confidence", "reason",
    ]:
        assert field in row.index


def test_granular_indicator_preferred_over_aggregated_domain():
    scored = _scored(INC=30)  # aggregated domain below threshold
    raw = pd.DataFrame({"region_code": ["A"], "INC_SCHOOL_COVERAGE": [95]})  # granular above threshold
    result = evaluate(scored, raw_indicators=raw)
    row = recommendations_to_frame(result)
    row = row[row["rule_id"] == "INC-SR-01"]
    assert len(row) == 1
    assert row.iloc[0]["trigger_source"] == "granular indicator"


def test_static_rule_status_summary_covers_all_rules():
    summary = static_rule_status_summary()
    assert set(summary["rule_id"]) == {r.rule_id for r in RULES}
    cap02 = summary[summary["rule_id"] == "CAP-SR-02"].iloc[0]
    assert cap02["rule_status"] == "data_unavailable"
    dig01 = summary[summary["rule_id"] == "DIG-SR-01"].iloc[0]
    assert dig01["rule_status"] == "unlinked_policy"
