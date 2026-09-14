import pandas as pd

from app.copilot.grounding import build_grounded_context, validate_citations, RULE_STATUS_GLOSS
import app.copilot.service as service


def _scored_df():
    return pd.DataFrame({"region_code": ["11.01", "11.02"], "SR_ICSS": [72.0, 30.0],
                          "score_mode": ["full", "full"], "confidence": [90.0, 85.0]})


def _recommendations_df():
    return pd.DataFrame({
        "region_code": ["11.01"], "rule_id": ["CAP-SR-01"], "source_id": ["POL-SR-02"],
        "rule_status": ["candidate"], "policy_effective_status": ["effective"],
        "data_confidence": [90.0], "intervention": ["Targeted competency development"],
        "capacity_need": ["Inclusive pedagogy"], "reason": ["candidate mapping, unverified"],
    })


def _signals_df():
    return pd.DataFrame({
        "region_code": ["11.02"], "rule_id": ["DIG-SR-01"], "trigger_value": [90.0],
        "reason": ["no policy source registered"],
    })


def test_grounded_context_includes_honest_status_wording():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    assert "KANDIDAT" in ctx.text
    assert "BELUM diverifikasi" in ctx.text
    assert "SINYAL DATA SAJA" in ctx.text


def test_grounded_context_collects_allowed_ids():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    assert ctx.allowed_rule_ids == {"CAP-SR-01", "DIG-SR-01"}
    assert ctx.allowed_source_ids == {"POL-SR-02"}
    assert "11.01" in ctx.allowed_region_ids and "11.02" in ctx.allowed_region_ids


def test_empty_recommendations_do_not_crash_and_say_so():
    ctx = build_grounded_context(_scored_df(), pd.DataFrame(), pd.DataFrame())
    assert "tidak ada rekomendasi" in ctx.text


def test_validate_citations_passes_clean_response():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Wilayah 11.01 punya rekomendasi CAP-SR-01 dari sumber POL-SR-02 (masih kandidat)."
    check = validate_citations(response, ctx)
    assert not check.has_invented_citations


def test_validate_citations_flags_invented_rule_id():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Ada juga rekomendasi INC-SR-99 yang relevan."
    check = validate_citations(response, ctx)
    assert "INC-SR-99" in check.invented_rule_ids


def test_validate_citations_flags_invented_source_id():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Ini didukung oleh POL-SR-99 yang sangat kuat."
    check = validate_citations(response, ctx)
    assert "POL-SR-99" in check.invented_source_ids


def test_validate_citations_flags_invented_region_id():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Wilayah 99.99 juga perlu perhatian."
    check = validate_citations(response, ctx)
    assert "99.99" in check.invented_region_ids


def test_ask_prepends_warning_when_model_hallucinates(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt: "Rekomendasi utama adalah INC-SR-99 dari POL-SR-99.")
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "PERINGATAN GROUNDING" in result
    assert "INC-SR-99" in result
    assert "POL-SR-99" in result


def test_ask_does_not_warn_on_clean_grounded_response(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt: "Rekomendasi CAP-SR-01 untuk wilayah 11.01 masih kandidat.")
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "PERINGATAN GROUNDING" not in result


def test_ask_passes_system_rules_into_prompt(monkeypatch):
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(service, "generate", fake_generate)
    service.ask("test", _scored_df(), _recommendations_df(), _signals_df())
    assert "Jangan mengarang" in captured["prompt"]
    assert "EVIDENCE:" in captured["prompt"]
