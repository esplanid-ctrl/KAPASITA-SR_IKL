import pandas as pd

from app.copilot.grounding import (
    build_grounded_context,
    validate_citations,
    detect_unsupported_computation,
)
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


# --- grounding context ------------------------------------------------------

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


# --- citation validation: identifier existence check, hardened -------------

def test_validate_citations_passes_clean_response():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Wilayah 11.01 punya rekomendasi CAP-SR-01 dari sumber POL-SR-02 (masih kandidat)."
    check = validate_citations(response, ctx)
    assert not check.has_invented_citations
    assert not check.has_out_of_context_citations
    assert not check.check_failed


def test_validate_citations_flags_id_unknown_to_whole_system():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Ada juga rekomendasi INC-SR-99 yang relevan."
    check = validate_citations(response, ctx)
    assert "INC-SR-99" in check.unknown_to_system_rule_ids


def test_validate_citations_flags_unknown_source_id():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Ini didukung oleh POL-SR-99 yang sangat kuat."
    check = validate_citations(response, ctx)
    assert "POL-SR-99" in check.unknown_to_system_source_ids


def test_validate_citations_distinguishes_out_of_context_from_unknown_rule_id():
    # INC-SR-01 is a REAL registered rule (app.policy.rules.RULES) but was
    # not part of THIS query's evidence (only CAP-SR-01/DIG-SR-01 were).
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Perlu diperhatikan juga INC-SR-01 untuk wilayah lain."
    check = validate_citations(response, ctx)
    assert "INC-SR-01" not in check.unknown_to_system_rule_ids
    assert "INC-SR-01" in check.out_of_context_rule_ids


def test_validate_citations_distinguishes_out_of_context_from_unknown_source_id():
    # POL-SR-01 is real (in POLICY_SOURCE_REGISTER.csv) but wasn't cited
    # in this query's evidence (only POL-SR-02 was).
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    response = "Ada juga dasar dari POL-SR-01."
    check = validate_citations(response, ctx)
    assert "POL-SR-01" not in check.unknown_to_system_source_ids
    assert "POL-SR-01" in check.out_of_context_source_ids


def test_validate_citations_region_check_with_canonical_master():
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())
    # 12.01 is a "real" region per this fake canonical master but wasn't
    # part of THIS query's evidence (only 11.01/11.02 were).
    response = "Wilayah 12.01 dan 99.99 juga relevan."
    check = validate_citations(response, ctx, region_master_ids={"11.01", "11.02", "12.01"})
    assert "12.01" in check.out_of_context_region_ids
    assert "99.99" in check.unknown_to_system_region_ids


def test_validate_citations_is_fail_safe_on_internal_error(monkeypatch):
    ctx = build_grounded_context(_scored_df(), _recommendations_df(), _signals_df())

    def boom():
        raise RuntimeError("registry unreachable")

    import app.copilot.grounding as grounding
    monkeypatch.setattr(grounding, "_canonical_rule_ids", boom)
    check = grounding.validate_citations("some response", ctx)
    assert check.check_failed is True
    assert "registry unreachable" in check.check_failure_reason


# --- unsupported cross-region computation guard -----------------------------

def test_detect_unsupported_computation_flags_aggregation_keywords():
    assert detect_unsupported_computation("Berapa rata-rata SR_ICSS se-Indonesia?") is not None
    assert detect_unsupported_computation("Provinsi mana yang paling butuh dukungan?") is not None
    assert detect_unsupported_computation("Bagaimana tren 5 tahun ke depan?") is not None


def test_detect_unsupported_computation_allows_simple_region_question():
    assert detect_unsupported_computation("Apa rekomendasi untuk wilayah 11.01?") is None


def test_ask_short_circuits_before_calling_gemini_for_unsupported_computation(monkeypatch):
    called = {"generate": False}
    monkeypatch.setattr(service, "generate", lambda prompt: called.__setitem__("generate", True) or "should not run")
    result = service.ask("Berapa rata-rata semua wilayah?", _scored_df(), _recommendations_df(), _signals_df())
    assert called["generate"] is False
    assert "Belum bisa menjawab" in result


# --- ask(): graceful degradation + grounded warnings ------------------------

def test_ask_prepends_warning_when_model_hallucinates(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt: "Rekomendasi utama adalah INC-SR-99 dari POL-SR-99.")
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "PERINGATAN GROUNDING" in result
    assert "identifier existence" not in result.lower() or "BUKAN" in result  # labeled, not silent
    assert "INC-SR-99" in result and "POL-SR-99" in result


def test_ask_does_not_warn_on_clean_grounded_response(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt: "Rekomendasi CAP-SR-01 untuk wilayah 11.01 masih kandidat.")
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "PERINGATAN GROUNDING" not in result
    assert "CATATAN GROUNDING" not in result


def test_ask_degrades_gracefully_when_gemini_raises(monkeypatch):
    def boom(prompt):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(service, "generate", boom)
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "gagal menghubungi Gemini" in result


def test_ask_degrades_gracefully_on_empty_gemini_response(monkeypatch):
    monkeypatch.setattr(service, "generate", lambda prompt: "")
    result = service.ask("Apa rekomendasi utama?", _scored_df(), _recommendations_df(), _signals_df())
    assert "gagal menghubungi Gemini" in result


def test_ask_passes_system_rules_into_prompt(monkeypatch):
    captured = {}

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(service, "generate", fake_generate)
    service.ask("test wilayah 11.01", _scored_df(), _recommendations_df(), _signals_df())
    assert "Jangan mengarang" in captured["prompt"]
    assert "EVIDENCE:" in captured["prompt"]
