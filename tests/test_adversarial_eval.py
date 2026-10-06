"""Adversarial eval harness: research_from_text (no fetch, no
provider fan-out) + run_adversarial_eval's pass/fail logic - all offline, ask_model
and screen_sanctions monkeypatched, no real LLM/network calls in this test file.
The real, budgeted run against evals/adversarial_cases.json is a separate,
manually-invoked step (see the design notes), not part of the test suite.
"""
from leadscout import compliance, evals, research
from leadscout.models import ComplianceVerdict, Lead, ResearchFacts
from leadscout.sanctions import SanctionsScreen


def test_research_from_text_uses_only_the_given_text(monkeypatch):
    captured = {}

    def fake_ask_model(system, user, schema, purpose=""):
        captured["user"] = user
        return ResearchFacts(summary="A short summary.", industry="SaaS",
                              headquarters_country="Canada", confidence="high")

    monkeypatch.setattr(research, "ask_model", fake_ask_model)
    lead = Lead(name="Eval Contact", email="eval@example.test", company="Acme",
                website="https://acme.example")
    result = research.research_from_text(lead, "Acme is based in Toronto, Canada.")
    assert result.website_text == "Acme is based in Toronto, Canada."
    assert result.website_ok is True
    assert result.headquarters_country == "Canada"
    assert result.summary == "A short summary."
    assert len(result.evidence) == 1
    assert "Acme is based in Toronto, Canada." in captured["user"]


def test_hq_matches_unknown_is_flexible():
    assert evals._hq_matches("unknown", None) is True
    assert evals._hq_matches("unknown", "") is True
    assert evals._hq_matches("unknown", "unknown") is True
    assert evals._hq_matches("unknown", "Canada") is False
    assert evals._hq_matches("Canada", "canada") is True
    assert evals._hq_matches("Canada", "United States") is False


def _patch_pipeline(monkeypatch, *, summary, hq, status):
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(
                             summary=summary, industry="SaaS", headquarters_country=hq, confidence="high"))
    monkeypatch.setattr(compliance, "ask_model",
                         lambda system, user, schema, purpose="": ComplianceVerdict(status=status, flagged=False))
    # "none" - a screen that ran and found nothing - rather than "skipped". These cases
    # test what the MODEL does with adversarial text, so screening has to be out of the
    # way, and "skipped" no longer is: a screen that did not complete cannot clear a lead
    # (compliance safety net 5), so it would decide every case before the model spoke.
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq_country, run=None: SanctionsScreen(status="none", hits=[], note="test"))


def test_run_adversarial_eval_pass_and_fail(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, summary="A clean summary about Acme in Canada.", hq="Canada", status="clear")
    cases = [
        {
            "id": "case-pass", "category": "instruction_override", "company": "Acme",
            "website": "https://acme.example", "website_text": "Acme is in Canada.",
            "expected": {"compliance_status": "clear", "headquarters_country": "Canada",
                         "must_not_contain_in_summary": ["OVERRIDE_ACCEPTED"]},
        },
        {
            "id": "case-fail-leak", "category": "instruction_override", "company": "Acme",
            "website": "https://acme.example", "website_text": "Acme is in Canada.",
            "expected": {"compliance_status": "clear", "headquarters_country": "Canada",
                         "must_not_contain_in_summary": ["clean summary"]},
        },
    ]
    path = tmp_path / "adversarial_cases.json"
    import json
    path.write_text(json.dumps(cases), encoding="utf-8")

    result = evals.run_adversarial_eval(path)
    assert result["n_cases"] == 2
    by_id = {r["id"]: r for r in result["rows"]}
    assert by_id["case-pass"]["pass"] is True
    assert by_id["case-fail-leak"]["pass"] is False
    assert "clean summary" in by_id["case-fail-leak"]["leaked_strings"]
    assert result["n_passed"] == 1
    assert result["n_failed"] == 1
    assert result["failing_ids"] == ["case-fail-leak"]

    report = evals.render_adversarial_report(result)
    assert "case-pass" in report and "PASS" in report
    assert "case-fail-leak" in report and "FAIL" in report


def test_run_adversarial_eval_wrong_status_fails(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, summary="ok", hq="Canada", status="blocked")
    cases = [{
        "id": "case-status-mismatch", "category": "fake_policy", "company": "Acme",
        "website": "https://acme.example", "website_text": "Acme is in Canada.",
        "expected": {"compliance_status": "clear", "headquarters_country": "Canada",
                     "must_not_contain_in_summary": []},
    }]
    import json
    path = tmp_path / "adversarial_cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    result = evals.run_adversarial_eval(path)
    assert result["n_passed"] == 0
    assert result["rows"][0]["got_compliance_status"] == "blocked"
