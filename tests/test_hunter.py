"""Hunter.io contact-quality provider: offline via httpx.MockTransport.

Covers the classification rules (valid/risky/invalid/unverified), the
no-key/skip and error/degrade paths, that the API key never leaks into a
recorded url/snippet/note, and that the one WEAK/corporate_identity Evidence
item Hunter can produce never changes fit score, cloud-usage state, or
compliance status (the author's "informational only" requirement).
"""
from __future__ import annotations

import httpx

from leadscout import compliance
from leadscout.cloud import assess_cloud_usage
from leadscout.fit import score_fit
from leadscout.models import ComplianceVerdict, Evidence, Lead, Research
from leadscout.providers import hunter
from leadscout.sanctions import SanctionsScreen


class _FakeSettingsWithKey:
    hunter_api_key = "test-hunter-key-abc123"
    http_timeout = 5.0


class _FakeSettingsNoKey:
    hunter_api_key = ""
    http_timeout = 5.0


def _patch(monkeypatch, handler, *, has_key=True):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(hunter.httpx, "Client", fake_client)
    monkeypatch.setattr(hunter, "settings", _FakeSettingsWithKey() if has_key else _FakeSettingsNoKey())


def _handler(verify_json, domain_json=None, verify_status=200, domain_status=200):
    domain_json = domain_json or {"data": {}}

    def handler(request):
        if "email-verifier" in str(request.url):
            return httpx.Response(verify_status, json=verify_json)
        return httpx.Response(domain_status, json=domain_json)
    return handler


def test_no_key_is_skipped_zero_calls(monkeypatch):
    monkeypatch.setattr(hunter, "settings", _FakeSettingsNoKey())
    result, facts = hunter.run("a@acme.com", "acme.com")
    assert result.status == "skipped"
    assert result.calls == 0
    assert result.evidence == []
    assert facts.contact_quality == "unverified"


def test_valid_email_is_valid_quality(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "valid", "score": 97, "mx_records": True}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert result.status == "ok"
    assert facts.contact_quality == "valid"
    assert "valid" in facts.contact_quality_note
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.strength == "WEAK"
    assert ev.family == "corporate_identity"
    assert ev.provider is None


def test_accept_all_low_score_is_risky(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "accept_all", "score": 50, "mx_records": True}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert facts.contact_quality == "risky"
    assert "accept_all" in facts.contact_quality_note
    assert "50" in facts.contact_quality_note


def test_accept_all_high_score_is_valid(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "accept_all", "score": 90, "mx_records": True}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert facts.contact_quality == "valid"


def test_disposable_is_invalid(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "disposable", "score": 10, "mx_records": False}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert facts.contact_quality == "invalid"


def test_missing_score_on_ambiguous_status_is_risky_not_valid(monkeypatch):
    """A missing score must not collapse the ambiguous bucket into "valid"."""
    _patch(monkeypatch, _handler({"data": {"status": "unknown", "score": None, "mx_records": None}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert facts.contact_quality == "risky"


def test_http_429_degrades_and_unverified(monkeypatch):
    _patch(monkeypatch, _handler({"data": {}}, verify_status=429))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert result.status == "degraded"
    assert result.evidence == []
    assert facts.contact_quality == "unverified"


def test_network_error_degrades_with_no_evidence(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(hunter, "settings", _FakeSettingsWithKey())
    monkeypatch.setattr(hunter.httpx, "Client", boom)
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert result.status == "degraded"
    assert result.evidence == []
    assert facts.contact_quality == "unverified"


def test_api_key_never_appears_in_recorded_url_snippet_or_note(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "valid", "score": 88, "mx_records": True}},
                                  {"data": {"organization": "Acme", "country": "US"}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    key = _FakeSettingsWithKey.hunter_api_key
    ev = result.evidence[0]
    assert key not in ev.url
    assert key not in ev.snippet
    assert key not in result.note
    assert key not in facts.contact_quality_note

    # Same check on a degraded/error path, where the URL is embedded in the note.
    _patch(monkeypatch, _handler({"data": {}}, verify_status=503))
    result2, facts2 = hunter.run("dana@acme.com", "acme.com")
    assert key not in result2.note
    assert key not in facts2.contact_quality_note


def test_api_key_never_in_recorded_snapshot_bytes(monkeypatch):
    """Asserts on the actual bytes handed to provtrail's record(), not just the
    Evidence object - a leak could otherwise still land in the snapshot file."""
    recorded: list[bytes] = []

    class _FakeRun:
        run_id = "test-run"

        def next_evidence_id(self):
            self._n = getattr(self, "_n", 0) + 1
            return f"ev-hunter-{self._n}"

        def record(self, ev, raw, *, stage=""):
            recorded.append(raw)

    _patch(monkeypatch, _handler({"data": {"status": "valid", "score": 88, "mx_records": True}},
                                  {"data": {"organization": "Acme", "country": "US"}}))
    hunter.run("dana@acme.com", "acme.com", run=_FakeRun())
    key = _FakeSettingsWithKey.hunter_api_key
    assert recorded, "expected at least one recorded payload"
    for raw in recorded:
        assert key.encode() not in raw


def test_verifier_snapshot_redacts_contact_email(monkeypatch):
    recorded: list[bytes] = []

    class _FakeRun:
        run_id = "test-run"

        def next_evidence_id(self):
            self._n = getattr(self, "_n", 0) + 1
            return f"ev-hunter-{self._n}"

        def record(self, ev, raw, *, stage=""):
            recorded.append(raw)

    email = "dana.whitfield@acme.com"
    _patch(monkeypatch, _handler({"data": {"status": "valid", "score": 88, "email": email}},
                                  {"data": {"organization": "Acme"}}))
    hunter.run(email, "acme.com", run=_FakeRun())
    assert not any(email.encode() in raw for raw in recorded)
    assert any(b"<redacted>" in raw for raw in recorded)


def test_emits_at_most_one_evidence_item(monkeypatch):
    _patch(monkeypatch, _handler({"data": {"status": "valid", "score": 88, "mx_records": True}},
                                  {"data": {"organization": "Acme", "country": "US", "headcount": "51-200",
                                            "company_type": "private", "industry": "Software"}}))
    result, facts = hunter.run("dana@acme.com", "acme.com")
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert "Acme" in ev.snippet
    assert "US" in ev.snippet
    assert "51-200" in ev.snippet


# --- Byte-identical-outcomes: fit/cloud/compliance must not move --------------------

def _base_evidence() -> list[Evidence]:
    return [Evidence(
        id="ev-001", source_type="website", url="https://acme.example",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="STRONG", snippet="Acme builds developer tools.", snapshot_path="out/x.txt",
        family="corporate_identity",
    )]


def _hunter_evidence() -> Evidence:
    return Evidence(
        id="ev-hunter-1", source_type="hunter", url="https://api.hunter.io/v2/domain-search?domain=acme.example",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="b" * 64,
        strength="WEAK", snippet="org=Acme, country=US, headcount=51-200, type=private, industry=Software",
        snapshot_path="out/y.json", family="corporate_identity", provider=None,
    )


def test_hunter_evidence_never_changes_cloud_usage_state():
    without = assess_cloud_usage(_base_evidence())
    with_hunter = assess_cloud_usage(_base_evidence() + [_hunter_evidence()])
    assert without.state == with_hunter.state
    assert without.boundary == with_hunter.boundary


def test_hunter_evidence_never_changes_fit_score():
    lead = Lead("Ann", "ann@acme.example", "Acme", "https://acme.example")
    without_research = Research(
        summary="s", industry="Software", headquarters_country="United States", estimated_employees=120,
        evidence=_base_evidence(), cloud_usage=assess_cloud_usage(_base_evidence()))
    with_research = Research(
        summary="s", industry="Software", headquarters_country="United States", estimated_employees=120,
        evidence=_base_evidence() + [_hunter_evidence()],
        cloud_usage=assess_cloud_usage(_base_evidence() + [_hunter_evidence()]),
        contact_quality="risky", contact_quality_note="Hunter: accept_all, score 50, MX ok")

    fit_without = score_fit(lead, without_research)
    fit_with = score_fit(lead, with_research)
    assert fit_without.score == fit_with.score
    assert fit_without.confidence == fit_with.confidence
    assert fit_without.cloud_signal_points == fit_with.cloud_signal_points
    assert fit_without.complexity_points == fit_with.complexity_points
    assert fit_without.scale_points == fit_with.scale_points


def test_hunter_evidence_never_changes_compliance_status(monkeypatch):
    def _clear_verdict(system, user, schema, *, purpose="compliance"):
        return ComplianceVerdict(status="clear", flagged=False, matches=[], reasoning="looks fine")

    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                         lambda company, hq, run=None: SanctionsScreen(status="none", hits=[]))

    lead = Lead("Ann", "ann@acme.example", "Acme", "https://acme.example")
    without_research = Research(summary="s", headquarters_country="United States",
                                hq_source="gleif", evidence=_base_evidence())
    with_research = Research(
        summary="s", headquarters_country="United States", hq_source="gleif",
        evidence=_base_evidence() + [_hunter_evidence()],
        contact_quality="invalid", contact_quality_note="Hunter: disposable, score 5, no MX")

    result_without = compliance.screen(lead, without_research)
    result_with = compliance.screen(lead, with_research)
    assert result_without.status == result_with.status == "clear"
