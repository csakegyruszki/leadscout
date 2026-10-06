"""GLEIF provider: multi-factor acceptance (name similarity + country + ACTIVE status),
offline via httpx.MockTransport. Fixtures are the real `lei-records` responses recorded
2026-09-19 (raw/gleif__zapier.json, raw/gleif__snapp.json - not truncated, used as-is).
"""
import json
from pathlib import Path

import httpx

from leadscout.providers import gleif

FIXTURES = Path(__file__).parent / "fixtures" / "gleif"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(gleif.httpx, "Client", fake_client)


def test_zapier_hq_us_accepted_strong(monkeypatch):
    fixture = _load("lei_records_zapier.json")

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Zapier", "United States of America", "https://zapier.com")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.strength == "STRONG"
    assert ev.family == "corporate_identity"
    assert "ZAPIER, INC." in ev.snippet
    assert "registered_jurisdiction=US-DE" in ev.snippet


def test_snapp_hq_iran_both_candidates_rejected(monkeypatch):
    fixture = _load("lei_records_snapp.json")

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Snapp", "Iran", "https://snapp.ir")
    assert result.status == "ok"
    assert result.evidence == []


def test_zapier_hq_unknown_generic_tld_accepted_weak(monkeypatch):
    """REVIEW-6bA-verified.md #5: a generic gTLD with HQ unknown confirms no
    country at all, so this is identity-only WEAK, never MEDIUM."""
    fixture = _load("lei_records_zapier.json")

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Zapier", None, "https://zapier.com")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].strength == "WEAK"


def test_hq_unknown_matching_cctld_accepted_medium(monkeypatch):
    """A specific ccTLD that DOES match the legal address is a real (if weaker than
    HQ-confirmed) signal - MEDIUM, not WEAK."""
    fixture = _load("lei_records_zapier.json")

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Zapier", None, "https://zapier.us")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].strength == "MEDIUM"


def test_hq_unknown_conflicting_cctld_rejected(monkeypatch):
    """Snapp's legal candidates are NO/IN; a website that itself resolves to a
    conflicting ccTLD (not IN/NO) should still reject even with HQ unknown."""
    fixture = _load("lei_records_snapp.json")

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Snapp", None, "https://snapp.us")
    assert result.evidence == []


def test_international_legal_suffix_hungarian_public_company_accepted(monkeypatch):
    """Item 3: GLEIF's own registered name for a well-known Hungarian company can
    carry the FULL Hungarian legal-form wording ("Nyilvánosan működő
    Részvénytársaság", not the "Nyrt" abbreviation) - the pre-fix, English-only
    suffix list scored "MASTERPLAST Nyilvánosan működő Részvénytársaság" vs.
    "Masterplast" well below the 85 name-similarity threshold and rejected a real
    match. `compliance.normalise_name` (shared by the competitor prescreen and
    this provider) now strips it."""
    record = {
        "data": [{
            "attributes": {
                "lei": "HU0000000000000MASTP",
                "entity": {
                    "legalName": {"name": "MASTERPLAST Nyilvánosan működő Részvénytársaság"},
                    "legalAddress": {"country": "HU"},
                    "jurisdiction": "HU",
                    "status": "ACTIVE",
                },
            },
        }],
    }

    def handler(request):
        return httpx.Response(200, json=record)

    _patch_client(monkeypatch, handler)
    result = gleif.run("Masterplast", "Hungary", "https://www.masterplast.hu")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].strength == "STRONG"
    assert "MASTERPLAST" in result.evidence[0].snippet


def test_no_matching_records_zero_evidence(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"data": []})

    _patch_client(monkeypatch, handler)
    result = gleif.run("Some Unknown Co", "United States of America", "https://example.com")
    assert result.status == "ok"
    assert result.evidence == []


def test_server_error_degrades(monkeypatch):
    def handler(request):
        return httpx.Response(503, text="service unavailable")

    _patch_client(monkeypatch, handler)
    result = gleif.run("Zapier", "United States of America", "https://zapier.com")
    assert result.status == "degraded"
    assert result.evidence == []


def test_network_error_degrades_with_no_evidence(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(gleif.httpx, "Client", boom)
    result = gleif.run("Zapier", "United States of America", "https://zapier.com")
    assert result.status == "degraded"
    assert result.evidence == []


def test_malformed_json_response_degrades_with_no_evidence(monkeypatch):
    """A 200 with an invalid JSON body must not crash research_lead - the provider
    boundary catches Exception broadly, not only httpx.HTTPError
    (REVIEW-6bA-verified.md #2)."""
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, text="not json{{{")

    _patch_client(monkeypatch, handler)
    result = gleif.run("Zapier", "United States of America", "https://zapier.com")
    assert result.status == "degraded"
    assert result.evidence == []


def test_a_non_200_is_degraded_not_a_clean_negative(monkeypatch):
    """A lookup that did not happen must not be reportable as a lookup that found
    nothing. Before the fix only >=500 was `degraded`; 429/401/400 returned `ok` with
    no evidence, byte-identical downstream to a real empty result - and cloud.py fills
    `missing_channels` from `degraded` alone, so a rate-limited call let the run claim
    NO_PUBLIC_CLOUD_EVIDENCE instead of INSUFFICIENT_EVIDENCE."""
    for code in (400, 401, 403, 429, 500, 503):
        def handler(request, code=code):
            return httpx.Response(code, json={"errors": [{"title": "nope"}]})

        _patch_client(monkeypatch, handler)
        result = gleif.run("Zapier", "United States of America", "https://zapier.com")
        assert result.status == "degraded", f"HTTP {code} reported as {result.status!r}"
        assert result.evidence == []


def test_an_empty_200_is_still_ok(monkeypatch):
    """The other side of the same rule: a lookup that ran and found nothing is `ok`."""
    def handler(request):
        return httpx.Response(200, json={"data": []})

    _patch_client(monkeypatch, handler)
    result = gleif.run("Nonexistent Co", "Hungary", "https://nonexistent.example")
    assert result.status == "ok"
    assert result.evidence == []
