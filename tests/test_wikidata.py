"""Wikidata website-matched entity lookup - offline, against recorded fixtures
(tests/fixtures/wikidata/*.json, fetched once from the real API on 2026-09-19).
"""
import json
from pathlib import Path

import httpx
import pytest

from leadscout.providers import wikidata

FIXTURES = Path(__file__).parent / "fixtures" / "wikidata"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _mock_transport():
    """Route wbsearchentities/EntityData/wbgetentities calls to the recorded fixtures
    by inspecting the request - no network."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "Special:EntityData/Q27150165" in url:
            return httpx.Response(200, json=_load("Q27150165.json"))
        if "Special:EntityData/Q109355620" in url:
            return httpx.Response(200, json=_load("Q109355620.json"))
        if "action=wbsearchentities" in url:
            if "Zapier" in url:
                return httpx.Response(200, json=_load("zapier_search.json"))
            if "Masterplast" in url:
                return httpx.Response(200, json=_load("masterplast_search.json"))
            return httpx.Response(200, json={"search": []})
        if "action=wbgetentities" in url:
            return httpx.Response(200, json=_load("labels.json"))
        return httpx.Response(404, json={"error": "unmapped fixture request"})

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    transport = _mock_transport()
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(wikidata.httpx, "Client", fake_client)


def test_zapier_matches_on_official_website_and_extracts_facts():
    facts, _outcome = wikidata.lookup_by_website("Zapier", "https://zapier.com")
    assert facts is not None
    assert facts.qid == "Q27150165"
    assert facts.country == "United States"
    assert facts.industries == ["cloud computing"]
    # Zapier's fixture has no P159/P1128/P749 claims - must not be invented.
    assert facts.headquarters is None
    assert facts.employees is None


def test_masterplast_matches_on_official_website_with_www_and_trailing_slash():
    # Wikidata's P856 for Masterplast is "https://www.masterplast.hu/" - the lead
    # submitted "masterplast.hu" with no scheme/www/slash; the host match must be
    # scheme/www/trailing-slash insensitive.
    facts, _outcome = wikidata.lookup_by_website("Masterplast", "masterplast.hu")
    assert facts is not None
    assert facts.qid == "Q109355620"
    assert facts.country == "Hungary"


def test_no_match_when_website_domain_differs():
    facts, _outcome = wikidata.lookup_by_website("Zapier", "https://totally-different-domain.example")
    assert facts is None


def test_registrable_host_normalises_scheme_www_and_trailing_slash():
    assert wikidata._registrable_host("https://www.masterplast.hu/") == "masterplast.hu"
    assert wikidata._registrable_host("masterplast.hu") == "masterplast.hu"
    assert wikidata._registrable_host("http://zapier.com") == "zapier.com"
    assert wikidata._registrable_host("") == ""


def test_latest_quantity_picks_the_most_recent_point_in_time():
    claims = [
        {"mainsnak": {"datavalue": {"value": {"amount": "+100"}}},
         "qualifiers": {"P585": [{"datavalue": {"value": {"time": "+2020-00-00T00:00:00Z"}}}]}},
        {"mainsnak": {"datavalue": {"value": {"amount": "+500"}}},
         "qualifiers": {"P585": [{"datavalue": {"value": {"time": "+2023-00-00T00:00:00Z"}}}]}},
    ]
    amount, date = wikidata._latest_quantity(claims)
    assert amount == "500"
    assert date == "+2023-00-00T00:00:00Z"


def test_latest_quantity_none_when_no_claims():
    assert wikidata._latest_quantity([]) is None


def test_lookup_never_raises_on_network_error(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(wikidata.httpx, "Client", boom)
    assert wikidata.lookup_by_website("Anything", "https://example.com") == (None, wikidata.DEGRADED)
