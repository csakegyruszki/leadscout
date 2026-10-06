"""One table for the failure class this audit hit three separate times.

Each time the shape was identical: the request failed, the provider reported `ok` with no
evidence, and downstream the absence became a finding about the company.

    gleif.py      every non-200 except 5xx returned `ok`
    github.py     401, 400, 500 and 503 all returned `ok` / "org not found"
    wikidata.py   an outage and a genuine no-match were both a bare None

Provider-specific unit tests did not catch it because each was written to the provider's
own idea of success. This is the contract, in one place, so a regression is visible as a
row rather than as a missing test:

    a completed measurement with an empty result   -> ok        (a real negative)
    a measurement that did not complete            -> degraded  (never evidence)

The expected value per row is the API's OWN contract, not a mechanical HTTP rule. GitHub's
`GET /orgs/{org}` answers 404 when the organisation does not exist, so that row is a real
negative; GLEIF has no such 404 semantics on a filter query, so a 400 there is a
measurement that did not happen even though the fault is in our request.
"""
from __future__ import annotations

import httpx
import pytest

from leadscout.providers import github, gleif

OK, DEGRADED = "ok", "degraded"

# provider, HTTP status, expected ProviderResult.status, why
MATRIX = [
    ("gleif", 200, OK, "a filter query that ran and matched nothing"),
    ("gleif", 400, DEGRADED, "GLEIF has no 400-means-absent semantics; the query did not run"),
    ("gleif", 401, DEGRADED, "authentication failure - nothing was measured"),
    ("gleif", 403, DEGRADED, "forbidden - nothing was measured"),
    ("gleif", 404, DEGRADED, "not a per-entity endpoint; a 404 here is not 'no such entity'"),
    ("gleif", 408, DEGRADED, "timeout"),
    ("gleif", 429, DEGRADED, "rate limited - the measurement was refused, not completed"),
    ("gleif", 500, DEGRADED, "server error"),
    ("gleif", 503, DEGRADED, "unavailable"),
    ("github", 404, OK, "GitHub's own contract: the organisation does not exist"),
    ("github", 400, DEGRADED, "bad request - the org lookup did not happen"),
    ("github", 401, DEGRADED, "bad credentials - the org lookup did not happen"),
    ("github", 403, DEGRADED, "rate limited / forbidden"),
    ("github", 429, DEGRADED, "rate limited"),
    ("github", 500, DEGRADED, "server error"),
    ("github", 503, DEGRADED, "unavailable"),
]


_REAL_CLIENT = httpx.Client   # captured once: a second patch must not wrap the first


def _patch_client(monkeypatch, module, handler):
    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_CLIENT(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", factory)


def _run(name: str, status: int, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if status == 200:
            body = {"data": []} if name == "gleif" else {"login": "someco"}
            return httpx.Response(200, json=body)
        return httpx.Response(status, json={"message": "no"})

    if name == "gleif":
        _patch_client(monkeypatch, gleif, handler)
        return gleif.run("SomeCo", "United States of America", "https://someco.example")
    _patch_client(monkeypatch, github, handler)
    return github.run("SomeCo", ["we use github.com/someco"], hrefs=["https://github.com/someco"])


@pytest.mark.parametrize("name,status,expected,why", MATRIX,
                         ids=[f"{n}-{s}" for n, s, _e, _w in MATRIX])
def test_provider_outcome_contract(name, status, expected, why, monkeypatch):
    result = _run(name, status, monkeypatch)
    assert result.status == expected, f"{name} HTTP {status}: {why} (got {result.status!r})"
    if expected == DEGRADED:
        assert result.evidence == [], "a failed measurement must never yield evidence"


@pytest.mark.parametrize("name", ["gleif", "github"])
def test_a_transport_failure_is_degraded(name, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    module = gleif if name == "gleif" else github
    _patch_client(monkeypatch, module, handler)
    result = _run_transport(name)
    assert result.status == DEGRADED
    assert result.evidence == []


def _run_transport(name: str):
    if name == "gleif":
        return gleif.run("SomeCo", "United States of America", "https://someco.example")
    return github.run("SomeCo", ["we use github.com/someco"], hrefs=["https://github.com/someco"])


@pytest.mark.parametrize("name", ["gleif", "github"])
def test_a_malformed_body_is_degraded_not_an_empty_result(name, monkeypatch):
    """Valid HTTP, unusable content: the measurement did not produce a reading."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json at all</html>",
                              headers={"content-type": "text/html"})

    module = gleif if name == "gleif" else github
    _patch_client(monkeypatch, module, handler)
    result = _run_transport(name)
    assert result.status == DEGRADED, f"{name} reported {result.status!r} on an unparseable body"
    assert result.evidence == []


def test_wikidata_separates_an_outage_from_a_genuine_no_match(monkeypatch):
    """Wikidata has no ProviderResult, so its contract is the three-value outcome."""
    from leadscout.providers import wikidata

    def searched_and_found_nothing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"search": []})

    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    _patch_client(monkeypatch, wikidata, searched_and_found_nothing)
    facts, outcome = wikidata.lookup_by_website("SomeCo", "https://someco.example")
    assert facts is None and outcome == wikidata.NOT_FOUND

    _patch_client(monkeypatch, wikidata, unavailable)
    facts, outcome = wikidata.lookup_by_website("SomeCo", "https://someco.example")
    assert facts is None and outcome == wikidata.DEGRADED
