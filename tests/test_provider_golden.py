"""Golden output test for item 1 (provider `_base.py` extraction, refactor-only):
captures each provider's full `ProviderResult` (serialised via `dataclasses.asdict`,
with `observed_at` normalised) against a fixed offline fixture, and pins it to the
exact dict produced by the pre-refactor code. If a future change to `_base.py` or a
provider alters any field - including id/snapshot_path fallback strings that have
no other test coverage - this test catches it.
"""
import dataclasses
import json
from pathlib import Path

import httpx

from leadscout.providers import ats, gleif

FIXTURES = Path(__file__).parent / "fixtures"


def _patch(monkeypatch, module, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "Client", fake_client)


def _normalised(result) -> dict:
    d = dataclasses.asdict(result)
    for ev in d["evidence"]:
        ev["observed_at"] = "FIXED"
    return d


GOLDEN_ATS_INFRA_JOB = {
    "provider_name": "ats", "status": "ok",
    "note": "boards checked: ['ashby']; 1 evidence job(s) naming a provider",
    "calls": 1, "latency_ms": 0,
    "evidence": [{
        "id": "ev-ats-1", "source_type": "ats", "url": "https://jobs.ashbyhq.com/acme",
        "observed_at": "FIXED",
        "content_sha256": "f5e3a136d513a7a2b04018e399dcad79962b1822594a5ab58290deab3951d0aa",
        "strength": "STRONG",
        "snippet": "Senior SRE: Own our AWS and Kubernetes infrastructure.",
        "snapshot_path": "out/provenance/snapshots/unrecorded/ev-ats-1.json",
        "origin": "provider:ats", "family": "ats_hiring", "provider": "AWS",
        "scope": "workload", "freshness": "current", "source_published_at": None,
        "derived_text": "",
    }],
}

GOLDEN_GLEIF_ZAPIER = {
    "provider_name": "gleif", "status": "ok",
    "note": "accepted ZAPIER, INC. (US-DE)",
    "calls": 1, "latency_ms": 0,
    "evidence": [{
        "id": "ev-gleif-1", "source_type": "gleif",
        "url": "https://search.gleif.org/#/record/254900XIXKZQ7A7N1M29",
        "observed_at": "FIXED",
        "content_sha256": "a841c6c7ca4c84d5ef69608fe69006cfd13a6813130fb9ee7b58dc3d0bc44cd4",
        "strength": "STRONG",
        "snippet": "ZAPIER, INC. (254900XIXKZQ7A7N1M29), US, ACTIVE, registered_jurisdiction=US-DE",
        "snapshot_path": "out/provenance/snapshots/unrecorded/ev-gleif-1.json",
        "origin": "provider:gleif", "family": "corporate_identity", "provider": None,
        "scope": "unknown", "freshness": "current", "source_published_at": None,
        "derived_text": "",
    }],
}


def test_ats_infra_job_golden(monkeypatch):
    fixture = {"jobs": [{"title": "Senior SRE",
                          "descriptionPlain": "Own our AWS and Kubernetes infrastructure."}],
               "apiVersion": "1"}

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch(monkeypatch, ats, handler)
    result = ats.run("Acme", ['<a href="https://jobs.ashbyhq.com/acme/xyz">Jobs</a>'])
    assert _normalised(result) == GOLDEN_ATS_INFRA_JOB


def test_gleif_zapier_golden(monkeypatch):
    fixture = json.loads((FIXTURES / "gleif" / "lei_records_zapier.json").read_text(encoding="utf-8"))

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch(monkeypatch, gleif, handler)
    result = gleif.run("Zapier", "United States of America", "https://zapier.com")
    assert _normalised(result) == GOLDEN_GLEIF_ZAPIER
