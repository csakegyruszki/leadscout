"""API test: monkeypatches process_lead so no network/LLM call happens."""
import pytest
from fastapi.testclient import TestClient

from leadscout import api
from leadscout.llm import LLMError
from leadscout.models import ComplianceResult, FitResult, Lead, LeadOutcome, Research


def _fake_outcome(lead: Lead) -> LeadOutcome:
    return LeadOutcome(
        lead=lead,
        research=Research(summary="s", industry="SaaS", headquarters_country="US"),
        compliance=ComplianceResult(False, "clear", [], "ok"),
        fit=FitResult(80, "HIGH", 25, 40, 15, "r", "1k-10k", "reasoning"),
        sales_ready=True, llm_calls=2, llm_cost_usd=0.0001, llm_latency_ms=500,
        run_id="run-1", evidence_count=3, provenance_status="ok", provenance_head="1:abcdef",
    )


@pytest.fixture(autouse=True)
def _pretend_llm_credentials(monkeypatch):
    """Tests must not depend on a local .env: the credential preflight is stubbed to True;
    the 503 test overrides it explicitly."""
    monkeypatch.setattr(api, "_has_usable_llm_credential", lambda: True)


def test_leads_endpoint_returns_outcome(monkeypatch):
    monkeypatch.setattr(api, "process_lead", lambda lead, *, verbose=True: _fake_outcome(lead))
    client = TestClient(api.app)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X Corp", "website": "https://x.com",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["sales_ready"] is True
    assert body["fit"]["confidence"] == "HIGH"
    assert body["lead"]["company"] == "X Corp"


def test_health_endpoint():
    client = TestClient(api.app)
    body = client.get("/health").json()
    assert body["status"] == "ok" and "build" in body


def test_invalid_size_band_is_rejected():
    client = TestClient(api.app)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
        "company_size_band": "huge",
    })
    assert resp.status_code == 422


def test_empty_size_band_is_allowed(monkeypatch):
    monkeypatch.setattr(api, "process_lead", lambda lead, *, verbose=True: _fake_outcome(lead))
    client = TestClient(api.app)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
        "company_size_band": "",
    })
    assert resp.status_code == 200


def test_no_llm_credential_returns_503(monkeypatch):
    monkeypatch.setattr(api, "_has_usable_llm_credential", lambda: False)
    client = TestClient(api.app)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
    })
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "configuration"
    assert ".env.example" in body["detail"]


def test_llm_error_returns_502_with_stripped_secrets(monkeypatch):
    def _raise(lead, *, verbose=True):
        raise LLMError("openrouter 401 from deepseek: Bearer sk-or-v1-abcdefghijklmno invalid")

    monkeypatch.setattr(api, "process_lead", _raise)
    client = TestClient(api.app)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
    })
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "llm_upstream"
    assert "sk-or-v1-abcdefghijklmno" not in body["detail"]
    assert "[REDACTED]" in body["detail"]


def test_unexpected_error_returns_500_without_traceback(monkeypatch):
    def _raise(lead, *, verbose=True):
        raise RuntimeError("boom, unexpected")

    monkeypatch.setattr(api, "process_lead", _raise)
    # raise_server_exceptions=False: a bare Exception is handled by Starlette's
    # ServerErrorMiddleware, which re-raises into the test by default (for
    # debugging) even though our handler still produces the real HTTP response
    # a live server would send - so this flag is needed to see that response here.
    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
    })
    assert resp.status_code == 500
    body = resp.json()
    assert body == {"error": "internal", "run_id": None}


def test_unexpected_error_includes_run_id_when_attached(monkeypatch):
    def _raise(lead, *, verbose=True):
        exc = RuntimeError("boom with run id")
        exc.leadscout_run_id = "run-abort-1"
        raise exc

    monkeypatch.setattr(api, "process_lead", _raise)
    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X", "website": "https://x.com",
    })
    assert resp.status_code == 500
    body = resp.json()
    assert body == {"error": "internal", "run_id": "run-abort-1"}
