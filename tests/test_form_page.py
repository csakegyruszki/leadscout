"""GET / serves the four-field intake form, and POST /leads carries the routed
action so the page never has to re-derive it.

An earlier version of the page had the routing
rule re-implemented in JavaScript, because the response had no `next_action` key. A
second copy of "compliance outranks fit" in the browser would drift from notify.py
silently, so the key was added to the response instead and the copy removed - these
tests hold both ends of that.
"""
import re

import pytest
from fastapi.testclient import TestClient

from leadscout import api
from leadscout.models import ComplianceResult, FitResult, Lead, LeadOutcome, Research

_FIELDS = ("name", "email", "company", "website")


def _outcome(lead: Lead, *, status: str = "blocked") -> LeadOutcome:
    return LeadOutcome(
        lead=lead,
        research=Research(summary="s", industry="SaaS", headquarters_country="US"),
        compliance=ComplianceResult(True, status, [], "competitor"),
        fit=FitResult(90, "HIGH", 30, 45, 15, "r", "1k-10k", "reasoning"),
        sales_ready=True, run_id="run-1", evidence_count=1, provenance_status="ok",
        provenance_head="1:abc",
    )


@pytest.fixture(autouse=True)
def _pretend_llm_credentials(monkeypatch):
    monkeypatch.setattr(api, "_has_usable_llm_credential", lambda: True)


def test_form_page_is_served_with_exactly_the_four_brief_fields():
    resp = TestClient(api.app).get("/")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    names = re.findall(r'<input[^>]*\bname="([^"]+)"', resp.text)
    assert tuple(names) == _FIELDS


def test_the_page_makes_no_external_request():
    """No CDN, font host or analytics: the demo has to work on a laptop with no
    internet, and a third party must not see who is being screened."""
    html = TestClient(api.app).get("/").text
    assert not re.search(r'(?:src|href)\s*=\s*"\s*(?:https?:)?//', html)


def test_the_page_renders_server_values_as_text_not_markup():
    """Every value shown comes from a fetched page or an LLM summary of one."""
    html = TestClient(api.app).get("/").text
    # The assignment, not the word - the page explains in a comment why it avoids it.
    assert not re.search(r"\.(?:inner|outer)HTML\s*=", html)
    assert not re.search(r"insertAdjacentHTML|document\.write", html)
    assert "textContent" in html


def test_the_page_does_not_re_derive_the_next_action():
    html = TestClient(api.app).get("/").text
    assert "payload.next_action" in html
    # The rule itself must live in notify.py alone.
    assert "'blocked'" not in html and '"blocked"' not in html


def test_leads_response_carries_the_routed_action(monkeypatch):
    monkeypatch.setattr(api, "process_lead", lambda lead, *, verbose=True: _outcome(lead))
    resp = TestClient(api.app).post("/leads", json={
        "name": "Ann", "email": "ann@x.com", "company": "X Corp", "website": "https://x.com"})
    assert resp.status_code == 200
    # Compliance outranks a sales_ready, fit-90 account - the precedence the page would
    # have got wrong on its own.
    assert resp.json()["next_action"].startswith("DO NOT ENGAGE")
