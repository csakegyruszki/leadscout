"""the ledger `tool` values a fake run produces must cover every
provider that actually produced evidence in that run - no silent gap between what a
provider claims to have recorded and what the ledger actually holds - and a payload
containing the lead's own contact name/email must never reach the ledger or any
snapshot file, exercised end-to-end through research_lead (not a sentinel string
check in isolation - see REVIEW-6bA-verified.md's "Not verified / low priority" note
on PII, folded in here as asked).

Every real provider (`ats`/`github`/`gleif`/`footprint`/`trust_pages`/`vendor`) is
stubbed to produce exactly one Evidence item and call `run.record()` itself, the same
way the real provider would - unlike test_research_ats_wiring.py's stubs (which
return an empty ProviderResult for the providers not under test), this file's stubs
are the ones actually exercising the ledger-write path for each provider.
"""
import hashlib
import json
from datetime import UTC, datetime

import pytest

from leadscout import provenance, research
from leadscout.models import Evidence, Lead, ProviderResult, ResearchFacts
from leadscout.providers.wikidata import WikidataFacts

CONTACT_NAME = "Priya LeakTest"
CONTACT_EMAIL = "priya@leak-test.example"


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


def _fake_provider_result(run, provider_name: str, family: str) -> ProviderResult:
    """Builds one real Evidence + actually calls run.record() (the way the real
    provider module would), so the ledger genuinely receives an entry for
    `provider_name` - stubbing the provider's `run` to just return a canned
    ProviderResult without calling run.record() would make this test vacuous."""
    eid = run.next_evidence_id()
    raw = f"{provider_name} fake payload, no PII".encode()
    ev = Evidence(
        id=eid, source_type=provider_name, url=f"https://example.com/{provider_name}",
        observed_at=datetime.now(UTC).isoformat(), content_sha256=hashlib.sha256(raw).hexdigest(),
        strength="MEDIUM", snippet=f"{provider_name} evidence, no contact PII",
        snapshot_path=f"out/provenance/snapshots/{run.run_id}/{eid}.txt",
        family=family, freshness="current", origin=f"provider:{provider_name}",
    )
    run.record(ev, raw, stage=provider_name)
    return ProviderResult(provider_name=provider_name, status="ok", evidence=[ev], calls=1)


@pytest.fixture
def _stub_all_providers(monkeypatch):
    """Every provider produces exactly one evidence item; website/wikipedia/wikidata
    go through research_lead's own recording path, not this helper."""
    monkeypatch.setattr(research.ats_provider, "run",
                         lambda company, texts, hrefs=None, run=None: _fake_provider_result(run, "ats", "ats_hiring"))
    monkeypatch.setattr(research.github_provider, "run",
                         lambda company, texts, hrefs=None, run=None: _fake_provider_result(
                             run, "github", "engineering_footprint"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: _fake_provider_result(
                             run, "gleif", "corporate_identity"))
    monkeypatch.setattr(research.footprint_provider, "run",
                         lambda domain, run=None: _fake_provider_result(run, "footprint", "network_footprint"))
    monkeypatch.setattr(research.trust_pages_provider, "run",
                         lambda domain, texts, run=None: _fake_provider_result(
                             run, "trust_pages", "first_party_statement"))
    monkeypatch.setattr(research.vendor_provider, "run",
                         lambda company, website_url="", run=None: _fake_provider_result(
                             run, "vendor", "vendor_case_study"))
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website",
                         lambda company, website: (WikidataFacts(qid="Q1", website=website, raw={"id": "Q1"}), "FOUND"))
    monkeypatch.setattr(research, "ask_model", lambda system, user, schema, purpose="": ResearchFacts())
    # crawl_subpages now always probes robots.txt/sitemap discovery (Fix #6) - a
    # real network call this offline test never wants; the landing page's own
    # website/website_subpage recording path is what this test actually covers.
    monkeypatch.setattr(research.website_provider, "crawl_subpages", lambda landing_url, landing_html: [])


def test_ledger_tool_values_cover_every_provider_that_produced_evidence(monkeypatch, _stub_all_providers):
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("Acme home page", True, "<html></html>"))
    monkeypatch.setattr(research, "fetch_wikipedia",
                         lambda company: ("Acme summary", "https://en.wikipedia.org/wiki/Acme"))

    run = provenance.ProvenanceRun.start("Acme")
    lead = Lead(name=CONTACT_NAME, email=CONTACT_EMAIL, company="Acme", website="https://acme.example")
    r = research.research_lead(lead, run=run)
    run.finish()

    produced_source_types = {ev.source_type for ev in r.evidence}
    # Every provider stubbed above, plus the two research.py-recorded source types
    # (website + wikipedia), must actually have produced evidence in this run -
    # otherwise the coverage assertion below would be vacuously true.
    expected = {"ats", "github", "gleif", "footprint", "trust_pages", "vendor", "wikidata", "website", "wikipedia"}
    assert expected <= produced_source_types, f"missing evidence for: {expected - produced_source_types}"

    ledger_text = provenance.LEDGER_PATH.read_text(encoding="utf-8")
    ledger_lines = [line for line in ledger_text.splitlines() if line.strip()]
    ledger_tools = {json.loads(line)["tool"] for line in ledger_lines}

    for source_type in produced_source_types:
        assert f"leadscout:{source_type}" in ledger_tools, (
            f"{source_type} produced evidence but has no ledger record")


def test_contact_name_and_email_never_reach_ledger_or_snapshots(monkeypatch, _stub_all_providers):
    """The lead's own submitted contact fields must never be threaded into any
    provider payload, snapshot, or ledger record - checked end-to-end, not by
    asserting a helper simply omits a parameter."""
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("Acme home page, generic content.", True, "<html></html>"))
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("Acme summary", "https://en.wikipedia.org/wiki/Acme"))

    run = provenance.ProvenanceRun.start("Acme")
    lead = Lead(name=CONTACT_NAME, email=CONTACT_EMAIL, company="Acme", website="https://acme.example")
    research.research_lead(lead, run=run)
    run.finish()

    ledger_text = provenance.LEDGER_PATH.read_text(encoding="utf-8")
    assert CONTACT_NAME not in ledger_text
    assert CONTACT_EMAIL not in ledger_text

    for snap in provenance.SNAPSHOT_DIR.rglob("*"):
        if snap.is_file():
            content = snap.read_bytes()
            assert CONTACT_NAME.encode() not in content
            assert CONTACT_EMAIL.encode() not in content
