"""cloud.assess_cloud_usage is wired into research_lead
(Research.cloud_usage) and into notify.render's email body, above the spend prior."""
import pytest

from leadscout import provenance, research
from leadscout.compliance import ComplianceResult
from leadscout.fit import score_fit
from leadscout.models import Evidence, Lead, LeadOutcome, ProviderResult, Research, ResearchFacts
from leadscout.notify import render


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


def test_research_lead_sets_cloud_usage_from_collected_evidence(monkeypatch):
    """A first-party trust-page hosting statement (DIRECT, current) should make
    research_lead's own Research.cloud_usage reach CONFIRMED - proving cloud.py is
    actually wired in, not left at the CloudUsageAssessment() default."""
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("home page, no cloud mentions", True, "<html></html>"))
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("", ""))
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.ats_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(provider_name="ats", status="ok"))
    monkeypatch.setattr(research.github_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(
                             provider_name="github", status="ok"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq, website, run=None: ProviderResult(provider_name="gleif", status="ok"))
    monkeypatch.setattr(research.footprint_provider, "run",
                         lambda domain, run=None: ProviderResult(provider_name="footprint", status="ok"))
    monkeypatch.setattr(research.website_provider, "crawl_subpages", lambda landing_url, landing_html: [])
    monkeypatch.setattr(research.vendor_provider, "run",
                         lambda company, website_url="", run=None: ProviderResult(
                             provider_name="vendor", status="skipped"))

    hosting_ev = Evidence(
        id="ev-trust-1", source_type="trust_pages", url="https://acme.example/trust",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="DIRECT", snippet="We host our platform on AWS.", snapshot_path="out/x.txt",
        family="first_party_statement", provider="AWS", scope="workload", freshness="current",
    )
    monkeypatch.setattr(research.trust_pages_provider, "run",
                         lambda domain, texts, run=None: ProviderResult(
                             provider_name="trust_pages", status="ok", evidence=[hosting_ev]))
    monkeypatch.setattr(research, "ask_model", lambda system, user, schema, purpose="": ResearchFacts())

    lead = Lead(name="Ann", email="a@b.c", company="Acme", website="https://acme.example")
    r = research.research_lead(lead, run=None)
    assert r.cloud_usage.state == "CONFIRMED"


def test_notify_render_includes_cloud_usage_above_spend_prior():
    lead = Lead("Ann", "ann@x.com", "Acme", "https://acme.example")
    research_result = Research(summary="s", headquarters_country="US", estimated_employees=50)
    from leadscout.cloud import assess_cloud_usage
    hosting_ev = Evidence(
        id="ev-1", source_type="trust_pages", url="https://acme.example/trust",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="DIRECT", snippet="We host on AWS.", snapshot_path="out/x.txt",
        family="first_party_statement", provider="AWS", scope="workload", freshness="current",
    )
    research_result.cloud_usage = assess_cloud_usage([hosting_ev])
    fit = score_fit(lead, research_result)
    outcome = LeadOutcome(
        lead=lead, research=research_result,
        compliance=ComplianceResult(flagged=False, status="clear", reasoning="ok", sanctions_status="none"),
        fit=fit, sales_ready=True,
    )
    subject, body = render(outcome)
    assert "Cloud usage: CONFIRMED" in body
    assert body.index("Cloud usage:") < body.index("Spend prior")


def test_notify_render_cloud_block_has_the_part4_addendum_format():
    """Part 4 addendum: Cloud usage / Confidence / Observed provider(s) / Evidence
    (one line per family, "<STRENGTH> · <family> · <observation> (ev-id)") /
    Boundary - in that order, above the spend prior."""
    lead = Lead("Ann", "ann@x.com", "Acme", "https://acme.example")
    research_result = Research(summary="s", headquarters_country="US", estimated_employees=50)
    from leadscout.cloud import assess_cloud_usage
    hosting_ev = Evidence(
        id="ev-1", source_type="trust_pages", url="https://acme.example/trust",
        observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
        strength="DIRECT", snippet="We host on AWS.", snapshot_path="out/x.txt",
        family="first_party_statement", provider="AWS", scope="workload", freshness="current",
    )
    research_result.evidence = [hosting_ev]
    research_result.cloud_usage = assess_cloud_usage([hosting_ev])
    fit = score_fit(lead, research_result)
    outcome = LeadOutcome(
        lead=lead, research=research_result,
        compliance=ComplianceResult(flagged=False, status="clear", reasoning="ok", sanctions_status="none"),
        fit=fit, sales_ready=True,
    )
    _, body = render(outcome)
    assert f"Confidence: {fit.confidence}" in body
    assert "Observed provider(s): AWS (CONFIRMED, 1 independent family)" in body
    assert "Evidence:" in body
    assert "- DIRECT · first_party_statement · We host on AWS. (ev-1)" in body
    assert "Boundary:" in body
    order = [body.index(s) for s in ("Cloud usage:", "Confidence:", "Observed provider(s):",
                                     "Evidence:", "Boundary:", "Spend prior")]
    assert order == sorted(order)
