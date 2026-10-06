"""research.py wiring for the ats provider: ats.run must
actually be called on the fetched pages, its evidence must reach Research.evidence /
Research.provider_results, and its raw payload must land in the provtrail ledger -
none of that was true before this wiring (ats.py existed but research_lead never
called it). Everything else research_lead touches (website fetch, Wikipedia,
Wikidata, GitHub, GLEIF, the LLM) is faked so this stays a fast, offline, single-
purpose test of the ats wiring path.
"""
import httpx
import pytest

from leadscout import provenance, research
from leadscout.models import Lead, ProviderResult, ResearchFacts
from leadscout.providers import ats

ASHBY_HTML = '<html><body><a href="https://jobs.ashbyhq.com/zapier/xyz">Careers</a></body></html>'


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


@pytest.fixture(autouse=True)
def _stub_everything_but_ats(monkeypatch):
    """research_lead also calls Wikipedia/Wikidata/GitHub/GLEIF/the LLM - stub each
    to a no-op so this test exercises only the ats path, with no real network call."""
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("", ""))
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.github_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(
                             provider_name="github", status="ok"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok"))
    monkeypatch.setattr(research.footprint_provider, "run",
                         lambda domain, run=None: ProviderResult(provider_name="footprint", status="ok"))
    monkeypatch.setattr(research.trust_pages_provider, "run",
                         lambda domain, texts, run=None: ProviderResult(provider_name="trust_pages", status="ok"))
    monkeypatch.setattr(research.vendor_provider, "run",
                         lambda company, website_url="", run=None: ProviderResult(
                             provider_name="vendor", status="skipped"))
    monkeypatch.setattr(research, "ask_model", lambda system, user, schema, purpose="": ResearchFacts())
    # website_provider.crawl_subpages now always probes robots.txt/sitemap discovery
    # (Fix #6) - a real network call this offline test suite never wants. Tests that
    # actually exercise the crawl (e.g. the href-in-subpage one below) override this
    # with their own monkeypatch.setattr call, which applies after this autouse one.
    monkeypatch.setattr(research.website_provider, "crawl_subpages", lambda landing_url, landing_html: [])


def _patch_ats_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ats.httpx, "Client", fake_client)


def test_zapier_like_run_ashby_link_seven_jobs_zero_infra_evidence(monkeypatch):
    fixture = {
        "jobs": [{"title": f"Role {i}", "descriptionPlain": "Help our customers automate their workflows."}
                 for i in range(7)],
        "apiVersion": "1",
    }

    def ashby_handler(request):
        assert "ashbyhq.com" in str(request.url)
        return httpx.Response(200, json=fixture)

    _patch_ats_transport(monkeypatch, ashby_handler)
    monkeypatch.setattr(research, "fetch_website", lambda url, max_chars=6000: ("Zapier home page", True, ASHBY_HTML))

    run = provenance.ProvenanceRun.start("Zapier")
    lead = Lead(name="Ann", email="a@b.c", company="Zapier", website="https://zapier.com")
    r = research.research_lead(lead, run=run)
    status, head, count = run.finish()

    ats_results = [pr for pr in r.provider_results if pr.provider_name == "ats"]
    assert len(ats_results) == 1
    assert ats_results[0].status == "ok"
    assert ats_results[0].evidence == []
    assert ats_results[0].calls == 1
    assert not any(ev.source_type == "ats" for ev in r.evidence)

    ledger_text = provenance.LEDGER_PATH.read_text(encoding="utf-8")
    assert '"tool": "leadscout:ats"' not in ledger_text  # zero evidence -> nothing recorded for ats
    assert status == "ok"


def test_no_link_run_ats_is_skipped_with_zero_calls(monkeypatch):
    def handler(request):
        raise AssertionError("must not call any board API without a detected link")

    _patch_ats_transport(monkeypatch, handler)
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("Plain company homepage, no ATS link.", True,
                                                       "<html><body>Plain company homepage</body></html>"))

    run = provenance.ProvenanceRun.start("Acme")
    lead = Lead(name="Ann", email="a@b.c", company="Acme", website="https://acme.example")
    r = research.research_lead(lead, run=run)

    ats_results = [pr for pr in r.provider_results if pr.provider_name == "ats"]
    assert len(ats_results) == 1
    assert ats_results[0].status == "skipped"
    assert ats_results[0].calls == 0


def test_ats_evidence_reaches_ledger_with_real_url(monkeypatch):
    """A board link WITH infra-term evidence: the real board URL (not a placeholder)
    is what gets recorded, and the Ashby payload lands in the ledger."""
    fixture = {"jobs": [{"title": "Senior SRE", "descriptionPlain": "Own our AWS and Kubernetes infrastructure."}],
               "apiVersion": "1"}

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_ats_transport(monkeypatch, handler)
    monkeypatch.setattr(research, "fetch_website", lambda url, max_chars=6000: ("home", True, ASHBY_HTML))

    run = provenance.ProvenanceRun.start("Acme")
    lead = Lead(name="Ann", email="a@b.c", company="Acme", website="https://acme.example")
    r = research.research_lead(lead, run=run)
    run.finish()

    ats_evidence = [ev for ev in r.evidence if ev.source_type == "ats"]
    assert len(ats_evidence) == 1
    assert ats_evidence[0].url == "https://jobs.ashbyhq.com/zapier"
    assert ats_evidence[0].url.startswith("https://jobs.ashbyhq.com/")
    assert "example" not in ats_evidence[0].url  # no placeholder host

    ledger_text = provenance.LEDGER_PATH.read_text(encoding="utf-8")
    assert '"tool": "leadscout:ats"' in ledger_text
    assert "jobs.ashbyhq.com" in ledger_text


def test_ats_link_only_on_a_crawled_subpage_href_is_still_detected(monkeypatch):
    """Item 1, end-to-end wiring: the landing page links NO board at all - the
    Ashby board link only lives in a crawled subpage's footer <a href>, whose
    trafilatura-extracted TEXT (what `research.py` used to pass to ats.run) is
    just "Careers - join our team", never the URL. research_lead must still find
    it via the subpage's `hrefs` list."""
    fixture = {"jobs": [{"title": "Senior SRE", "descriptionPlain": "Own our AWS infrastructure."}],
               "apiVersion": "1"}

    def ashby_handler(request):
        assert "ashbyhq.com" in str(request.url)
        return httpx.Response(200, json=fixture)

    _patch_ats_transport(monkeypatch, ashby_handler)
    monkeypatch.setattr(
        research, "fetch_website",
        lambda url, max_chars=6000: ("home page", True, '<a href="/careers">Careers</a>'))
    monkeypatch.setattr(
        research.website_provider, "crawl_subpages",
        lambda landing_url, landing_html: [{
            "url": "https://acme.example/careers", "text": "Careers - join our team", "strength": "MEDIUM",
            "hrefs": ["https://jobs.ashbyhq.com/zapier/xyz"],
        }])

    run = provenance.ProvenanceRun.start("Acme")
    lead = Lead(name="Ann", email="a@b.c", company="Acme", website="https://acme.example")
    r = research.research_lead(lead, run=run)
    run.finish()

    ats_evidence = [ev for ev in r.evidence if ev.source_type == "ats"]
    assert len(ats_evidence) == 1
    assert ats_evidence[0].url == "https://jobs.ashbyhq.com/zapier"
