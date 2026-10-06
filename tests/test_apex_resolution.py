"""Fix #2 (v0.2 PART B): unknown_reason=NO_RESOLVABLE_DOMAIN now comes from an
actual Cloudflare DoH Status check (Status 3 = NXDOMAIN), not from sniffing the
website fetch's exception type name. Offline via httpx.MockTransport.
"""
import httpx
import pytest

from leadscout import provenance, research
from leadscout.models import Lead, ProviderResult, ResearchFacts


def test_apex_resolves_true_on_normal_answer(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"Status": 0, "Answer": [{"type": 1, "data": "1.2.3.4"}]})

    real_client = httpx.Client
    monkeypatch.setattr(research.httpx, "Client",
                         lambda *a, **k: real_client(*a, transport=httpx.MockTransport(handler), **k))
    assert research._apex_resolves("zapier.com") is True


def test_apex_resolves_false_on_nxdomain_status_3(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"Status": 3})

    real_client = httpx.Client
    monkeypatch.setattr(research.httpx, "Client",
                         lambda *a, **k: real_client(*a, transport=httpx.MockTransport(handler), **k))
    assert research._apex_resolves("cloud-trim.io") is False


def test_apex_resolves_true_when_doh_itself_fails(monkeypatch):
    """A DoH timeout/error must never be asserted as a live NXDOMAIN fact - the
    default on any failure is "resolvable, or at least not provably not"."""

    def boom(*a, **k):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(research.httpx, "Client", boom)
    assert research._apex_resolves("example.com") is True


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


def test_cloud_trim_style_failed_fetch_and_nxdomain_yields_no_resolvable_domain(monkeypatch):
    """End-to-end wiring: a failed website fetch + a DoH Status-3 apex answer must
    reach Research.cloud_usage.unknown_reason == "NO_RESOLVABLE_DOMAIN"."""
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("[unreachable: ConnectError]", False, ""))
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("", ""))
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.ats_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(provider_name="ats", status="ok"))
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
    monkeypatch.setattr(research, "_apex_resolves", lambda domain: False)

    lead = Lead(name="Priya Nair", email="priya@cloud-trim.io", company="Cloud-Trim Ltd.",
               website="https://cloud-trim.io", job_title="VP Sales", company_size_band="51-200")
    r = research.research_lead(lead, run=None)
    assert r.cloud_usage.state == "UNKNOWN"
    assert r.cloud_usage.unknown_reason == "NO_RESOLVABLE_DOMAIN"
