"""Defect: when the LLM's own headquarters_country extraction
comes back empty/unknown, research_lead falls back to a structured source - an
ACCEPTED GLEIF record's legal-address country, or a website-matched Wikidata
entity's country - and records which one in the new Research.hq_source field.
If both exist and disagree, HQ stays unknown rather than silently picking one
(CLAUDE.md's "unknown must not collapse into valid"). Offline: research_lead's
website/Wikipedia/ATS/GitHub/footprint/trust_pages/vendor calls are all stubbed to
no-ops so only the GLEIF/Wikidata/LLM interaction under test runs.
"""
import pytest

from leadscout import provenance, research
from leadscout.models import Evidence, Lead, ProviderResult, ResearchFacts
from leadscout.providers.wikidata import WikidataFacts

GLEIF_US_EVIDENCE = Evidence(
    id="ev-gleif-1", source_type="gleif", url="https://search.gleif.org/#/record/x",
    observed_at="2026-09-19T00:00:00+00:00", content_sha256="a" * 64,
    strength="STRONG", snippet="ZAPIER, INC. (549300O8AVGWCXWF1584), US, ACTIVE, "
                                "registered_jurisdiction=US-DE",
    snapshot_path="out/x.json", family="corporate_identity", freshness="current",
    origin="provider:gleif",
)


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "LEDGER_DIR", tmp_path / "provenance")
    monkeypatch.setattr(provenance, "LEDGER_PATH", tmp_path / "provenance" / "ledger.jsonl")
    monkeypatch.setattr(provenance, "SNAPSHOT_DIR", tmp_path / "provenance" / "snapshots")


@pytest.fixture(autouse=True)
def _stub_non_hq_providers(monkeypatch):
    monkeypatch.setattr(research, "fetch_website",
                         lambda url, max_chars=6000: ("home page", True, "<html></html>"))
    monkeypatch.setattr(research, "fetch_wikipedia", lambda company: ("", ""))
    monkeypatch.setattr(research.ats_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(provider_name="ats", status="ok"))
    monkeypatch.setattr(research.github_provider, "run",
                         lambda company, texts, hrefs=None, run=None: ProviderResult(
                             provider_name="github", status="ok"))
    monkeypatch.setattr(research.footprint_provider, "run",
                         lambda domain, run=None: ProviderResult(provider_name="footprint", status="ok"))
    monkeypatch.setattr(research.trust_pages_provider, "run",
                         lambda domain, texts, run=None: ProviderResult(provider_name="trust_pages", status="ok"))
    monkeypatch.setattr(research.website_provider, "crawl_subpages", lambda landing_url, landing_html: [])
    monkeypatch.setattr(research.vendor_provider, "run",
                         lambda company, website_url="", run=None: ProviderResult(
                             provider_name="vendor", status="skipped"))


def _lead() -> Lead:
    return Lead(name="Jo Contact", email="jo@zapier.com", company="Zapier", website="https://zapier.com")


def test_gleif_resolves_hq_when_llm_and_wikidata_are_both_unknown(monkeypatch):
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok", evidence=[GLEIF_US_EVIDENCE]))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="unknown"))

    r = research.research_lead(_lead())

    assert r.headquarters_country == "United States"
    assert r.hq_source == "gleif"


def test_wikidata_resolves_hq_when_llm_is_unknown_and_gleif_found_nothing(monkeypatch):
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website",
                         lambda company, website: (WikidataFacts(
                             qid="Q1", website=website, country="Hungary", raw={"id": "Q1"}), "FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok"))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country=""))

    r = research.research_lead(_lead())

    assert r.headquarters_country == "Hungary"
    assert r.hq_source == "wikidata"


def test_gleif_and_wikidata_disagreement_leaves_hq_unknown(monkeypatch):
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website",
                         lambda company, website: (WikidataFacts(
                             qid="Q1", website=website, country="Hungary", raw={"id": "Q1"}), "FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok", evidence=[GLEIF_US_EVIDENCE]))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="unknown"))

    r = research.research_lead(_lead())

    assert r.headquarters_country.strip().lower() in ("", "unknown")
    # No country, so no source either: labelling an unknown HQ "llm" names a rung that
    # produced nothing. Changed with the waterfall-precedence fix (2026-09-20).
    assert r.hq_source == ""
    assert any("disagree" in u for u in r.uncertainties)


def test_website_cctld_resolves_hq_when_llm_gleif_and_wikidata_are_all_unknown(monkeypatch):
    """Item 4: a ccTLD is the last, weakest rung of the HQ waterfall - used only
    when the LLM, GLEIF and Wikidata all came back empty. Artizan Bakery
    (artizan.hu) is the real case that motivated this: no GLEIF/Wikidata record,
    generic-website LLM extraction -> HQ used to stay "unknown" forever even
    though the submitted website itself carries a country signal."""
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok"))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="unknown"))

    lead = Lead(name="Jo Contact", email="jo@artizan.hu", company="Artizan Bakery",
                website="https://www.artizan.hu")
    r = research.research_lead(lead)

    assert r.headquarters_country == "Hungary"
    assert r.hq_source == "website_tld"
    assert r.hq_confidence == "low"


def test_generic_tld_leaves_hq_unknown_when_llm_gleif_and_wikidata_are_all_unknown(monkeypatch):
    """A generic gTLD (.com/.io/...) carries no country signal at all - the
    waterfall must not invent one; HQ stays unknown, same as before item 4."""
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website", lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok"))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="unknown"))

    r = research.research_lead(_lead())  # zapier.com - generic TLD

    assert r.headquarters_country.strip().lower() in ("", "unknown")
    assert r.hq_source == "llm"
    assert r.hq_confidence == ""


def test_a_registry_conflict_is_not_resolved_by_falling_back_to_the_llm(monkeypatch):
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website",
                         lambda company, website: (WikidataFacts(
                             qid="Q1", website=website, country="Hungary", raw={"id": "Q1"}), "FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                         lambda company, hq_country, website, run=None: ProviderResult(
                             provider_name="gleif", status="ok", evidence=[GLEIF_US_EVIDENCE]))
    monkeypatch.setattr(research, "ask_model",
                         lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="France"))

    r = research.research_lead(_lead())

    # GLEIF (US) and Wikidata (Hungary) disagree, so neither is adopted - and the LLM's
    # "France" is not a fallback for a registry-level conflict either. Before the
    # precedence fix this returned France/llm: the weakest rung won outright because the
    # waterfall returned early whenever the LLM had produced any non-"unknown" string.
    assert r.headquarters_country.strip().lower() in ("", "unknown")
    assert r.hq_source == ""
    assert any("disagree" in u for u in r.uncertainties)


def test_a_weaker_source_disagreeing_does_not_erase_the_stronger_one(monkeypatch):
    """Conflicting weak evidence may lower confidence; it must not delete a stronger
    finding merely by disagreeing with it. GLEIF says US, the model says France: the
    result is the registry's answer, the model's is retained as an uncertainty for audit,
    and the field does NOT collapse to unknown."""
    monkeypatch.setattr(research.wikidata_provider, "lookup_by_website",
                        lambda company, website: (None, "NOT_FOUND"))
    monkeypatch.setattr(research.gleif_provider, "run",
                        lambda company, hq_country, website, run=None: ProviderResult(
                            provider_name="gleif", status="ok", evidence=[GLEIF_US_EVIDENCE]))
    monkeypatch.setattr(research, "ask_model",
                        lambda system, user, schema, purpose="": ResearchFacts(headquarters_country="France"))

    r = research.research_lead(_lead())

    assert r.hq_source == "gleif"
    assert r.headquarters_country.strip().lower() not in ("", "unknown")
    assert any("France" in u and "overrides" in u for u in r.uncertainties), r.uncertainties
