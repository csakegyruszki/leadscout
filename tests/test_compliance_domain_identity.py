"""The submitted WEBSITE/E-MAIL domain is a second competitor-identity surface.

Found by an adversarial test battery: a lead submitted as company "Unimedia
Technology" with website `cloud-trim.com` - Cloud-Trim is that company's own AWS
cost-optimization product, i.e. the listed competitor under a different legal name -
was screened CLEAR. The LLM layer had the URL in its prompt and summarised the page
correctly ("a free tool focused on AWS cost optimization"), then reasoned the flag
away: "clearly a different company based in Spain". A name-only pre-screen cannot see
this, so the domain is matched deterministically and cannot be cleared by the model.
"""
from leadscout import compliance
from leadscout.models import ComplianceVerdict, Lead, Research
from leadscout.sanctions import SanctionsScreen


def _clear_verdict(system, user, schema, *, purpose="compliance"):
    return ComplianceVerdict(status="clear", flagged=False, matches=[], reasoning="different company")


def _offline(monkeypatch):
    monkeypatch.setattr(compliance, "ask_model", _clear_verdict)
    monkeypatch.setattr(compliance, "screen_sanctions",
                        lambda company, hq, run=None: SanctionsScreen(status="none", hits=[]))


def test_competitor_domain_under_a_different_company_name_is_flagged():
    lead = Lead("Press Team", "info@unimedia.tech", "Unimedia Technology",
                "https://www.cloud-trim.com/how-it-works/")
    hits = compliance.prescreen(lead, Research(headquarters_country="Spain"))
    domain_hits = [h for h in hits if h.get("origin") == "prescreen:domain"]
    assert [h["term"] for h in domain_hits] == ["CloudTrim Inc"]
    assert domain_hits[0]["score"] == 100
    # The flag must say which surface matched, and carry the domain as the raw value.
    assert domain_hits[0]["original_value"] == "cloud-trim.com"
    assert "domain 'cloud-trim.com'" in domain_hits[0]["reason"]


def test_a_hyphen_as_space_reading_is_needed_too():
    """`cloud-trim` only reaches 100 with the hyphen stripped, `spendwise-cloud` only
    with the hyphen read as a space - hence both readings, deduplicated per domain."""
    lead = Lead("Acme", "a@acme.com", "Acme", "https://spendwise-cloud.io")
    hits = [h for h in compliance.prescreen(lead, Research()) if h.get("origin") == "prescreen:domain"]
    assert any(h["term"] == "SpendWise Cloud" and h["score"] == 100 for h in hits)
    # One hit per (competitor, domain) - the two readings are one surface, not two.
    assert len({(h["term"], h["original_value"]) for h in hits}) == len(hits)


def test_the_llm_cannot_clear_a_confident_domain_hit(monkeypatch):
    _offline(monkeypatch)
    result = compliance.screen(
        Lead("Press Team", "info@unimedia.tech", "Unimedia Technology",
             "https://www.cloud-trim.com/how-it-works/"),
        Research(headquarters_country="Spain", summary="Free AWS cost optimization tool."))
    assert result.status == "review" and result.flagged
    assert any(m.get("origin") == "prescreen:domain" for m in result.matches)


def test_ordinary_lead_domains_produce_no_competitor_hit(monkeypatch):
    """The check must not fire on unrelated business domains - measured on the real
    leads of the 2026-09-20 battery plus the demo corpus."""
    _offline(monkeypatch)
    for company, website, email in [
        ("Zapier", "https://zapier.com", "ops@zapier.com"),
        ("NISSHA", "https://www.nissha.com/company/outline.html", "a@zonnebodo.co.jp"),
        ("ClosetMaid PRO", "https://closetmaidpro.com/privacy-policy/", "help@ames.com"),
        ("Notion", "https://notiontechnologies.com/about-us", "sales@notiontechnologies.com"),
        ("PostHog", "https://posthog.com", "a@posthog.com"),
    ]:
        hits = compliance.prescreen(Lead("A", email, company, website), Research())
        assert not [h for h in hits if h.get("origin") == "prescreen:domain"], f"{website} flagged"


def test_a_lead_without_a_usable_domain_is_not_an_error():
    hits = compliance.prescreen(Lead("A", "not-an-email", "Acme", ""), Research())
    assert not [h for h in hits if h.get("origin") == "prescreen:domain"]
