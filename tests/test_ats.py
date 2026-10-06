"""ATS provider: link detection (no slug guessing) and job-content classification,
offline via httpx.MockTransport. The Zapier fixture is the real Ashby response
recorded 2026-09-19 (see PROVIDERS.md/SUMMARY.md) - 7 jobs, none naming infra terms,
which must yield zero evidence without lowering anything (a provider absence is not
negative evidence)."""
import json
from pathlib import Path

import httpx

from leadscout.providers import ats

FIXTURES = Path(__file__).parent / "fixtures" / "ats"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_find_board_links_detects_ashby():
    links = ats.find_board_links("some text", '<a href="https://jobs.ashbyhq.com/zapier/abc">Careers</a>')
    assert links == {"ashby": "zapier"}


def test_find_board_links_none_found():
    assert ats.find_board_links("no links here", "<a href='/pricing'>Pricing</a>") == {}


def test_find_board_links_from_hrefs_when_visible_text_has_no_url():
    """Item 1: a footer/nav anchor whose VISIBLE TEXT is just "Careers" (the real
    Zapier/PostHog shape - trafilatura's extracted text for a crawled subpage keeps
    "Careers", not the href) must still be detected via the raw href list, even
    when no text argument mentions the board URL at all."""
    links = ats.find_board_links("Careers", "Join our team", hrefs=["https://jobs.ashbyhq.com/posthog"])
    assert links == {"ashby": "posthog"}


def test_find_board_links_hrefs_and_texts_both_searched_no_duplicate_board():
    links = ats.find_board_links(
        "no url here", hrefs=["https://jobs.ashbyhq.com/zapier", "https://example.com/other"])
    assert links == {"ashby": "zapier"}


def test_job_signal_none_without_a_named_provider():
    assert ats._job_signal("Sales Assist Representative helping customers") is None


def test_job_signal_none_for_kubernetes_alone_no_named_provider():
    """REVIEW-6bA-verified.md #7: Kubernetes/Terraform/SRE text without a named
    cloud provider is not evidence, even though it used to be."""
    assert ats._job_signal("We use Kubernetes for internal tools") is None


def test_job_signal_medium_with_provider_and_no_role_word():
    assert ats._job_signal("We use AWS for internal tools") == ("AWS", "MEDIUM")


def test_job_signal_strong_with_provider_and_role_word():
    assert ats._job_signal("Senior Platform Engineer - AWS and Kubernetes") == ("AWS", "STRONG")


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ats.httpx, "Client", fake_client)


def test_no_link_skips_any_api_call(monkeypatch):
    def handler(request):
        raise AssertionError("must not call any board API without a detected link")

    _patch_client(monkeypatch, handler)
    result = ats.run("Some Co", ["plain landing page text, no ATS link"])
    assert result.status == "skipped"
    assert result.evidence == []
    assert result.calls == 0


def test_zapier_ashby_board_seven_jobs_zero_infra_evidence(monkeypatch):
    """The measured negative case: a real board with real jobs, none of which are
    evidence for cloud usage. Must not degrade the provider or the assessment."""
    fixture = _load("ashby_zapier.json")

    def handler(request):
        if "ashbyhq.com" in str(request.url):
            return httpx.Response(200, json=fixture)
        raise AssertionError(f"unexpected call to {request.url}")

    _patch_client(monkeypatch, handler)
    result = ats.run("Zapier", ["landing page", '<a href="https://jobs.ashbyhq.com/zapier/xyz">Jobs</a>'])
    assert result.status == "ok"
    assert result.evidence == []
    assert result.calls == 1


def test_infra_job_becomes_ats_hiring_evidence(monkeypatch):
    fixture = {"jobs": [{"title": "Senior SRE", "descriptionPlain": "Own our AWS and Kubernetes infrastructure."}],
               "apiVersion": "1"}

    def handler(request):
        return httpx.Response(200, json=fixture)

    _patch_client(monkeypatch, handler)
    result = ats.run("Acme", ['<a href="https://jobs.ashbyhq.com/acme/xyz">Jobs</a>'])
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "ats_hiring"
    assert ev.strength == "STRONG"
    assert ev.freshness == "current"
    assert ev.url == "https://jobs.ashbyhq.com/acme"
    assert ev.provider == "AWS"
    assert ev.scope == "workload"


def test_malformed_json_response_degrades_provider_with_no_evidence(monkeypatch):
    """A 200 with a body that isn't valid JSON must not crash research_lead - the
    provider boundary catches Exception broadly, not only httpx.HTTPError
    (REVIEW-6bA-verified.md #2)."""
    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, text="not json{{{")

    _patch_client(monkeypatch, handler)
    result = ats.run("Acme", ['<a href="https://jobs.ashbyhq.com/acme/xyz">Jobs</a>'])
    assert result.status == "degraded"
    assert result.evidence == []


def test_404_board_is_ok_not_degraded(monkeypatch):
    def handler(request):
        return httpx.Response(404, text="Not Found")

    _patch_client(monkeypatch, handler)
    result = ats.run("Acme", ['<a href="https://jobs.lever.co/acme">Jobs</a>'])
    assert result.status == "ok"
    assert result.evidence == []


def test_network_error_degrades_provider_with_no_evidence(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(ats.httpx, "Client", boom)
    result = ats.run("Acme", ['<a href="https://jobs.lever.co/acme">Jobs</a>'])
    assert result.status == "degraded"
    assert result.evidence == []
