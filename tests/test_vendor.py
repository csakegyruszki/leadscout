"""vendor provider: a Brave search hit is discovery only, never evidence by itself -
offline via httpx.MockTransport for both the Brave call and the page fetch; the
fixtures are synthetic.
"""
import httpx

from leadscout.providers import vendor


class _FakeSettingsWithKey:
    brave_api_key = "test-key"
    http_timeout = 5.0


class _FakeSettingsNoKey:
    brave_api_key = ""
    http_timeout = 5.0


def _patch_client(monkeypatch, handler, *, has_key=True):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(vendor.httpx, "Client", fake_client)
    monkeypatch.setattr(vendor, "settings", _FakeSettingsWithKey() if has_key else _FakeSettingsNoKey())


def _brave_response(url: str) -> httpx.Response:
    return httpx.Response(200, json={"web": {"results": [{"url": url, "title": "Acme case study"}]}})


def test_no_key_is_skipped(monkeypatch):
    monkeypatch.setattr(vendor, "settings", _FakeSettingsNoKey())
    result = vendor.run("Acme", "https://acme.com")
    assert result.status == "skipped"
    assert result.evidence == []


_LONG_FILLER = "<p>Acme is a customer case study about cloud infrastructure and scale.</p>" * 8
_ACME_URL = "https://acme.com"


def _acme_page(body: str) -> str:
    """A single-token company name ("Acme") requires the lead's own registrable
    domain (acme.com) somewhere on the page, plus a title/H1 that is actually
    about Acme - not just a body mention."""
    return f"<html><head><title>Acme Case Study | AWS</title></head><body>{body} acme.com{_LONG_FILLER}</body></html>"


def test_allowlisted_hit_with_company_name_is_direct(monkeypatch):
    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/acme")
        return httpx.Response(200, html=_acme_page("Acme runs its platform on AWS. Published: January 5, 2024"))

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "vendor_case_study"
    assert ev.strength == "DIRECT"
    assert ev.provider == "AWS"
    assert ev.freshness == "current"


def test_redirect_off_allowlist_is_ignored(monkeypatch):
    """The Brave result URL is allowlisted, but if the fetch redirects to a host that
    is NOT allowlisted, the final host must be re-validated - the search-result host
    alone is not enough (a vanity/short link could point anywhere)."""

    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/acme")
        if str(request.url) == "https://aws.amazon.com/solutions/case-studies/acme":
            return httpx.Response(302, headers={"location": "https://some-reseller-blog.example/acme"})
        return httpx.Response(200, html=_acme_page("Acme runs on AWS."))

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert result.status == "ok"
    assert result.evidence == []


def test_short_page_is_ignored(monkeypatch):
    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/acme")
        return httpx.Response(200, html="<html>Acme on AWS.</html>")

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert result.status == "ok"
    assert result.evidence == []


def test_non_allowlisted_host_is_ignored(monkeypatch):
    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://some-reseller-blog.example/aws-case-studies/acme")
        raise AssertionError("must not fetch a non-allowlisted host")

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert result.status == "ok"
    assert result.evidence == []


def test_allowlisted_page_without_company_name_is_ignored(monkeypatch):
    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/other-co")
        return httpx.Response(200, text="<html>Some unrelated company runs on AWS.</html>")

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert result.status == "ok"
    assert result.evidence == []


def test_four_year_old_date_is_historical(monkeypatch):
    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/acme")
        return httpx.Response(
            200,
            html=_acme_page('Acme runs on AWS. <time datetime="2020-01-01">2020-01-01</time>'),
        )

    _patch_client(monkeypatch, handler)
    result = vendor.run("Acme", _ACME_URL)
    assert len(result.evidence) == 1
    assert result.evidence[0].freshness == "historical"


# --- Regression tests for the three real false positives in commit 020c40e's
# out/results/*.json (verified against a live run) ---

def test_snapp_does_not_substring_match_snappet(monkeypatch):
    """Lead "Snapp" (snapp.ir) must not match a "Snappet" case study - "snapp" is a
    substring of "snappet" but not a whole word."""

    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/snappet/")
        return httpx.Response(
            200,
            html=(
                "<html><head><title>Snappet Case Study | AWS</title></head><body>"
                "Snappet is an educational technology company running its platform on AWS. "
                "Snappet uses AWS for scale." + _LONG_FILLER + "</body></html>"
            ),
        )

    _patch_client(monkeypatch, handler)
    result = vendor.run("Snapp", "https://snapp.ir")
    assert result.status == "ok"
    assert result.evidence == []


def test_snapp_does_not_match_unrelated_customer_wall_hit(monkeypatch):
    """A customer-wall page (real subject "PPC Samurai") that happens to list many
    company names, including "Snapp" as plain text among the logos, must be
    rejected because the title/H1 is not about Snapp and the page never mentions
    snapp.ir."""

    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://cloud.google.com/customers/ppc-samurai")
        return httpx.Response(
            200,
            html=(
                "<html><head><title>PPC Samurai | Google Cloud</title></head><body>"
                "PPC Samurai runs on Google Cloud. Other customers include Snapp, Acme, "
                "and dozens of other logos on our customer wall." + _LONG_FILLER + "</body></html>"
            ),
        )

    _patch_client(monkeypatch, handler)
    result = vendor.run("Snapp", "https://snapp.ir")
    assert result.status == "ok"
    assert result.evidence == []


def test_posthog_does_not_match_smartproxy_customer_page(monkeypatch):
    """Lead "PostHog" must not match a Google Cloud customer page whose real
    subject is "Decodo" (formerly Smartproxy) even if PostHog is named somewhere
    on the page (e.g. a "related customers" rail)."""

    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://cloud.google.com/customers/smartproxy")
        return httpx.Response(
            200,
            html=(
                "<html><head><title>Decodo (formerly Smartproxy) | Google Cloud</title></head><body>"
                "Decodo, formerly known as Smartproxy, scales its proxy network on Google Cloud. "
                "See also: PostHog, another Google Cloud customer." + _LONG_FILLER + "</body></html>"
            ),
        )

    _patch_client(monkeypatch, handler)
    result = vendor.run("PostHog", "https://posthog.com")
    assert result.status == "ok"
    assert result.evidence == []


def test_zapier_case_study_with_matching_title_and_domain_is_accepted(monkeypatch):
    """The true-positive control: a real case study whose title, body, and domain
    all name Zapier must still be accepted."""

    def handler(request):
        if "api.search.brave.com" in str(request.url):
            return _brave_response("https://aws.amazon.com/solutions/case-studies/zapier")
        return httpx.Response(
            200,
            html=(
                "<html><head><title>Zapier Case Study | AWS</title></head><body>"
                "Zapier runs its automation platform on AWS. Learn more at zapier.com."
                + _LONG_FILLER + "</body></html>"
            ),
        )

    _patch_client(monkeypatch, handler)
    result = vendor.run("Zapier", "https://zapier.com")
    assert result.status == "ok"
    assert len(result.evidence) == 1
    ev = result.evidence[0]
    assert ev.family == "vendor_case_study"
    assert ev.strength == "DIRECT"
    assert ev.provider == "AWS"
