"""Bounded same-origin crawl: link selection, priority order, robots.txt, budgets -
all offline via httpx.MockTransport."""
import httpx

from leadscout.providers import website

_LANDING_HTML = """
<html><body>
<a href="/about">About us</a>
<a href="/careers">Careers</a>
<a href="/engineering">Engineering blog</a>
<a href="/pricing">Pricing</a>
<a href="https://other-domain.example/about">Off-site about</a>
<a href="/about">About us (dup)</a>
<a href="/team">Team</a>
<a href="/jobs">Jobs</a>
</body></html>
"""


def test_select_subpages_respects_priority_and_dedupes():
    links = website.select_subpages(_LANDING_HTML, "https://example.com")
    # about (priority 0) first, then team is priority 2 but careers is priority 3 -
    # so the first 4 by priority are about, team is NOT before careers... check order
    assert links[0] == "https://example.com/about"
    assert len(links) == website.MAX_PAGES
    assert len(set(links)) == len(links)  # deduped
    assert all(website._same_origin("https://example.com", link) for link in links)


def test_select_subpages_ignores_off_site_links():
    links = website.select_subpages(_LANDING_HTML, "https://example.com")
    assert not any("other-domain.example" in link for link in links)


def test_select_subpages_empty_when_no_matching_links():
    html = '<html><body><a href="/pricing">Pricing</a><a href="/blog">Blog</a></body></html>'
    assert website.select_subpages(html, "https://example.com") == []


def test_select_subpages_no_longer_reserves_a_fixed_slot():
    """Fix #6 (v0.2 PART B): the old "reserve one slot for a trust-family link"
    hack is gone from `select_subpages` itself - a trust link that doesn't score
    into the landing page's own top MAX_PAGES stays out of THIS function's result;
    `crawl_subpages`'s frontier (sitemap discovery + keyword-hrefs found on fetched
    pages) is what now gives it a fair shot, not a static exception here."""
    html = """
    <html><body>
    <a href="/about">About</a>
    <a href="/company">Company</a>
    <a href="/team">Team</a>
    <a href="/careers">Careers</a>
    <a href="/jobs">Jobs</a>
    <a href="/engineering">Engineering</a>
    <a href="/technology">Technology</a>
    <a href="/platform">Platform</a>
    <a href="/security">Security</a>
    <a href="/trust">Trust Center</a>
    </body></html>
    """
    links = website.select_subpages(html, "https://example.com")
    assert len(links) == website.MAX_PAGES
    assert not any(link.endswith("/trust") for link in links)


def test_crawl_subpages_frontier_discovers_a_trust_link_via_a_fetched_keyword_page(monkeypatch):
    """Fix #6: the positive case for the new mechanism - a dpa link that isn't
    among the landing page's own links at all is still crawled, because a
    DISCOVERY-keyword page the crawl DID fetch (security) links to it, mirroring
    capture_universe.py's own d1->d2 rule: hrefs are only harvested for the
    frontier from a page whose OWN path is itself a keyword match."""
    landing_html = """
    <html><body>
    <a href="/about">About</a><a href="/security">Security</a>
    </body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(404)
        if path in ("/sitemap.xml", "/sitemap_index.xml"):
            return httpx.Response(404)
        if path == "/security":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text='<html><body><p>Security</p><a href="/dpa">DPA</a></body></html>')
        if path == "/dpa":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text="<html><body><p>We host on AWS.</p></body></html>")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", landing_html)
    assert any(r["url"].endswith("/dpa") for r in results)


def test_select_subpages_no_reservation_needed_when_trust_link_already_ranks_in():
    html = '<html><body><a href="/trust">Trust</a><a href="/about">About</a></body></html>'
    links = website.select_subpages(html, "https://example.com")
    assert any(link.endswith("/trust") for link in links)
    assert len(links) == 2


def _patch_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(website.httpx, "Client", fake_client)


def test_crawl_subpages_fetches_and_extracts_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/about":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text="<html><body><p>We are a great company.</p></body></html>")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", '<a href="/about">About</a>')
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/about"
    assert "great company" in results[0]["text"]
    assert results[0]["strength"] == "MEDIUM"


def test_extract_hrefs_returns_off_domain_links_too():
    """Item 1: extract_hrefs (unlike select_subpages's same-origin filter) must
    keep off-domain links - an ATS board or a GitHub org is never same-origin with
    the company's own site."""
    html = ('<html><body><a href="https://jobs.ashbyhq.com/posthog">Careers</a>'
            '<a href="https://github.com/zapier">GitHub</a>'
            '<a href="mailto:hi@example.com">Email</a></body></html>')
    hrefs = website.extract_hrefs(html, "https://example.com")
    assert "https://jobs.ashbyhq.com/posthog" in hrefs
    assert "https://github.com/zapier" in hrefs
    assert not any(h.startswith("mailto:") for h in hrefs)


def test_crawl_subpages_captures_hrefs_from_a_footer_anchor_with_no_visible_url_text(monkeypatch):
    """Item 1 (the reference fixture case): a careers subpage whose
    footer anchor points to an Ashby board with no visible URL text anywhere in
    the page - trafilatura's extracted `text` says only "Careers"/"Join our team",
    never the URL. `crawl_subpages` must still surface the href so
    providers/ats.py can find the board link."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/careers":
            return httpx.Response(200, headers={"content-type": "text/html"}, text=(
                '<html><body><p>Join our team</p>'
                '<footer><a href="https://jobs.ashbyhq.com/posthog">Careers</a></footer>'
                '</body></html>'))
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", '<a href="/careers">Careers</a>')
    assert len(results) == 1
    assert "ashbyhq.com" not in results[0]["text"]  # confirms the text alone hides it
    assert "https://jobs.ashbyhq.com/posthog" in results[0]["hrefs"]


def test_crawl_subpages_strong_when_hiring_page_names_cloud_tech(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/careers":
            return httpx.Response(200, headers={"content-type": "text/html"},
                                  text="<html><body><p>Join our SRE team running Kubernetes on AWS.</p></body></html>")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", '<a href="/careers">Careers</a>')
    assert results[0]["strength"] == "STRONG"


def test_crawl_subpages_respects_robots_disallow(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /careers\n")
        if request.url.path == "/careers":
            raise AssertionError("must not fetch a disallowed path")
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>ok</p>")

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", '<a href="/careers">Careers</a>')
    assert results == []


def test_crawl_subpages_skips_non_html_content_type(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/about":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")
        return httpx.Response(404)

    _patch_client(monkeypatch, handler)
    results = website.crawl_subpages("https://example.com", '<a href="/about">About</a>')
    assert results == []


def test_crawl_subpages_no_links_fetches_no_pages(monkeypatch):
    """Fix #6: robots.txt/sitemap discovery still runs (it's origin-based, not
    landing-href-based), but with no matching candidate from ANY of the three
    sources, no page is actually fetched."""
    requested_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path in ("/robots.txt", "/sitemap.xml", "/sitemap_index.xml"):
            return httpx.Response(404)
        requested_pages.append(request.url.path)
        raise AssertionError(f"should not fetch a page when there are no matching links: {request.url}")

    _patch_client(monkeypatch, handler)
    assert website.crawl_subpages("https://example.com", "<html><body>no links</body></html>") == []
    assert requested_pages == []


def test_crawl_subpages_never_raises_on_network_error(monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(website.httpx, "Client", boom)
    assert website.crawl_subpages("https://example.com", '<a href="/about">About</a>') == []
