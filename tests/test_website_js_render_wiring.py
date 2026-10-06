"""REVIEW-6bB-verified.md #3: `maybe_render_with_crawl4ai` must actually be called
from research.py's landing-page fetch path (previously 0 references - dead code).
Offline via httpx.MockTransport for the fetch + a monkeypatched
`website.maybe_render_with_crawl4ai` standing in for the fake renderer, so this test
proves the WIRING (research.py calls it and uses its return value), not the
renderer's own internals - those are covered in test_js_render_trigger.py.
"""
import httpx

from leadscout import research
from leadscout.providers import website as website_provider


def _patch_client(monkeypatch, html: str):
    def handler(request):
        return httpx.Response(200, html=html)

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(research.httpx, "Client", fake_client)


def test_fetch_website_uses_rendered_text_when_escalation_fires(monkeypatch):
    _patch_client(monkeypatch, '<html><div id="root"></div></html>')
    monkeypatch.setattr(
        website_provider, "maybe_render_with_crawl4ai",
        lambda url, html, text: "Rendered prose that replaces the empty JS shell.",
    )
    text, ok, raw_html = research.fetch_website("https://example.com")
    assert text == "Rendered prose that replaces the empty JS shell."
    assert ok is True


def test_fetch_website_keeps_static_text_when_no_escalation(monkeypatch):
    html = "<html><body>" + ("<p>Real prose about the company. </p>" * 20) + "</body></html>"
    _patch_client(monkeypatch, html)
    monkeypatch.setattr(website_provider, "maybe_render_with_crawl4ai", lambda url, html, text: None)
    text, ok, raw_html = research.fetch_website("https://example.com")
    assert "Real prose about the company" in text
    assert ok is True
